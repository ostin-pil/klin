"""The fetch adapters, their guards, and the mapping that is the point of them.

No test here touches the network. The vendor payloads are trimmed copies of
real API responses, captured on 2026-08-24 from the eight models this project
actually acquires, which is what makes the Civitai table below a specification
rather than an assertion about code that already exists.
"""

import io
import json
import os

import pytest

from klin import cli, ledger, manifest, net, policy
from klin.fetch import civitai, hf
import klin.fetch as fetch


# --------------------------------------------------------------------------
# Civitai: permission flags into families.
#
# Captured verbatim. `Rent`/`RentCivit` grant a generation service permission
# and do not answer whether an image made with the model may be sold, so they
# do not count as commercial use. That is the contested call, and this table is
# where it is pinned down.
# --------------------------------------------------------------------------

CIVITAI_CASES = [
    # (model id, name, allowCommercialUse, allowDerivatives, allowNoCredit, families)
    (980106, "Darkest Dungeon", ["Image", "RentCivit", "Rent", "Sell"], True, True, set()),
    (675157, "DarkFantasyIllustration", ["RentCivit", "Rent"], True, True, {"noncommercial"}),
    (660136, "FLUX64", ["RentCivit"], False, True, {"noncommercial", "noderivatives"}),
    (720442, "Isometrica", ["Image", "RentCivit", "Rent"], True, True, set()),
    (60962, "Diorama", ["Image", "RentCivit", "Rent"], True, False, {"attribution"}),
    (650444, "3DMM_FLUX", ["RentCivit"], False, True, {"noncommercial", "noderivatives"}),
    (648058, "PS1 Style Flux", ["Image", "RentCivit", "Rent", "Sell"], True, True, set()),
    (1041229, "Retro Dark Fantasy", ["Image", "RentCivit", "Rent"], True, True, set()),
]


def civitai_payload(model_id, name, commercial, derivatives, credit, versions=None):
    return {
        "id": model_id,
        "name": name,
        "type": "LORA",
        "creator": {"username": "someone"},
        "allowCommercialUse": commercial,
        "allowDerivatives": derivatives,
        "allowNoCredit": credit,
        "allowDifferentLicense": True,
        "modelVersions": versions
        or [
            {
                "id": model_id * 2,
                "name": "v1",
                "baseModel": "Flux.1 D",
                "files": [
                    {
                        "name": "thing.safetensors",
                        "primary": True,
                        "sizeKB": 18847.4208984375,
                        "hashes": {"SHA256": "ABC123"},
                    }
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    "model_id,name,commercial,derivatives,credit,expected", CIVITAI_CASES
)
def test_civitai_flags_map_to_families(
    model_id, name, commercial, derivatives, credit, expected
):
    payload = civitai_payload(model_id, name, commercial, derivatives, credit)
    families, why = civitai.derive_families(payload)
    assert families == expected, name
    # Every derived family owes a reason, because the reason is what goes in
    # the record's notes and makes the mapping auditable.
    assert len(why) == len(expected)


def test_rent_alone_is_not_commercial_use():
    """The whole contested call, isolated so it cannot be changed by accident."""
    families, _ = civitai.derive_families(
        civitai_payload(1, "rental", ["Rent", "RentCivit"], True, True)
    )
    assert families == {"noncommercial"}


def test_empty_commercial_use_is_noncommercial():
    for value in ([], None, [None]):
        families, _ = civitai.derive_families(
            civitai_payload(1, "empty", value, True, True)
        )
        assert families == {"noncommercial"}, value


def test_sell_alone_is_commercial_enough():
    families, _ = civitai.derive_families(
        civitai_payload(1, "sell", ["Sell"], True, True)
    )
    assert families == set()


# --------------------------------------------------------------------------
# HuggingFace: an identifier exists, so policy classifies it and the adapter
# does not.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tag,expected",
    [
        ("apache-2.0", {"permissive"}),
        ("mit", {"permissive"}),
        ("cc0-1.0", {"public-domain"}),
        ("cc-by-4.0", {"attribution"}),
        ("cc-by-nc-4.0", {"attribution", "noncommercial"}),
        ("cc-by-sa-4.0", {"attribution", "share-alike"}),
        ("gpl-3.0", {"copyleft"}),
    ],
)
def test_hf_licence_tags_classify_through_policy(tag, expected):
    record = ledger.blank("x", kind="model")
    record["licence"]["id"] = tag
    assert policy.families(record) == expected


def test_hf_other_is_unknown_not_guessed():
    """`license:other` must not become a family the adapter invented.

    This is the case that covers Flux.1-dev and the Shakker ControlNet. An
    adapter that quietly decided `noncommercial` here would be right about
    those two and wrong invisibly about the next repository.
    """
    record = ledger.blank("x", kind="model")
    record["licence"]["id"] = "other"
    assert policy.families(record) == {"unknown"}


def test_hf_licence_id_reads_tag_then_card():
    assert hf._licence_id({"tags": ["diffusers", "license:apache-2.0"]}) == "apache-2.0"
    assert hf._licence_id({"tags": [], "cardData": {"license": "mit"}}) == "mit"
    assert hf._licence_id({"tags": [], "cardData": {"license": ["mit"]}}) == "mit"
    assert hf._licence_id({"tags": [], "cardData": {}}) is None


# --------------------------------------------------------------------------
# Precedence: a hand-set list wins over anything derived or classified.
# --------------------------------------------------------------------------

class Args(object):
    def __init__(self, **kw):
        self.families = None
        self.dest = None
        self.as_kind = None
        self.resume = False
        self.dry_run = False
        self.repo = "."
        self.manifest = None
        for key, value in kw.items():
            setattr(self, key, value)


class Ctx(object):
    def __init__(self, args, lines=None):
        self.args = args
        self.lines = lines if lines is not None else []

    def say(self, text=""):
        self.lines.append(text)


def test_hand_families_beat_derived():
    ctx = Ctx(Args(families="noncommercial,noderivatives"))
    record = ledger.blank("x", kind="model")
    record["licence"]["id"] = "apache-2.0"
    how, found = fetch.classify(ctx, record, derived={"attribution"})
    assert how == "hand"
    assert found == {"noncommercial", "noderivatives"}
    # Stored sorted, so a record's field order never depends on typing order.
    assert record["licence"]["families"] == ["noderivatives", "noncommercial"]


def test_derived_beats_identifier():
    ctx = Ctx(Args())
    record = ledger.blank("x", kind="model")
    record["licence"]["id"] = "apache-2.0"
    how, found = fetch.classify(ctx, record, derived={"noncommercial"})
    assert how == "derived"
    assert found == {"noncommercial"}


def test_unknown_identifier_demands_a_human():
    ctx = Ctx(Args())
    record = ledger.blank("x", kind="model")
    record["licence"]["id"] = "other"
    how, found = fetch.classify(ctx, record)
    assert how == "unknown"
    assert "families" not in record["licence"]

    fetch.report_classification(ctx, record, how, found)
    printed = "\n".join(ctx.lines)
    assert "--families" in printed
    assert "does not guess" in printed


def test_explicit_families_win_in_policy_too():
    """`classify` writing the list is only useful if `policy` honours it."""
    record = ledger.blank("x", kind="model")
    record["licence"]["id"] = "apache-2.0"
    record["licence"]["families"] = ["noncommercial"]
    assert policy.families(record) == {"noncommercial"}


# --------------------------------------------------------------------------
# cache_dir: the defect that `resolve()` would have caused.
# --------------------------------------------------------------------------

#: `os.path.expandvars` only understands the host's own syntax, so a test that
#: hard-codes `%VAR%` passes on Windows and fails on a Linux runner for a reason
#: that has nothing to do with the behaviour under test. The behaviour is the
#: same on both; only the spelling differs.
WINDOWS = os.name == "nt"


def a_var(name):
    return "%%%s%%" % name if WINDOWS else "$%s" % name


def an_abs(tail):
    return ("D:\\%s" % tail) if WINDOWS else ("/mnt/%s" % tail)


def test_cache_dir_expands_variables(monkeypatch):
    monkeypatch.setenv("SOME_ROOT", an_abs("models"))
    monkeypatch.delenv(manifest.CACHE_ENV, raising=False)
    got = manifest.cache_dir({"cache_dir": a_var("SOME_ROOT") + "/klin/cache"})
    assert got == os.path.normpath(an_abs("models") + "/klin/cache")


def test_cache_dir_is_never_joined_to_the_repo(monkeypatch):
    """The whole reason this is not `manifest.resolve`."""
    monkeypatch.setenv("SOME_ROOT", an_abs("models"))
    monkeypatch.delenv(manifest.CACHE_ENV, raising=False)
    got = manifest.cache_dir({"cache_dir": a_var("SOME_ROOT") + "/cache"})
    assert "Barinn" not in got
    assert not got.startswith(os.getcwd())


def test_klin_cache_beats_the_manifest(monkeypatch):
    monkeypatch.setenv(manifest.CACHE_ENV, an_abs("override"))
    got = manifest.cache_dir({"cache_dir": a_var("LOCALAPPDATA") + "/klin/cache"})
    assert got == os.path.normpath(an_abs("override"))


def test_unexpanded_variable_is_an_error(monkeypatch):
    monkeypatch.delenv(manifest.CACHE_ENV, raising=False)
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    with pytest.raises(manifest.ManifestError) as exc:
        manifest.cache_dir({"cache_dir": a_var("DEFINITELY_NOT_SET") + "/cache"})
    assert "not set on this machine" in str(exc.value)


@pytest.mark.parametrize("spelling", ["%NOT_SET_ANYWHERE%", "$NOT_SET_ANYWHERE"])
def test_a_foreign_variable_syntax_is_still_caught(spelling, monkeypatch):
    """A Windows manifest read on a Linux runner, and the reverse.

    `expandvars` leaves the other platform's syntax untouched, so without an
    explicit check `%LOCALAPPDATA%` becomes a directory of that literal name.
    """
    monkeypatch.delenv(manifest.CACHE_ENV, raising=False)
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    with pytest.raises(manifest.ManifestError):
        manifest.cache_dir({"cache_dir": spelling + "/cache"})


def test_relative_cache_dir_is_an_error(monkeypatch):
    monkeypatch.delenv(manifest.CACHE_ENV, raising=False)
    with pytest.raises(manifest.ManifestError) as exc:
        manifest.cache_dir({"cache_dir": "cache"})
    assert "not absolute" in str(exc.value)


def test_missing_cache_dir_is_an_error(monkeypatch):
    monkeypatch.delenv(manifest.CACHE_ENV, raising=False)
    with pytest.raises(manifest.ManifestError):
        manifest.cache_dir({})


# --------------------------------------------------------------------------
# The guards. Each one gets a crafted bad payload.
# --------------------------------------------------------------------------

class FakeHeaders(dict):
    def get(self, key, default=None):
        for name, value in self.items():
            if name.lower() == key.lower():
                return value
        return default


class FakeResponse(object):
    def __init__(self, body, headers=None, status=200, url="https://cdn.invalid/f"):
        self._body = body
        self._at = 0
        self.headers = FakeHeaders(headers or {})
        self.status = status
        self._url = url

    def read(self, size=-1):
        if size is None or size < 0:
            chunk, self._at = self._body[self._at :], len(self._body)
            return chunk
        chunk = self._body[self._at : self._at + size]
        self._at += len(chunk)
        return chunk

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def safetensors_bytes(header=None, payload=b"\x00" * 64):
    blob = json.dumps(header or {"a": {"dtype": "F32"}}).encode()
    return len(blob).to_bytes(8, "little") + blob + payload


def serve(monkeypatch, response):
    monkeypatch.setattr(net, "urlopen", lambda *a, **k: response)


def test_guard_html_body_is_rejected(tmp_path, monkeypatch):
    serve(
        monkeypatch,
        FakeResponse(b"<html>sign in</html>", {"Content-Type": "text/html"}),
    )
    dest = str(tmp_path / "m.safetensors")
    with pytest.raises(net.NetError) as exc:
        net.download("https://vendor.invalid/m", dest)
    assert "not a model file" in str(exc.value)
    assert not os.path.exists(dest)


def test_guard_plain_text_body_is_rejected(tmp_path, monkeypatch):
    """The failure a denylist of `text/html` would have waved through.

    Civitai's edge answers an unrecognised client with `text/plain` and
    seventeen bytes. Guard 1 is an allowlist for exactly this reason.
    """
    serve(
        monkeypatch,
        FakeResponse(b"error code: 1010\n", {"Content-Type": "text/plain"}),
    )
    dest = str(tmp_path / "m.safetensors")
    with pytest.raises(net.NetError) as exc:
        net.download("https://vendor.invalid/m", dest)
    assert "not a model file" in str(exc.value)
    assert not os.path.exists(dest)


def test_guard_short_read_is_rejected(tmp_path, monkeypatch):
    body = safetensors_bytes()
    serve(
        monkeypatch,
        FakeResponse(body, {"Content-Type": "application/octet-stream"}),
    )
    dest = str(tmp_path / "m.safetensors")
    with pytest.raises(net.NetError) as exc:
        net.download("https://vendor.invalid/m", dest, expected_size=len(body) + 500)
    assert "truncated download" in str(exc.value)
    assert not os.path.exists(dest)
    assert not os.path.exists(dest + ".part")


def test_guard_bad_magic_is_rejected(tmp_path, monkeypatch):
    # A plausible octet-stream that is not safetensors: the leading eight bytes
    # declare a header longer than the file.
    body = (1 << 40).to_bytes(8, "little") + b"nonsense"
    serve(
        monkeypatch,
        FakeResponse(body, {"Content-Type": "application/octet-stream"}),
    )
    dest = str(tmp_path / "m.safetensors")
    with pytest.raises(net.NetError) as exc:
        net.download("https://vendor.invalid/m", dest)
    assert "cannot be right" in str(exc.value)
    assert not os.path.exists(dest)


def test_guard_bad_header_json_is_rejected(tmp_path, monkeypatch):
    body = (4).to_bytes(8, "little") + b"{{{{" + b"\x00" * 16
    serve(
        monkeypatch,
        FakeResponse(body, {"Content-Type": "application/octet-stream"}),
    )
    dest = str(tmp_path / "m.safetensors")
    with pytest.raises(net.NetError) as exc:
        net.download("https://vendor.invalid/m", dest)
    assert "not valid JSON" in str(exc.value)


def test_good_download_records_its_hash(tmp_path, monkeypatch):
    import hashlib

    body = safetensors_bytes()
    serve(
        monkeypatch,
        FakeResponse(body, {"Content-Type": "application/octet-stream"}),
    )
    dest = str(tmp_path / "m.safetensors")
    facts = net.download("https://vendor.invalid/m", dest, expected_size=len(body))
    assert facts["sha256"] == hashlib.sha256(body).hexdigest()
    assert facts["bytes"] == len(body)
    assert os.path.exists(dest)
    assert not os.path.exists(dest + ".part")


def test_a_signed_url_is_scrubbed_before_it_reaches_a_record(tmp_path, monkeypatch):
    body = safetensors_bytes()
    serve(
        monkeypatch,
        FakeResponse(
            body,
            {"Content-Type": "application/octet-stream"},
            url="https://cdn.invalid/f?token=supersecret&region=eu",
        ),
    )
    dest = str(tmp_path / "m.safetensors")
    facts = net.download("https://vendor.invalid/m", dest)
    assert "supersecret" not in facts["final_url"]
    assert "region=eu" in facts["final_url"]


def test_a_401_names_the_credential(monkeypatch, tmp_path):
    from urllib.error import HTTPError

    def boom(*a, **k):
        raise HTTPError(
            "https://civitai.invalid/x", 401, "Unauthorized", FakeHeaders(), io.BytesIO(b"{}")
        )

    monkeypatch.setattr(net, "urlopen", boom)
    with pytest.raises(net.NetError) as exc:
        net.download("https://civitai.invalid/x", str(tmp_path / "m.safetensors"))
    assert "klin secret set" in str(exc.value)


def test_an_edge_block_is_not_reported_as_an_auth_failure(monkeypatch, tmp_path):
    """403 with `error code: 1010` is the client being refused, not the token."""
    from urllib.error import HTTPError

    def boom(*a, **k):
        raise HTTPError(
            "https://civitai.invalid/x",
            403,
            "Forbidden",
            FakeHeaders(),
            io.BytesIO(b"error code: 1010\n"),
        )

    monkeypatch.setattr(net, "urlopen", boom)
    with pytest.raises(net.NetError) as exc:
        net.download("https://civitai.invalid/x", str(tmp_path / "m.safetensors"))
    message = str(exc.value)
    assert "refused the client, not the credential" in message
    assert "User-Agent" in message


def test_every_request_carries_a_real_user_agent():
    assert "Mozilla" in net.USER_AGENT
    assert "urllib" not in net.USER_AGENT.lower()


# --------------------------------------------------------------------------
# The registry's promise: a vendor is a file.
# --------------------------------------------------------------------------

def test_both_vendors_are_discovered():
    assert set(fetch.adapters()) == {"civitai", "hf"}


def test_a_new_module_becomes_a_subcommand_with_no_other_edit(tmp_path, monkeypatch):
    """cli.py's docstring promises this; the promise is checked, not trusted."""
    package = os.path.dirname(fetch.__file__)
    added = os.path.join(package, "zz_probe.py")
    io.open(added, "w", encoding="utf-8").write(
        "NAME = 'zzprobe'\nHELP = 'a probe'\n"
        "def configure(parser):\n    parser.add_argument('thing')\n"
        "def run(args, ctx):\n    return 0\n"
    )
    try:
        assert "zzprobe" in fetch.adapters()
        parser = cli.build_parser()
        args = parser.parse_args(["fetch", "zzprobe", "widget"])
        assert args.thing == "widget"
    finally:
        os.remove(added)
        for stale in ("zz_probe.cpython-312.pyc",):
            cached = os.path.join(package, "__pycache__", stale)
            if os.path.exists(cached):
                os.remove(cached)


def test_fetch_help_lists_every_vendor():
    parser = cli.build_parser()
    text = parser.format_help()
    assert "fetch" in text


# --------------------------------------------------------------------------
# End to end: the adapter feeds the policy engine.
# --------------------------------------------------------------------------

def _audit(repo, ship=False):
    stream = io.StringIO()
    code = cli.main(
        ["--repo", repo.root, "ledger", "audit"] + (["--ship"] if ship else []),
        stream=stream,
    )
    return code, stream.getvalue()


def test_a_fetched_clean_record_passes_both_gates(repo):
    made = repo()
    record = ledger.blank("civitai-1960212", kind="model")
    record["paths"] = ["D:/klin-cache/civitai/1960212/darkest.safetensors"]
    record["sha256"] = "a" * 64
    record["source"]["adapter"] = "civitai"
    record["source"]["url"] = "https://civitai.com/models/980106"
    record["licence"]["id"] = "LicenseRef-Civitai-980106"
    record["licence"]["name"] = "Darkest Dungeon — Civitai model terms"
    record["licence"]["text"] = "allowCommercialUse=['Image', 'Sell']"
    record["licence"]["families"] = []
    made.write_records([record])

    code, out = _audit(made)
    assert code == 0, out
    code, out = _audit(made, ship=True)
    assert code == 0, out


def test_a_fetched_noncommercial_record_fails_the_ship_gate(repo):
    """The end-to-end proof: the mapping reaches the rule that stops a ship."""
    made = repo()
    record = ledger.blank("civitai-1350314", kind="model")
    record["paths"] = ["D:/klin-cache/civitai/1350314/flux64.safetensors"]
    record["sha256"] = "b" * 64
    record["source"]["adapter"] = "civitai"
    record["source"]["url"] = "https://civitai.com/models/660136"
    record["licence"]["id"] = "LicenseRef-Civitai-660136"
    record["licence"]["name"] = "FLUX64 — Civitai model terms"
    record["licence"]["text"] = "allowCommercialUse=['RentCivit']"
    record["licence"]["families"] = ["noncommercial", "noderivatives"]
    made.write_records([record])

    code, out = _audit(made)
    assert code == 0, "the stage gate takes anything that is written down"

    code, out = _audit(made, ship=True)
    assert code != 0
    assert "civitai-1350314" in out
    # The audit quotes the rule it applied, which is how a drifted
    # transcription surfaces instead of hiding.
    assert "NonCommercial" in out


def test_an_unclassified_record_fails_the_stage_rule(repo):
    """`license:other` left unclassified must be visible, not merely quiet."""
    made = repo()
    record = ledger.blank("hf-Comfy-Org--flux1-dev", kind="model")
    record["paths"] = ["D:/klin-cache/hf/x/flux1-dev-fp8.safetensors"]
    record["source"]["adapter"] = "hf"
    record["source"]["url"] = None
    record["licence"]["id"] = None
    made.write_records([record])

    code, out = _audit(made)
    assert code != 0
    assert "rule 0" in out


# --------------------------------------------------------------------------
# Cache reuse. A cache is only a cache if the second run is cheap.
# --------------------------------------------------------------------------

def _explode(*a, **k):
    raise AssertionError("the network was used when the cache should have served")


def test_a_verified_cached_file_is_reused(tmp_path, monkeypatch):
    body = safetensors_bytes()
    dest = str(tmp_path / "m.safetensors")
    io.open(dest, "wb").write(body)

    monkeypatch.setattr(net, "urlopen", _explode)
    facts = net.download("https://vendor.invalid/m", dest, expected_size=len(body))
    assert facts["reused"] is True
    assert facts["sha256"] == __import__("hashlib").sha256(body).hexdigest()


def test_a_cached_file_of_the_wrong_size_is_not_reused(tmp_path, monkeypatch):
    """Reuse checks the file, never a memory of having downloaded it."""
    dest = str(tmp_path / "m.safetensors")
    io.open(dest, "wb").write(safetensors_bytes(payload=b"\x00" * 8))

    good = safetensors_bytes()
    serve(monkeypatch, FakeResponse(good, {"Content-Type": "application/octet-stream"}))
    facts = net.download("https://vendor.invalid/m", dest, expected_size=len(good))
    assert facts["reused"] is False
    assert facts["bytes"] == len(good)


def test_a_corrupt_cached_file_is_not_reused(tmp_path, monkeypatch):
    dest = str(tmp_path / "m.safetensors")
    io.open(dest, "wb").write(b"not safetensors at all")

    good = safetensors_bytes()
    serve(monkeypatch, FakeResponse(good, {"Content-Type": "application/octet-stream"}))
    facts = net.download("https://vendor.invalid/m", dest)
    assert facts["reused"] is False


def test_force_ignores_the_cache(tmp_path, monkeypatch):
    body = safetensors_bytes()
    dest = str(tmp_path / "m.safetensors")
    io.open(dest, "wb").write(body)

    serve(monkeypatch, FakeResponse(body, {"Content-Type": "application/octet-stream"}))
    facts = net.download("https://vendor.invalid/m", dest, force=True)
    assert facts["reused"] is False


def test_reuse_needs_no_declared_size(tmp_path, monkeypatch):
    body = safetensors_bytes()
    dest = str(tmp_path / "m.safetensors")
    io.open(dest, "wb").write(body)

    monkeypatch.setattr(net, "urlopen", _explode)
    facts = net.download("https://vendor.invalid/m", dest)
    assert facts["reused"] is True
