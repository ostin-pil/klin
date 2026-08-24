import io

import pytest

from klin import cli, ledger, manifest, secrets

from conftest import record


class Fake(object):
    """An in-memory credential store.

    No test touches a real vault on any platform. `max_chars` mirrors the
    Windows blob limit so the rejection path is exercised everywhere, not only
    on the one operating system that enforces it.
    """

    def __init__(self, max_chars=None, broken=False):
        self.values = {}
        self.max_chars = max_chars
        self.broken = broken
        self.backend_name = "tests.Fake"

    def _check(self):
        if self.broken:
            raise secrets.SecretError("the credential store is unavailable (test)")

    def get(self, name):
        self._check()
        return self.values.get(name)

    def set(self, name, value):
        self._check()
        self.values[name] = value

    def delete(self, name):
        self._check()
        return self.values.pop(name, None) is not None


@pytest.fixture
def store(monkeypatch):
    def build(**kwargs):
        made = Fake(**kwargs)
        monkeypatch.setattr(secrets, "_store", lambda: made)
        monkeypatch.setattr(
            secrets,
            "backend_report",
            lambda: {"backend": made.backend_name, "recommended": True, "note": ""},
        )
        return made

    return build


def run(made, argv, stream=None):
    stream = stream if stream is not None else io.StringIO()
    code = cli.main(["--repo", made.root] + argv, stream=stream)
    return code, stream.getvalue()


# --- names and references ---------------------------------------------------


def test_a_name_becomes_an_environment_variable():
    assert secrets.env_name("civitai") == "KLIN_SECRET_CIVITAI"
    assert secrets.env_name("hugging-face") == "KLIN_SECRET_HUGGING_FACE"


def test_the_index_name_is_reserved():
    with pytest.raises(secrets.SecretError):
        secrets.normalise(secrets.INDEX)


@pytest.mark.parametrize("bad", ["", "Has Space", "-leading", "üñî"])
def test_an_unusable_name_is_refused(bad):
    with pytest.raises(secrets.SecretError):
        secrets.normalise(bad)


def test_a_reference_is_validated_without_being_resolved():
    assert secrets.ref_scheme("keepassxc://klin/huggingface") == "keepassxc"
    with pytest.raises(secrets.SecretError):
        secrets.ref_scheme("just-a-string")


# --- lookup order -----------------------------------------------------------


def test_the_environment_wins_over_the_store(store, monkeypatch):
    made = store()
    made.set("civitai", "from-the-store")
    monkeypatch.setenv("KLIN_SECRET_CIVITAI", "from-the-environment")
    assert secrets.lookup("civitai") == ("env", "from-the-environment")


def test_a_declared_alias_is_honoured(store, monkeypatch):
    store()
    monkeypatch.setenv("HF_TOKEN", "already-here")
    source, value = secrets.lookup("huggingface", {"env": "HF_TOKEN"})
    assert (source, value) == ("env:HF_TOKEN", "already-here")


def test_the_store_is_the_last_resort(store):
    made = store()
    made.set("civitai", "stored")
    assert secrets.lookup("civitai") == ("store", "stored")


def test_backend_env_never_reaches_the_store(store, monkeypatch):
    made = store(broken=True)
    monkeypatch.setenv("KLIN_SECRET_BACKEND", "env")
    assert secrets.lookup("civitai") == (None, None)
    assert made.values == {}


def test_a_declared_reference_says_what_to_do_instead(store):
    store()
    with pytest.raises(secrets.SecretError) as caught:
        secrets.lookup("huggingface", {"ref": "keepassxc://klin/hf"})
    assert "klin secret set huggingface" in str(caught.value)


def test_a_missing_value_names_both_ways_to_supply_it(store):
    store()
    with pytest.raises(secrets.SecretError) as caught:
        secrets.resolve("civitai")
    assert "klin secret set civitai" in str(caught.value)
    assert "KLIN_SECRET_CIVITAI" in str(caught.value)


def test_an_unavailable_store_is_an_error_not_a_traceback(store):
    store(broken=True)
    with pytest.raises(secrets.SecretError):
        secrets.resolve("civitai")


# --- storing ----------------------------------------------------------------


def test_an_oversize_value_is_refused_before_the_store_sees_it(store):
    made = store(max_chars=1280)
    with pytest.raises(secrets.SecretError) as caught:
        secrets.store_secret("civitai", "x" * 1281)
    assert "1281" in str(caught.value)
    assert made.values == {}


def test_an_empty_value_is_refused(store):
    store()
    with pytest.raises(secrets.SecretError):
        secrets.store_secret("civitai", "")


def test_stored_names_come_from_the_index(store):
    store()
    secrets.store_secret("civitai", "one")
    secrets.store_secret("huggingface", "two")
    assert secrets.stored_names() == ["civitai", "huggingface"]
    secrets.delete_secret("civitai")
    assert secrets.stored_names() == ["huggingface"]


def test_the_index_is_not_itself_listable_as_a_secret(store):
    made = store()
    secrets.store_secret("civitai", "one")
    assert secrets.INDEX in made.values
    assert secrets.INDEX not in secrets.stored_names()


# --- keeping secrets out of the tree ----------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        (
            "https://example.invalid/model.glb?token=abc123&version=4",
            "https://example.invalid/model.glb?version=4",
        ),
        (
            "https://example.invalid/f.glb?X-Amz-Signature=deadbeef&X-Amz-Expires=900",
            "https://example.invalid/f.glb",
        ),
        ("https://example.invalid/model.glb?version=4", None),
        ("https://example.invalid/model.glb", None),
        (None, None),
    ],
)
def test_credential_shaped_parameters_are_stripped_from_urls(url, expected):
    assert secrets.scrub_url(url) == (url if expected is None else expected)


def test_a_tokenised_url_does_not_survive_a_ledger_round_trip(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.add(
        path,
        record(
            "downloaded",
            source={"url": "https://example.invalid/a.glb?api_key=secret&id=7"},
        ),
    )
    stored = ledger.load(path)[0]["source"]["url"]
    assert "secret" not in stored
    assert "id=7" in stored


def test_scrub_replaces_known_values():
    assert secrets.scrub("bearer hunter2hunter2", ["hunter2hunter2"]) == (
        "bearer %s" % secrets.REDACTED
    )
    assert secrets.scrub("nothing here", ["hunter2hunter2"]) == "nothing here"


# --- the manifest block -----------------------------------------------------


def test_a_manifest_without_secrets_is_fine():
    assert manifest.secrets({"product_name": "Fixture"}) == {}


def test_a_declared_secret_keeps_its_shape():
    block = {"secrets": {"huggingface": {"env": "HF_TOKEN", "description": "read"}}}
    assert manifest.secrets(block)["huggingface"]["env"] == "HF_TOKEN"


@pytest.mark.parametrize(
    "block",
    [
        {"secrets": ["civitai"]},
        {"secrets": {"civitai": "KLIN_SECRET_CIVITAI"}},
        {"secrets": {"civitai": {"value": "never-do-this"}}},
        {"secrets": {"civitai": {"env": 7}}},
    ],
)
def test_a_malformed_secrets_block_is_refused(block):
    with pytest.raises(manifest.ManifestError):
        manifest.secrets(block)


# --- the command line -------------------------------------------------------


def test_list_shows_a_declared_secret_without_its_value(repo, store):
    made = repo(secrets_block=True)
    backing = store()
    backing.set("huggingface", "a-real-token")
    code, out = run(made, ["secret", "list"])
    assert code == 0
    assert "huggingface" in out
    assert "a-real-token" not in out


def test_list_works_without_a_secrets_block(repo, store):
    made = repo()
    store()
    secrets.store_secret("civitai", "a-real-token")
    code, out = run(made, ["secret", "list"])
    assert code == 0
    assert "civitai" in out
    assert "a-real-token" not in out


def test_get_refuses_a_terminal(repo, store):
    made = repo(secrets_block=True)
    backing = store()
    backing.set("huggingface", "a-real-token")

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    code, out = run(made, ["secret", "get", "huggingface"], stream=Terminal())
    assert code == 2
    assert "a-real-token" not in out
    assert "--reveal" in out


def test_get_pipes_cleanly(repo, store):
    made = repo(secrets_block=True)
    backing = store()
    backing.set("huggingface", "a-real-token")
    code, out = run(made, ["secret", "get", "huggingface"])
    assert code == 0
    assert out.strip() == "a-real-token"


def test_set_reads_from_stdin_and_never_from_argv(repo, store, monkeypatch):
    made = repo(secrets_block=True)
    backing = store()
    monkeypatch.setattr("sys.stdin", io.StringIO("piped-token\n"))
    code, out = run(made, ["secret", "set", "huggingface"])
    assert code == 0
    assert backing.values["huggingface"] == "piped-token"
    assert "piped-token" not in out


def test_set_warns_when_the_environment_will_shadow_the_store(
    repo, store, monkeypatch
):
    made = repo(secrets_block=True)
    store()
    monkeypatch.setattr("sys.stdin", io.StringIO("piped-token\n"))
    monkeypatch.setenv("HF_TOKEN", "the-one-that-wins")
    code, out = run(made, ["secret", "set", "huggingface"])
    assert code == 0
    assert "takes precedence" in out
    assert "the-one-that-wins" not in out


def test_rm_reports_what_it_could_not_unset(repo, store, monkeypatch):
    made = repo(secrets_block=True)
    backing = store()
    backing.set("huggingface", "stored")
    monkeypatch.setenv("HF_TOKEN", "from-the-environment")
    code, out = run(made, ["secret", "rm", "huggingface"])
    assert code == 0
    assert "removed huggingface" in out
    assert "cannot unset" in out
    assert "from-the-environment" not in out


def test_doctor_fails_while_a_declared_secret_is_unset(repo, store):
    made = repo(secrets_block=True)
    store()
    code, out = run(made, ["secret", "doctor"])
    assert code == 1
    assert "unresolved" in out


def test_doctor_passes_once_everything_resolves(repo, store):
    made = repo(secrets_block=True)
    backing = store()
    backing.set("huggingface", "a-real-token")
    backing.set("civitai", "another-real-token")
    code, out = run(made, ["secret", "doctor"])
    assert code == 0
    assert "a-real-token" not in out
    assert "another-real-token" not in out


def test_set_succeeds_even_when_the_manifest_declares_a_reference(
    repo, store, monkeypatch
):
    """A reference klin has no resolver for must not turn a stored value into a
    failed command. The store is exactly the fallback the error recommends."""
    made = repo(secrets_block=True)
    backing = store()
    manifest_path = made.manifest
    text = io.open(manifest_path, encoding="utf-8").read().replace(
        "    env: HF_TOKEN", "    env: HF_TOKEN\n    ref: keepassxc://klin/hf"
    )
    io.open(manifest_path, "w", encoding="utf-8", newline="\n").write(text)
    monkeypatch.setattr("sys.stdin", io.StringIO("piped-token\n"))
    code, out = run(made, ["secret", "set", "huggingface"])
    assert code == 0
    assert backing.values["huggingface"] == "piped-token"
    assert "piped-token" not in out


def test_doctor_never_prints_a_value(repo, store, monkeypatch):
    made = repo(secrets_block=True)
    store()
    monkeypatch.setenv("KLIN_SECRET_CIVITAI", "a-real-token")
    monkeypatch.setenv("HF_TOKEN", "another-real-token")
    code, out = run(made, ["secret", "doctor"])
    assert "a-real-token" not in out
    assert "another-real-token" not in out
