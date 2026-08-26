"""The index, tested against synthesised files rather than a real corpus.

Every PNG here is built byte by byte in-process. That matters more than it
usually would: this module's whole claim is that a generated image describes
itself, so a test that leaned on a real ComfyUI output would be checking one
machine's directory rather than the format. Building the chunks by hand also
makes the awkward cases reachable — a compressed text chunk, a graph with two
samplers, a LoRA loader wired to nothing — which a corpus supplies only by
luck.
"""

import io
import json
import os
import struct
import zlib

import pytest

from klin import cli, index
from klin.index import comfy, png

from conftest import record


# ---------------------------------------------------------------- fixtures


def chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def png_bytes(width=8, height=4, text=None, ztxt=None, itxt=None):
    """A valid one-colour greyscale PNG carrying whatever chunks are asked for."""
    out = png.MAGIC + chunk(
        b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    )
    for name, value in (text or {}).items():
        out += chunk(b"tEXt", name.encode("latin-1") + b"\x00" + value)
    for name, value in (ztxt or {}).items():
        out += chunk(
            b"zTXt", name.encode("latin-1") + b"\x00\x00" + zlib.compress(value)
        )
    for name, value in (itxt or {}).items():
        out += chunk(
            b"iTXt", name.encode("latin-1") + b"\x00\x00\x00\x00\x00" + value
        )
    raw = b"".join(b"\x00" + b"\x00" * width for _ in range(height))
    return out + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def graph(seed=101, loras=(), base="flux1-dev-fp8.safetensors", extra=None):
    """A ComfyUI API-format graph: loader, optional LoRA chain, sampler, save."""
    nodes = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": base}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "a warm bar car"}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry"}},
        "20": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": 768, "height": 448},
        },
    }
    upstream = "1"
    for number, (name, strength) in enumerate(loras, start=30):
        nodes[str(number)] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": name,
                "strength_model": strength,
                "model": [upstream, 0],
            },
        }
        upstream = str(number)
    nodes["50"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": 24,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "model": [upstream, 0],
            "positive": ["10", 0],
            "negative": ["11", 0],
            "latent_image": ["20", 0],
        },
    }
    nodes["60"] = {"class_type": "VAEDecode", "inputs": {"samples": ["50", 0]}}
    nodes.update(extra or {})
    return nodes


def write_png(path, **kwargs):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "wb") as handle:
        handle.write(png_bytes(**kwargs))
    return path


def write_generated(path, **kwargs):
    return write_png(
        path, text={"prompt": json.dumps(graph(**kwargs)).encode("utf-8")}
    )


@pytest.fixture
def db(tmp_path):
    return index.connect(str(tmp_path / "index.sqlite3"))


# ------------------------------------------------------------ png framing


def test_dimensions_come_from_the_header(tmp_path):
    path = write_png(str(tmp_path / "a.png"), width=640, height=368)
    assert png.dimensions(path) == (640, 368)


def test_a_file_that_is_not_a_png_reads_as_nothing(tmp_path):
    path = str(tmp_path / "b.png")
    io.open(path, "w").write("this is not a png")
    assert png.dimensions(path) is None
    assert png.text_chunks(path) == {}


def test_compressed_text_chunks_are_read_too(tmp_path):
    path = write_png(
        str(tmp_path / "c.png"),
        ztxt={"zipped": b"hello"},
        itxt={"international": b"hi"},
    )
    got = png.text_chunks(path)
    assert got["zipped"] == b"hello"
    assert got["international"] == b"hi"


# ------------------------------------------------------- the comfy reader


def test_an_image_with_no_graph_is_not_an_error(tmp_path):
    assert comfy.read(write_png(str(tmp_path / "sheet.png"))) is None


def test_a_graph_that_does_not_parse_says_so(tmp_path):
    path = write_png(str(tmp_path / "bad.png"), text={"prompt": b"{not json"})
    with pytest.raises(comfy.ReadError):
        comfy.read(path)


def test_the_lora_stack_keeps_its_order_and_strengths(tmp_path):
    path = write_generated(
        str(tmp_path / "d.png"),
        loras=(("first.safetensors", 0.7), ("second.safetensors", 0.4)),
    )
    got = comfy.read(path)
    assert [(l["name"], l["strength"]) for l in got["loras"]] == [
        ("first.safetensors", 0.7),
        ("second.safetensors", 0.4),
    ]
    assert got["model"]["name"] == "flux1-dev-fp8.safetensors"
    assert got["seed"] == 101
    assert got["prompt"] == "a warm bar car"
    assert got["negative"] == "blurry"
    assert got["latent_size"] == [768, 448]


def test_a_lora_loader_wired_to_nothing_is_not_in_the_stack(tmp_path):
    """The reason the reader walks the chain instead of collecting by name."""
    orphan = {
        "99": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"lora_name": "unused.safetensors", "strength_model": 1.0},
        }
    }
    path = write_generated(
        str(tmp_path / "e.png"),
        loras=(("used.safetensors", 0.5),),
        extra=orphan,
    )
    names = [l["name"] for l in comfy.read(path)["loras"]]
    assert names == ["used.safetensors"]


def test_two_samplers_are_reported_rather_than_silently_picked(tmp_path):
    second = {
        "51": {
            "class_type": "KSampler",
            "inputs": {"seed": 7, "model": ["1", 0], "positive": ["10", 0]},
        }
    }
    path = write_generated(str(tmp_path / "f.png"), extra=second)
    got = comfy.read(path)
    assert any("2 sampler nodes" in note for note in got["notes"])


def test_the_workflow_hash_ignores_the_seed(tmp_path):
    one = write_generated(str(tmp_path / "g1.png"), seed=101)
    two = write_generated(str(tmp_path / "g2.png"), seed=202)
    three = write_generated(str(tmp_path / "g3.png"), seed=101, base="other.safetensors")
    assert comfy.read(one)["workflow_sha256"] == comfy.read(two)["workflow_sha256"]
    assert comfy.read(one)["workflow_sha256"] != comfy.read(three)["workflow_sha256"]


def test_an_img2img_graph_has_no_declared_latent_size(tmp_path):
    encode = {
        "21": {"class_type": "VAEEncode", "inputs": {"pixels": ["22", 0]}},
        "22": {"class_type": "LoadImage", "inputs": {"image": "in.png"}},
    }
    nodes = graph(extra=encode)
    nodes["50"]["inputs"]["latent_image"] = ["21", 0]
    path = write_png(
        str(tmp_path / "h.png"), text={"prompt": json.dumps(nodes).encode("utf-8")}
    )
    got = comfy.read(path)
    assert got["latent_size"] is None
    assert got["model"]["name"] == "flux1-dev-fp8.safetensors"


# ------------------------------------------------------------- the scanner


def test_a_scan_records_provenance_and_dimensions(db, tmp_path):
    root = str(tmp_path / "out")
    write_generated(os.path.join(root, "one.png"), loras=(("psx.safetensors", 0.6),))
    stats = index.scan(db, root, "Fixture", [])
    assert stats["added"] == 1
    assert stats["provenance"] == 1
    row = index.query(db)[0]
    assert row["model"] == "flux1-dev-fp8.safetensors"
    assert row["seed"] == 101
    assert (row["width"], row["height"]) == (8, 4)
    assert [l["name"] for l in index.loras_of(db, row["path"])] == ["psx.safetensors"]


def test_a_second_scan_skips_what_has_not_changed(db, tmp_path):
    root = str(tmp_path / "out")
    write_generated(os.path.join(root, "one.png"))
    index.scan(db, root, "Fixture", [])
    again = index.scan(db, root, "Fixture", [])
    assert again["skipped"] == 1
    assert again["added"] == 0
    forced = index.scan(db, root, "Fixture", [], rescan=True)
    assert forced["updated"] == 1


def test_claim_patterns_tag_a_project_and_leave_the_rest_visible(db, tmp_path):
    root = str(tmp_path / "out")
    write_generated(os.path.join(root, "mock", "mine.png"))
    write_generated(os.path.join(root, "elsewhere", "theirs.png"))
    stats = index.scan(db, root, "Fixture", ["mock/*"])
    assert stats["unclaimed"] == 1
    assert len(index.query(db, project="Fixture")) == 1
    assert len(index.query(db, unclaimed=True)) == 1
    assert len(index.query(db)) == 2


def test_a_second_project_claiming_the_same_file_is_a_conflict(db, tmp_path):
    root = str(tmp_path / "out")
    write_generated(os.path.join(root, "mock", "shared.png"))
    index.scan(db, root, "Fixture", ["mock/*"])
    stats = index.scan(db, root, "Other", ["mock/*"])
    assert len(stats["conflicts"]) == 1
    # The first claim stands. Overwriting silently would make the winner depend
    # on which project happened to scan last.
    assert index.query(db)[0]["project"] == "Fixture"


def test_pruning_drops_rows_whose_file_has_gone(db, tmp_path):
    root = str(tmp_path / "out")
    path = write_generated(os.path.join(root, "gone.png"))
    index.scan(db, root, "Fixture", [])
    os.remove(path)
    assert index.forget_missing(db) == [os.path.normpath(path)]
    assert index.query(db) == []


def test_a_file_over_the_hash_limit_is_indexed_without_one(db, tmp_path):
    root = str(tmp_path / "out")
    write_generated(os.path.join(root, "big.png"))
    index.scan(db, root, "Fixture", [], hash_limit=1)
    assert index.query(db)[0]["sha256"] is None
    assert index.status(db)["unhashed"] == 1


def test_filters_narrow_the_listing(db, tmp_path):
    root = str(tmp_path / "out")
    write_generated(os.path.join(root, "a.png"), seed=101, loras=(("psx.safetensors", 0.6),))
    write_generated(os.path.join(root, "b.png"), seed=202, base="other.safetensors")
    index.scan(db, root, "Fixture", [])
    assert len(index.query(db, seed=101)) == 1
    assert len(index.query(db, lora="psx")) == 1
    assert len(index.query(db, model="other")) == 1
    assert len(index.query(db, prompt="warm bar")) == 2
    assert len(index.query(db, kind="image")) == 2


def test_a_malformed_since_is_refused_rather_than_ignored(db):
    with pytest.raises(index.IndexingError):
        index.query(db, since="last tuesday")


# --------------------------------------------------- resolving to a ledger


def weights(path, body=b"weights"):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "wb") as handle:
        handle.write(body)
    return path


def test_a_hardlink_resolves_by_identity_even_when_renamed(tmp_path):
    """The case that decided the design.

    A weights tree is where files get renamed to something readable, so the
    name in the tree and the name in the ledger routinely differ. `--as` links
    rather than copies, so identity still ties them together.
    """
    cache = weights(str(tmp_path / "cache" / "Original_Name.safetensors"))
    tree = str(tmp_path / "models" / "loras" / "renamed.safetensors")
    os.makedirs(os.path.dirname(tree))
    os.link(cache, tree)

    models = index.model_map(
        str(tmp_path / "models"), [record("civitai-1", paths=[cache])]
    )
    got = index.lookup(models, "renamed.safetensors")
    assert got["record"] == "civitai-1"
    assert got["how"] == "identity"


def test_a_name_match_is_offered_but_labelled_as_one(tmp_path):
    cache = weights(str(tmp_path / "cache" / "same.safetensors"))
    tree = weights(str(tmp_path / "models" / "loras" / "same.safetensors"), b"copy")
    models = index.model_map(
        str(tmp_path / "models"), [record("civitai-2", paths=[cache])]
    )
    assert os.path.exists(tree)
    got = index.lookup(models, "same.safetensors")
    assert got["record"] == "civitai-2"
    assert got["how"] == "name"


def test_one_name_matching_two_records_resolves_to_neither(tmp_path):
    one = weights(str(tmp_path / "cache" / "a" / "dup.safetensors"))
    two = weights(str(tmp_path / "cache" / "b" / "dup.safetensors"), b"other")
    weights(str(tmp_path / "models" / "loras" / "dup.safetensors"), b"third")
    models = index.model_map(
        str(tmp_path / "models"),
        [record("first", paths=[one]), record("second", paths=[two])],
    )
    got = index.lookup(models, "dup.safetensors")
    assert got["how"] == "ambiguous"
    assert got["record"] is None


def test_a_subdirectory_qualified_name_still_resolves(tmp_path):
    cache = weights(str(tmp_path / "cache" / "q.safetensors"))
    tree = str(tmp_path / "models" / "loras" / "q.safetensors")
    os.makedirs(os.path.dirname(tree))
    os.link(cache, tree)
    models = index.model_map(
        str(tmp_path / "models"), [record("civitai-3", paths=[cache])]
    )
    assert index.lookup(models, "loras/q.safetensors")["record"] == "civitai-3"
    assert index.lookup(models, "d_drive\\q.safetensors")["record"] == "civitai-3"


def test_an_untraceable_model_never_counts_as_clean(db, tmp_path, repo):
    made = repo()
    root = str(tmp_path / "out")
    write_generated(os.path.join(root, "mystery.png"), base="nobody_knows.safetensors")
    index.scan(db, root, "Fixture", [])
    row = index.query(db)[0]

    got = index.verdict(db, row["path"], {}, [], [], {})
    assert got["ship"] is False
    assert got["unresolved"][0]["name"] == "nobody_knows.safetensors"
    assert made is not None


def test_a_noncommercial_model_fails_the_gate_it_produced(db, tmp_path):
    cache = weights(str(tmp_path / "cache" / "nc.safetensors"))
    tree = str(tmp_path / "models" / "loras" / "nc.safetensors")
    os.makedirs(os.path.dirname(tree))
    os.link(cache, tree)

    root = str(tmp_path / "out")
    write_generated(os.path.join(root, "tainted.png"), base="nc.safetensors")
    index.scan(db, root, "Fixture", [])
    row = index.query(db)[0]

    records = [record("nc-model", "CC-BY-NC-4.0", paths=[cache])]
    models = index.model_map(str(tmp_path / "models"), records)
    rules = [
        {
            "rule": 4,
            "id": "excluded-licences",
            "kind": "deny",
            "when": "ship",
            "families": ["noncommercial"],
            "text": "Exclude outright NonCommercial.",
        }
    ]
    got = index.verdict(db, row["path"], models, records, rules, {})
    assert got["ship"] is False
    assert got["models"][0]["record"] == "nc-model"
    assert [f.rule_id for f in got["findings"]] == ["excluded-licences"]


def test_a_derived_empty_family_list_travels_beside_the_classification(db, tmp_path):
    """`policy.families` cannot tell "checked, nothing applies" from "unknown".

    A Civitai adapter that reads permission flags and finds no restriction
    writes `families: []`, and an empty list is falsy, so classification falls
    through to a `LicenseRef-` identifier and reports `unknown`. The verdict
    carries the raw declaration so a caller can say which of the two it is.
    """
    cache = weights(str(tmp_path / "cache" / "open.safetensors"))
    tree = str(tmp_path / "models" / "loras" / "open.safetensors")
    os.makedirs(os.path.dirname(tree))
    os.link(cache, tree)

    root = str(tmp_path / "out")
    write_generated(os.path.join(root, "fine.png"), base="open.safetensors")
    index.scan(db, root, "Fixture", [])
    row = index.query(db)[0]

    permissive = record("civitai-open", "LicenseRef-Civitai-1", paths=[cache])
    permissive["licence"]["families"] = []
    models = index.model_map(str(tmp_path / "models"), [permissive])
    got = index.verdict(db, row["path"], models, [permissive], [], {})
    entry = got["models"][0]
    assert entry["families"] == ["unknown"]
    assert entry["declared"] == []
    assert cli._families(entry).startswith("no family applies")


# ------------------------------------------------------------------- cli


def run(made, argv, env=None):
    stream = io.StringIO()
    code = cli.main(["--repo", made.root] + argv, stream=stream)
    return code, stream.getvalue()


@pytest.fixture(autouse=True)
def index_in_tmp(tmp_path, monkeypatch):
    """Never the developer's real index, and never their real weights tree."""
    monkeypatch.setenv(index.DB_ENV, str(tmp_path / "cli-index.sqlite3"))
    monkeypatch.delenv("KLIN_MODELS", raising=False)


def test_build_refuses_to_guess_where_outputs_live(repo):
    made = repo()
    code, out = run(made, ["index", "build"])
    assert code == 2
    assert "will not guess" in out


def test_build_then_status_reports_what_was_found(repo, tmp_path):
    made = repo()
    root = str(tmp_path / "out")
    write_generated(os.path.join(root, "one.png"))
    code, out = run(made, ["index", "build", "--root", root])
    assert code == 0
    assert "1 added" in out
    code, out = run(made, ["index", "status"])
    assert code == 0
    assert "1 item(s) | 1 with provenance" in out


def test_ls_marks_an_untraceable_model_with_a_question_mark(repo, tmp_path):
    made = repo()
    root = str(tmp_path / "out")
    write_generated(os.path.join(root, "one.png"))
    run(made, ["index", "build", "--root", root])
    code, out = run(made, ["ls"])
    assert code == 0
    assert out.splitlines()[3].startswith("?")
    assert "That is not a pass" in out


def test_ls_check_exits_non_zero_when_something_is_untraceable(repo, tmp_path):
    made = repo()
    root = str(tmp_path / "out")
    write_generated(os.path.join(root, "one.png"))
    run(made, ["index", "build", "--root", root])
    code, _ = run(made, ["ls", "--check"])
    assert code == 1


def test_ls_json_carries_the_stack_and_leaves_out_the_graph(repo, tmp_path):
    made = repo()
    root = str(tmp_path / "out")
    write_generated(os.path.join(root, "one.png"), loras=(("psx.safetensors", 0.6),))
    run(made, ["index", "build", "--root", root])
    code, out = run(made, ["ls", "--json"])
    assert code == 0
    payload = json.loads(out)
    assert payload[0]["loras"][0]["name"] == "psx.safetensors"
    assert "graph" not in payload[0]


def test_show_refuses_to_pick_between_matches(repo, tmp_path):
    made = repo()
    root = str(tmp_path / "out")
    write_generated(os.path.join(root, "twin-a.png"))
    write_generated(os.path.join(root, "twin-b.png"))
    run(made, ["index", "build", "--root", root])
    code, out = run(made, ["show", "twin"])
    assert code == 2
    assert "will not pick one for you" in out


def test_show_prints_the_settings_and_the_prompt(repo, tmp_path):
    made = repo()
    root = str(tmp_path / "out")
    write_generated(os.path.join(root, "only.png"))
    run(made, ["index", "build", "--root", root])
    code, out = run(made, ["show", "only"])
    assert code == 1  # untraceable model, so not a pass
    assert "seed 101" in out
    assert "a warm bar car" in out
    assert "could not be traced" in out


def test_show_says_when_nothing_matches(repo, tmp_path):
    made = repo()
    root = str(tmp_path / "out")
    write_generated(os.path.join(root, "only.png"))
    run(made, ["index", "build", "--root", root])
    code, out = run(made, ["show", "no-such-thing"])
    assert code == 2
    assert "nothing in the index matches" in out
