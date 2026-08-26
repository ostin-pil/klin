import io
import os

import pytest

from klin import cli, ledger

from conftest import record


def run(made, argv):
    stream = io.StringIO()
    code = cli.main(["--repo", made.root] + argv, stream=stream)
    return code, stream.getvalue()


def test_roundtrip_sorts_by_id(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.save(path, [record("zebra"), record("apple")])
    assert [r["id"] for r in ledger.load(path)] == ["apple", "zebra"]


def test_a_missing_ledger_is_empty_not_an_error(tmp_path):
    assert ledger.load(str(tmp_path / "nothing.jsonl")) == []


def test_a_duplicate_id_is_rejected_on_read(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    io.open(path, "w", encoding="utf-8", newline="\n").write(
        '{"id": "same"}\n{"id": "same"}\n'
    )
    with pytest.raises(ledger.LedgerError):
        ledger.load(path)


def test_add_refuses_to_clobber_without_replace(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.add(path, record("one"))
    with pytest.raises(ledger.LedgerError):
        ledger.add(path, record("one"))
    ledger.add(path, record("one", "MIT"), replace=True)
    assert ledger.load(path)[0]["licence"]["id"] == "MIT"


def test_dotted_field_lookup_never_raises():
    assert ledger.field(record("a"), "source.url").endswith("/a")
    assert ledger.field(record("a"), "source.nope") is None
    assert ledger.field(record("a"), "nope.nope.nope") is None


def test_audit_exits_zero_on_a_clean_ledger(repo):
    made = repo()
    made.write_records([record("clean", "CC0-1.0")])
    code, out = run(made, ["ledger", "audit", "--ship"])
    assert code == 0
    assert "0 failure(s)" in out


def test_audit_exits_one_and_prints_the_rule_text(repo):
    made = repo()
    made.write_records([record("tavern-tools", "CC-BY-SA-4.0")])
    code, out = run(made, ["ledger", "audit", "--ship"])
    assert code == 1
    assert "FAIL  rule 2 (no-share-alike)" in out
    assert "Don't take it." in out


def test_the_stage_gate_says_it_is_only_the_stage_gate(repo):
    made = repo()
    made.write_records([record("tavern-tools", "CC-BY-SA-4.0")])
    code, out = run(made, ["ledger", "audit"])
    assert code == 0
    assert "before anything goes public" in out


def test_render_then_check_is_clean(repo):
    made = repo()
    made.write_records([record("clean", "CC0-1.0")])
    code, _ = run(made, ["ledger", "render"])
    assert code == 0
    assert "`clean`" in made.licenses()
    code, out = run(made, ["ledger", "render", "--check"])
    assert code == 0
    assert "up to date" in out


def test_check_fails_when_a_record_is_added_but_not_rendered(repo):
    made = repo()
    made.write_records([record("clean", "CC0-1.0")])
    run(made, ["ledger", "render"])
    made.write_records([record("clean", "CC0-1.0"), record("later", "CC0-1.0")])
    code, out = run(made, ["ledger", "render", "--check"])
    assert code == 1
    assert "stale" in out


def test_a_missing_manifest_is_a_message_not_a_traceback(tmp_path):
    stream = io.StringIO()
    code = cli.main(["--repo", str(tmp_path), "ledger", "list"], stream=stream)
    assert code == 2
    assert "no manifest at" in stream.getvalue()


# ------------------------------------------------------------ cache drift


def test_drift_is_measured_against_the_cache_in_force(tmp_path):
    """Not "do the files exist", which stays true through the whole failure."""
    cache = str(tmp_path / "cache")
    elsewhere = str(tmp_path / "somewhere-else" / "civitai" / "1" / "a.safetensors")
    got = ledger.cache_drift([record("one", paths=[elsewhere])], cache)
    assert got["recorded"] == 1
    assert got["cache"] == os.path.normpath(cache)


def test_one_file_under_the_cache_is_enough_to_settle_it(tmp_path):
    cache = str(tmp_path / "cache")
    inside = os.path.join(cache, "civitai", "1", "a.safetensors")
    outside = str(tmp_path / "elsewhere" / "b.safetensors")
    records = [record("one", paths=[inside]), record("two", paths=[outside])]
    assert ledger.cache_drift(records, cache) is None


def test_repo_relative_paths_are_not_evidence_either_way(tmp_path):
    """A pack committed into the tree says nothing about where the cache is."""
    records = [record("kaykit", paths=["game/assets/_vendor/kaykit/"])]
    assert ledger.cache_drift(records, str(tmp_path / "cache")) is None


def test_no_cache_configured_is_not_drift():
    assert ledger.cache_drift([record("one", paths=["/abs/x"])], None) is None


def test_the_audit_says_so_when_the_cache_has_drifted(repo, tmp_path, monkeypatch):
    made = repo()
    monkeypatch.setenv("KLIN_CACHE", str(tmp_path / "empty-cache"))
    made.write_records(
        [record("one", "CC0-1.0", paths=[str(tmp_path / "real" / "a.safetensors")])]
    )
    code, out = run(made, ["ledger", "audit"])
    assert code == 0  # a note, never a failure
    assert "none of the 1 recorded file(s) are under it" in out
    assert "Set KLIN_CACHE" in out
