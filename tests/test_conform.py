"""Conform: resolving a project's atlas, and the gates a file passes to be staged.

No test here starts Blender. The subprocess is a stub returning the json a real
run prints, which is what makes the gates testable at all: each one is a pure
function of that report and the project's swatch table.

What a stub cannot prove is that a loop uv actually moved, that Blender's uv
origin and glTF's were reconciled the right way round, or that the exported
file references the atlas rather than embedding a copy of it. Those are what
the session log's real run is for, and the report shape here is chosen so that
a real run answers them by measurement rather than by assertion.
"""

import io
import json
import os

import pytest

from klin import cli, conform, ledger
from klin.conform import blender

from conftest import record


SWATCHES = {
    "_note": "the atlas cell map",
    "atlas": {
        "res_path": "res://assets/atlas.png",
        "repo_path": "assets/atlas.png",
    },
    "swatches": {
        "wood_mid": {"uv": [0.6094, 0.0234], "use": "counter, stools"},
        "iron": {"uv": [0.4844, 0.0234], "use": "bands, hoops"},
    },
}

REPORT = {
    "ok": True,
    "tool": "blender",
    "blender": "5.2.0",
    "objects": ["Counter"],
    "slots": [
        {"object": "Counter", "name": "wood_mid", "swatch": "wood_mid", "polygons": 84},
        {"object": "Counter", "name": "iron", "swatch": "iron", "polygons": 12},
    ],
    "pinned": {"wood_mid": [0.6094, 0.0234], "iron": [0.4844, 0.0234]},
    "triangles": 312,
    "vertices": 480,
    "uvs_measured": [[0.4844, 0.0234], [0.6094, 0.0234]],
    "image_uri": "../assets/atlas.png",
    "mag_filter": 9728,
    "flip_v": True,
    "warnings": [],
}

BANNER = "Blender 5.2.0 (hash 0123456789ab)\nRead prefs: userpref.blend\n"
FAREWELL = "\nBlender quit\n"


def said(report=None, notes=(), banner=True):
    """Stdout as Blender really prints it: a result buried in its own chatter."""
    lines = [BANNER] if banner else []
    for note in notes:
        lines.append("%s%s\n" % (conform.NOTE, note))
    if report is not None:
        lines.append("%s%s\n" % (conform.PREFIX, json.dumps(report)))
    if banner:
        lines.append(FAREWELL)
    return "".join(lines)


@pytest.fixture(autouse=True)
def no_real_machine(tmp_path, monkeypatch):
    """A developer's own Blender must never be what makes a test pass."""
    monkeypatch.delenv("BLENDER", raising=False)
    monkeypatch.delenv("GLTF_VALIDATOR", raising=False)
    monkeypatch.setenv("KLIN_CACHE", str(tmp_path / "cache"))


@pytest.fixture
def project(repo, tmp_path):
    """A repo whose manifest carries a conform block and a swatch table."""
    def make(**over):
        made = repo()
        assets = os.path.join(made.root, "assets")
        os.makedirs(assets, exist_ok=True)
        io.open(os.path.join(assets, "atlas.png"), "wb").write(b"\x89PNG\r\n\x1a\n")
        table = dict(SWATCHES)
        table.update(over.pop("table", {}))
        io.open(os.path.join(assets, "swatches.json"), "w", encoding="utf-8").write(
            json.dumps(table))

        block = {"swatches": "assets/swatches.json"}
        block.update(over.pop("conform", {}))
        text = io.open(made.manifest, encoding="utf-8").read()
        extra = ["staging_dir: assets/_staging", "conform:"]
        for key in sorted(block):
            value = block[key]
            extra.append("  %s: %s" % (key, "null" if value is None else value))
        # The closing fence, not the opening one: "```yaml" also ends in "```".
        head, sep, tail = text.rpartition("```")
        text = head + "\n".join(extra) + "\n" + sep + tail
        io.open(made.manifest, "w", encoding="utf-8", newline="\n").write(text)
        return made
    return make


@pytest.fixture
def fake_blender(tmp_path, monkeypatch):
    """Stand in for the executable and for the process it would start."""
    exe = tmp_path / "blender.exe"
    exe.write_bytes(b"not really blender")

    class Stub(object):
        def __init__(self):
            self.calls = []
            self.code = 0
            self.stdout = said(REPORT)
            self.stderr = ""
            self.writes = None

        def spawn(self, argv):
            self.calls.append(argv)
            job = json.loads(io.open(argv[-1], encoding="utf-8").read())
            self.job = job
            if self.writes is not False:
                io.open(job["output"], "wb").write(self.writes or b"glTF-ish bytes")
            return self.code, self.stdout, self.stderr

    stub = Stub()
    stub.exe = str(exe)
    monkeypatch.setattr(blender, "spawn", stub.spawn)
    return stub


def run(made, argv, stream=None):
    stream = stream or io.StringIO()
    code = cli.main(["--repo", made.root] + argv, stream=stream)
    return code, stream.getvalue()


def conforming(made, stub, *extra):
    return run(made, ["conform", "blender", os.path.join(made.root, "counter.glb"),
                      "--blender", stub.exe] + list(extra))


@pytest.fixture
def mesh(project):
    def make(**kw):
        made = project(**kw)
        io.open(os.path.join(made.root, "counter.glb"), "wb").write(b"glTF")
        return made
    return make


# ---------------------------------------------------------------- the registry


def test_only_blender_is_discovered():
    assert set(conform.adapters()) == {"blender"}


def test_the_script_for_the_other_interpreter_is_present_but_not_a_subcommand():
    """Both halves asserted. Without the first this passes when the file is gone."""
    import sys

    here = os.path.dirname(conform.__file__)
    assert os.path.isfile(os.path.join(here, "_inside_blender.py"))
    assert "_inside_blender" not in conform.adapters()
    conform.adapters()
    assert "klin.conform._inside_blender" not in sys.modules


def test_a_new_module_becomes_a_subcommand_with_no_other_edit():
    """The package docstring promises this; the promise is checked, not trusted."""
    package = os.path.dirname(conform.__file__)
    added = os.path.join(package, "zz_probe.py")
    io.open(added, "w", encoding="utf-8").write(
        "NAME = 'zzprobe'\nHELP = 'a probe'\n"
        "def configure(parser):\n    parser.add_argument('thing')\n"
        "def run(args, ctx):\n    return 0\n"
    )
    try:
        assert "zzprobe" in conform.adapters()
        parser = cli.build_parser()
        args = parser.parse_args(["conform", "zzprobe", "widget"])
        assert args.thing == "widget"
    finally:
        os.remove(added)
        for cached in list(os.listdir(os.path.join(package, "__pycache__"))
                           if os.path.isdir(os.path.join(package, "__pycache__"))
                           else []):
            if cached.startswith("zz_probe"):
                os.remove(os.path.join(package, "__pycache__", cached))


def test_a_missing_manifest_is_a_message_not_a_traceback(tmp_path):
    stream = io.StringIO()
    code = cli.main(["--repo", str(tmp_path), "conform", "blender", "x.glb"],
                    stream=stream)
    assert code == 2
    assert "klin:" in stream.getvalue()


# ------------------------------------------------------------- the swatch table


def test_no_swatches_key_names_the_key_rather_than_guessing_a_path(repo):
    made = repo()
    ctx = _context(made)
    with pytest.raises(conform.ConformError) as exc:
        conform.swatch_table(ctx)
    assert "conform.swatches" in str(exc.value)


def test_a_table_with_no_swatches_block_is_refused(project):
    made = project(table={"swatches": {}})
    with pytest.raises(conform.ConformError) as exc:
        conform.swatch_table(_context(made))
    assert "no 'swatches' block" in str(exc.value)


def test_a_uv_that_is_not_two_numbers_names_the_swatch(project):
    made = project(table={"swatches": {"wood_mid": {"uv": [0.5]}}})
    with pytest.raises(conform.ConformError) as exc:
        conform.swatch_table(_context(made))
    assert "wood_mid" in str(exc.value)


def test_a_uv_outside_the_unit_square_is_refused(project):
    """It wraps, so it renders as a plausible colour from the wrong cell."""
    made = project(table={"swatches": {"wood_mid": {"uv": [1.4, 0.02]}}})
    with pytest.raises(conform.ConformError) as exc:
        conform.swatch_table(_context(made))
    assert "outside the unit square" in str(exc.value)


def test_the_atlas_comes_from_the_path_on_disk_not_the_engine_one(project):
    made = project()
    table = conform.swatch_table(_context(made))
    assert table["atlas"].endswith(os.path.join("assets", "atlas.png"))
    assert "://" not in table["atlas"]


def test_an_engine_resource_path_alone_is_refused_by_name(project):
    made = project(table={"atlas": {"res_path": "res://assets/atlas.png"}})
    with pytest.raises(conform.ConformError) as exc:
        conform.swatch_table(_context(made))
    assert "repo_path" in str(exc.value)


# ---------------------------------------------------------------- finding tools


def test_the_flag_beats_the_environment_which_beats_the_manifest(mesh, monkeypatch,
                                                                tmp_path):
    made = mesh(conform={"blender": str(tmp_path / "from-manifest")})
    for name in ("from-manifest", "from-env", "from-flag"):
        (tmp_path / name).write_bytes(b"x")
    ctx = _context(made)
    block = conform.settings(ctx)

    monkeypatch.setenv("BLENDER", str(tmp_path / "from-env"))
    args = _args(blender=str(tmp_path / "from-flag"))
    assert blender.executable(args, ctx, block).endswith("from-flag")
    assert blender.executable(_args(), ctx, block).endswith("from-env")
    monkeypatch.delenv("BLENDER")
    assert blender.executable(_args(), ctx, block).endswith("from-manifest")


def test_no_blender_anywhere_names_all_three_ways_to_supply_one(mesh):
    made = mesh()
    code, out = run(made, ["conform", "blender", os.path.join(made.root, "counter.glb")])
    assert code == 2
    assert "--blender" in out and "BLENDER" in out and "conform.blender" in out


def test_a_configured_blender_that_is_not_there_is_refused(mesh, tmp_path):
    made = mesh()
    code, out = run(made, ["conform", "blender",
                           os.path.join(made.root, "counter.glb"),
                           "--blender", str(tmp_path / "nope.exe")])
    assert code == 2
    assert "no Blender executable at" in out


# ------------------------------------------------------------ job and command


def test_the_job_carries_the_resolved_table_rather_than_a_path_to_it(mesh, fake_blender):
    made = mesh()
    conforming(made, fake_blender)
    assert fake_blender.job["swatches"] == {
        "wood_mid": [0.6094, 0.0234], "iron": [0.4844, 0.0234]}
    assert os.path.isabs(fake_blender.job["atlas"])


def test_the_command_is_reproducible_and_names_the_script_by_path(mesh, fake_blender):
    made = mesh()
    conforming(made, fake_blender)
    argv = fake_blender.calls[0]
    assert argv[1:6] == ["--background", "--factory-startup",
                         "--python-exit-code", "1", "--python"]
    assert argv[6].endswith("_inside_blender.py")
    assert os.path.isfile(argv[6])
    assert argv[7] == "--"


def test_staging_is_created_on_first_conform(mesh, fake_blender):
    made = mesh()
    staging = os.path.join(made.root, "assets", "_staging")
    assert not os.path.isdir(staging)
    code, _ = conforming(made, fake_blender)
    assert code == 0
    assert os.path.isfile(os.path.join(staging, "counter.glb"))


def test_dry_run_resolves_everything_and_starts_nothing(mesh, fake_blender):
    made = mesh()
    code, out = conforming(made, fake_blender, "--dry-run")
    assert code == 0
    assert fake_blender.calls == []
    assert "dry run" in out
    assert "wood_mid" in out


def test_a_source_in_a_format_blender_does_not_open_is_refused_by_name(mesh,
                                                                      fake_blender):
    made = mesh()
    weird = os.path.join(made.root, "counter.3dm")
    io.open(weird, "wb").write(b"x")
    code, out = run(made, ["conform", "blender", weird, "--blender", fake_blender.exe])
    assert code == 2
    assert ".glb" in out


# --------------------------------------------------------------- parsing a run


def test_the_result_is_found_among_blender_s_own_chatter():
    report, notes = conform.parse_report(said(REPORT), "", "blender")
    assert report["triangles"] == 312
    assert notes == []


def test_notes_are_carried_through_in_order():
    _, notes = conform.parse_report(said(REPORT, notes=["one", "two"]), "", "blender")
    assert notes == ["one", "two"]


def test_no_result_line_surfaces_both_streams():
    """A bare exit code says nothing about which of Blender's many lines failed."""
    with pytest.raises(conform.ConformError) as exc:
        conform.parse_report(BANNER, "Error: cannot read file", "blender")
    assert "cannot read file" in str(exc.value)
    assert "Read prefs" in str(exc.value)


def test_two_result_lines_are_refused_rather_than_guessed_between():
    doubled = said(REPORT) + said(REPORT, banner=False)
    with pytest.raises(conform.ConformError) as exc:
        conform.parse_report(doubled, "", "blender")
    assert "will not pick one" in str(exc.value)


def test_an_unreadable_result_names_where_it_came_from():
    with pytest.raises(conform.ConformError) as exc:
        conform.parse_report(conform.PREFIX + "{not json", "", "blender")
    assert "blender" in str(exc.value)


def test_the_script_s_own_error_reaches_the_user_verbatim():
    broken = said({"ok": False, "error": "RuntimeError: the source holds no mesh"})
    with pytest.raises(conform.ConformError) as exc:
        conform.parse_report(broken, "", "blender")
    assert "the source holds no mesh" in str(exc.value)


# ------------------------------------------------------------------- the gates


def test_a_slot_naming_no_swatch_fails_and_names_the_slot(mesh, fake_blender):
    made = mesh()
    report = _with(slots=REPORT["slots"] + [
        {"object": "Counter", "name": "Material.003", "swatch": None, "polygons": 4}])
    fake_blender.stdout = said(report)
    code, out = conforming(made, fake_blender)
    assert code == 1
    assert "Material.003" in out
    assert "swatches.json" in out


def test_an_unknown_slot_can_be_downgraded_to_a_warning(mesh, fake_blender):
    made = mesh()
    fake_blender.stdout = said(_with(slots=REPORT["slots"] + [
        {"object": "Counter", "name": "Material.003", "swatch": None, "polygons": 4}]))
    code, out = conforming(made, fake_blender, "--allow-unknown-slots")
    assert code == 0
    assert "warn" in out


def test_a_uv_that_never_moved_fails_and_says_so(mesh, fake_blender):
    made = mesh()
    fake_blender.stdout = said(_with(
        uvs_measured=REPORT["uvs_measured"] + [[0.11, 0.87]]))
    code, out = conforming(made, fake_blender)
    assert code == 1
    assert "kept the layout it arrived with" in out


def test_a_swatch_with_no_uv_at_all_fails(mesh, fake_blender):
    made = mesh()
    fake_blender.stdout = said(_with(uvs_measured=[[0.6094, 0.0234]]))
    code, out = conforming(made, fake_blender)
    assert code == 1
    assert "no uv came back" in out


def test_a_layout_written_upside_down_is_caught_by_the_measurement(mesh, fake_blender):
    """The whole reason the exported file is read back rather than the scene."""
    made = mesh()
    flipped = [[u, 1.0 - v] for u, v in REPORT["uvs_measured"]]
    fake_blender.stdout = said(_with(uvs_measured=flipped))
    code, out = conforming(made, fake_blender)
    assert code == 1
    assert "kept the layout it arrived with" in out


def test_bilinear_filtering_fails_because_it_bleeds_the_next_swatch(mesh, fake_blender):
    made = mesh()
    fake_blender.stdout = said(_with(mag_filter=9729))
    code, out = conforming(made, fake_blender)
    assert code == 1
    assert "point filtering" in out


def test_the_manifest_budget_fails_a_mesh_over_it(mesh, fake_blender):
    made = mesh(conform={"max_triangles": 100})
    code, out = conforming(made, fake_blender)
    assert code == 1
    assert "312 triangles against a budget of 100" in out


def test_the_flag_overrides_the_manifest_budget(mesh, fake_blender):
    made = mesh(conform={"max_triangles": 100})
    code, _ = conforming(made, fake_blender, "--max-tris", "400")
    assert code == 0


def test_no_budget_configured_means_no_budget_gate(mesh, fake_blender):
    """klin holds no more opinion about triangle counts than about licences."""
    made = mesh()
    code, _ = conforming(made, fake_blender)
    assert code == 0


def test_zero_disables_the_budget_gate(mesh, fake_blender):
    made = mesh(conform={"max_triangles": 100})
    code, _ = conforming(made, fake_blender, "--max-tris", "0")
    assert code == 0


# --------------------------------------------------------------- the validator


def test_an_absent_validator_warns_and_says_how_to_supply_one(mesh, fake_blender):
    made = mesh()
    code, out = conforming(made, fake_blender)
    assert code == 0
    assert "not validated" in out
    assert "GLTF_VALIDATOR" in out


def test_an_absent_validator_under_strict_is_a_failure(mesh, fake_blender):
    made = mesh()
    code, out = conforming(made, fake_blender, "--strict")
    assert code == 1
    assert "--strict" in out


def test_that_the_export_was_not_validated_reaches_the_record(mesh, fake_blender):
    made = mesh()
    conforming(made, fake_blender)
    row = _row(made, "counter")
    assert "not validated" in (row["notes"] or "").lower()


def test_a_validator_reporting_errors_fails_and_says_where_the_file_is(mesh,
                                                                      fake_blender,
                                                                      monkeypatch):
    made = mesh()
    monkeypatch.setattr(conform, "validate", lambda path, exe: {
        "errors": 3, "warnings": 0, "messages": []})
    code, out = conforming(made, fake_blender, "--validator", fake_blender.exe)
    assert code == 1
    assert "3 error(s)" in out


def test_a_validator_reporting_only_warnings_passes(mesh, fake_blender, monkeypatch):
    made = mesh()
    monkeypatch.setattr(conform, "validate", lambda path, exe: {
        "errors": 0, "warnings": 2, "messages": []})
    code, out = conforming(made, fake_blender, "--validator", fake_blender.exe)
    assert code == 0
    assert "2 warning(s)" in out


def test_a_validator_that_printed_nothing_readable_is_not_a_pass(tmp_path):
    """Unreadable is not clean, and reporting it as clean is the quiet failure."""
    fake = tmp_path / "validator.py"
    exe = tmp_path / "v.cmd"
    exe.write_text("@echo off\r\necho not json\r\n")
    with pytest.raises(conform.ConformError) as exc:
        conform.validate(str(tmp_path / "x.glb"), str(exe))
    assert "readable json" in str(exc.value)


# ------------------------------------------------------- the staging discipline


@pytest.mark.parametrize("mutation,flag", [
    ({"slots": REPORT["slots"] + [{"object": "C", "name": "M", "swatch": None,
                                   "polygons": 1}]}, None),
    ({"uvs_measured": [[0.11, 0.87]]}, None),
    ({"mag_filter": 9729}, None),
    ({}, "--strict"),
])
def test_a_file_that_failed_a_gate_never_reaches_staging(mesh, fake_blender,
                                                         mutation, flag):
    """The manifest's contract sentence, executed rather than asserted."""
    made = mesh()
    if mutation:
        fake_blender.stdout = said(_with(**mutation))
    code, out = conforming(made, fake_blender, *( [flag] if flag else [] ))
    assert code == 1
    staging = os.path.join(made.root, "assets", "_staging")
    assert not os.path.isdir(staging) or os.listdir(staging) == []
    assert "nothing was staged" in out


# ------------------------------------------------------------------ the record


def test_the_record_says_a_tool_conformed_it_and_not_that_a_model_made_it(mesh,
                                                                         fake_blender):
    made = mesh()
    conforming(made, fake_blender)
    row = _row(made, "counter")
    assert row["produced_by"] == {"tool": "blender", "version": "5.2.0"}


def test_the_derivation_uses_its_own_field_and_leaves_mirror_of_alone(mesh,
                                                                     fake_blender):
    """`mirror_of` means the same bytes, and a conformed mesh is not those."""
    made = mesh()
    made.write_records([record("battleroach")])
    conforming(made, fake_blender, "--from", "battleroach")
    row = _row(made, "counter")
    assert row["source"]["derived_from"] == "battleroach"
    assert row["source"]["mirror_of"] is None


def test_the_licence_is_the_source_s_rather_than_one_klin_invented(mesh, fake_blender):
    made = mesh()
    origin = record("battleroach", licence_id="CC0-1.0")
    made.write_records([origin])
    conforming(made, fake_blender, "--from", "battleroach")
    assert _row(made, "counter")["licence"] == origin["licence"]


def test_with_no_source_the_licence_stays_blank_rather_than_defaulting(mesh,
                                                                      fake_blender):
    """A conform that quietly wrote a licence would be inventing provenance."""
    made = mesh()
    conforming(made, fake_blender)
    assert _row(made, "counter")["licence"]["id"] is None


def test_the_staged_path_is_repo_relative_so_drift_never_counts_it(mesh, fake_blender):
    made = mesh()
    conforming(made, fake_blender)
    row = _row(made, "counter")
    assert row["paths"] == ["assets/_staging/counter.glb"]
    assert ledger.cache_drift([row], os.path.join(made.root, "cache")) is None


def test_the_modification_line_comes_from_the_report_and_not_the_arguments(mesh,
                                                                          fake_blender):
    made = mesh()
    conforming(made, fake_blender)
    line = _row(made, "counter")["modifications"][-1]
    assert "wood_mid" in line and "iron" in line and "312 triangles" in line


def test_a_re_conform_keeps_what_a_person_wrote(mesh, fake_blender):
    made = mesh()
    conforming(made, fake_blender)
    row = _row(made, "counter")
    row["notes"] = "Hand-written and not klin's to delete."
    row["used_for"] = "The bar counter."
    row["reviewed_at"] = "2026-08-01"
    row["modifications"] = ["An earlier hand-written line."] + row["modifications"]
    made.write_records([row])

    conforming(made, fake_blender, "--replace")
    again = _row(made, "counter")
    assert "not klin's to delete" in again["notes"]
    assert again["used_for"] == "The bar counter."
    assert again["reviewed_at"] == "2026-08-01"
    assert "An earlier hand-written line." in again["modifications"]


def test_an_existing_record_without_replace_is_refused(mesh, fake_blender):
    made = mesh()
    conforming(made, fake_blender)
    code, out = conforming(made, fake_blender)
    assert code == 2
    assert "--replace" in out


def test_a_machine_never_sets_reviewed_at(mesh, fake_blender):
    made = mesh()
    conforming(made, fake_blender)
    assert _row(made, "counter")["reviewed_at"] is None


def test_no_record_stages_the_file_and_writes_nothing(mesh, fake_blender):
    made = mesh()
    code, _ = conforming(made, fake_blender, "--no-record")
    assert code == 0
    assert os.path.isfile(os.path.join(made.root, "assets", "_staging", "counter.glb"))
    assert ledger.load(os.path.join(made.root, "assets", "ledger.jsonl")) == []


def test_the_id_defaults_to_the_output_stem_and_the_flag_overrides_it(mesh,
                                                                     fake_blender):
    made = mesh()
    conforming(made, fake_blender, "--id", "bar-counter")
    assert _row(made, "bar-counter") is not None


# ------------------------------------------------------------------ end to end


def test_a_conformed_record_passes_the_ship_gate_when_its_source_does(mesh,
                                                                     fake_blender):
    made = mesh()
    made.write_records([record("battleroach", licence_id="CC0-1.0")])
    conforming(made, fake_blender, "--from", "battleroach")
    code, out = run(made, ["ledger", "audit", "--ship"])
    assert code == 0, out


def test_a_share_alike_source_makes_the_derivative_fail_the_ship_gate(mesh,
                                                                     fake_blender):
    made = mesh()
    made.write_records([record("tavern", licence_id="CC-BY-SA-4.0")])
    conforming(made, fake_blender, "--from", "tavern")
    code, out = run(made, ["ledger", "audit", "--ship"])
    assert code == 1
    assert "counter" in out


def test_a_source_that_cannot_ship_is_reported_before_any_work(mesh, fake_blender):
    made = mesh()
    made.write_records([record("tavern", licence_id="CC-BY-SA-4.0")])
    code, out = conforming(made, fake_blender, "--from", "tavern", "--check")
    assert code == 1
    assert fake_blender.calls == []
    assert "could not ship" in out


def test_without_check_a_risky_source_still_conforms_and_says_what_will_happen(
        mesh, fake_blender):
    """A prototype conform is as legitimate as a prototype plate; the gate catches it."""
    made = mesh()
    made.write_records([record("tavern", licence_id="CC-BY-SA-4.0")])
    code, out = conforming(made, fake_blender, "--from", "tavern")
    assert code == 0
    assert "will fail the ship gate" in out


def test_an_unknown_source_record_lists_what_the_ledger_holds(mesh, fake_blender):
    made = mesh()
    made.write_records([record("battleroach")])
    code, out = conforming(made, fake_blender, "--from", "nope")
    assert code == 2
    assert "battleroach" in out


# ----------------------------------------------------------------------- helpers


def _with(**over):
    report = json.loads(json.dumps(REPORT))
    report.update(over)
    return report


def _context(made):
    from klin import manifest
    from klin.fetch import Context

    return Context(_args(repo=made.root), manifest.load(made.manifest), io.StringIO())


def _args(**kw):
    class Args(object):
        pass

    args = Args()
    args.repo = kw.get("repo", ".")
    args.blender = kw.get("blender")
    args.validator = kw.get("validator")
    args.max_tris = kw.get("max_tris")
    args.strict = kw.get("strict", False)
    return args


def _row(made, ident):
    path = os.path.join(made.root, "assets", "ledger.jsonl")
    for row in ledger.load(path):
        if row["id"] == ident:
            return row
    return None
