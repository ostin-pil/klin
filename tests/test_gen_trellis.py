"""TRELLIS: the geometry-only patch, and the pre-flight around a run.

No test here starts TRELLIS. It needs a GPU, several gigabytes of weights and
an interpreter klin does not own, so the subprocess is stubbed and what is
exercised is the half that decides whether to start it at all.

The patch itself is different, and it is fully covered: it is text
manipulation over a vendor file, which is exactly the kind of thing that can
be checked without the tool it patches being present. That matters more than
usual here, because the patch is what a licence claim rests on, and the
failure it guards against is a source tree that reads as patched while the
copy python actually imports still bakes.
"""

import io
import json
import os

import pytest

from klin import cli, gen
from klin.gen import trellis

from conftest import record


#: The lines of the upstream exporter this patches, in the order it meets them.
#: Enough of the real file to exercise every edit, and no more.
UPSTREAM = """import numpy as np
import trimesh
import nvdiffrast.torch as dr


def to_glb(
    mesh,
    texture_size: int = 2048,
    verbose: bool = True,
):
    \"\"\"Export.

    Args:
        mesh: the thing
        texture_size: size of the texture for baking
    \"\"\"
    out_vertices, out_faces, out_uvs, out_normals = unwrap(mesh)

    # --- Texture Baking (Attribute Sampling) ---
    # Setup differentiable rasterizer context
    ctx = dr.RasterizeCudaContext()
    return baked
"""


@pytest.fixture(autouse=True)
def no_real_machine(tmp_path, monkeypatch):
    monkeypatch.delenv("TRELLIS_INSTALL", raising=False)
    monkeypatch.delenv("TRELLIS_PYTHON", raising=False)
    monkeypatch.setenv("KLIN_CACHE", str(tmp_path / "cache"))


@pytest.fixture
def install(tmp_path):
    """An install with all three copies of the exporter, in CRLF like the real one."""
    def make(copies=None, text=UPSTREAM, newline="\r\n"):
        root = tmp_path / "trellis"
        for rel in (copies if copies is not None else trellis.TARGETS):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            io.open(str(path), "w", encoding="utf-8", newline=newline).write(text)
        venv = root / "venv" / "Scripts"
        venv.mkdir(parents=True, exist_ok=True)
        (venv / "python.exe").write_bytes(b"not really python")
        return str(root)
    return make


def run(made, argv, stream=None):
    stream = stream or io.StringIO()
    return cli.main(["--repo", made.root] + argv, stream=stream), stream.getvalue()


# ------------------------------------------------------------------- the patch


def test_a_stock_install_reads_as_clean(install):
    text, _ = trellis.read(os.path.join(install(), trellis.TARGETS[0]))
    assert trellis.state(text) == "clean"


def test_applying_reaches_every_edit(install):
    text, _ = trellis.read(os.path.join(install(), trellis.TARGETS[0]))
    assert trellis.state(trellis.apply_to(text)) == "patched"


def test_applying_twice_changes_nothing_the_second_time(install):
    """A reinstall reverts it, so this has to be re-runnable without thought."""
    text, _ = trellis.read(os.path.join(install(), trellis.TARGETS[0]))
    once = trellis.apply_to(text)
    assert trellis.apply_to(once) == once


def test_reverting_restores_the_file_byte_for_byte(install):
    text, _ = trellis.read(os.path.join(install(), trellis.TARGETS[0]))
    assert trellis.revert_to(trellis.apply_to(text)) == text


def test_the_module_scope_import_is_gone_and_a_lazy_one_takes_its_place(install):
    text, _ = trellis.read(os.path.join(install(), trellis.TARGETS[0]))
    patched = trellis.apply_to(text)
    assert "\nimport nvdiffrast.torch as dr\n" not in patched
    assert "    import nvdiffrast.torch as dr\n" in patched


def test_the_geometry_exit_lands_above_the_bake_and_not_below_it(install):
    text, _ = trellis.read(os.path.join(install(), trellis.TARGETS[0]))
    patched = trellis.apply_to(text)
    assert patched.index("if not bake_texture:") < patched.index("Texture Baking")


def test_a_half_patched_file_is_broken_rather_than_either(install):
    """The state that matters: it is neither safe to run nor safe to re-patch."""
    text, _ = trellis.read(os.path.join(install(), trellis.TARGETS[0]))
    half = text.replace(trellis.PARAM_OLD, trellis.PARAM_NEW, 1)
    assert trellis.state(half) == "broken"


def test_an_upstream_that_moved_refuses_rather_than_patching_something_else(install):
    moved = UPSTREAM.replace("    texture_size: int = 2048,\n", "    tex: int = 2048,\n")
    text, _ = trellis.read(os.path.join(install(text=moved), trellis.TARGETS[0]))
    with pytest.raises(gen.GenError) as exc:
        trellis.apply_to(text)
    assert "anchor" in str(exc.value)


def test_crlf_survives_a_patch(install):
    """The audit diff is the artifact; rewriting every line turns it into noise."""
    path = os.path.join(install(), trellis.TARGETS[0])
    text, newline = trellis.read(path)
    assert newline == "\r\n"
    trellis.write(path, trellis.apply_to(text), newline)
    raw = io.open(path, "rb").read()
    assert b"\r\n" in raw
    assert raw.count(b"\n") == raw.count(b"\r\n")


def test_a_patch_written_by_another_tool_reads_as_patched(install):
    """Detection ignores the mark, so klin never patches somebody else's patch twice."""
    text, _ = trellis.read(os.path.join(install(), trellis.TARGETS[0]))
    foreign = trellis.apply_to(text).replace(trellis.MARK, "some other tool")
    assert trellis.state(foreign) == "patched"
    assert trellis.apply_to(foreign) == foreign


def test_reverting_a_patch_klin_did_not_write_refuses_rather_than_guesses(install):
    text, _ = trellis.read(os.path.join(install(), trellis.TARGETS[0]))
    foreign = trellis.apply_to(text).replace(trellis.MARK, "some other tool")
    with pytest.raises(gen.GenError) as exc:
        trellis.revert_to(foreign)
    assert "by hand" in str(exc.value)


# -------------------------------------------------------- the patch, end to end


def test_apply_reports_every_copy_and_leaves_none_behind(repo, install):
    made = repo()
    root = install()
    code, out = run(made, ["gen", "trellis", "--install", root, "--patch", "apply"])
    assert code == 0
    for rel in trellis.TARGETS:
        assert rel in out
        text, _ = trellis.read(os.path.join(root, rel))
        assert trellis.state(text) == "patched"


def test_the_copy_python_imports_is_the_one_that_must_not_be_missed(repo, install):
    """Patching the source alone leaves a clean-looking tree and a run that bakes."""
    assert trellis.TARGETS[0].startswith("venv/Lib/site-packages/")
    made = repo()
    root = install(copies=trellis.TARGETS[1:])
    code, out = run(made, ["gen", "trellis", "--install", root, "--patch", "apply"])
    assert "absent" in out
    assert trellis.TARGETS[0] in out


def test_a_split_state_fails_the_check_and_says_why(repo, install):
    made = repo()
    root = install()
    text, newline = trellis.read(os.path.join(root, trellis.TARGETS[0]))
    trellis.write(os.path.join(root, trellis.TARGETS[0]),
                  trellis.apply_to(text), newline)
    code, out = run(made, ["gen", "trellis", "--install", root, "--patch", "check"])
    assert code == 1
    assert "split state" in out


def test_an_install_with_no_exporter_at_all_is_a_message(repo, tmp_path):
    made = repo()
    empty = tmp_path / "empty"
    empty.mkdir()
    code, out = run(made, ["gen", "trellis", "--install", str(empty),
                           "--patch", "check"])
    assert code == 2
    assert "no copy of the exporter" in out


def test_no_install_configured_names_all_three_ways_to_supply_one(repo):
    made = repo()
    code, out = run(made, ["gen", "trellis", "--patch", "check"])
    assert code == 2
    assert "--install" in out and "TRELLIS_INSTALL" in out


# --------------------------------------------------------------- the pre-flight


def test_a_geometry_run_refuses_an_unpatched_install(repo, install):
    """The clean path is the non-default argument, so it has to be checked for."""
    made = repo()
    code, out = run(made, ["gen", "trellis", "--install", install(),
                           "--image", __file__])
    assert code == 2
    assert "--patch apply" in out


def test_the_patch_state_is_reported_before_a_run_rather_than_assumed(repo, install,
                                                                     monkeypatch):
    made = repo()
    root = install()
    run(made, ["gen", "trellis", "--install", root, "--patch", "apply"])
    monkeypatch.setattr(trellis.subprocess, "run", _never)
    code, out = run(made, ["gen", "trellis", "--install", root,
                           "--image", __file__, "--dry-run"])
    assert code == 0
    assert "patch    patched" in out


def test_dry_run_starts_nothing(repo, install, monkeypatch):
    made = repo()
    root = install()
    run(made, ["gen", "trellis", "--install", root, "--patch", "apply"])
    monkeypatch.setattr(trellis.subprocess, "run", _never)
    code, out = run(made, ["gen", "trellis", "--install", root,
                           "--image", __file__, "--dry-run"])
    assert code == 0
    assert "dry run" in out


def test_a_run_that_printed_no_result_surfaces_both_streams(repo, install,
                                                            monkeypatch):
    made = repo()
    root = install()
    run(made, ["gen", "trellis", "--install", root, "--patch", "apply"])
    monkeypatch.setattr(trellis.subprocess, "run",
                        _proc(1, "loading weights", "CUDA out of memory"))
    code, out = run(made, ["gen", "trellis", "--install", root, "--image", __file__])
    assert code == 2
    assert "CUDA out of memory" in out


def test_a_geometry_run_that_loaded_restricted_code_is_a_failure(repo, install,
                                                                 monkeypatch):
    """The whole claim. If this passes quietly the licence label is a lie."""
    made = repo()
    root = install()
    run(made, ["gen", "trellis", "--install", root, "--patch", "apply"])
    report = dict(_REPORT, restricted_loaded=["nvdiffrast"])
    monkeypatch.setattr(trellis.subprocess, "run",
                        _proc(0, "TRELLIS " + json.dumps(report), ""))
    code, out = run(made, ["gen", "trellis", "--install", root, "--image", __file__])
    assert code == 2
    assert "nvdiffrast" in out


def test_a_clean_geometry_run_says_what_it_proved_and_what_comes_next(repo, install,
                                                                     monkeypatch):
    made = repo()
    root = install()
    run(made, ["gen", "trellis", "--install", root, "--patch", "apply"])
    monkeypatch.setattr(trellis.subprocess, "run",
                        _proc(0, "TRELLIS " + json.dumps(_REPORT), ""))
    code, out = run(made, ["gen", "trellis", "--install", root, "--image", __file__])
    assert code == 0
    assert "no restricted module could be imported" in out
    assert "klin conform blender" in out


def test_the_generator_reaches_the_same_licence_check_as_a_node_graph(repo, install,
                                                                     monkeypatch):
    """It has no graph, so posture takes the pairs directly."""
    made = repo()
    root = install()
    run(made, ["gen", "trellis", "--install", root, "--patch", "apply"])
    monkeypatch.setattr(trellis.subprocess, "run", _never)
    code, out = run(made, ["gen", "trellis", "--install", root,
                           "--image", __file__, "--dry-run"])
    # The checkpoint, which is what a ledger could hold a record of. The tier
    # is a run parameter and asking the ledger about it produced an
    # untraceable model every time, for no reason but the name.
    assert trellis.MODEL in out
    assert "1024_cascade" not in out.split("checkpoint")[0]


def test_a_run_is_offline_unless_it_is_asked_to_reach_the_network(repo, install,
                                                                 monkeypatch):
    """`from_pretrained` resolves every model through `hf_hub_download`.

    So a warm cache is not the same as no network: the run still sends HEAD
    requests and takes whatever answers them. Barinn's session 36 lost a day
    to that shape in the Hi3DGen environment, where nothing was pinned and two
    of five runtime fetches had moved.
    """
    made = repo()
    root = install()
    run(made, ["gen", "trellis", "--install", root, "--patch", "apply"])
    seen = _capture(0, "TRELLIS " + json.dumps(_REPORT), "")
    monkeypatch.setattr(trellis.subprocess, "run", seen)
    code, out = run(made, ["gen", "trellis", "--install", root, "--image", __file__])
    assert code == 0
    assert seen.env["HF_HUB_OFFLINE"] == "1"
    assert seen.env["TRANSFORMERS_OFFLINE"] == "1"
    assert "offline" in out


def test_asking_for_the_network_out_loud_clears_the_offline_switches(repo, install,
                                                                    monkeypatch):
    """A first fetch has to be possible, and has to be visible in the output."""
    made = repo()
    root = install()
    run(made, ["gen", "trellis", "--install", root, "--patch", "apply"])
    # Set on the machine, so clearing it is a real removal rather than an
    # absence that would have looked the same either way.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    seen = _capture(0, "TRELLIS " + json.dumps(_REPORT), "")
    monkeypatch.setattr(trellis.subprocess, "run", seen)
    code, out = run(made, ["gen", "trellis", "--install", root,
                           "--image", __file__, "--online"])
    assert code == 0
    assert "HF_HUB_OFFLINE" not in seen.env
    assert "online" in out


# ---------------------------------------------------------- the background remover


def _patched(repo, install):
    made = repo()
    root = install()
    run(made, ["gen", "trellis", "--install", root, "--patch", "apply"])
    return made, root


def _remover_dir(tmp_path):
    path = tmp_path / "BiRefNet"
    path.mkdir()
    (path / "config.json").write_text("{}")
    return str(path)


def test_no_remover_configured_says_an_opaque_image_will_be_refused(repo, install,
                                                                    monkeypatch):
    made, root = _patched(repo, install)
    monkeypatch.setattr(trellis.subprocess, "run", _never)
    code, out = run(made, ["gen", "trellis", "--install", root,
                           "--image", __file__, "--dry-run"])
    assert code == 0
    assert "remover  none: an image without its own alpha will be refused" in out
    assert '"rembg": null' in out


def test_the_remover_travels_in_the_job_from_the_flag(repo, install, monkeypatch,
                                                      tmp_path):
    made, root = _patched(repo, install)
    birefnet = _remover_dir(tmp_path)
    monkeypatch.setattr(trellis.subprocess, "run", _never)
    code, out = run(made, ["gen", "trellis", "--install", root, "--image", __file__,
                           "--rembg", birefnet, "--dry-run"])
    assert code == 0
    assert "remover  %s" % os.path.normpath(birefnet) in out
    assert json.dumps(os.path.normpath(birefnet)) in out


def test_the_remover_can_come_from_the_environment(repo, install, monkeypatch,
                                                   tmp_path):
    made, root = _patched(repo, install)
    monkeypatch.setenv("TRELLIS_REMBG", _remover_dir(tmp_path))
    monkeypatch.setattr(trellis.subprocess, "run", _never)
    code, out = run(made, ["gen", "trellis", "--install", root,
                           "--image", __file__, "--dry-run"])
    assert code == 0
    assert "will be refused" not in out


def test_a_remover_that_is_not_a_model_directory_is_refused_before_anything_runs(
        repo, install, monkeypatch, tmp_path):
    made, root = _patched(repo, install)
    monkeypatch.setattr(trellis.subprocess, "run", _never)
    code, out = run(made, ["gen", "trellis", "--install", root, "--image", __file__,
                           "--rembg", str(tmp_path / "nowhere")])
    assert code == 2
    assert "config.json" in out


def test_a_run_that_needed_no_remover_says_so_and_that_rmbg_was_never_built(
        repo, install, monkeypatch):
    made, root = _patched(repo, install)
    monkeypatch.setattr(trellis.subprocess, "run",
                        _proc(0, "TRELLIS " + json.dumps(_REPORT), ""))
    code, out = run(made, ["gen", "trellis", "--install", root, "--image", __file__])
    assert code == 0
    assert "the image carried its own alpha, so no remover ran" in out
    assert "briaai/RMBG-2.0 was never built" in out


def test_a_run_that_masked_the_image_names_the_remover_it_used(repo, install,
                                                               monkeypatch, tmp_path):
    made, root = _patched(repo, install)
    birefnet = os.path.normpath(_remover_dir(tmp_path))
    report = dict(_REPORT, remover={"configured": "briaai/RMBG-2.0",
                                    "loaded": birefnet, "called": True})
    monkeypatch.setattr(trellis.subprocess, "run",
                        _proc(0, "TRELLIS " + json.dumps(report), ""))
    code, out = run(made, ["gen", "trellis", "--install", root, "--image", __file__,
                           "--rembg", birefnet])
    assert code == 0
    assert "background removed with %s" % birefnet in out


def test_a_run_that_loaded_some_other_remover_is_a_failure(repo, install,
                                                           monkeypatch, tmp_path):
    made, root = _patched(repo, install)
    report = dict(_REPORT, remover={"configured": "briaai/RMBG-2.0",
                                    "loaded": "models/RMBG-2.0", "called": True})
    monkeypatch.setattr(trellis.subprocess, "run",
                        _proc(0, "TRELLIS " + json.dumps(report), ""))
    code, out = run(made, ["gen", "trellis", "--install", root, "--image", __file__,
                           "--rembg", _remover_dir(tmp_path)])
    assert code == 2
    assert "models/RMBG-2.0" in out


def test_a_run_that_does_not_report_its_remover_claims_nothing(repo, install,
                                                               monkeypatch):
    made, root = _patched(repo, install)
    report = dict(_REPORT)
    del report["remover"]
    monkeypatch.setattr(trellis.subprocess, "run",
                        _proc(0, "TRELLIS " + json.dumps(report), ""))
    code, out = run(made, ["gen", "trellis", "--install", root, "--image", __file__])
    assert code == 2
    assert "did not say which background remover" in out


def _inside():
    import importlib.util

    path = os.path.join(os.path.dirname(gen.__file__), "_inside_trellis.py")
    spec = importlib.util.spec_from_file_location("_inside_trellis_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Stock(object):
    """Stands in for the install's wrapper, which must never be constructed."""

    def __init__(self, *a, **kw):
        raise AssertionError("the install's remover was built")


def test_the_guard_replaces_the_install_class_so_rmbg_is_never_built():
    inside = _inside()
    rembg = type("rembg", (), {"BiRefNet": _Stock})
    record = {}
    inside.guard_remover({"rembg": None}, rembg, record)
    remover = rembg.BiRefNet(model_name="briaai/RMBG-2.0")
    assert isinstance(remover, inside.Remover)
    assert record == {"configured": "briaai/RMBG-2.0", "loaded": None, "called": False}


def test_without_a_remover_an_opaque_image_is_refused_by_name():
    inside = _inside()
    rembg = type("rembg", (), {"BiRefNet": _Stock})
    record = {}
    inside.guard_remover({"rembg": None}, rembg, record)
    remover = rembg.BiRefNet(model_name="briaai/RMBG-2.0")
    remover.to("cuda")
    remover.cpu()
    with pytest.raises(inside.RemoverRefused) as refused:
        remover(object())
    assert "briaai/RMBG-2.0" in str(refused.value)
    assert "CC BY-NC" in str(refused.value)
    assert record["called"] is True


def test_a_pipeline_holding_a_remover_klin_did_not_build_is_refused():
    inside = _inside()
    pipeline = type("pipeline", (), {"rembg_model": object()})()
    with pytest.raises(inside.RemoverRefused):
        inside.confirm_remover(pipeline)
    inside.confirm_remover(type("pipeline", (), {"rembg_model": None})())


def test_every_generator_is_discovered():
    assert set(gen.adapters()) == {"comfy", "hi3dgen", "trellis"}


def test_the_harness_for_the_other_interpreter_ships_but_is_not_a_subcommand():
    here = os.path.dirname(gen.__file__)
    assert os.path.isfile(os.path.join(here, "_inside_trellis.py"))
    assert "_inside_trellis" not in gen.adapters()


# ----------------------------------------------------------------------- helpers


_REPORT = {
    "ok": True,
    "glb": "out/stove.glb",
    "faces": 192327,
    "vertices": 127752,
    "seconds": 47.8,
    "mode": "geometry",
    "restricted_loaded": [],
    "restricted_attempts": [],
    "restricted_blocked": True,
    "has_baked_texture": False,
    "has_uvs": True,
    "remover": {"configured": "briaai/RMBG-2.0", "loaded": None, "called": False},
}


def _never(*a, **kw):
    raise AssertionError("nothing should have been started")


def _capture(code, out, err):
    """Like `_proc`, but keeps the environment the subprocess was handed.

    What the run is allowed to reach is part of what it proves, so it is
    asserted rather than read off the machine that happens to run the suite.
    """

    class Done(object):
        returncode = code
        stdout = out
        stderr = err

    class Call(object):
        env = None

        def __call__(self, *a, **kw):
            Call.env = kw.get("env")
            return Done()

    call = Call()
    return call


def _proc(code, out, err):
    class Done(object):
        returncode = code
        stdout = out
        stderr = err

    return lambda *a, **kw: Done()
