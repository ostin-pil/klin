"""Hi3DGen: the pins, and the pre-flight around a run.

No test here starts Hi3DGen. It needs a GPU, seven gigabytes of pinned weights
and an interpreter klin does not own, so the subprocess is stubbed and what is
exercised is the half that decides whether to start it at all.

The pins are different, and they are covered properly: they are a JSON file
read off disk and compared with what is beside it, which is exactly the kind of
thing that can be checked without the tool being present. That matters here for
the same reason the patch matters in the TRELLIS tests. The patch is what makes
that tool's licence claim true; the pins are what make this tool's *offline*
claim mean anything, and the failure they guard against is a run that quietly
uses a file nobody recorded.
"""

import io
import json
import os

import pytest

from klin import cli, gen
from klin.gen import hi3dgen


PINS = {
    "pinned_as_of": "2025-07-02, Stable-X/Stable3DGen c29f668",
    "hf": [
        {
            "repo": "Stable-X/trellis-normal-v0-1",
            "revision": "fdfac1d0e0cb1111111111111111111111111111",
            "licence": "MIT",
            "dir": "weights/trellis-normal-v0-1",
            "files": {"ckpts/model.safetensors": {"bytes": 11}},
        },
        {
            "repo": "Stable-X/yoso-normal-v1-8-1",
            "revision": "4dd7d000d76f2222222222222222222222222222",
            "licence": "Apache-2.0",
            "dir": "weights/yoso-normal-v1-8-1",
            "files": {"unet/diffusion.safetensors": {"bytes": 7}},
        },
    ],
    "github": [
        {
            "repo": "hugoycj/StableNormal",
            "commit": "1b7203c988384cb62286a25865d7354be8fdbbf7",
            "licence": "Apache-2.0",
            "dir": ".torch/hub/hugoycj_StableNormal_main",
        },
    ],
    "dinov2_weights": {
        "path": ".torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth",
        "bytes": 5,
        "sha256": "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51",
        "licence": "Apache-2.0",
    },
}

#: The pinned paths and the bytes each holds, matching PINS above.
CONTENT = {
    "weights/trellis-normal-v0-1/ckpts/model.safetensors": b"eleven byte",
    "weights/yoso-normal-v1-8-1/unet/diffusion.safetensors": b"7 bytes",
    ".torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth": b"five!",
}


@pytest.fixture(autouse=True)
def no_real_machine(tmp_path, monkeypatch):
    monkeypatch.delenv("HI3DGEN_INSTALL", raising=False)
    monkeypatch.delenv("HI3DGEN_PYTHON", raising=False)
    monkeypatch.setenv("KLIN_CACHE", str(tmp_path / "cache"))


@pytest.fixture
def install(tmp_path):
    """A clone with its pins and every pinned file at the right size."""
    def make(pins=PINS, content=None):
        root = tmp_path / "hi3dgen"
        root.mkdir(parents=True, exist_ok=True)
        if pins is not None:
            io.open(str(root / hi3dgen.PINS), "w", encoding="utf-8").write(
                json.dumps(pins))
        for relative, blob in (CONTENT if content is None else content).items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        (root / ".torch" / "hub" / "hugoycj_StableNormal_main").mkdir(
            parents=True, exist_ok=True)
        venv = root / ".venv" / "Scripts"
        venv.mkdir(parents=True, exist_ok=True)
        (venv / "python.exe").write_bytes(b"not really python")
        return str(root)
    return make


def run(made, argv, stream=None):
    stream = stream or io.StringIO()
    return cli.main(["--repo", made.root] + argv, stream=stream), stream.getvalue()


# --------------------------------------------------------------------- the pins


def test_the_pins_name_every_source_with_its_licence(install):
    pins = hi3dgen.load_pins(install())
    named = dict((name, licence) for name, licence, _rev, _dir in hi3dgen.sources(pins))
    assert named == {
        "Stable-X/trellis-normal-v0-1": "MIT",
        "Stable-X/yoso-normal-v1-8-1": "Apache-2.0",
        "hugoycj/StableNormal": "Apache-2.0",
        "dinov2_vitl14_reg4 weights": "Apache-2.0",
    }


def test_a_complete_install_has_nothing_missing(install):
    root = install()
    assert hi3dgen.missing(root, hi3dgen.load_pins(root)) == []


def test_a_file_that_is_gone_is_reported(install):
    root = install()
    os.remove(os.path.join(root, "weights", "yoso-normal-v1-8-1", "unet",
                           "diffusion.safetensors"))
    wrong = hi3dgen.missing(root, hi3dgen.load_pins(root))
    assert wrong == [("weights/yoso-normal-v1-8-1/unet/diffusion.safetensors", "absent")]


def test_a_file_that_changed_size_is_reported_rather_than_accepted(install):
    """The pinned file being replaced by a different one is the whole worry.

    Barinn's session 36 found two of five runtime fetches had moved, and the
    breakage read as a code bug. A size that does not match is the cheap half
    of noticing that; the fetch step's hash pass is the other half.
    """
    root = install()
    path = os.path.join(root, "weights", "trellis-normal-v0-1", "ckpts",
                        "model.safetensors")
    io.open(path, "wb").write(b"different length entirely")
    wrong = hi3dgen.missing(root, hi3dgen.load_pins(root))
    assert len(wrong) == 1
    assert "expected 11" in wrong[0][1]


def test_an_install_with_no_pins_will_not_run(install):
    root = install(pins=None)
    with pytest.raises(gen.GenError) as raised:
        hi3dgen.load_pins(root)
    assert "pins.json" in str(raised.value)


def test_checking_the_pins_prints_them_and_says_nothing_is_wrong(repo, install):
    made = repo()
    code, out = run(made, ["gen", "hi3dgen", "--install", install(), "--pins", "check"])
    assert code == 0
    assert "Stable-X/trellis-normal-v0-1" in out
    assert "Apache-2.0" in out
    assert "0 not as pinned" in out


def test_checking_the_pins_fails_when_one_is_gone(repo, install):
    made = repo()
    root = install()
    os.remove(os.path.join(root, ".torch", "hub", "checkpoints",
                           "dinov2_vitl14_reg4_pretrain.pth"))
    code, out = run(made, ["gen", "hi3dgen", "--install", root, "--pins", "check"])
    assert code == 1
    assert "1 not as pinned" in out


# ---------------------------------------------------------------------- the run


def test_a_run_refuses_to_start_when_a_pinned_file_is_not_as_pinned(repo, install,
                                                                    monkeypatch):
    """An offline run against unrecorded files proves nothing about what ran."""
    made = repo()
    root = install()
    os.remove(os.path.join(root, "weights", "trellis-normal-v0-1", "ckpts",
                           "model.safetensors"))
    monkeypatch.setattr(hi3dgen.subprocess, "run", _never)
    code, out = run(made, ["gen", "hi3dgen", "--install", root, "--image", __file__])
    assert code == 2
    assert "not as pinned" in out


def test_the_licence_pre_flight_asks_about_every_model_not_just_one(repo, install,
                                                                   monkeypatch):
    """Four models make one mesh, so a licence question is about all of them."""
    made = repo()
    monkeypatch.setattr(hi3dgen.subprocess, "run", _never)
    code, out = run(made, ["gen", "hi3dgen", "--install", install(),
                           "--image", __file__, "--dry-run"])
    assert code == 0
    for name in ("Stable-X/trellis-normal-v0-1", "Stable-X/yoso-normal-v1-8-1",
                 "hugoycj/StableNormal", "dinov2_vitl14_reg4 weights"):
        assert name in out


def test_an_untraceable_model_stops_a_checked_run(repo, install, monkeypatch):
    """Nothing here is in the ledger, which is exactly what --check is for."""
    made = repo()
    monkeypatch.setattr(hi3dgen.subprocess, "run", _never)
    code, out = run(made, ["gen", "hi3dgen", "--install", install(),
                           "--image", __file__, "--check"])
    assert code == 1
    assert "refusing to run" in out


def test_a_clean_run_says_what_it_proved_and_what_comes_next(repo, install,
                                                             monkeypatch):
    made = repo()
    monkeypatch.setattr(hi3dgen.subprocess, "run",
                        _proc(0, "HI3DGEN " + json.dumps(_REPORT), ""))
    code, out = run(made, ["gen", "hi3dgen", "--install", install(),
                           "--image", __file__])
    assert code == 0
    assert "watertight" in out
    assert "no restricted module could be imported" in out
    assert "klin conform blender" in out


def test_a_run_that_reached_for_restricted_code_is_a_failure(repo, install,
                                                             monkeypatch):
    made = repo()
    report = dict(_REPORT, ok=False,
                  error="restricted code was reached for; see restricted_attempts")
    monkeypatch.setattr(hi3dgen.subprocess, "run",
                        _proc(0, "HI3DGEN " + json.dumps(report), ""))
    code, out = run(made, ["gen", "hi3dgen", "--install", install(),
                           "--image", __file__])
    assert code == 2
    assert "restricted code was reached for" in out


def test_a_frame_with_no_object_found_is_said_out_loud(repo, install, monkeypatch):
    """The stove crop took this path in session 36 and made a slab of the frame."""
    made = repo()
    report = dict(_REPORT, foreground_found=False)
    monkeypatch.setattr(hi3dgen.subprocess, "run",
                        _proc(0, "HI3DGEN " + json.dumps(report), ""))
    code, out = run(made, ["gen", "hi3dgen", "--install", install(),
                           "--image", __file__])
    assert code == 0
    assert "found no object" in out


def test_a_clone_that_moved_off_the_pinned_revision_is_noted_not_refused(repo, install,
                                                                        monkeypatch):
    """Upstream may legitimately move; the pins then describe a different program."""
    made = repo()
    report = dict(_REPORT, code_revision="0123456789abcdef" * 2 + "0123")
    monkeypatch.setattr(hi3dgen.subprocess, "run",
                        _proc(0, "HI3DGEN " + json.dumps(report), ""))
    code, out = run(made, ["gen", "hi3dgen", "--install", install(),
                           "--image", __file__])
    assert code == 0
    assert "the pins were taken against" in out


def test_a_run_that_prints_no_result_says_what_the_interpreter_said(repo, install,
                                                                    monkeypatch):
    made = repo()
    monkeypatch.setattr(hi3dgen.subprocess, "run",
                        _proc(1, "loading weights", "CUDA out of memory"))
    code, out = run(made, ["gen", "hi3dgen", "--install", install(),
                           "--image", __file__])
    assert code == 2
    assert "CUDA out of memory" in out


def test_the_harness_for_the_other_interpreter_ships_but_is_not_a_subcommand():
    here = os.path.dirname(gen.__file__)
    assert os.path.isfile(os.path.join(here, "_inside_hi3dgen.py"))
    assert "_inside_hi3dgen" not in gen.adapters()


# ----------------------------------------------------------------------- helpers


_REPORT = {
    "ok": True,
    "glb": "out/stove.glb",
    "faces": 415386,
    "vertices": 207694,
    "watertight": True,
    "foreground_found": True,
    "load_seconds": 21.4,
    "pipeline_seconds": 12.6,
    "export_seconds": 0.4,
    "peak_vram_gb": 6.1,
    "peak_process_gb": 11.4,
    "mode": "geometry",
    "code_revision": hi3dgen.PINNED_REVISION,
    "code_modified": [],
    "restricted_loaded": [],
    "restricted_attempts": [],
    "restricted_blocked": True,
}


def _never(*a, **kw):
    raise AssertionError("nothing should have been started")


def _proc(code, out, err):
    class Done(object):
        returncode = code
        stdout = out
        stderr = err

    return lambda *a, **kw: Done()
