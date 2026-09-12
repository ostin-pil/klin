"""The half of the TRELLIS adapter that runs inside TRELLIS's own interpreter.

klin never imports this. It needs torch, trimesh and a compiled voxel
extension, none of which klin has or wants; klin ships two dependencies and
keeps it that way. The leading underscore keeps the module out of adapter
discovery, where importing it would fail on the first thing it touches.

Its only caller is `klin/gen/trellis.py`, as

    <install>/venv/Scripts/python.exe <this> '<job json>'

**The proof here is a removal, not a reading.** The claim is that a
geometry-only run loads no NVIDIA non-commercial code, and reading the diff is
a weak way to check that: a transitive import three modules deep does not
appear in one. So this takes nvdiffrast, nvdiffrec and nvdiffrec_render out of
the import system for the whole process and generates anyway. If a mesh comes
out, no restricted code executed, because none of it could be loaded.

Refusing the import is deliberately blunter than stubbing the modules. A stub
would let the pipeline believe it had a rasterizer and leave open the question
of what it did with it. Refusing means any use at all is a loud traceback, and
the absence of that traceback is the evidence.
"""

import importlib.abc
import json
import os
import sys
import time

# Set before torch is imported anywhere, which is why it sits at module scope
# rather than beside the import that needs it. The install's own scripts set
# it too: without it the 1024 cascade fragments the allocator and a run that
# fits can still fail on a large prop.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PREFIX = "TRELLIS "
NOTE = "trellis-note: "

BLOCKED = ("nvdiffrast", "nvdiffrec", "nvdiffrec_render")

#: What `to_glb` is allowed to keep. The install's own batch script uses this
#: number, so a mesh out of klin is the same mesh rounds 14 and 16 measured.
#: It is not a triangle budget for the game: TRELLIS.2's exports are a hollow
#: double skin and do not decimate below about 900 to 2,700 triangles without
#: being rebuilt first, which is a step that belongs downstream of generating.
DECIMATION_TARGET = 200000


def note(text):
    print("%s%s" % (NOTE, text), flush=True)


def answer(payload):
    print("%s%s" % (PREFIX, json.dumps(payload, sort_keys=True)), flush=True)


class BlockedImport(ImportError):
    pass


class Blocker(importlib.abc.MetaPathFinder):
    """Refuse the restricted modules, and remember that they were asked for.

    Recording the attempt matters as much as refusing it. A run that never
    tried is the claim; a run that tried and was stopped is a different fact
    and would mean the patch had missed a path.
    """

    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED:
            self.attempts.append(fullname)
            raise BlockedImport(
                "%s is blocked for this run: it is NVIDIA non-commercial code "
                "and this run claims not to use it" % fullname
            )
        return None


def guard():
    """Install the blocker, refusing to continue if it is already too late."""
    already = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
    if already:
        raise RuntimeError(
            "restricted modules were imported before the blocker: %s" % already
        )
    blocker = Blocker()
    sys.meta_path.insert(0, blocker)
    return blocker


def verify(path):
    """Read back what shipped, not what was left in memory.

    Two different claims, and only one of them is about the file anybody else
    will open. trimesh attaches a default material to any TextureVisuals, so a
    geometry-only export still lands carrying a placeholder base colour: two
    pixels square, uniform, invented by the exporter and holding no scene
    information. A real bake is texture_size square and varied. Checking the
    in-memory mesh reports no texture and misses the placeholder entirely,
    which reads as a discrepancy to anybody who later opens the file.
    """
    import numpy as np
    import trimesh

    scene = trimesh.load(path)
    mesh = list(scene.geometry.values())[0] if hasattr(scene, "geometry") else scene
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    texture = getattr(material, "baseColorTexture", None)
    facts = {
        "glb_bytes": os.path.getsize(path),
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "has_uvs": getattr(visual, "uv", None) is not None,
        "texture_size": list(texture.size) if texture is not None else None,
        "texture_distinct_colours": None,
    }
    if texture is not None:
        pixels = np.asarray(texture.convert("RGB")).reshape(-1, 3)
        facts["texture_distinct_colours"] = len(set(tuple(p) for p in pixels))
    # A placeholder is tiny and flat. Anything bigger or varied is a bake.
    facts["has_baked_texture"] = bool(
        texture is not None
        and not (max(texture.size) <= 4 and facts["texture_distinct_colours"] == 1)
    )
    return facts


def install_triton_shim():
    """The shim the install's own `run_batch.py` carries.

    Nothing to do with licensing: without it the pipeline fails on this
    machine's `flex_gemm` build, which has no triton kernels compiled in. It is
    installed only when the attribute is genuinely absent, so a build that has
    them keeps them.
    """
    import types

    import torch
    import flex_gemm.kernels as kernels

    if hasattr(kernels, "triton"):
        return
    module = types.ModuleType("flex_gemm.kernels.triton")

    def fwd(feats, indices, weight):
        idx = indices.long().clamp_(0, feats.shape[0] - 1)
        return (feats[idx] * weight.to(feats.dtype).unsqueeze(-1)).sum(1)

    def bwd(go, indices, weight, count):
        channels = go.shape[-1]
        idx = indices.long().clamp(0, count - 1).reshape(-1)
        contrib = (go.unsqueeze(1) * weight.to(go.dtype).unsqueeze(-1)).reshape(-1, channels)
        return torch.zeros(
            (count, channels), device=go.device, dtype=go.dtype
        ).index_add_(0, idx, contrib)

    module.indice_weighed_sum_fwd = fwd
    module.indice_weighed_sum_bwd_input = bwd
    kernels.triton = module
    note("triton shim installed")


def generate(job):
    """Load, run, export. The three steps are timed apart, on purpose.

    The boundaries are the ones round 14 measured and the generation benchmark
    records, so a number out of this adapter is comparable with the numbers
    already in the plan:

    - `pipeline_seconds` covers `pipeline.run` alone, after a peak-memory reset
      that puts model loading outside the measurement;
    - `export_seconds` covers the simplify and the `to_glb`;
    - `peak_vram_gb` is decimal GB, not GiB, read after that same reset.

    Reporting one end-to-end number instead would not be wrong, but it would
    not be the number the plan holds, and a comparison across the two would
    quietly be against different things.
    """
    import torch
    from PIL import Image

    # trellis2 and o_voxel are imported from the install directory rather than
    # pip-installed, which is why the install's own scripts only run from
    # inside it.
    if job["install"] not in sys.path:
        sys.path.insert(0, job["install"])

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    install_triton_shim()

    note("loading %s" % job["model"])
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(job["model"])
    pipeline.cuda()

    image = Image.open(job["image"])
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    # The tier is a run parameter and not a checkpoint: one model serves every
    # tier, and `pipeline_type` chooses the resolution cascade.
    mesh = pipeline.run(image, seed=job["seed"], pipeline_type=job["tier"])[0]
    pipeline_seconds = time.time() - started
    peak = torch.cuda.max_memory_allocated() / 1e9

    note("exporting %s" % job["mode"])
    export_started = time.time()
    # 16777216 is the install's own ceiling: high enough to be no cut at this
    # resolution, and required because `to_glb` reads a simplified mesh.
    mesh.simplify(16777216)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=DECIMATION_TARGET,
        texture_size=2048,
        bake_texture=(job["mode"] == "textured"),
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )
    export_seconds = time.time() - export_started

    glb.export(job["output"], extension_webp=True)
    return {
        "pipeline_seconds": round(pipeline_seconds, 1),
        "export_seconds": round(export_seconds, 1),
        "seconds": round(pipeline_seconds + export_seconds, 1),
        "peak_vram_gb": round(peak, 2),
    }


def main():
    job = json.loads(sys.argv[1])
    blocker = guard() if job["mode"] == "geometry" else None

    timings = generate(job)
    facts = verify(job["output"])

    payload = {
        "ok": True,
        "glb": job["output"],
        "image": job["image"],
        "mode": job["mode"],
        "seed": job["seed"],
        "tier": job["tier"],
        "model": job["model"],
        "restricted_loaded": sorted(
            m for m in sys.modules if m.split(".")[0] in BLOCKED),
        "restricted_attempts": blocker.attempts if blocker else None,
        "restricted_blocked": blocker is not None,
    }
    payload.update(timings)
    payload.update(facts)
    # A patch that made the tool clean by making it useless would pass every
    # check above and fail the project, so the output has to be plausible too.
    if payload["faces"] < 1000:
        payload["ok"] = False
        payload["error"] = "the mesh is implausibly small: %d faces" % payload["faces"]
    answer(payload)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        answer({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": traceback.format_exc()})
        sys.exit(1)
