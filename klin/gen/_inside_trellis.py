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

PREFIX = "TRELLIS "
NOTE = "trellis-note: "

BLOCKED = ("nvdiffrast", "nvdiffrec", "nvdiffrec_render")


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


def generate(job, blocker):
    import torch
    from PIL import Image

    sys.path.insert(0, job["install"])
    from o_voxel.pipelines import OVoxelImageTo3DPipeline

    note("loading the pipeline")
    pipeline = OVoxelImageTo3DPipeline.from_pretrained(job.get("tier"))
    pipeline.cuda()

    image = Image.open(job["image"])
    started = time.time()
    outputs = pipeline.run(image, seed=job["seed"])
    mesh = outputs["mesh"][0]

    note("exporting %s" % job["mode"])
    glb = mesh.to_glb(bake_texture=(job["mode"] == "textured"))
    glb.export(job["output"])
    seconds = time.time() - started

    peak = None
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / float(1 << 30)
    return seconds, peak


def main():
    job = json.loads(sys.argv[1])
    blocker = guard() if job["mode"] == "geometry" else None

    seconds, peak = generate(job, blocker)
    facts = verify(job["output"])

    payload = {
        "ok": True,
        "glb": job["output"],
        "image": job["image"],
        "mode": job["mode"],
        "seed": job["seed"],
        "tier": job["tier"],
        "seconds": round(seconds, 1),
        "peak_vram_gb": round(peak, 2) if peak is not None else None,
        "restricted_loaded": sorted(
            m for m in sys.modules if m.split(".")[0] in BLOCKED),
        "restricted_attempts": blocker.attempts if blocker else None,
        "restricted_blocked": blocker is not None,
    }
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
