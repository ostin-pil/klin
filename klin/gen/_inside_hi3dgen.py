"""The half of the Hi3DGen adapter that runs inside Hi3DGen's own interpreter.

klin never imports this. It needs torch, xformers, spconv, trimesh and a clone
of Stable3DGen on `sys.path`, none of which klin has or wants. The leading
underscore keeps the module out of adapter discovery, where importing it would
fail on the first thing it touches.

Its only caller is `klin/gen/hi3dgen.py`, as

    <install>/.venv/Scripts/python.exe <this> '<job json>'

**The flow is the upstream demo's** (`app.py generate_3d` at Stable3DGen
c29f668) with its defaults: BiRefNet background removal at 1024, a
StableNormal-turbo normal map at 768, then the sparse-structure sampler at 50
steps and the latent sampler at 6, both at guidance 3.0. Two upstream paths are
stepped around rather than copied, and both cost a day to find:

- **The demo never loads its own StableNormal.** Its local load passes
  `pretrained=True`, which `StableNormal_turbo` does not accept, and a bare
  `except` then falls through to downloading GitHub's main branch. This calls
  the local copy without it.
- **When background removal finds no object**, the pipeline hands StableNormal
  the bare image and StableNormal masks it with a BiRefNet it fetches by hub
  id. Here it is given the pipeline's pinned BiRefNet instead, so that path
  runs offline like every other. `foreground_found` says which path a run took.

**The proof is a removal**, as in the TRELLIS adapter. Before torch is
imported, a meta-path finder refuses nvdiffrast, nvdiffrec, nvdiffrec_render,
kaolin and anything named flexicube(s), and records every attempt with the line
that made it. Upstream removed all of it; this is what makes that checkable
rather than believed.

**Nothing is fetched.** Both Hugging Face hubs are offline and torch.hub's
downloader is replaced by one that raises, so every weight and every line of
code that runs is one the pins put on disk. A missing pin is a traceback,
never whatever main holds today.

**Two switches this card needs.** xformers' FlashAttention-3 path is off: it is
Hopper-only and dies on sm_120, and the CUTLASS kernel it falls back to was
checked against the CPU before any of this was trusted. spconv runs its Native
algorithm, which the demo sets too. Both are recorded in the output rather than
left as machine lore.

**The timing boundaries are the TRELLIS adapter's**, so the two generators are
comparable: `pipeline_seconds` is the sampler alone after a peak-memory reset
placed past loading, `export_seconds` is the GLB write, and `peak_vram_gb` is
decimal GB read at the same boundary. `peak_process_gb` is the whole process,
which is the different question of whether a run fits on the card at all.
"""

import importlib.abc
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback

PREFIX = "HI3DGEN "
NOTE = "hi3dgen-note: "

BLOCKED_ROOTS = ("nvdiffrast", "nvdiffrec", "nvdiffrec_render", "kaolin")

#: The demo's defaults: `app.py generate_3d` and its sliders.
PREPROCESS_RESOLUTION = 1024
NORMAL_RESOLUTION = 768
SS_PARAMS = {"steps": 50, "cfg_strength": 3.0}
SLAT_PARAMS = {"steps": 6, "cfg_strength": 3.0}

#: What the cleanliness check covers. Upstream's PNGs always read as modified:
#: its `.gitattributes` routes `*.png` through LFS but they were committed as
#: plain blobs, so including them would make every run report a dirty tree.
CODE_PATHS = ("hi3dgen", "app.py", "requirements.txt")


def note(text):
    print("%s%s" % (NOTE, text), flush=True)


def answer(payload):
    print("%s%s" % (PREFIX, json.dumps(payload, sort_keys=True)), flush=True)


def restricted(fullname):
    parts = fullname.split(".")
    return parts[0] in BLOCKED_ROOTS or any(p.startswith("flexicube") for p in parts)


class BlockedImport(ImportError):
    pass


class Blocker(importlib.abc.MetaPathFinder):
    """Refuse the restricted modules, and remember who asked.

    A run that never tried is the claim. A run that tried and was stopped is a
    different fact: some path still reaches for the code, which is worth
    knowing even when the mesh comes out anyway.
    """

    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if restricted(fullname):
            self.attempts.append({"module": fullname, "from": caller()})
            raise BlockedImport(
                "%s is blocked for this run: it is NVIDIA non-commercial code"
                % fullname
            )
        return None


def caller():
    for frame in reversed(traceback.extract_stack()[:-3]):
        if "importlib" not in frame.filename and not frame.filename.startswith("<frozen"):
            return "%s:%d" % (frame.filename, frame.lineno)
    return None


def guard():
    """Install the blocker, refusing to continue if it is already too late."""
    already = sorted(m for m in sys.modules if restricted(m))
    if already:
        raise RuntimeError(
            "restricted modules were imported before the blocker: %s" % already
        )
    blocker = Blocker()
    sys.meta_path.insert(0, blocker)
    return blocker


def environment(install):
    """Offline, pinned, and rooted where the relative weight paths point."""
    os.environ["TORCH_HOME"] = os.path.join(install, ".torch")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["SPCONV_ALGO"] = "native"
    os.chdir(install)
    sys.path.insert(0, install)


def refuse_downloads(torch):
    def refuse(url, *args, **kwargs):
        raise RuntimeError(
            "torch.hub tried to download %s. Every weight this run may use is "
            "pinned; re-run the fetch step rather than taking today's copy" % url
        )

    torch.hub.download_url_to_file = refuse


def code_state(install):
    """The clone's revision, and whether the code that runs was edited."""
    try:
        rev = subprocess.run(
            ["git", "-C", install, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", install, "status", "--porcelain", "--"] + list(CODE_PATHS),
            capture_output=True, text=True, check=True).stdout.split("\n")
        return rev, [line for line in dirty if line.strip()]
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, ["git unavailable: %s" % exc]


def loaded_modules():
    """Every non-stdlib top-level module in the process, and where it came from.

    The licence surface of the run, recorded rather than reasoned about. A
    module with no distribution behind it is reported by its file, because that
    is the case worth looking at.
    """
    dists = importlib.metadata.packages_distributions()
    stdlib = set(sys.stdlib_module_names)
    loaded = {}
    for name in sorted(set(m.split(".")[0] for m in list(sys.modules))):
        if name in stdlib or name.startswith("_") or name == "__main__":
            continue
        if name in dists:
            dist = dists[name][0]
            try:
                loaded[name] = "%s %s" % (dist, importlib.metadata.version(dist))
            except importlib.metadata.PackageNotFoundError:
                loaded[name] = dist
            continue
        where = getattr(sys.modules.get(name), "__file__", None) or ""
        loaded[name] = "local %s" % where.replace("\\", "/")
    return loaded


def verify(path):
    """Read back what shipped, not what was left in memory.

    Hi3DGen exports geometry and never a texture, so unlike the TRELLIS adapter
    there is no bake to tell apart from a placeholder. What is worth reporting
    instead is whether the surface closed: a watertight mesh is the one that
    collapses to a prop budget in a single pass, and an open one is the shape
    that will not.
    """
    import trimesh

    mesh = trimesh.load(path, force="mesh")
    return {
        "glb_bytes": os.path.getsize(path),
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "watertight": bool(mesh.is_watertight),
        "extents": [round(float(x), 4) for x in mesh.extents],
    }


def generate(job):
    """Load every model, then generate once with the clock started after.

    BiRefNet and StableNormal are loaded before the clock deliberately: they
    are per-process costs, and folding them into a per-image number would make
    a batch of one look like the per-image cost of a batch of ten.
    """
    import torch

    refuse_downloads(torch)
    # Hopper-only, and it dies on sm_120. Off before anything builds an
    # attention op, not after.
    from xformers.ops.fmha import dispatch

    dispatch._set_use_fa3(False)

    from PIL import Image
    from hi3dgen.pipelines import Hi3DGenPipeline

    note("loading the pipeline and the normal predictor")
    started = time.time()
    pipeline = Hi3DGenPipeline.from_pretrained("weights/trellis-normal-v0-1")
    pipeline.cuda()
    pipeline._lazy_load_birefnet()
    normal_predictor = torch.hub.load(
        os.path.join(torch.hub.get_dir(), "hugoycj_StableNormal_main"),
        "StableNormal_turbo", source="local",
        yoso_version="yoso-normal-v1-8-1", local_cache_dir="./weights")
    # StableNormal fetches its own BiRefNet only while this is None.
    normal_predictor.segmentation_handler.birefnet_model = pipeline.birefnet_model
    load_seconds = time.time() - started

    image = Image.open(job["image"])
    stem = os.path.splitext(job["output"])[0]

    # Read before the reset, because the reset is what makes the generation
    # peak comparable with the TRELLIS adapter's and it destroys the loading
    # peak in the same move. Whether a run fits on the card is the other
    # question, and it needs both halves.
    load_peak = torch.cuda.max_memory_allocated() / 1e9
    load_reserved = torch.cuda.max_memory_reserved() / 1e9
    torch.cuda.reset_peak_memory_stats()
    note("preprocessing and predicting normals")
    t0 = time.time()
    rgba = pipeline.preprocess_image(image, resolution=PREPROCESS_RESOLUTION)
    t1 = time.time()
    normal = normal_predictor(rgba, resolution=NORMAL_RESOLUTION,
                              match_input_resolution=True, data_type="object")
    t2 = time.time()
    note("sampling")
    outputs = pipeline.run(
        normal, seed=job["seed"], formats=["mesh"], preprocess_image=False,
        sparse_structure_sampler_params=dict(SS_PARAMS),
        slat_sampler_params=dict(SLAT_PARAMS))
    t3 = time.time()
    peak = torch.cuda.max_memory_allocated() / 1e9

    note("exporting")
    outputs["mesh"][0].to_trimesh(transform_pose=True).export(job["output"])
    t4 = time.time()

    rgba.save(stem + ".input.png")
    normal.save(stem + ".normal.png")

    return {
        "foreground_found": rgba.mode == "RGBA",
        "load_seconds": round(load_seconds, 1),
        "preprocess_seconds": round(t1 - t0, 2),
        "normal_seconds": round(t2 - t1, 2),
        "pipeline_seconds": round(t3 - t2, 1),
        "export_seconds": round(t4 - t3, 1),
        "seconds": round(t4 - t0, 1),
        "peak_vram_gb": round(peak, 2),
        "load_peak_gb": round(load_peak, 2),
        "peak_process_gb": round(max(load_peak, peak), 2),
        "peak_reserved_gb": round(
            max(load_reserved, torch.cuda.max_memory_reserved() / 1e9), 2),
        "input_png": stem + ".input.png",
        "normal_png": stem + ".normal.png",
    }


def main():
    job = json.loads(sys.argv[1])
    blocker = guard()
    environment(job["install"])

    facts = generate(job)
    rev, dirty = code_state(job["install"])

    payload = {
        "ok": True,
        "glb": job["output"],
        "image": job["image"],
        "mode": "geometry",
        "seed": job["seed"],
        "code_revision": rev,
        "code_modified": dirty,
        "settings": {
            "preprocess_resolution": PREPROCESS_RESOLUTION,
            "normal_resolution": NORMAL_RESOLUTION,
            "sparse_structure_sampler": SS_PARAMS,
            "slat_sampler": SLAT_PARAMS,
            "attention": "xformers, FlashAttention-3 off",
            "sparse_conv": "spconv, native algorithm",
            "stablenormal_birefnet": "the pipeline's pinned BiRefNet, not the hub's",
        },
        "restricted_blocked": True,
        "restricted_attempts": blocker.attempts,
        "restricted_loaded": sorted(m for m in sys.modules if restricted(m)),
        "loaded_modules": loaded_modules(),
    }
    payload.update(facts)
    payload.update(verify(job["output"]))

    # A clean run that made nothing usable would pass every other check here.
    if payload["faces"] < 1000:
        payload["ok"] = False
        payload["error"] = "the mesh is implausibly small: %d faces" % payload["faces"]
    if payload["restricted_attempts"] or payload["restricted_loaded"]:
        payload["ok"] = False
        payload["error"] = "restricted code was reached for; see restricted_attempts"
    answer(payload)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback as tb

        answer({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": tb.format_exc()})
        sys.exit(1)
