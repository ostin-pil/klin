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

#: The remover the stock `pipeline.json` names, for the report. It is
#: briaai's RMBG-2.0, whose weights are CC BY-NC 4.0, and it is never loaded.
STOCK_REMOVER = "briaai/RMBG-2.0"

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


class RemoverRefused(RuntimeError):
    pass


class Remover(object):
    """The background remover this run uses, which is never the install's.

    TRELLIS.2 removes the background of any input that has no alpha of its
    own, and its `pipeline.json` builds the remover from `briaai/RMBG-2.0`,
    whose weights are CC BY-NC 4.0. So the geometry path that carried no NVIDIA
    code still ran a non-commercial model over every opaque image. Blocking an
    import could not have caught it, because nothing restricted is imported:
    the restricted part is weights, loaded through the same `transformers`
    call as any other.

    Changing the name the pipeline asks for is not enough either. The wrapper
    in the stableprojectorz install loads a local `MODELS/RMBG-2.0` whenever
    one exists and ignores `model_name`, and that install keeps one there. So
    the wrapper class is replaced outright and this one is built instead.

    With a directory, this loads that BiRefNet and does what the wrapper did.
    Without one it loads nothing and refuses when called, which is the same
    choice the import blocker makes: a stub that quietly returned the image
    would leave open what masked it, and a refusal leaves nothing open.
    """

    def __init__(self, path, record):
        self.record = record
        self.model = None
        record["loaded"] = None
        record["called"] = False
        if path:
            self.model = load_birefnet(path)
            record["loaded"] = path

    def to(self, device):
        if self.model is not None:
            self.model.to(device)

    def cuda(self):
        if self.model is not None:
            self.model.cuda()

    def cpu(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, image):
        self.record["called"] = True
        if self.model is None:
            raise RemoverRefused(
                "this image has no alpha of its own, so TRELLIS.2 would remove "
                "its background with %s, which is CC BY-NC 4.0. Pass --rembg "
                "with a BiRefNet directory whose licence the project accepts, "
                "or hand it an image that is already masked"
                % self.record.get("configured", STOCK_REMOVER))
        import torch
        from torchvision import transforms

        size = image.size
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        weight = next(self.model.parameters())
        batch = transform(image.convert("RGB")).unsqueeze(0).to(
            device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            preds = self.model(batch)[-1].sigmoid().float().cpu()
        mask = transforms.ToPILImage()(preds[0].squeeze()).resize(size)
        image.putalpha(mask)
        return image


def load_birefnet(path):
    """Build a BiRefNet from its own directory's code, and load its weights strictly.

    Not `AutoModelForImageSegmentation.from_pretrained`, which is what the
    install's wrapper calls: under this install's transformers (4.57) it ties
    weights through `config.get_text_config`, and BiRefNet's own `Config`
    has no `is_encoder_decoder`, so upstream's code does not load at all. The
    install's RMBG-2.0 copy carries a fix for that, but it is briaai's file,
    and borrowing it would bring back the licence this replaces. So the class
    comes from the directory being loaded, the weights go in by name, and a
    single missing or unexpected key refuses: a partial load masks images
    with a network that is partly random, and the masks look plausible.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(path, trust_remote_code=True,
                                        local_files_only=True)
    model_class = get_class_from_dynamic_module("birefnet.BiRefNet", path,
                                                local_files_only=True)
    model = model_class(bb_pretrained=False, config=config)
    result = model.load_state_dict(
        load_file(os.path.join(path, "model.safetensors")), strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RemoverRefused(
            "the remover at %s did not load cleanly: %d missing and %d "
            "unexpected keys" % (path, len(result.missing_keys),
                                 len(result.unexpected_keys)))
    return model.eval()


def guard_remover(job, rembg, record):
    """Replace the install's remover class before `from_pretrained` looks it up.

    The pipeline finds it with `getattr(rembg, name)`, so the module attribute
    is the one seam, and `BiRefNet` is the only class the package exports.
    Whatever name it is asked for, the remover is the one klin was given,
    because the wrapper being replaced ignores the name anyway. A config that
    named some other class would slip past this, which is why `confirm_remover`
    checks what the pipeline actually holds once it is built.
    """
    def build(model_name=STOCK_REMOVER, **kwargs):
        record["configured"] = model_name
        return Remover(job.get("rembg"), record)

    rembg.BiRefNet = build
    return build


def confirm_remover(pipeline):
    """Refuse a pipeline whose remover klin did not build."""
    held = getattr(pipeline, "rembg_model", None)
    if held is not None and not isinstance(held, Remover):
        raise RemoverRefused(
            "the pipeline built its background remover as %s, which klin did "
            "not replace, so what it would load is unknown"
            % type(held).__name__)


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

    from trellis2.pipelines import Trellis2ImageTo3DPipeline, rembg
    import o_voxel

    install_triton_shim()
    remover = {}
    guard_remover(job, rembg, remover)

    note("loading %s" % job["model"])
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(job["model"])
    confirm_remover(pipeline)
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
        "remover": remover,
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
