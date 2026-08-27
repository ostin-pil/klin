"""TRELLIS.2: image to mesh, on the geometry path that carries no NVIDIA code.

TRELLIS.2's own code and its 4B weights are MIT. The non-commercial taint comes
from two optional NVIDIA dependencies, nvdiffrast and nvdiffrec, which are
source-licensed for non-commercial use, and from one module-scope import that
drags them in whether or not anything uses them. A project that discards
generated textures and re-pins its meshes to its own atlas never wants the
texture bake those dependencies serve, and yet as installed it loaded them on
every run. That is what kept the tool labelled evaluation-only.

**So this adapter carries two jobs, and the second is what makes the first
legitimate.** It generates, and it patches the install so that generating is
clean. Keeping them apart would mean a generator whose licence status depended
on a step somebody else remembered to run.

Three things about the patch are worth knowing before changing any of it, each
of which cost a session to learn:

- **The file exists in three places and the one Python imports is the copy in
  `site-packages`.** Patching the source tree alone leaves a repository that
  reads as patched and a run that still bakes, which is a false provenance
  claim rather than a missing feature.
- **The install is not a git repository**, so a reinstall silently reverts
  everything. The patch is therefore idempotent and re-runnable rather than an
  edit somebody made once.
- **The clean path is the non-default argument.** `bake_texture` still defaults
  to `True` upstream, so this makes a clean path available without making the
  tool clean by default, and a run that does not ask for geometry only is
  exactly as restricted as it ever was.

TRELLIS runs under its own interpreter, with torch and trimesh and a compiled
voxel extension, none of which klin will ever import: klin ships two
dependencies and intends to keep doing so. So generation is a subprocess into
that interpreter, and the script it runs is `_inside_trellis.py`, kept out of
adapter discovery by its leading underscore.

The proof that a run touched no restricted code is not a reading of the diff. A
transitive import three modules deep does not appear in one. The harness blocks
nvdiffrast, nvdiffrec and nvdiffrec_render at the import system for the whole
process and generates anyway, so the absence of a traceback is the evidence.
"""

import io
import json
import os
import subprocess

from . import GenError, posture
from .. import index, ledger, manifest

NAME = "trellis"
HELP = "TRELLIS.2 image-to-mesh, geometry only (needs its own python)"

INSTALL_ENV = "TRELLIS_INSTALL"
PYTHON_ENV = "TRELLIS_PYTHON"

#: Every copy of the exporter, most important first. site-packages is the one
#: that runs; the others keep a reinstall or a rebuild from quietly undoing it.
TARGETS = (
    "venv/Lib/site-packages/o_voxel/postprocess.py",
    "o-voxel/o_voxel/postprocess.py",
    "o-voxel/build/lib.win-amd64-cpython-311/o_voxel/postprocess.py",
)

MARK = "klin geometry-only patch"

IMPORT_OLD = "import nvdiffrast.torch as dr\n"

IMPORT_NEW = """# nvdiffrast is imported lazily, inside the texture-baking block that uses it.
#
# {mark}. nvdiffrast and nvdiffrec are NVIDIA source-code licensed for
# non-commercial use; TRELLIS.2 itself is MIT. A module-scope import means
# every consumer of this exporter loads the restricted code even when it only
# wants geometry. Moving it into the bake block is what lets a geometry-only
# run prove it never touched NVIDIA code: block the module at the import
# system and the run still succeeds. MIT permits the modification.
""".format(mark=MARK)

PARAM_OLD = "    texture_size: int = 2048,\n"
PARAM_NEW = (
    "    texture_size: int = 2048,\n"
    "    bake_texture: bool = True,\n"
)

DOC_OLD = "        texture_size: size of the texture for baking\n"
DOC_NEW = (
    "        texture_size: size of the texture for baking\n"
    "        bake_texture: if False, return after UV unwrapping with geometry,\n"
    "            normals and UVs but no material, and never import nvdiffrast\n"
    "            ({mark})\n".format(mark=MARK)
)

EXIT_ANCHOR = "    # --- Texture Baking (Attribute Sampling) ---\n"

EXIT_NEW = """    # --- Geometry-only exit ({mark}) ---
    #
    # Everything above this line runs on MIT dependencies: CuMesh for
    # extraction, cleaning, simplification and remeshing, xatlas for the UV
    # unwrap. Everything below it is the texture bake, and the bake is the only
    # thing in this file that touches nvdiffrast. A consumer that retextures
    # against its own atlas wants precisely what is on the stack right here.
    if not bake_texture:
        vertices_np = out_vertices.cpu().numpy()
        faces_np = out_faces.cpu().numpy()
        uvs_np = out_uvs.cpu().numpy()
        normals_np = out_normals.cpu().numpy()
        # The same axis and UV conventions the textured path applies below. A
        # geometry-only export that came out on a different convention than the
        # textured one would be a silent trap for anything consuming both.
        vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
        normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
        uvs_np[:, 1] = 1 - uvs_np[:, 1]
        geometry_mesh = trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            vertex_normals=normals_np,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=uvs_np),
        )
        if use_tqdm:
            pbar.update(2)
            pbar.close()
        if verbose:
            print("Geometry-only export: skipped texture baking")
        return geometry_mesh

""".format(mark=MARK) + EXIT_ANCHOR

LAZY_OLD = (
    "    # Setup differentiable rasterizer context\n"
    "    ctx = dr.RasterizeCudaContext()\n"
)
LAZY_NEW = (
    "    # Setup differentiable rasterizer context\n"
    "    # Imported here rather than at module scope, so a geometry-only run\n"
    "    # never loads it ({mark}).\n".format(mark=MARK) +
    "    import nvdiffrast.torch as dr\n"
    "    ctx = dr.RasterizeCudaContext()\n"
)

#: label, the text to find, the text to write, and a fingerprint that says the
#: edit is present. The fingerprint is separate from the replacement, and none
#: of them mentions the mark, so an install patched under another tool's name
#: still reads as patched rather than being patched a second time.
EDITS = (
    ("module-scope nvdiffrast import", IMPORT_OLD, IMPORT_NEW,
     "nvdiffrast is imported lazily"),
    ("bake_texture parameter", PARAM_OLD, PARAM_NEW,
     "    bake_texture: bool = True,\n"),
    ("bake_texture docstring", DOC_OLD, DOC_NEW,
     "bake_texture: if False, return after UV unwrapping"),
    ("geometry-only exit", EXIT_ANCHOR, EXIT_NEW, "    if not bake_texture:\n"),
    ("lazy import at the bake", LAZY_OLD, LAZY_NEW,
     "    import nvdiffrast.torch as dr\n    ctx = dr.RasterizeCudaContext()\n"),
)


def configure(parser):
    parser.add_argument("--image", default=None, help="the image to lift into a mesh")
    parser.add_argument("--out", default=None, help="where the glb goes")
    parser.add_argument("--name", default=None, help="the output's stem")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tier", default="1024_cascade")
    parser.add_argument("--install", default=None, help="the TRELLIS.2 install")
    parser.add_argument("--python", dest="interpreter", default=None,
                        help="the interpreter inside that install")
    parser.add_argument("--patch", choices=("check", "apply", "revert"), default=None,
                        help="inspect or change the install's geometry-only patch")
    parser.add_argument("--textured", action="store_true",
                        help="bake a texture, which loads the restricted code")
    return parser


# ------------------------------------------------------------------- the patch


def read(path):
    r"""Text with newlines normalised, plus the convention to write back.

    The install is a vendor tree in CRLF. Writing LF back rewrites every line,
    which turns the audit diff, the artifact that makes the licence claim
    checkable by somebody who does not trust this code, into noise.
    """
    raw = io.open(path, "rb").read()
    crlf = raw.count(b"\r\n")
    newline = "\r\n" if crlf > raw.count(b"\n") - crlf else "\n"
    return raw.decode("utf-8").replace("\r\n", "\n"), newline


def write(path, text, newline):
    io.open(path, "w", encoding="utf-8", newline=newline).write(text)


def state(text):
    """patched, clean or broken. Never guessed from one marker alone.

    A split state is the dangerous one, because the source tree can read as
    patched while the copy Python imports still bakes.
    """
    applied = sum(1 for _, _, _, fingerprint in EDITS if fingerprint in text)
    if applied == len(EDITS):
        return "patched"
    if applied == 0:
        return "clean"
    return "broken"


def apply_to(text):
    for label, old, new, fingerprint in EDITS:
        if fingerprint in text:
            continue
        if old not in text:
            raise GenError(
                "cannot apply %r: the anchor is not in the file. Upstream has "
                "changed; read it before trusting this patch" % label
            )
        text = text.replace(old, new, 1)
    return text


def revert_to(text):
    for label, old, new, fingerprint in reversed(EDITS):
        if new in text:
            text = text.replace(new, old, 1)
        elif fingerprint in text:
            raise GenError(
                "cannot revert %r: it is patched, but not with text klin "
                "wrote. Reverting by hand is safer than guessing" % label
            )
    return text


def copies(install):
    return [(rel, os.path.join(install, rel.replace("/", os.sep))) for rel in TARGETS]


def patch(ctx, install, mode):
    """Report or change the geometry-only patch across every copy."""
    rows = []
    missing = []
    for rel, path in copies(install):
        if not os.path.isfile(path):
            missing.append(rel)
            continue
        text, newline = read(path)
        before = state(text)
        if mode == "apply" and before != "patched":
            write(path, apply_to(text), newline)
        elif mode == "revert" and before != "clean":
            write(path, revert_to(text), newline)
        rows.append((rel, before, state(read(path)[0])))

    for rel, before, after in rows:
        ctx.say("  %-9s to %-9s %s" % (before, after, rel))
    for rel in missing:
        ctx.say("  %-9s    %-9s %s" % ("absent", "", rel))
    if not rows:
        raise GenError("no copy of the exporter under %s" % install)

    want = {"apply": "patched", "revert": "clean"}.get(mode)
    if want and [row for row in rows if row[2] != want]:
        ctx.say("")
        ctx.say("not every copy reached %s" % want)
        return 1
    states = set(row[2] for row in rows)
    ctx.say("")
    ctx.say("state: %s" % "/".join(sorted(states)))
    if len(states) > 1:
        ctx.say("a split state is the dangerous one: the source can read as "
                "patched while the copy python imports still bakes")
        return 1
    return 0


# -------------------------------------------------------------------- the run


def install_dir(args, ctx):
    value = (args.install or os.environ.get(INSTALL_ENV)
             or ctx.manifest.get("trellis_install"))
    if not value:
        raise GenError(
            "no TRELLIS.2 install configured. Set --install, %s, or "
            "trellis_install in the manifest" % INSTALL_ENV
        )
    path = os.path.normpath(os.path.expanduser(os.path.expandvars(str(value))))
    if not os.path.isdir(path):
        raise GenError("no TRELLIS.2 install at %s" % path)
    return path


def interpreter(args, ctx, install):
    """The python inside the install, which is the only one that can run it."""
    value = (args.interpreter or os.environ.get(PYTHON_ENV)
             or ctx.manifest.get("trellis_python"))
    if not value:
        value = os.path.join(install, "venv", "Scripts", "python.exe")
        if not os.path.isfile(value):
            value = os.path.join(install, "venv", "bin", "python")
    path = os.path.normpath(os.path.expanduser(os.path.expandvars(str(value))))
    if not os.path.isfile(path):
        raise GenError(
            "no interpreter at %s. klin will not run TRELLIS under its own "
            "python, which has none of what TRELLIS needs" % path
        )
    return path


def script_path():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "_inside_trellis.py")
    if not os.path.isfile(path):
        raise GenError("klin is installed without %s" % os.path.basename(path))
    return path


def weights(tier):
    """What a run loads, as `(name, role)`, for the pre-flight.

    TRELLIS names one model and no LoRAs, which is the whole of its stack. The
    pair shape is what `gen.posture` takes, so a generator with no node graph
    reaches the same licence check as one that has.
    """
    return [("trellis2-%s" % tier, "checkpoint")]


def run(args, ctx):
    install = install_dir(args, ctx)

    if args.patch:
        ctx.say("install %s" % install)
        return patch(ctx, install, args.patch)

    if not args.image:
        raise GenError("nothing to generate from: pass --image")
    image = os.path.abspath(os.path.expanduser(args.image))
    if not os.path.isfile(image):
        raise GenError("no image at %s" % image)

    rows, findings, ok = posture(ctx, weights(args.tier))
    for row in rows:
        ctx.say("  %-28s %-11s %s" % (
            row["name"], row["role"], row.get("record") or "(no record)"))
    for finding in findings:
        ctx.say("  %-5s rule %s (%s)" % (
            finding.level, finding.number or "-", finding.rule_id))
    if args.check and not ok:
        ctx.say("")
        ctx.say("refusing to run, because --check was given")
        return 1

    # The state of the patch is part of the licence posture, not a detail of
    # the install, so it is reported before the run rather than assumed.
    states = set()
    for rel, path in copies(install):
        if os.path.isfile(path):
            states.add(state(read(path)[0]))
    ctx.say("")
    ctx.say("patch    %s" % ("/".join(sorted(states)) or "no copies found"))
    if not args.textured and states != {"patched"}:
        raise GenError(
            "the geometry-only path needs the install patched, and it reads "
            "%s. Run `klin gen trellis --patch apply`"
            % ("/".join(sorted(states)) or "absent")
        )

    exe = interpreter(args, ctx, install)
    name = args.name or os.path.splitext(os.path.basename(image))[0]
    out_dir = os.path.abspath(args.out or os.path.join(install, "klin_out"))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    job = {
        "install": install,
        "image": image,
        "output": os.path.join(out_dir, "%s.glb" % name),
        "seed": args.seed,
        "tier": args.tier,
        "mode": "textured" if args.textured else "geometry",
    }
    argv = [exe, script_path(), json.dumps(job)]

    if args.dry_run:
        ctx.say("")
        ctx.say("dry run: nothing generated")
        ctx.say("would run %s" % " ".join(argv[:2]))
        ctx.say(json.dumps(job, indent=2, sort_keys=True))
        return 0

    ctx.say("")
    ctx.say("running %s (%s)" % (os.path.basename(exe), job["mode"]))
    proc = subprocess.run(argv, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    report = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("TRELLIS "):
            report = json.loads(line[len("TRELLIS "):])
        elif line.startswith("trellis-note: "):
            ctx.say("  %s" % line[len("trellis-note: "):])
    if report is None:
        raise GenError(
            "the run printed no result.\n--- stdout ---\n%s\n--- stderr ---\n%s"
            % ((proc.stdout or "").strip()[-2000:], (proc.stderr or "").strip()[-2000:])
        )
    if report.get("error"):
        raise GenError("TRELLIS failed: %s" % report["error"])

    ctx.say("")
    ctx.say("%s" % report["glb"])
    ctx.say("%d faces, %d vertices, %.1f s" % (
        report.get("faces") or 0, report.get("vertices") or 0,
        report.get("seconds") or 0.0))
    if job["mode"] == "geometry":
        blocked = report.get("restricted_loaded") or []
        if blocked:
            raise GenError(
                "restricted modules loaded during a geometry-only run: %s"
                % ", ".join(blocked)
            )
        ctx.say("no restricted module could be imported for the whole run, "
                "and a mesh came out anyway")

    ctx.say("")
    ctx.say("next: klin conform blender %s" % report["glb"])
    return 0
