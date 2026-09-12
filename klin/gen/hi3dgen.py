"""Hi3DGen: image to mesh, on a stack that is MIT and Apache all the way down.

Stable-X's Hi3DGen is TRELLIS-derived, and the interesting part is what its
authors removed: kaolin, nvdiffrast and flexicubes are gone upstream, so there
is no restricted dependency to patch out of it and no texture bake to avoid.
Where the TRELLIS adapter has to make its tool clean before it can use it, this
one only has to prove the claim it inherited.

**So the two adapters carry the same evidence and a different burden.** Both
block the NVIDIA modules at the import system and generate anyway, because a
transitive import three modules deep does not show up in a licence file either.
This one blocks kaolin and flexicubes as well, which are precisely the
dependencies upstream says it dropped.

**What this adapter adds over the TRELLIS one is the pins.** Hi3DGen pins
nothing it downloads: `from_pretrained`, `torch.hub.load` and StableNormal's
own masking step each reach for a name and take whatever is behind it today.
Barinn's session 36 built the environment and found two of the five had moved
far enough to break the pipeline, which surfaced as a code bug rather than as a
supply change. A `pins.json` beside the clone records every one of them with a
revision, a licence and a hash, and this adapter refuses to start when the
files it names are not the files on disk.

That file is also the licence surface, so the pre-flight has something real to
ask about: five sources, three from Hugging Face and two from GitHub, each with
its licence. `klin gen trellis` can only name one checkpoint; this names the
whole stack, which is the shape the ledger wants anyway.

Hi3DGen runs under its own interpreter, with torch, xformers and a compiled
sparse-convolution extension, none of which klin will ever import. So
generation is a subprocess into that interpreter, and the script it runs is
`_inside_hi3dgen.py`, kept out of adapter discovery by its leading underscore.
"""

import io
import json
import os
import subprocess

from . import GenError, posture

NAME = "hi3dgen"
HELP = "Hi3DGen image-to-mesh, geometry only (needs its own python)"

INSTALL_ENV = "HI3DGEN_INSTALL"
PYTHON_ENV = "HI3DGEN_PYTHON"

PINS = "pins.json"

#: The clone's revision the pins were taken against. A different revision is
#: not an error — upstream may legitimately move — but it means the pins
#: describe a different program than the one about to run, and the run says so.
PINNED_REVISION = "c29f668ecec44b197275e9bf77f823c0c8a21076"


def configure(parser):
    parser.add_argument("--image", default=None, help="the image to lift into a mesh")
    parser.add_argument("--out", default=None, help="where the glb goes")
    parser.add_argument("--name", default=None, help="the output's stem")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--install", default=None, help="the Stable3DGen clone")
    parser.add_argument("--python", dest="interpreter", default=None,
                        help="the interpreter inside that install")
    parser.add_argument("--pins", choices=("check",), default=None,
                        help="report the pinned sources and stop")
    return parser


# --------------------------------------------------------------------- the pins


def install_dir(args, ctx):
    value = (args.install or os.environ.get(INSTALL_ENV)
             or ctx.manifest.get("hi3dgen_install"))
    if not value:
        raise GenError(
            "no Hi3DGen install configured. Set --install, %s, or "
            "hi3dgen_install in the manifest" % INSTALL_ENV
        )
    path = os.path.normpath(os.path.expanduser(os.path.expandvars(str(value))))
    if not os.path.isdir(path):
        raise GenError("no Hi3DGen install at %s" % path)
    return path


def interpreter(args, ctx, install):
    """The python inside the install, which is the only one that can run it."""
    value = (args.interpreter or os.environ.get(PYTHON_ENV)
             or ctx.manifest.get("hi3dgen_python"))
    if not value:
        value = os.path.join(install, ".venv", "Scripts", "python.exe")
        if not os.path.isfile(value):
            value = os.path.join(install, ".venv", "bin", "python")
    path = os.path.normpath(os.path.expanduser(os.path.expandvars(str(value))))
    if not os.path.isfile(path):
        raise GenError(
            "no interpreter at %s. klin will not run Hi3DGen under its own "
            "python, which has none of what Hi3DGen needs" % path
        )
    return path


def script_path():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "_inside_hi3dgen.py")
    if not os.path.isfile(path):
        raise GenError("klin is installed without %s" % os.path.basename(path))
    return path


def load_pins(install):
    path = os.path.join(install, PINS)
    if not os.path.isfile(path):
        raise GenError(
            "no %s under %s. The pins are what makes an offline run mean "
            "anything; fetch them before generating" % (PINS, install)
        )
    try:
        return json.loads(io.open(path, encoding="utf-8").read())
    except ValueError as exc:
        raise GenError("%s does not parse: %s" % (path, exc))


def sources(pins):
    """Every pinned source as `(name, licence, revision, directory)`.

    One shape for both hubs. What differs between a Hugging Face repo and a
    GitHub clone is how it is fetched, and that is the fetch step's problem
    rather than something a licence pre-flight should have to know about.
    """
    found = []
    for entry in pins.get("hf") or []:
        found.append((entry["repo"], entry.get("licence"),
                      (entry.get("revision") or "")[:12], entry.get("dir")))
    for entry in pins.get("github") or []:
        found.append((entry["repo"], entry.get("licence"),
                      (entry.get("commit") or "")[:12], entry.get("dir")))
    weights = pins.get("dinov2_weights")
    if weights:
        found.append(("dinov2_vitl14_reg4 weights", weights.get("licence"),
                      (weights.get("sha256") or "")[:12], weights.get("path")))
    return found


def pinned_files(pins):
    """Every file the pins name, as `(relative path, expected bytes or None)`."""
    found = []
    for entry in pins.get("hf") or []:
        for name, fact in (entry.get("files") or {}).items():
            found.append(("%s/%s" % (entry["dir"], name), fact.get("bytes")))
    weights = pins.get("dinov2_weights")
    if weights and weights.get("path"):
        found.append((weights["path"], weights.get("bytes")))
    for entry in pins.get("github") or []:
        if entry.get("dir"):
            found.append((entry["dir"], None))
    return found


def missing(install, pins):
    """Which pinned files are absent or the wrong size.

    Sizes, not hashes. Hashing 7 GB before every run would buy an assurance the
    fetch step already gives and this step cannot afford; what this catches is
    the failure that actually happens, which is a file gone or replaced by a
    different one. `hi3dgen_fetch.py --check` is the hash pass.
    """
    wrong = []
    for relative, expected in pinned_files(pins):
        path = os.path.join(install, relative.replace("/", os.sep))
        if not os.path.exists(path):
            wrong.append((relative, "absent"))
            continue
        if expected is not None and os.path.isfile(path):
            actual = os.path.getsize(path)
            if actual != expected:
                wrong.append((relative, "%d bytes, expected %d" % (actual, expected)))
    return wrong


def report_pins(ctx, install, pins):
    """What is pinned, and whether it is still there. Returns the bad files."""
    ctx.say("pins     as of %s" % pins.get("pinned_as_of", "(undated)"))
    for name, licence, revision, where in sources(pins):
        ctx.say("  %-38s %-12s %-13s %s" % (
            name, licence or "(no licence)", revision, where or ""))
    wrong = missing(install, pins)
    files = pinned_files(pins)
    ctx.say("")
    ctx.say("%d pinned paths, %d not as pinned" % (len(files), len(wrong)))
    for relative, why in wrong[:10]:
        ctx.say("  %-9s %s" % (why.split(",")[0], relative))
    if len(wrong) > 10:
        ctx.say("  ... and %d more" % (len(wrong) - 10))
    return wrong


def weights(pins):
    """What a run loads, as `(name, role)`, for the pre-flight.

    Every pinned source, not just the checkpoint. Hi3DGen runs four models to
    make one mesh — background removal, the normal predictor, its DINOv2
    backbone and the sparse-structure stack — and a licence question about the
    output is a question about all of them.
    """
    found = []
    for name, _licence, _revision, where in sources(pins):
        # Where it lands says what it is: the weights tree holds models, the
        # torch.hub tree holds the code that drives them, except for DINOv2's
        # checkpoint which lands there too and is named as a checkpoint.
        model = (where or "").startswith("weights") or "checkpoints" in (where or "")
        found.append((name, "checkpoint" if model else "code"))
    return found


# ---------------------------------------------------------------------- the run


def run(args, ctx):
    install = install_dir(args, ctx)
    pins = load_pins(install)

    if args.pins:
        ctx.say("install %s" % install)
        return 1 if report_pins(ctx, install, pins) else 0

    if not args.image:
        raise GenError("nothing to generate from: pass --image")
    image = os.path.abspath(os.path.expanduser(args.image))
    if not os.path.isfile(image):
        raise GenError("no image at %s" % image)

    rows, findings, ok = posture(ctx, weights(pins))
    for row in rows:
        ctx.say("  %-38s %-11s %s" % (
            row["name"], row["role"], row.get("record") or "(no record)"))
    for finding in findings:
        ctx.say("  %-5s rule %s (%s)" % (
            finding.level, finding.number or "-", finding.rule_id))
    if args.check and not ok:
        ctx.say("")
        ctx.say("refusing to run, because --check was given")
        return 1

    # The state of the pins is part of the licence posture, not a detail of the
    # install, so it is reported before the run rather than assumed. This is
    # the same reasoning the TRELLIS adapter applies to its patch.
    ctx.say("")
    wrong = report_pins(ctx, install, pins)
    if wrong:
        raise GenError(
            "%d pinned path(s) are not as pinned, so an offline run would be "
            "running something other than what was recorded. Re-run the fetch "
            "step" % len(wrong)
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
    }
    argv = [exe, script_path(), json.dumps(job)]

    if args.dry_run:
        ctx.say("")
        ctx.say("dry run: nothing generated")
        ctx.say("would run %s" % " ".join(argv[:2]))
        ctx.say(json.dumps(job, indent=2, sort_keys=True))
        return 0

    ctx.say("")
    ctx.say("running %s (geometry, offline)" % os.path.basename(exe))
    proc = subprocess.run(argv, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    report = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("HI3DGEN "):
            report = json.loads(line[len("HI3DGEN "):])
        elif line.startswith("hi3dgen-note: "):
            ctx.say("  %s" % line[len("hi3dgen-note: "):])
    if report is None:
        raise GenError(
            "the run printed no result.\n--- stdout ---\n%s\n--- stderr ---\n%s"
            % ((proc.stdout or "").strip()[-2000:], (proc.stderr or "").strip()[-2000:])
        )
    if report.get("error"):
        raise GenError("Hi3DGen failed: %s" % report["error"])

    revision = report.get("code_revision") or ""
    if revision and not revision.startswith(PINNED_REVISION[:12]):
        ctx.say("  note: the clone is at %s, and the pins were taken against %s"
                % (revision[:12], PINNED_REVISION[:12]))
    for line in report.get("code_modified") or []:
        ctx.say("  note: modified in the clone: %s" % line.strip())

    ctx.say("")
    ctx.say("%s" % report["glb"])
    ctx.say("%d faces, %d vertices, %s" % (
        report.get("faces") or 0, report.get("vertices") or 0,
        "watertight" if report.get("watertight") else "not watertight"))
    # The same named boundaries the TRELLIS adapter reports, so the two
    # generators can be compared without converting anything by hand.
    ctx.say("%.1f s in the pipeline, %.1f s exporting, %.2f GB peak "
            "(%.1f s loading, %.2f GB whole process)" % (
                report.get("pipeline_seconds") or 0.0,
                report.get("export_seconds") or 0.0,
                report.get("peak_vram_gb") or 0.0,
                report.get("load_seconds") or 0.0,
                report.get("peak_process_gb") or 0.0))
    if not report.get("foreground_found"):
        ctx.say("background removal found no object, so the whole frame was "
                "handed on: read the mesh before trusting it")
    ctx.say("no restricted module could be imported for the whole run, and a "
            "mesh came out anyway")

    ctx.say("")
    ctx.say("next: klin conform blender %s" % report["glb"])
    return 0
