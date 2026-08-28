"""Conform a mesh through headless Blender.

Blender is here for two capabilities and no others: it reads every mesh format
this is likely to meet, and it writes glTF. The work between those two points
is a loop over material slots setting a UV, which is not interesting enough to
justify a dependency, and this module carries no opinion about geometry at all.

**Blender is a subprocess, never an import.** `bpy` exists only inside Blender's
own interpreter, which is not the one klin runs under and cannot be made to be.
So the half of this that touches meshes lives in `_inside_blender.py`, gets
handed a json job file and prints back a json result. The underscore keeps it
out of adapter discovery, because importing it here would fail on its first
line.

That split is also what makes this testable. Everything on this side is path
resolution, argument construction, result parsing and gates, all of which run
against a stubbed subprocess. What no stub can prove is that a loop UV actually
moved, and that is what the real run and its record are for.
"""

import io
import json
import os
import subprocess
import tempfile

from . import (
    ConformError, NOTE, budget, check_atlas, check_slots, check_triangles,
    check_uvs, check_validator, finish, mapping, parse_report, posture,
    record_for,
    relink_atlas, report_findings, source_record, staging_dir, swatch_table,
    tool_path, validation_stage,
)

NAME = "blender"
HELP = "headless Blender (reads most mesh formats, writes glTF)"

#: Where the executable comes from when no flag names it. Not `KLIN_BLENDER`:
#: that prefix marks a tree klin owns, and klin does not own Blender.
BLENDER_ENV = "BLENDER"
VALIDATOR_ENV = "GLTF_VALIDATOR"

#: Formats Blender will open for us. Refused by name rather than attempted,
#: because Blender's failure on an unknown extension is a python traceback
#: inside a subprocess and this is a sentence.
READABLE = (".blend", ".glb", ".gltf", ".obj", ".fbx", ".dae", ".stl", ".ply")


def configure(parser):
    parser.add_argument("source", help="the mesh to conform")
    parser.add_argument("--blender", default=None,
                        help="path to the Blender executable")
    parser.add_argument("--validator", default=None,
                        help="path to the Khronos gltf validator")
    parser.add_argument("--object", dest="objects", action="append", default=None,
                        help="conform only this object; repeatable")
    parser.add_argument("--keep-job", action="store_true",
                        help="leave the job file on disk for inspection")
    return parser


def executable(args, ctx, block):
    """Blender's path, or a refusal naming all three ways to supply it."""
    found = tool_path(args, "blender", BLENDER_ENV, "blender", block)
    if not found:
        raise ConformError(
            "no Blender configured. Set --blender, the %s environment "
            "variable, or conform.blender in the manifest" % BLENDER_ENV
        )
    if not os.path.isfile(found):
        raise ConformError("no Blender executable at %s" % found)
    return found


def script_path():
    """The script Blender runs, which ships beside this module.

    Asserted rather than assumed. It is loaded by path and never imported, so
    a packaging mistake that dropped it would pass every test in the suite and
    fail only on a real run.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "_inside_blender.py")
    if not os.path.isfile(path):
        raise ConformError(
            "klin is installed without %s, which is the half of conform that "
            "runs inside Blender" % os.path.basename(path)
        )
    return path


def command(exe, script, job):
    """The argv, with startup made reproducible.

    `--factory-startup` so a developer's own preferences, addons and unit
    settings cannot change what a conform produces. `--python-exit-code` so a
    failure inside the script is a non-zero exit rather than a zero one with a
    traceback in the log.
    """
    return [
        exe, "--background", "--factory-startup",
        "--python-exit-code", "1",
        "--python", script,
        "--", job,
    ]


def spawn(argv):
    """The seam every test replaces. Returns (returncode, stdout, stderr)."""
    proc = subprocess.run(
        argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return proc.returncode, proc.stdout, proc.stderr


def output_name(args, source):
    """What the conformed file is called: the source's stem, as glTF binary."""
    if args.out and not os.path.isdir(args.out):
        return os.path.basename(args.out)
    stem = os.path.splitext(os.path.basename(source))[0]
    return "%s.glb" % stem


def job(args, table, source, output, slots=None):
    """Everything the script inside Blender needs, resolved.

    The whole swatch table travels rather than a path to it, because resolving
    a repo-relative path is this side's job and the script has no repository.
    A file rather than argv: the table is structured, Windows argv quoting on
    backslash-heavy paths is a hazard a json file does not have, and a job left
    on disk after a failure is evidence.
    """
    return {
        "source": source,
        "output": output,
        "atlas": table["atlas"],
        "swatches": dict((name, list(uv)) for name, uv in table["swatches"].items()),
        "material_name": "klin_atlas",
        "slot_map": dict(slots or {}),
        "objects": list(args.objects or []),
        # glTF puts the UV origin at the top left and Blender at the bottom
        # left, so a coordinate measured in one space needs flipping to be
        # written in the other. The verify pass below measures what actually
        # shipped, which turns this from a convention argument into a number.
        "flip_v": True,
        "filter": "nearest",
        "keep_image_external": True,
        "verify": True,
    }


def run(args, ctx):
    from . import settings

    block = settings(ctx)
    source = os.path.abspath(os.path.expanduser(args.source))
    if not os.path.isfile(source):
        raise ConformError("no mesh at %s" % source)
    if os.path.splitext(source)[1].lower() not in READABLE:
        raise ConformError(
            "%s is not a format Blender opens here (%s)"
            % (os.path.basename(source), ", ".join(READABLE))
        )

    table = swatch_table(ctx)
    ctx.say("swatches %s (%d)" % (
        os.path.relpath(table["source"], ctx.repo), len(table["swatches"])))
    ctx.say("atlas    %s" % os.path.relpath(table["atlas"], ctx.repo))

    origin = None
    if args.derived_from:
        origin = source_record(ctx, args.derived_from)
        findings, ok = posture(ctx, origin)
        for finding in findings:
            ctx.say("  %-5s rule %s (%s)" % (
                finding.level, finding.number or "-", finding.rule_id))
        if not ok:
            ctx.say("")
            ctx.say("%s could not ship, so nothing conformed from it can either."
                    % origin["id"])
            if args.check:
                ctx.say("refusing, because --check was given")
                return 1
            ctx.say("continuing anyway; the record will fail the ship gate")

    exe = executable(args, ctx, block)
    script = script_path()

    if args.out and os.path.isdir(args.out):
        out_dir = os.path.abspath(args.out)
    elif args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out)) or os.getcwd()
    else:
        out_dir = staging_dir(ctx)

    # Blender writes to a temporary file and klin moves it into staging only
    # once every gate has passed, so the promise that nothing unvalidated
    # reaches the staging directory is structural rather than a matter of
    # remembering to check.
    work = tempfile.mkdtemp(prefix="klin-conform-")
    staged = os.path.join(out_dir, output_name(args, source))
    produced = os.path.join(work, output_name(args, source))
    payload = job(args, table, source, produced, mapping(args, table))
    job_file = os.path.join(work, "job.json")
    io.open(job_file, "w", encoding="utf-8", newline="\n").write(
        json.dumps(payload, indent=2, sort_keys=True))

    argv = command(exe, script, job_file)
    if args.dry_run:
        ctx.say("")
        ctx.say("dry run: nothing conformed")
        ctx.say("would run %s" % " ".join(argv))
        ctx.say(io.open(job_file, encoding="utf-8").read())
        return 0

    code, out, err = spawn(argv)
    report, notes = parse_report(out, err, os.path.basename(exe))
    for note in notes:
        ctx.say("  %s" % note)
    report.setdefault("tool", NAME)
    ctx.say("")
    ctx.say("%d triangle(s), %d slot(s)" % (
        report.get("triangles") or 0, len(report.get("slots") or [])))

    # Before the gates, so what they judge is the file that will be staged
    # rather than the one Blender happened to write. The landing directory is
    # already known even though nothing has moved yet, which is what lets the
    # atlas uri be written relative to where the file ends up.
    uri = relink_atlas(produced, table["atlas"], out_dir)
    if uri:
        ctx.say("atlas    referenced as %s" % uri)
    report["image_uri"] = uri

    findings = []
    findings.extend(check_atlas(report, uri))
    findings.extend(check_slots(report, table, args.allow_unknown_slots))
    findings.extend(check_uvs(report, table))
    findings.extend(check_triangles(report, budget(ctx, args)))
    validator = tool_path(args, "validator", VALIDATOR_ENV, "gltf_validator", block)
    # The copy, not the original: the uri points at where the file will land,
    # so in the working directory it resolves to nothing and the validator
    # reports an error klin caused. validation_stage gives the same bytes the
    # context they will really have.
    to_validate = produced
    if validator and uri:
        to_validate = validation_stage(produced, uri, table["atlas"])
    checked, result = check_validator(ctx, args, to_validate, validator)
    findings.extend(checked)

    if not report_findings(ctx, findings):
        ctx.say("")
        ctx.say("nothing was staged. The rejected file is at %s" % produced)
        return 1

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if os.path.exists(staged):
        os.remove(staged)
    os.replace(produced, staged)

    if args.no_record:
        ctx.say("")
        ctx.say("%s" % os.path.relpath(staged, ctx.repo))
        ctx.say("no record written, because --no-record was given")
        return 0

    ident = args.record_id or os.path.splitext(os.path.basename(staged))[0]
    record = record_for(ctx, args, ident, origin)
    report["tool_version"] = report.get("blender")
    extra = []
    if result is None:
        extra.append("Not validated: no gltf validator was configured when "
                     "this was conformed.")
    return finish(ctx, args, record, staged, report, table, origin, notes=extra)
