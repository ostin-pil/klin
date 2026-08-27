"""Conform adapters: making a mesh somebody else authored native to a project.

A generator makes pixels and a fetcher makes bytes appear. Conform does neither.
It takes a mesh that already exists, with its own materials and its own UV
layout, and re-points it at the shared texture atlas the consuming project
styles everything from. What comes out is the same geometry wearing the
project's palette.

**The interesting half is that this is a UV operation, not a texture one.** The
consuming project's atlas is a grid of flat swatches, so a material is not an
image to author but a coordinate to point at: every loop of every face using a
slot named `wood_mid` gets the single UV that names the wood_mid texel. That
collapses "re-texture this asset" to "look up eleven numbers", and it is why a
mesh from any source becomes style-native without anybody painting anything.

**A single texel per face group is what makes a re-grade possible.** The project
grades its atlas as one image, so a prop pinned to a texel moves with the grade
and a prop carrying a baked colour does not. That is the whole reason this verb
pins to coordinates rather than assigning materials, and the reason the UV gate
below is a hard failure rather than a warning.

klin holds no opinion about which swatches exist or how many triangles is too
many. The table is the project's file, the budget is the project's number, and
both arrive through the manifest. What klin owns is that a file which failed a
gate never reaches the staging directory, and that whatever does reach it
carries a ledger record naming what it was made from.

Adapters are discovered the way `klin.fetch` discovers vendors and `klin.gen`
discovers generators: a module here declaring `NAME` and `configure` becomes a
subcommand, so a second conformer is a new file and no edit anywhere else. A
module whose name starts with an underscore is skipped, which is what keeps
`_inside_blender.py` out of the CLI: it is a script for another interpreter
entirely and importing it here would fail on `import bpy`.

One gap worth naming rather than assuming covered. A conformed record copies
its source's licence at the moment it is made, so amending the source record
later leaves the derivative holding a stale copy. `source.derived_from` is the
field a rule that noticed such drift would key on. No such rule kind exists.
"""

import hashlib
import importlib
import io
import json
import os
import pkgutil
import subprocess
import time

from .. import ledger, manifest, policy
from ..fetch import Context

#: The one line a conformer script prints that klin reads as its result.
#: Everything else on its stdout is the tool's own chatter, and a conformer
#: runs inside somebody else's interpreter which prints a banner klin cannot
#: suppress. A prefix is how the signal is found in that noise.
PREFIX = "CONFORM "

#: Progress lines klin echoes through to the user as they arrive.
NOTE = "conform-note: "

#: How far a measured UV may sit from the swatch it claims to be pinned to.
#: A texel on a 1024-square atlas is a bit under 0.001, so this is a tenth of
#: one: far too tight for a UV to have landed in a neighbouring cell, and far
#: too loose to be tripped by float32 in a glTF accessor.
EPSILON = 1e-4

#: What a nearest-filtered sampler is called in the glTF spec. The project's
#: swatch table states that point filtering is required, because bilinear
#: sampling bleeds a neighbouring swatch into any UV near a cell boundary, and
#: every UV this verb writes is near a boundary by construction.
NEAREST = 9728


class ConformError(Exception):
    pass


def adapters():
    """Every conform adapter in this package, by name."""
    found = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module("%s.%s" % (__name__, info.name))
        if getattr(module, "NAME", None) and hasattr(module, "configure"):
            found[module.NAME] = module
    return found


def configure(parser):
    """Wire every discovered adapter in as a subcommand."""
    tools = parser.add_subparsers(dest="conformer")
    for name in sorted(adapters()):
        module = adapters()[name]
        sub = tools.add_parser(name, help=getattr(module, "HELP", None))
        sub.add_argument("--out", default=None,
                         help="write here instead of the staging directory")
        sub.add_argument("--id", dest="record_id", default=None,
                         help="ledger record id (default: the output's stem)")
        sub.add_argument("--from", dest="derived_from", default=None,
                         help="the record this derives from; its licence is inherited")
        sub.add_argument("--replace", action="store_true",
                         help="overwrite an existing record, keeping hand-written fields")
        sub.add_argument("--no-record", action="store_true",
                         help="conform and gate, but write no ledger record")
        sub.add_argument("--dry-run", action="store_true",
                         help="resolve and validate, but conform nothing")
        sub.add_argument("--check", action="store_true",
                         help="refuse to run when the source could not ship")
        sub.add_argument("--strict", action="store_true",
                         help="a missing validator is a failure, not a warning")
        sub.add_argument("--max-tris", type=int, default=None,
                         help="triangle budget; 0 disables the gate")
        sub.add_argument("--allow-unknown-slots", action="store_true",
                         help="a slot naming no swatch warns instead of failing")
        sub.add_argument("--map", dest="mapping", action="append", default=None,
                         metavar="SLOT=SWATCH",
                         help="which swatch a slot means; repeatable")
        module.configure(sub)
        sub.set_defaults(func=_run_adapter, module=module)
    return parser


def _run_adapter(args, stream):
    data = {}
    path = args.manifest or os.path.join(args.repo, manifest.DEFAULT_MANIFEST)
    if os.path.isfile(path):
        data = manifest.load(path)
    return args.module.run(args, Context(args, data, stream))


# ------------------------------------------------------- what the project says


def settings(ctx):
    """The manifest's `conform` block, or an empty one."""
    block = ctx.manifest.get("conform") or {}
    if not isinstance(block, dict):
        raise ConformError("manifest: 'conform' is not a mapping")
    return block


def staging_dir(ctx):
    """Where a conformed file lands, created if it does not exist.

    The manifest key has been declared by at least one project since before
    this verb existed, carrying an open item saying klin creates the directory
    on first conform. This is that.
    """
    path = manifest.resolve(ctx.manifest, "staging_dir", ctx.repo)
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def swatch_table(ctx):
    """The project's atlas cell map, resolved and checked.

    Every path in the table is resolved here rather than inside the conformer,
    which receives absolute paths and numbers and makes no decisions about
    layout. That keeps the half needing a repository on this side of the
    process boundary, where the test suite can reach it.
    """
    block = settings(ctx)
    if not block.get("swatches"):
        raise ConformError(
            "manifest has no 'conform.swatches'; klin needs the project's "
            "atlas cell map and will not guess where it lives"
        )
    path = os.path.normpath(os.path.join(ctx.repo, str(block["swatches"])))
    if not os.path.isfile(path):
        raise ConformError("no swatch table at %s" % path)
    try:
        data = json.loads(io.open(path, encoding="utf-8").read())
    except ValueError as exc:
        raise ConformError("%s is not valid json: %s" % (path, exc))

    swatches = data.get("swatches")
    if not isinstance(swatches, dict) or not swatches:
        raise ConformError("%s has no 'swatches' block" % path)

    table = {}
    for name in sorted(swatches):
        entry = swatches[name] or {}
        uv = entry.get("uv") if isinstance(entry, dict) else None
        if not isinstance(uv, list) or len(uv) != 2:
            raise ConformError("%s: swatch %r has no two-number 'uv'" % (path, name))
        try:
            u, v = float(uv[0]), float(uv[1])
        except (TypeError, ValueError):
            raise ConformError("%s: swatch %r has a non-numeric 'uv'" % (path, name))
        # Outside the unit square a UV wraps, which renders as a plausible
        # colour from the wrong cell rather than as an error.
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            raise ConformError(
                "%s: swatch %r sits outside the unit square at (%s, %s)"
                % (path, name, u, v)
            )
        table[name] = (u, v)

    return {
        "atlas": _atlas_path(ctx, path, data.get("atlas") or {}),
        "swatches": table,
        "source": path,
    }


def _atlas_path(ctx, table_path, atlas):
    """The atlas image, from the repo-relative key rather than the engine one.

    A table written for a game engine carries the same image twice, once as an
    engine resource path and once as a path on disk. Only the second means
    anything to a tool outside that engine, and picking the wrong one produces
    a missing-file error three steps from its cause.
    """
    value = atlas.get("repo_path") if isinstance(atlas, dict) else None
    if not value:
        raise ConformError(
            "%s: the 'atlas' block has no 'repo_path'. klin needs the image's "
            "path on disk; an engine resource path is not one" % table_path
        )
    if "://" in str(value):
        raise ConformError(
            "%s: atlas.repo_path is an engine resource path (%s). klin needs "
            "the path on disk" % (table_path, value)
        )
    path = os.path.normpath(os.path.join(ctx.repo, str(value)))
    if not os.path.isfile(path):
        raise ConformError("no atlas image at %s" % path)
    return path


def mapping(args, table):
    """Which swatch each material slot means, for a mesh nobody here authored.

    Conform was specified against slots already named after swatches, which is
    true of a mesh authored for this project and false of every mesh bought
    from anybody. A bought pack names its materials for what they look like,
    `Wood` and `Metal` and `Fabric`, and no amount of walking the slots turns
    those into a project's own vocabulary. The same document that specified the
    naming also lists a CC0 tavern pack as needing re-pinning through this
    verb, so both cases were always in scope and only one of them had a
    mechanism.

    A flag rather than a manifest key, because the mapping is a fact about one
    pack rather than about the project, and a project may take props from
    several. Matching is case-insensitive on the slot, since a pack that
    writes `Wood` and a person who types `wood` mean the same thing and being
    strict there buys nothing.
    """
    pairs = {}
    for entry in args.mapping or []:
        if "=" not in entry:
            raise ConformError(
                "--map wants SLOT=SWATCH, and %r has no '='" % entry)
        slot, _, swatch = entry.partition("=")
        slot, swatch = slot.strip(), swatch.strip()
        if not slot or not swatch:
            raise ConformError("--map wants SLOT=SWATCH, and %r is missing one" % entry)
        if swatch not in table["swatches"]:
            raise ConformError(
                "--map names swatch %r, which is not in %s. It has: %s"
                % (swatch, os.path.basename(table["source"]),
                   ", ".join(sorted(table["swatches"])))
            )
        pairs[slot.lower()] = swatch
    return pairs


def budget(ctx, args):
    """The triangle ceiling: the flag, else the manifest, else none.

    No budget configured means no gate, because a triangle count is an art
    direction decision and klin holds no more opinion about those than it does
    about licences.
    """
    if args.max_tris is not None:
        return args.max_tris or None
    value = settings(ctx).get("max_triangles")
    return int(value) if value else None


def tool_path(args, flag, env, key, block):
    """A third-party executable: the flag, then the environment, then the manifest.

    The same precedence `klin.gen` uses for a server URL, and for the same
    reason: which volume holds a tool is a fact about a machine rather than
    about a project, so the committed file is the last resort and not the
    first.

    Named for the tool rather than prefixed `KLIN_`. That prefix marks a tree
    klin itself owns, and klin does not own Blender.
    """
    value = getattr(args, flag, None) or os.environ.get(env) or block.get(key)
    if not value:
        return None
    return os.path.normpath(os.path.expanduser(os.path.expandvars(str(value))))


# --------------------------------------------------------------- the reporting


def parse_report(stdout, stderr, where):
    """The conformer's one result line, and the notes it printed along the way.

    A conformer runs inside another program's interpreter, and that program
    prints a version banner, extension chatter and a farewell that klin cannot
    turn off. So the result is a prefixed line found among the noise rather
    than the whole of stdout.

    When no such line appears, both streams are surfaced. A bare exit code
    says nothing about which of those many lines went wrong, and the failure
    this hits in practice is an interpreter that died before reaching the
    script at all.
    """
    found = []
    notes = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if line.startswith(PREFIX):
            found.append(line[len(PREFIX):].strip())
        elif line.startswith(NOTE):
            notes.append(line[len(NOTE):].strip())

    if not found:
        raise ConformError(
            "%s printed no %sline.\n--- stdout ---\n%s\n--- stderr ---\n%s"
            % (where, PREFIX, (stdout or "").strip(), (stderr or "").strip())
        )
    # Refusing rather than taking the last one. Two results mean the script ran
    # twice or printed a stale line, and either way klin cannot tell which
    # describes the file on disk.
    if len(found) > 1:
        raise ConformError(
            "%s printed %d %slines; klin will not pick one" % (where, len(found), PREFIX)
        )
    try:
        report = json.loads(found[0])
    except ValueError as exc:
        raise ConformError("%s printed an unreadable result: %s" % (where, exc))
    if not isinstance(report, dict):
        raise ConformError("%s printed a result that is not an object" % where)
    if report.get("error"):
        raise ConformError("%s failed: %s" % (where, report["error"]))
    return report, notes


# ------------------------------------------------------------------- the gates


def _fail(text):
    return {"level": "fail", "text": text}


def _warn(text):
    return {"level": "warn", "text": text}


def check_slots(report, table, allow_unknown=False):
    """Every material slot names a swatch the table knows.

    A slot naming nothing keeps whatever UVs it arrived with, so it survives
    the export looking like a prop with a shading bug rather than one that was
    never conformed. Failing by default is what turns that into a sentence
    naming the slot.
    """
    findings = []
    for slot in report.get("slots") or []:
        if slot.get("swatch"):
            continue
        text = ("slot %r names no swatch in %s; say which one it means with "
                "--map %s=<swatch>"
                % (slot.get("name"), os.path.basename(table["source"]),
                   slot.get("name")))
        findings.append(_warn(text) if allow_unknown else _fail(text))
    return findings


def check_uvs(report, table):
    """Every uv in the exported file is one of the texels it was meant to be.

    Compared as sets rather than per swatch, because the conform collapses
    every material slot into one shared atlas material, so a re-imported mesh
    has nothing left to say which face used to be which slot. The set is the
    stronger claim anyway: it catches a uv that never moved, a uv that moved to
    the wrong cell, and a whole layout written with the v axis inverted, none
    of which a per-slot check would separate.

    Measured by re-importing the file that was exported, so this is the
    coordinate space that shipped rather than the one the conformer believed it
    was writing.
    """
    findings = []
    pinned = report.get("pinned") or {}
    measured = [tuple(uv) for uv in (report.get("uvs_measured") or [])]

    if not pinned:
        return findings
    if not measured:
        findings.append(_warn(
            "the conformer measured no uvs, so the pin was not verified"
        ))
        return findings

    def near(a, b):
        return abs(a[0] - b[0]) <= EPSILON and abs(a[1] - b[1]) <= EPSILON

    wanted = dict((name, tuple(uv)) for name, uv in pinned.items())
    for got in sorted(measured):
        if not [n for n, uv in wanted.items() if near(got, uv)]:
            findings.append(_fail(
                "a uv at (%.5f, %.5f) is not any swatch this pinned, so a face "
                "kept the layout it arrived with" % got
            ))
    for name in sorted(wanted):
        if not [uv for uv in measured if near(uv, wanted[name])]:
            findings.append(_fail(
                "swatch %r was pinned to (%.5f, %.5f) and no uv came back "
                "there" % (name, wanted[name][0], wanted[name][1])
            ))

    filtering = report.get("mag_filter")
    if filtering is not None and filtering != NEAREST:
        findings.append(_fail(
            "the atlas sampler is %s and point filtering (%s) is required, "
            "because bilinear sampling bleeds a neighbouring swatch into every "
            "uv near a cell boundary" % (filtering, NEAREST)
        ))
    return findings


def check_triangles(report, ceiling):
    """The triangle count against the project's budget, when it set one."""
    if not ceiling:
        return []
    count = report.get("triangles")
    if count is None:
        return [_warn("the conformer reported no triangle count")]
    if count > ceiling:
        return [_fail("%d triangles against a budget of %d" % (count, ceiling))]
    return []


def report_findings(ctx, findings):
    """Print what the gates found. True when nothing failed."""
    for finding in findings:
        ctx.say("  %-5s %s" % (finding["level"], finding["text"]))
    return not [f for f in findings if f["level"] == "fail"]


# ----------------------------------------------------------------- the atlas


#: glTF's container: a 12-byte header, then length-prefixed chunks. Only the
#: json one is rewritten here, and only its `images` array.
_JSON_CHUNK = 0x4E4F534A


def _chunks(raw):
    import struct

    magic, version, length = struct.unpack("<III", raw[:12])
    if magic != 0x46546C67:
        raise ConformError("not a glb: bad magic")
    out = []
    offset = 12
    while offset < length:
        size, kind = struct.unpack("<II", raw[offset:offset + 8])
        out.append([kind, raw[offset + 8:offset + 8 + size]])
        offset += 8 + size
    return out


def relink_atlas(path, atlas, landing):
    """Point the exported file at the shared atlas, by a path that will resolve.

    Blender decides for itself how a texture leaves the exporter, and what it
    decided here was both at once: a one-pixel placeholder embedded in a buffer
    view *and* a uri, which the specification forbids, with the uri written
    relative to the temporary directory the export happened in. Either half
    alone is wrong for this. An embedded copy is a private image, so a project
    that re-grades its one shared atlas would leave every conformed prop behind
    on the old palette, which is the whole thing pinning to a texel exists to
    prevent. A uri into a temporary directory resolves to nothing at all.

    So klin fixes it afterwards rather than asking Blender more politely. The
    invariant belongs to klin either way, and an exporter keyword that means
    one thing this release and another next is exactly what should not be load
    bearing.

    Returns the uri written.
    """
    import struct

    raw = io.open(path, "rb").read()
    chunks = _chunks(raw)
    index = [i for i, (kind, _) in enumerate(chunks) if kind == _JSON_CHUNK]
    if not index:
        raise ConformError("%s has no json chunk" % path)

    document = json.loads(chunks[index[0]][1].decode("utf-8"))
    images = document.get("images") or []
    if not images:
        return None

    uri = os.path.relpath(atlas, landing).replace("\\", "/")
    for image in images:
        # Exactly one of the two, per the specification. The buffer view is
        # dropped rather than the uri because the bytes it holds are a
        # placeholder and the file on disk is the real atlas.
        image.pop("bufferView", None)
        image.pop("mimeType", None)
        image["uri"] = uri

    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((4 - len(encoded) % 4) % 4)
    chunks[index[0]][1] = encoded

    body = b""
    for kind, payload in chunks:
        body += struct.pack("<II", len(payload), kind) + payload
    header = struct.pack("<III", 0x46546C67, 2, 12 + len(body))
    io.open(path, "wb").write(header + body)
    return uri


def check_atlas(report, uri):
    """The exported file references the atlas rather than carrying a copy."""
    findings = []
    if uri is None:
        findings.append(_warn("the export carries no image at all"))
    layers = report.get("uv_layers")
    if layers and layers > 1:
        findings.append(_fail(
            "%d uv layers came back, and only one can be the pinned one; the "
            "others ship the layout the mesh arrived with" % layers
        ))
    return findings


# --------------------------------------------------------------- the validator


def validate(path, executable):
    """What the glTF validator says about a file, as counts and issues.

    Isolated in one function on purpose. The validator is a third-party binary
    whose command line and json shape are not klin's to define, so confirming
    them against a real install should be a change to this function and to
    nothing else.
    """
    try:
        proc = subprocess.run(
            [executable, path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except OSError as exc:
        raise ConformError("could not run %s: %s" % (executable, exc))

    text = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    try:
        data = json.loads(text)
    except ValueError:
        # A validator that said something klin cannot read is not the same as
        # a clean file, and reporting it as one would be the quiet failure the
        # gate exists to prevent.
        raise ConformError(
            "%s printed no readable json (exit %d):\n%s"
            % (executable, proc.returncode, text[:2000])
        )
    issues = data.get("issues") or {}
    return {
        "errors": int(issues.get("numErrors") or 0),
        "warnings": int(issues.get("numWarnings") or 0),
        "messages": issues.get("messages") or [],
    }


def check_validator(ctx, args, path, executable):
    """Run the validator when there is one, and be loud when there is not.

    Absent by default is a warning rather than a failure, because requiring a
    separate binary before the verb works at all would make it unusable on
    every machine that has not installed one, including all four continuous
    integration legs. Silence is the option not taken: a run that says
    `conformed` while skipping its own gate has misreported its coverage, which
    is exactly the failure klin's one rule of its own exists to catch.
    """
    if not executable:
        text = (
            "no gltf validator configured, so the export was not validated. "
            "Set --validator, GLTF_VALIDATOR or conform.gltf_validator"
        )
        if args.strict:
            return [_fail(text + " (--strict)")], None
        return [_warn(text)], None

    result = validate(path, executable)
    findings = []
    if result["errors"]:
        findings.append(_fail(
            "the validator reports %d error(s); the rejected file is at %s"
            % (result["errors"], path)
        ))
    if result["warnings"]:
        findings.append(_warn("the validator reports %d warning(s)" % result["warnings"]))
    return findings, result


# ---------------------------------------------------------------- the provenance


def source_record(ctx, ident):
    """The record a conform derives from, or a refusal naming what exists."""
    records = ledger.load(ctx.ledger_path())
    for record in records:
        if record["id"] == ident:
            return record
    raise ConformError(
        "no record %r in the ledger. It holds: %s"
        % (ident, ", ".join(sorted(r["id"] for r in records)) or "nothing")
    )


def posture(ctx, record):
    """What the project's policy says about the record a conform starts from.

    The same question `klin.gen` asks of a workflow's weights, asked of one
    record instead. A conformed derivative inherits its source's licence, so a
    source that cannot ship produces a derivative that cannot ship either, and
    the cheap moment to learn that is before the tool starts.
    """
    findings = policy.evaluate(
        [record], manifest.rules(ctx.manifest), manifest.build_facts(ctx.manifest),
        ship=True,
    )
    return findings, not policy.failed(findings)


def inherit_licence(record, source):
    """Copy the source's licence onto a derivative, whole.

    A copy rather than a reference because `klin.policy` reads the record in
    front of it and has no resolver. Whole rather than by field because a
    licence with its text left behind classifies differently from one with it.
    """
    record["licence"] = dict(source.get("licence") or {})
    return record


def sentence(report, table, source_id):
    """One line saying what was done, for the record's modifications list."""
    pinned = []
    for slot in report.get("slots") or []:
        if slot.get("swatch"):
            pinned.append("%s to %s" % (slot.get("name"), slot["swatch"]))
    parts = ["Conformed onto %s" % os.path.basename(table["atlas"])]
    if source_id:
        parts.append(" from `%s`" % source_id)
    if pinned:
        parts.append(": pinned %s" % ", ".join(sorted(pinned)))
    if report.get("triangles") is not None:
        parts.append(" (%d triangles)" % report["triangles"])
    return "".join(parts) + "."


def sha256_of(path):
    digest = hashlib.sha256()
    with io.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record_for(ctx, args, ident, source):
    """The record a conform writes, merged over whatever was already there.

    `ledger.add` replaces a record whole, and a record that has been in a
    repository for a while carries hand-written notes and modifications nobody
    wants a re-conform to delete. So the fields a machine owns are overwritten
    and the fields a person wrote are carried forward.
    """
    existing = None
    for candidate in ledger.load(ctx.ledger_path()):
        if candidate["id"] == ident:
            existing = candidate
            break
    if existing and not args.replace:
        raise ConformError(
            "%s is already in the ledger; pass --replace to overwrite it" % ident
        )

    record = existing or ledger.blank(ident, kind="mesh")
    record["source"]["adapter"] = "conform"
    record["source"]["retrieved"] = time.strftime("%Y-%m-%d")
    if source:
        record["source"]["derived_from"] = source["id"]
        if not record["source"].get("url"):
            record["source"]["url"] = ledger.field(source, "source.url")
        if not (record.get("author") or {}).get("name"):
            record["author"] = dict(source.get("author") or {})
        inherit_licence(record, source)
    return record


def finish(ctx, args, record, staged, report, table, source, notes=None):
    """Write the record and say what to run next."""
    record["paths"] = [os.path.relpath(staged, ctx.repo).replace("\\", "/")]
    record["sha256"] = sha256_of(staged)
    record["produced_by"] = {"tool": report.get("tool") or "unknown",
                             "version": report.get("tool_version")}
    mods = list(record.get("modifications") or [])
    mods.append(sentence(report, table, source["id"] if source else None))
    record["modifications"] = mods
    if notes:
        record["notes"] = "\n\n".join(
            [n for n in [record.get("notes")] + list(notes) if n]
        )

    path = ctx.ledger_path()
    ledger.add(path, record, replace=True)

    ctx.say("")
    ctx.say("%s" % os.path.relpath(staged, ctx.repo))
    ctx.say("sha256  %s" % record["sha256"])
    ctx.say("recorded %s in %s" % (record["id"], os.path.relpath(path, ctx.repo)))
    ctx.say("")
    ctx.say("next: klin ledger audit --ship")
    return 0
