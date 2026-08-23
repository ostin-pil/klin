"""The klin command line.

    klin ledger list
    klin ledger add --id <id> [--from record.json] [--replace]
    klin ledger audit [--ship]
    klin ledger render [--check]

Verbs are roles. Vendors arrive later as adapters under `fetch` and `gen`, and
adding one must never require touching this file's structure.
"""

import argparse
import io
import json
import os
import sys
import textwrap

from . import ledger, manifest, policy, render

WRAP = 78


def _out(stream, text=""):
    stream.write(text + "\n")


def _load(args):
    path = args.manifest or os.path.join(args.repo, manifest.DEFAULT_MANIFEST)
    data = manifest.load(path)
    return data, path


def _records(args, data):
    return ledger.load(manifest.resolve(data, "ledger", args.repo))


def cmd_list(args, stream):
    data, _ = _load(args)
    records = _records(args, data)
    if args.json:
        _out(stream, json.dumps(records, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    if not records:
        _out(stream, "ledger is empty")
        return 0
    for record in records:
        _out(
            stream,
            "%-40s %-14s %s"
            % (
                record["id"],
                ledger.field(record, "licence.id") or "unrecorded",
                ", ".join(record.get("paths") or []),
            ),
        )
    _out(stream, "")
    _out(stream, "%d record(s)" % len(records))
    return 0


def cmd_add(args, stream):
    data, _ = _load(args)
    path = manifest.resolve(data, "ledger", args.repo)
    if args.from_file:
        record = json.loads(io.open(args.from_file, encoding="utf-8").read())
        if args.id:
            record["id"] = args.id
    else:
        if not args.id:
            _out(stream, "klin: --id is required unless --from is given")
            return 2
        record = ledger.blank(args.id, args.kind)
    ledger.add(path, record, replace=args.replace)
    _out(stream, "recorded %s in %s" % (record["id"], os.path.relpath(path, args.repo)))
    if not args.from_file:
        _out(stream, "fill in source, licence and paths, then run: klin ledger audit")
    return 0


def _print_finding(stream, finding):
    label = {"fail": "FAIL ", "warn": "warn ", "waived": "waived"}.get(
        finding.level, finding.level
    )
    where = finding.record_id or "(whole set)"
    _out(
        stream,
        "%s rule %s (%s)  %s" % (label, finding.number, finding.rule_id, where),
    )
    _out(stream, "      %s" % finding.summary)
    if finding.text:
        for line in textwrap.wrap(finding.text, WRAP - 6):
            _out(stream, "      %s" % line)
    _out(stream)


def cmd_audit(args, stream):
    data, manifest_path = _load(args)
    records = _records(args, data)
    rules = manifest.rules(data)
    facts = manifest.build_facts(data)

    gate = "ship gate" if args.ship else "stage gate (%s)" % data.get("stage", "unset")
    _out(
        stream,
        "klin audit: %s | %d record(s) | policy transcribed from %s"
        % (gate, len(records), data.get("policy_doc", "(unset)")),
    )
    _out(stream)

    findings = policy.evaluate(records, rules, facts, ship=args.ship)
    for finding in findings:
        _print_finding(stream, finding)

    fails = len([f for f in findings if f.level == "fail"])
    warns = len([f for f in findings if f.level == "warn"])
    waived = len([f for f in findings if f.level == "waived"])
    summary = "%d failure(s), %d warning(s)" % (fails, warns)
    if waived:
        summary += ", %d waived" % waived
    _out(stream, summary)
    if not args.ship and fails == 0:
        _out(
            stream,
            "stage gate only - run `klin ledger audit --ship` before anything goes public",
        )
    return 1 if fails else 0


def cmd_render(args, stream):
    data, _ = _load(args)
    records = _records(args, data)
    target = manifest.resolve(data, "render_target", args.repo)
    name = data.get("render_marker", "records")

    text = io.open(target, encoding="utf-8").read()
    block = render.body(records)
    updated = render.normalise(render.splice(text, name, block))

    shown = os.path.relpath(target, args.repo)
    if args.check:
        if render.normalise(text) == updated:
            _out(stream, "%s is up to date (%d record(s))" % (shown, len(records)))
            return 0
        _out(stream, "%s is stale - run `klin ledger render`" % shown)
        return 1

    if render.normalise(text) == updated:
        _out(stream, "%s already up to date" % shown)
        return 0
    io.open(target, "w", encoding="utf-8", newline="\n").write(updated)
    _out(stream, "rendered %d record(s) into %s" % (len(records), shown))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="klin", description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=".", help="project root (default: cwd)")
    parser.add_argument("--manifest", default=None, help="path to the klin manifest")

    verbs = parser.add_subparsers(dest="verb")
    led = verbs.add_parser("ledger", help="provenance and licence records")
    actions = led.add_subparsers(dest="action")

    listing = actions.add_parser("list", help="show every record")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=cmd_list)

    adding = actions.add_parser("add", help="add or replace a record")
    adding.add_argument("--id")
    adding.add_argument("--kind", default="mesh")
    adding.add_argument("--from", dest="from_file", help="read the record from JSON")
    adding.add_argument("--replace", action="store_true")
    adding.set_defaults(func=cmd_add)

    audit = actions.add_parser("audit", help="apply the policy rules")
    audit.add_argument(
        "--ship",
        action="store_true",
        help="apply the ship gate, not just the stage rule",
    )
    audit.set_defaults(func=cmd_audit)

    rendering = actions.add_parser("render", help="write records into the policy doc")
    rendering.add_argument(
        "--check", action="store_true", help="fail if the document is stale"
    )
    rendering.set_defaults(func=cmd_render)

    return parser


def main(argv=None, stream=None):
    if stream is None:
        stream = sys.stdout
        # Rule text is quoted verbatim from a policy document, and those are
        # written by people who use em dashes. A cp1252 console would mangle
        # them into noise that reads like a klin bug.
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help(stream)
        return 2
    args.repo = os.path.abspath(args.repo)
    try:
        return args.func(args, stream)
    except (manifest.ManifestError, ledger.LedgerError, render.RenderError) as exc:
        _out(stream, "klin: %s" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
