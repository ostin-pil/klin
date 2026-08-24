"""The klin command line.

    klin ledger list
    klin ledger add --id <id> [--from record.json] [--replace]
    klin ledger audit [--ship]
    klin ledger render [--check]
    klin secret set <name> | get <name> | list | rm <name> | doctor
    klin fetch <vendor> ...

Verbs are roles. Vendors arrive later as adapters under `fetch` and `gen`, and
adding one must never require touching this file's structure. `fetch` keeps
that promise by delegating: it asks `klin.fetch` for its subcommands, and that
package discovers them from the modules present. Vendor three is a new file in
`klin/fetch/` and no edit here.
"""

import argparse
import getpass
import io
import json
import os
import sys
import textwrap

from . import fetch, ledger, manifest, net, policy, render, secrets

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


def _declared(args):
    """The manifest's secrets block, or nothing when there is no manifest.

    Storing a credential is useful before a project exists, so the secret verb
    does not insist on a manifest the way the ledger verb does. A manifest that
    exists and is malformed still raises.
    """
    path = args.manifest or os.path.join(args.repo, manifest.DEFAULT_MANIFEST)
    if not os.path.isfile(path):
        return {}
    return manifest.secrets(manifest.load(path))


def cmd_secret_set(args, stream):
    if sys.stdin is not None and not sys.stdin.isatty():
        value = sys.stdin.read().strip()
    else:
        # Never an argument. argv reaches the shell history and the process
        # list, and both outlive the command.
        value = getpass.getpass("value for %s: " % args.name)
    name = secrets.store_secret(args.name, value)
    report = secrets.backend_report()
    _out(stream, "stored %s in %s" % (name, report["backend"]))
    if not report["recommended"]:
        _out(stream, "warning: %s" % report["note"])
    # status rather than lookup: a name whose manifest entry declares a
    # reference klin cannot resolve yet must not turn a successful store into a
    # failure.
    state = secrets.status(name, _declared(args).get(name))
    if state["source"] not in ("store", "unset", "error"):
        _out(
            stream,
            "note: %s resolves from %s, which takes precedence over the store"
            % (name, state["source"]),
        )
    return 0


def cmd_secret_get(args, stream):
    name = secrets.normalise(args.name)
    if getattr(stream, "isatty", lambda: False)() and not args.reveal:
        _out(
            stream,
            "klin: refusing to print %s to a terminal; pipe it, or pass --reveal"
            % name,
        )
        return 2
    _out(stream, secrets.resolve(name, _declared(args).get(name)))
    return 0


def cmd_secret_list(args, stream):
    declared = _declared(args)
    names = sorted(set(declared) | set(_stored_names_or_empty()))
    if not names:
        _out(stream, "no secrets declared in the manifest and none stored")
        return 0
    for name in names:
        state = secrets.status(name, declared.get(name))
        where = state["detail"] or ("declared" if name in declared else "stored")
        _out(stream, "%-24s %-22s %s" % (state["name"], state["source"], where))
    _out(stream, "")
    _out(stream, "%d secret(s); values are never printed by list" % len(names))
    return 0


def _stored_names_or_empty():
    """The store's own index, or nothing when the store cannot be reached.

    Listing is a diagnostic. It should still show what the manifest declares on
    a machine with no vault at all.
    """
    try:
        return secrets.stored_names()
    except secrets.SecretError:
        return []


def cmd_secret_rm(args, stream):
    name = secrets.normalise(args.name)
    existed = secrets.delete_secret(name)
    _out(stream, "removed %s" % name if existed else "%s was not in the store" % name)
    state = secrets.status(name, _declared(args).get(name))
    if state["source"] not in ("unset", "error"):
        _out(
            stream,
            "note: %s still resolves from %s, which klin cannot unset"
            % (name, state["source"]),
        )
    return 0


def cmd_secret_doctor(args, stream):
    report = secrets.backend_report()
    _out(stream, "backend: %s" % report["backend"])
    if report["note"]:
        for line in textwrap.wrap(report["note"], WRAP - 9):
            _out(stream, "         %s" % line)
    _out(stream, "override: %s, or KLIN_SECRET_BACKEND=env" % secrets.env_name("name"))
    _out(stream)

    declared = _declared(args)
    if not declared:
        _out(stream, "no secrets declared in the manifest")
    unresolved = 0
    for name in sorted(declared):
        state = secrets.status(name, declared[name])
        if state["source"] in ("unset", "error"):
            unresolved += 1
        note = declared[name].get("description") or state["detail"]
        _out(stream, "%-24s %-22s %s" % (state["name"], state["source"], note))
    if unresolved:
        _out(stream, "")
        _out(stream, "%d declared secret(s) unresolved; `klin secret set <name>`" % unresolved)
    return 1 if unresolved or not report["recommended"] else 0


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

    fetch.configure(
        verbs.add_parser("fetch", help="acquire an asset from a vendor")
    )

    sec = verbs.add_parser("secret", help="credentials an adapter needs")
    kinds = sec.add_subparsers(dest="action")

    setting = kinds.add_parser("set", help="store a value, read from stdin or a prompt")
    setting.add_argument("name")
    setting.set_defaults(func=cmd_secret_set)

    getting = kinds.add_parser("get", help="print a value, for piping")
    getting.add_argument("name")
    getting.add_argument(
        "--reveal", action="store_true", help="allow printing to a terminal"
    )
    getting.set_defaults(func=cmd_secret_get)

    listing_secrets = kinds.add_parser("list", help="names and where they resolve from")
    listing_secrets.set_defaults(func=cmd_secret_list)

    removing = kinds.add_parser("rm", help="delete a value from the store")
    removing.add_argument("name")
    removing.set_defaults(func=cmd_secret_rm)

    doctor = kinds.add_parser("doctor", help="what is holding the credentials")
    doctor.set_defaults(func=cmd_secret_doctor)

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
    except (
        manifest.ManifestError,
        ledger.LedgerError,
        render.RenderError,
        secrets.SecretError,
        net.NetError,
        fetch.FetchError,
    ) as exc:
        _out(stream, "klin: %s" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
