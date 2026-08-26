"""The klin command line.

    klin ledger list
    klin ledger add --id <id> [--from record.json] [--replace]
    klin ledger audit [--ship]
    klin ledger render [--check]
    klin secret set <name> | get <name> | list | rm <name> | doctor
    klin fetch <vendor> ...
    klin gen comfy --workflow <api.json> --prompt "..." [--check]
    klin index build [--rescan] [--prune] | status
    klin ls [--lora X] [--model X] [--seed N] [--since D] [--check]
    klin show <path fragment>

`ls` and `show` sit at the top level rather than under `index` because they are
the interface to the corpus, not to the scanner. Somebody browsing what they
have should not have to know that a database is what makes it possible.

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
import re
import sys
import textwrap

from . import fetch, gen, index, ledger, manifest, net, policy, render, secrets

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
    # klin's own coverage finding carries no rule number, because it was not
    # transcribed from the project's policy document and citing a number it
    # never had would misattribute it.
    cited = "rule %s (%s)" % (finding.number, finding.rule_id)
    if finding.number is None:
        cited = "klin (%s)" % finding.rule_id
    _out(stream, "%s %s  %s" % (label, cited, where))
    _out(stream, "      %s" % finding.summary)
    if finding.text:
        for line in textwrap.wrap(finding.text, WRAP - 6):
            _out(stream, "      %s" % line)
    _out(stream)


def _report_drift(stream, data, records):
    """Say when the resolved cache is not the one the ledger was written in.

    A note rather than a failure. A project may legitimately keep assets
    outside the cache, and this cannot tell that apart from a variable that has
    gone missing. What it can do is stop the discrepancy being invisible, which
    is the only reason the state persists.
    """
    try:
        cache = manifest.cache_dir(data, default=None)
    except manifest.ManifestError:
        return
    drift = ledger.cache_drift(records, cache)
    if not drift:
        return
    _out(stream, "note: cache_dir resolves to %s," % drift["cache"])
    _out(
        stream,
        "      and none of the %d recorded file(s) are under it. They are in:"
        % drift["recorded"],
    )
    for where in drift["elsewhere"]:
        _out(stream, "        %s" % where)
    for line in textwrap.wrap(
        "Set %s to the tree that is actually in use, or correct cache_dir. "
        "Left alone, the next fetch downloads into the empty one and nothing "
        "reports a problem." % manifest.CACHE_ENV,
        WRAP - 6,
    ):
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

    _report_drift(stream, data, records)

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


def _index(args):
    """Open the index, and hand back the manifest that located it."""
    data, _ = _load(args)
    return index.connect(index.db_path(data)), data


def _resolver(args, data):
    """The model map and the ledger, which every licence verdict needs.

    Built once per command rather than once per row: it walks the weights tree
    and stats each file, which is cheap for thirty models and wasteful for a
    thousand images.
    """
    records = _records(args, data)
    return index.model_map(index.models_dir(data), records), records


def cmd_index_build(args, stream):
    conn, data = _index(args)
    project = data.get("product_name") or "(unnamed)"
    wanted = [os.path.abspath(r) for r in (args.root or [])] or index.roots(data)
    if not wanted:
        _out(stream, "no index roots. Add an `index.roots` list to the manifest,")
        _out(stream, "or pass --root; klin will not guess where outputs live.")
        return 2

    patterns = index.claims(data)
    limit = None if args.hash_large else index.HASH_LIMIT
    conflicts = []
    for root in wanted:
        _out(stream, "scanning %s" % root)
        got = index.scan(
            conn,
            root,
            project,
            patterns,
            rescan=args.rescan,
            hash_limit=limit,
            say=lambda text: _out(stream, text),
        )
        conflicts.extend(got["conflicts"])
        _out(
            stream,
            "  %d file(s): %d added, %d updated, %d unchanged, %d with provenance"
            % (
                got["seen"],
                got["added"],
                got["updated"],
                got["skipped"],
                got["provenance"],
            ),
        )
        if got["unclaimed"]:
            _out(
                stream,
                "  %d unclaimed by %s, and indexed all the same"
                % (got["unclaimed"], project),
            )

    if args.prune:
        gone = index.forget_missing(conn)
        _out(stream, "pruned %d row(s) whose file has gone" % len(gone))

    for path, held, also in conflicts:
        _out(stream)
        _out(stream, "conflict: %s" % path)
        _out(
            stream,
            "  claimed by %s and now also by %s; kept %s. Narrow one of the "
            "`index.claim` patterns rather than leaving klin to pick."
            % (held, also, held),
        )

    _out(stream)
    _out(stream, "next: klin ls")
    return 0


def cmd_index_status(args, stream):
    conn, data = _index(args)
    got = index.status(conn)
    _out(stream, "klin index: %s" % index.db_path(data))
    _out(stream)
    _out(
        stream,
        "%d item(s) | %d with provenance | %d unclaimed | %d distinct workflow(s)"
        % (got["items"], got["with_provenance"], got["unclaimed"], got["workflows"]),
    )
    if got["unhashed"]:
        _out(
            stream,
            "%d indexed without a hash, being over %d MiB; --hash-large includes them"
            % (got["unhashed"], index.HASH_LIMIT // (1024 * 1024)),
        )
    for label, rows in (("projects", got["projects"]), ("roots", got["roots"])):
        if not rows:
            continue
        _out(stream)
        _out(stream, label)
        for name, count in rows:
            _out(stream, "  %6d  %s" % (count, name if name else "(unclaimed)"))
    if got["models"]:
        _out(stream)
        _out(stream, "base models")
        for name, count in got["models"]:
            _out(stream, "  %6d  %s" % (count, name))
    return 0


def _short(name):
    return re.sub(r"\.(safetensors|ckpt|pt|pth|gguf|sft)$", "", str(name or ""))


def _families(entry):
    """How to render a classification, including the one that is empty.

    An empty set is a real answer: the licence was read and nothing klin tracks
    restricts it. Rendering that as a blank column would read as missing data,
    which is the opposite of what it means.
    """
    return ", ".join(entry["families"]) or "no family applies"


def _stack(conn, path):
    """The LoRA stack as one readable string, strengths included."""
    return " + ".join(
        "%s@%s" % (_short(os.path.basename(str(l["name"] or "?"))), l["strength"])
        for l in index.loras_of(conn, path)
    )


def _filters(args):
    return {
        "project": args.project,
        "unclaimed": args.unclaimed,
        "model": args.model,
        "lora": args.lora,
        "seed": args.seed,
        "prompt": args.prompt,
        "workflow": args.workflow,
        "since": args.since,
        "kind": args.kind,
        "limit": args.limit,
    }


def cmd_ls(args, stream):
    conn, data = _index(args)
    rows = index.query(conn, **_filters(args))
    if not rows:
        _out(stream, "nothing in the index matches. `klin index status` shows what is.")
        return 0

    models, records = _resolver(args, data)
    rules = manifest.rules(data)
    facts = manifest.build_facts(data)
    verdicts = [
        index.verdict(conn, row["path"], models, records, rules, facts) for row in rows
    ]

    if args.json:
        out = []
        for row, got in zip(rows, verdicts):
            item = dict(row)
            item.pop("graph", None)
            item["loras"] = [dict(l) for l in index.loras_of(conn, row["path"])]
            item["ship"] = got["ship"]
            item["licences"] = got["models"]
            item["unresolved"] = got["unresolved"]
            out.append(item)
        _out(stream, json.dumps(out, indent=2, default=str))
        return 0

    failing = untraced = 0
    _out(stream, "%d item(s)" % len(rows))
    _out(stream)
    _out(stream, "SHIP  %-42s %-11s %-8s %s" % ("FILE", "SIZE", "SEED", "MODEL"))
    for row, got in zip(rows, verdicts):
        if got["unresolved"]:
            mark, _ = "?", untraced
            untraced += 1
        elif got["ship"]:
            mark = "yes"
        else:
            mark = "NO"
            failing += 1
        model = _short(os.path.basename(str(row["model"] or "(no graph)")))
        stack = _stack(conn, row["path"])
        if stack:
            model += " + " + stack
        _out(
            stream,
            "%-5s %-42s %-11s %-8s %s"
            % (
                mark,
                os.path.basename(row["path"])[:42],
                "%sx%s" % (row["width"], row["height"]) if row["width"] else "-",
                row["seed"] if row["seed"] is not None else "-",
                model,
            ),
        )

    _out(stream)
    if untraced:
        _out(
            stream,
            "%d marked ? : a model klin cannot trace to a ledger record. That is "
            "not a pass, and `klin show <file>` says which model." % untraced,
        )
    if args.check:
        _out(stream, "%d item(s) fail the ship gate" % failing)
        return 1 if (failing or untraced) else 0
    return 0


def _match(conn, target):
    """Find items by exact path, then by substring or sha256 prefix."""
    target = str(target)
    exact = conn.execute(
        "SELECT * FROM item WHERE path = ?", (os.path.normpath(target),)
    ).fetchall()
    if exact:
        return exact
    return conn.execute(
        "SELECT * FROM item WHERE REPLACE(LOWER(path), '\\', '/') LIKE ? "
        "OR LOWER(sha256) LIKE ? ORDER BY mtime DESC",
        ("%" + target.replace("\\", "/").lower() + "%", target.lower() + "%"),
    ).fetchall()


def cmd_show(args, stream):
    conn, data = _index(args)
    rows = _match(conn, args.target)
    if not rows:
        _out(stream, "nothing in the index matches %r" % args.target)
        return 2
    if len(rows) > 1:
        _out(stream, "%r matches %d items:" % (args.target, len(rows)))
        for row in rows[:20]:
            _out(stream, "  %s" % row["path"])
        if len(rows) > 20:
            _out(stream, "  ... and %d more" % (len(rows) - 20))
        _out(stream, "Narrow it; klin will not pick one for you.")
        return 2

    row = rows[0]
    models, records = _resolver(args, data)
    got = index.verdict(
        conn,
        row["path"],
        models,
        records,
        manifest.rules(data),
        manifest.build_facts(data),
    )

    _out(stream, row["path"])
    _out(
        stream,
        "  %s  %.1f MiB  sha256 %s"
        % (
            "%sx%s" % (row["width"], row["height"]) if row["width"] else "-",
            (row["bytes"] or 0) / float(1 << 20),
            (row["sha256"] or "(not hashed)")[:16],
        ),
    )
    _out(stream, "  project   %s" % (row["project"] or "(unclaimed)"))
    if row["workflow"]:
        _out(stream, "  workflow  %s" % row["workflow"][:16])
    settings = "  ".join(
        "%s %s" % (key, row[key])
        for key in ("seed", "steps", "cfg", "sampler", "scheduler")
        if row[key] is not None
    )
    if settings:
        _out(stream, "  %s" % settings)
    for note in json.loads(row["notes"] or "[]"):
        _out(stream, "  note: %s" % note)

    _out(stream)
    _out(stream, "models")
    for entry in got["models"]:
        label = entry["role"]
        if entry["strength"] is not None:
            label += "@%s" % entry["strength"]
        _out(
            stream,
            "  %-12s %-38s -> %s (%s)"
            % (
                label,
                _short(os.path.basename(str(entry["name"])))[:38],
                entry["record"],
                entry["how"],
            ),
        )
        _out(
            stream,
            "  %-12s %s  [%s]"
            % ("", entry["licence"] or "(unrecorded)", _families(entry)),
        )
    for entry in got["unresolved"]:
        label = entry["role"]
        if entry["strength"] is not None:
            label += "@%s" % entry["strength"]
        _out(
            stream,
            "  %-12s %-38s -> %s"
            % (label, _short(os.path.basename(str(entry["name"])))[:38], entry["why"]),
        )

    _out(stream)
    _out(stream, "ship: %s" % ("yes" if got["ship"] else "NO"))
    _out(stream)
    for finding in got["findings"]:
        _print_finding(stream, finding)
    if got["unresolved"]:
        for line in textwrap.wrap(
            "%d model(s) here could not be traced to a ledger record. klin "
            "reports that rather than passing them: an asset whose origin is "
            "unknown is exactly the case a ship gate exists to catch. Fetch it "
            "through klin, or add a record by hand."
            % len(got["unresolved"]),
            WRAP,
        ):
            _out(stream, line)
        _out(stream)

    if row["prompt"] and not args.no_prompt:
        _out(stream, "prompt")
        for line in textwrap.wrap(row["prompt"], WRAP - 2):
            _out(stream, "  %s" % line)
    return 0 if got["ship"] else 1


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

    gen.configure(
        verbs.add_parser("gen", help="produce an asset with a local generator")
    )

    idx = verbs.add_parser("index", help="scan what is on this machine")
    scans = idx.add_subparsers(dest="action")

    building = scans.add_parser("build", help="scan the roots and update the index")
    building.add_argument(
        "--root",
        action="append",
        help="scan this directory instead of the manifest's roots",
    )
    building.add_argument(
        "--rescan", action="store_true", help="re-read every file, changed or not"
    )
    building.add_argument(
        "--prune", action="store_true", help="drop rows whose file has gone"
    )
    building.add_argument(
        "--hash-large",
        action="store_true",
        help="hash files over the size limit too, which is slow",
    )
    building.set_defaults(func=cmd_index_build)

    reporting = scans.add_parser("status", help="what the index holds")
    reporting.set_defaults(func=cmd_index_status)

    listing_items = verbs.add_parser("ls", help="list indexed items")
    for name, help_text in (
        ("--project", "only items claimed by this project"),
        ("--model", "base model name, substring match"),
        ("--lora", "a LoRA in the stack, substring match"),
        ("--prompt", "text anywhere in the positive prompt"),
        ("--workflow", "workflow hash, or a prefix of one"),
        ("--since", "modified on or after YYYY-MM-DD"),
        ("--kind", "image, mesh or model"),
    ):
        listing_items.add_argument(name, default=None, help=help_text)
    listing_items.add_argument("--seed", type=int, default=None)
    listing_items.add_argument("--limit", type=int, default=None)
    listing_items.add_argument(
        "--unclaimed", action="store_true", help="only items no project claims"
    )
    listing_items.add_argument("--json", action="store_true")
    listing_items.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if anything listed fails the ship gate",
    )
    listing_items.set_defaults(func=cmd_ls)

    showing = verbs.add_parser("show", help="everything known about one item")
    showing.add_argument("target", help="a path, a filename fragment, or a sha256")
    showing.add_argument(
        "--no-prompt", action="store_true", help="leave the prompt text out"
    )
    showing.set_defaults(func=cmd_show)

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
        gen.GenError,
        index.IndexingError,
    ) as exc:
        _out(stream, "klin: %s" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
