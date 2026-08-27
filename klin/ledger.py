"""The ledger: one JSON object per line, one line per asset.

JSONL rather than a JSON array so that adding an asset is a one-line diff. A
re-indented array would churn the whole file on every append, which is exactly
the version-control friction this project measures elsewhere.

`ship_ok` is deliberately absent from the schema. A ship verdict is computed
from policy at audit time; storing it would let a policy amendment leave stale
verdicts behind. Only `reviewed_at` and an explicit `waiver` persist.
"""

import io
import json
import os

from . import secrets

FIELDS = (
    "id",
    "kind",
    "paths",
    "sha256",
    "author",
    "source",
    "licence",
    "produced_by",
    "modifications",
    "notes",
    "used_for",
    "reviewed_at",
    "waiver",
)


class LedgerError(Exception):
    pass


def blank(record_id, kind="mesh"):
    """A record with every field present, so a hand-edit has a shape to follow."""
    return {
        "id": record_id,
        "kind": kind,
        "paths": [],
        "sha256": None,
        "author": {"name": None, "url": None},
        "source": {
            "adapter": "manual",
            "url": None,
            "mirror_of": None,
            # Which record this one was made from, when it is a derivative. A
            # conformed mesh is not the same bytes as its source, which is what
            # `mirror_of` asserts, so it needs a field of its own. Deliberately
            # not in `sanitise`'s tuple below: this holds a record id and never
            # a URL, and scrubbing it would be a no-op today and a corruption
            # the day an id contains a question mark.
            "derived_from": None,
            "retrieved": None,
            "upstream_version": None,
        },
        "licence": {
            "id": None,
            "name": None,
            "url": None,
            "text": None,
            "attribution_required": None,
        },
        "produced_by": None,
        "modifications": [],
        "notes": None,
        "used_for": None,
        "reviewed_at": None,
        "waiver": None,
    }


def load(path):
    """Read every record. A missing ledger is an empty one, not an error."""
    if not os.path.isfile(path):
        return []
    records = []
    with io.open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise LedgerError("%s:%d: %s" % (path, number, exc))
            if not isinstance(record, dict):
                raise LedgerError("%s:%d: record is not an object" % (path, number))
            if not record.get("id"):
                raise LedgerError("%s:%d: record has no id" % (path, number))
            records.append(record)
    seen = {}
    for record in records:
        if record["id"] in seen:
            raise LedgerError("duplicate id %r in %s" % (record["id"], path))
        seen[record["id"]] = True
    return records


def save(path, records):
    """Rewrite the ledger, sorted by id so the file order never depends on
    the order things were added."""
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with io.open(path, "w", encoding="utf-8", newline="\n") as handle:
        for record in sorted(records, key=lambda r: r["id"]):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def sanitise(record):
    """Strip credential-shaped query parameters out of a record's URLs.

    Every record here is committed. A fetch adapter that follows a presigned or
    tokenised download link would otherwise write that link into the tree, and
    a URL is the field where that happens by accident rather than by mistake.
    The path and host are the provenance and survive.
    """
    for section, key in (("source", "url"), ("source", "mirror_of"), ("licence", "url")):
        node = record.get(section)
        if isinstance(node, dict) and node.get(key):
            node[key] = secrets.scrub_url(node[key])
    return record


def add(path, record, replace=False):
    record = sanitise(record)
    records = load(path)
    existing = [r for r in records if r["id"] == record["id"]]
    if existing and not replace:
        raise LedgerError(
            "%s is already in the ledger; pass --replace to overwrite" % record["id"]
        )
    records = [r for r in records if r["id"] != record["id"]]
    records.append(record)
    save(path, records)
    return record


def field(record, dotted):
    """Fetch a dotted path such as 'source.url'. Missing is None, never a raise."""
    node = record
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
        if node is None:
            return None
    return node


def cache_drift(records, cache):
    """Whether the cache klin resolves to is the one the ledger was written in.

    The failure this catches is silent by construction. `cache_dir` resolves
    from an environment variable so that one committed manifest can describe
    several machines, and a variable that was set when the files were fetched
    and is unset now resolves somewhere else entirely. Nothing appears to
    break: the records still point at real files, the audit still passes, and
    the next fetch quietly re-downloads seventeen gigabytes into a second tree.
    Barinn spent two sessions in that state, every weight under `D:/klin-cache`
    and a manifest naming `%LOCALAPPDATA%/klin/cache`.

    So the test cannot be "do the recorded files exist", which stays true
    throughout. It is whether any of them live under the cache now in force.
    Returns None when there is nothing to compare.
    """
    if not cache:
        return None
    absolute = []
    for item in records:
        for path in item.get("paths") or []:
            if os.path.isabs(path):
                absolute.append(path)
    if not absolute:
        return None

    root = os.path.normcase(os.path.normpath(cache)) + os.sep
    under = [
        path
        for path in absolute
        if os.path.normcase(os.path.normpath(path)).startswith(root)
    ]
    if under:
        return None
    return {
        "cache": os.path.normpath(cache),
        "recorded": len(absolute),
        "elsewhere": sorted(set(os.path.dirname(p) for p in absolute))[:3],
    }


def is_empty(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False
