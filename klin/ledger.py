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


def add(path, record, replace=False):
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


def is_empty(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False
