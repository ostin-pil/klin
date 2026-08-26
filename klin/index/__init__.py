"""The index: a scan of what is on this machine, and what made it.

**The index is derived, the ledger is truth.** A ledger record is a committed
statement about an asset the project consumes, written once and reviewed by a
person. The index is a cache of what a scan found in some directories, and
deleting it costs nothing but the time to rebuild. That boundary is why the
database lives in the cache rather than the repository: the decision that no
model weights and no raw generation output ever enter a game repo applies just
as much to a table describing them.

**Machine-scoped, project-tagged.** One ComfyUI output directory serves every
project on a machine, so splitting the index per project would scan the same
files twice and answer "what else made this" with silence. Instead there is one
database, each row tagged with whichever project claimed its path, and rows no
project claims are indexed and reported rather than dropped. An unclaimed file
is the normal state of a corpus that predates the index; making it invisible
would defeat the point of scanning at all.

**The licence verdict is computed at query time, never stored.** klin already
refuses to persist `ship_ok` on a ledger record, because a policy amendment
would leave stale verdicts behind. The same reasoning applies here with more
force: the index is a scan of files that do not change, while the ledger and
the policy around them change constantly. A verdict cached in this database
would be wrong the first time somebody classified a licence by hand.
"""

import fnmatch
import hashlib
import importlib
import json
import os
import pkgutil
import sqlite3
import time

from .. import ledger, manifest, policy
from ..fetch import MODELS_ENV

#: Where the database goes when a machine wants it somewhere specific. Same
#: reasoning as `KLIN_CACHE`: which volume has room is a fact about the
#: machine, not about the project, so it does not belong in a committed file.
DB_ENV = "KLIN_INDEX"

#: Bumped when the schema changes in a way a rebuild has to notice. The scanner
#: drops and recreates rather than migrating: every row is derived from a file
#: that is still on disk, so a rebuild is the cheapest possible migration.
SCHEMA_VERSION = 1

#: Files bigger than this are indexed without a hash unless asked. A scan of an
#: output tree should not stall for half an hour on a seventeen-gigabyte
#: checkpoint, and identity resolution below never needs the hash anyway.
HASH_LIMIT = 256 * 1024 * 1024

KINDS = {
    "image": (".png", ".jpg", ".jpeg", ".webp"),
    "mesh": (".glb", ".gltf", ".fbx", ".obj", ".blend"),
    "model": (".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".sft"),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS item (
    path       TEXT PRIMARY KEY,
    kind       TEXT,
    bytes      INTEGER,
    mtime      REAL,
    sha256     TEXT,
    width      INTEGER,
    height     INTEGER,
    project    TEXT,
    root       TEXT,
    source     TEXT,
    model      TEXT,
    seed       INTEGER,
    steps      INTEGER,
    cfg        REAL,
    sampler    TEXT,
    scheduler  TEXT,
    denoise    REAL,
    prompt     TEXT,
    negative   TEXT,
    workflow   TEXT,
    graph      TEXT,
    notes      TEXT,
    scanned_at TEXT
);

CREATE TABLE IF NOT EXISTS item_lora (
    path     TEXT,
    ord      INTEGER,
    name     TEXT,
    strength REAL,
    PRIMARY KEY (path, ord)
);

CREATE INDEX IF NOT EXISTS item_project  ON item (project);
CREATE INDEX IF NOT EXISTS item_model    ON item (model);
CREATE INDEX IF NOT EXISTS item_workflow ON item (workflow);
CREATE INDEX IF NOT EXISTS lora_name     ON item_lora (name);
"""


class IndexingError(Exception):
    pass


def readers():
    """Every provenance reader in this package, by name.

    Discovery, not a list, for the same reason `klin.fetch` discovers its
    adapters: a reader for a new generator should be a new file and no edit
    anywhere else. A module qualifies by declaring `NAME` and `read`.
    """
    found = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_") or info.name == "png":
            continue
        module = importlib.import_module("%s.%s" % (__name__, info.name))
        if getattr(module, "NAME", None) and hasattr(module, "read"):
            found[module.NAME] = module
    return found


def db_path(data):
    """Where the database lives: `KLIN_INDEX`, else beside the cache."""
    override = os.environ.get(DB_ENV)
    if override:
        return os.path.normpath(os.path.expanduser(os.path.expandvars(override)))
    return os.path.join(manifest.cache_dir(data), "index.sqlite3")


def models_dir(data):
    """The weights tree, which is what a generator's model names refer to.

    Same resolution `klin fetch --as` uses, and deliberately the same constant,
    so a machine configures the tree once. Absent is not an error: without it
    the index still records which weight names produced a file, and only the
    join back to a ledger record goes missing.
    """
    value = os.environ.get(MODELS_ENV) or data.get("models_dir")
    if not value:
        return None
    return os.path.normpath(os.path.expanduser(os.path.expandvars(str(value))))


def connect(path):
    """Open the database, creating it and its schema when absent."""
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    got = conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
    if got is None:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
    elif int(got["value"]) != SCHEMA_VERSION:
        raise IndexingError(
            "index at %s is schema v%s, this klin writes v%d. Delete it and "
            "run `klin index build`: every row is derived from a file still on "
            "disk, so a rebuild is the cheapest migration there is."
            % (path, got["value"], SCHEMA_VERSION)
        )
    return conn


def roots(data):
    """The directories this project asks klin to scan, absolute."""
    block = data.get("index") or {}
    if not isinstance(block, dict):
        raise IndexingError("manifest 'index' is not a mapping")
    got = block.get("roots") or []
    if not isinstance(got, list):
        raise IndexingError("manifest 'index.roots' is not a list")
    out = []
    for value in got:
        expanded = os.path.expanduser(os.path.expandvars(str(value)))
        if manifest.UNEXPANDED.search(expanded):
            raise IndexingError(
                "index root %r still holds an unset variable after expansion"
                % expanded
            )
        out.append(os.path.normpath(expanded))
    return out


def claims(data):
    """Glob patterns naming which scanned paths belong to this project.

    Matched with `fnmatch` against the path relative to its root, separators
    normalised to `/`. Note that `*` crosses directory separators here, so
    `mock/*` claims everything below `mock/` and there is no separate `**`.
    """
    block = data.get("index") or {}
    got = block.get("claim") or []
    if not isinstance(got, list):
        raise IndexingError("manifest 'index.claim' is not a list")
    return [str(value).replace("\\", "/") for value in got]


def _kind(path):
    ext = os.path.splitext(path)[1].lower()
    for name, extensions in KINDS.items():
        if ext in extensions:
            return name
    return None


def _sha256(path, limit=HASH_LIMIT):
    """The file's hash, or None when it is too big to be worth one here."""
    if limit is not None and os.path.getsize(path) > limit:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _claimed(relative, patterns):
    relative = relative.replace("\\", "/")
    return any(fnmatch.fnmatch(relative, pattern) for pattern in patterns)


def _provenance(path, modules):
    """Ask each reader in turn; the first that recognises the file wins."""
    for name in sorted(modules):
        module = modules[name]
        try:
            got = module.read(path)
        except Exception as exc:  # a malformed file is data, not a crash
            return {"source": name, "notes": ["%s could not read it: %s" % (name, exc)]}
        if got is not None:
            return got
    return None


def scan(conn, root, project, patterns, rescan=False, hash_limit=HASH_LIMIT, say=None):
    """Walk one root, updating rows for files that changed.

    Unchanged is `(bytes, mtime)` matching what is already stored, which is the
    same test every build tool uses and is wrong only if a file is rewritten
    within the filesystem's timestamp resolution at exactly its old length.
    `--rescan` exists for that, and for a reader that has since learned to
    understand a file it previously could not.
    """
    from . import png as png_reader

    modules = readers()
    stats = {
        "seen": 0,
        "added": 0,
        "updated": 0,
        "skipped": 0,
        "unclaimed": 0,
        "conflicts": [],
        "provenance": 0,
        "unhashed": 0,
    }
    if not os.path.isdir(root):
        raise IndexingError("index root does not exist: %s" % root)

    known = dict(
        (row["path"], row)
        for row in conn.execute(
            "SELECT path, bytes, mtime, project FROM item WHERE root = ?", (root,)
        )
    )
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for filename in filenames:
            path = os.path.normpath(os.path.join(dirpath, filename))
            kind = _kind(path)
            if kind is None:
                continue
            stats["seen"] += 1
            try:
                info = os.stat(path)
            except OSError:
                continue

            relative = os.path.relpath(path, root)
            claimed = project if _claimed(relative, patterns) else None
            if claimed is None:
                stats["unclaimed"] += 1

            previous = known.get(path)
            if previous is not None:
                held = previous["project"]
                if held and claimed and held != claimed:
                    stats["conflicts"].append((path, held, claimed))
                    claimed = held
                elif claimed is None:
                    claimed = held
                if (
                    not rescan
                    and previous["bytes"] == info.st_size
                    and previous["mtime"] == info.st_mtime
                ):
                    if claimed != previous["project"]:
                        conn.execute(
                            "UPDATE item SET project = ? WHERE path = ?",
                            (claimed, path),
                        )
                    stats["skipped"] += 1
                    continue

            got = _provenance(path, modules) if kind == "image" else None
            if got and got.get("model"):
                stats["provenance"] += 1
            digest = _sha256(path, hash_limit)
            if digest is None:
                stats["unhashed"] += 1
            size = png_reader.dimensions(path) if path.lower().endswith(".png") else None
            base = (got or {}).get("model") or {}

            conn.execute(
                "INSERT OR REPLACE INTO item (path, kind, bytes, mtime, sha256, "
                "width, height, project, root, source, model, seed, steps, cfg, "
                "sampler, scheduler, denoise, prompt, negative, workflow, graph, "
                "notes, scanned_at) VALUES (%s)" % ",".join("?" * 23),
                (
                    path,
                    kind,
                    info.st_size,
                    info.st_mtime,
                    digest,
                    size[0] if size else None,
                    size[1] if size else None,
                    claimed,
                    root,
                    (got or {}).get("source"),
                    base.get("name"),
                    (got or {}).get("seed"),
                    (got or {}).get("steps"),
                    (got or {}).get("cfg"),
                    (got or {}).get("sampler"),
                    (got or {}).get("scheduler"),
                    (got or {}).get("denoise"),
                    (got or {}).get("prompt"),
                    (got or {}).get("negative"),
                    (got or {}).get("workflow_sha256"),
                    json.dumps((got or {}).get("graph")) if got else None,
                    json.dumps((got or {}).get("notes") or []) if got else None,
                    now,
                ),
            )
            conn.execute("DELETE FROM item_lora WHERE path = ?", (path,))
            for order, lora in enumerate((got or {}).get("loras") or []):
                conn.execute(
                    "INSERT INTO item_lora (path, ord, name, strength) "
                    "VALUES (?, ?, ?, ?)",
                    (path, order, lora.get("name"), lora.get("strength")),
                )
            stats["updated" if previous is not None else "added"] += 1
            if say and stats["seen"] % 500 == 0:
                say("  ... %d files" % stats["seen"])

    conn.commit()
    return stats


def forget_missing(conn):
    """Drop rows whose file is gone. A scan adds and updates; this removes."""
    gone = [
        row["path"]
        for row in conn.execute("SELECT path FROM item")
        if not os.path.exists(row["path"])
    ]
    for path in gone:
        conn.execute("DELETE FROM item WHERE path = ?", (path,))
        conn.execute("DELETE FROM item_lora WHERE path = ?", (path,))
    conn.commit()
    return gone


def model_map(models_dir, records):
    """Every weight file in the models tree, mapped to a ledger record.

    Resolution is by **filesystem identity**, not by name. `klin fetch --as`
    hardlinks a cached file into the models tree, so the tree's entry and the
    path recorded in the ledger are one file with two names, and `(st_dev,
    st_ino)` identifies it exactly and for free.

    Names cannot do this job, and the evidence is in the tree this was built
    against: of fourteen recorded models, not one had the same filename in both
    places, because the tree is where files get renamed to something readable.
    Two of them are the same file linked under two names, which a name match
    would have reported as two different models. So a name match is offered
    only as a labelled fallback, never silently.

    Returns `name -> {"record": id, "how": "identity" | "name", "path": ...}`,
    keyed by both the bare filename and any subdirectory-qualified form a
    generator might use. A name resolving to more than one distinct record is
    left out with `how: "ambiguous"`, because picking one would be a guess.
    """
    by_identity = {}
    by_name = {}
    for record in records:
        for path in record.get("paths") or []:
            if not os.path.isfile(path):
                continue
            try:
                info = os.stat(path)
            except OSError:
                continue
            by_identity.setdefault((info.st_dev, info.st_ino), set()).add(record["id"])
            by_name.setdefault(os.path.basename(path).lower(), set()).add(record["id"])

    found = {}
    if not models_dir or not os.path.isdir(models_dir):
        return found

    for dirpath, _dirnames, filenames in os.walk(models_dir):
        for filename in filenames:
            if _kind(filename) != "model":
                continue
            path = os.path.join(dirpath, filename)
            try:
                info = os.stat(path)
            except OSError:
                continue
            hit = by_identity.get((info.st_dev, info.st_ino))
            how = "identity"
            if not hit:
                hit = by_name.get(filename.lower())
                how = "name"
            if not hit:
                continue
            entry = {
                "record": sorted(hit)[0] if len(hit) == 1 else None,
                "records": sorted(hit),
                "how": how if len(hit) == 1 else "ambiguous",
                "path": path,
            }
            # A generator names a weight relative to its category directory, so
            # both the bare name and the qualified form have to resolve.
            keys = {filename.lower()}
            relative = os.path.relpath(path, models_dir).replace("\\", "/")
            keys.add(relative.lower())
            if "/" in relative:
                keys.add(relative.split("/", 1)[1].lower())
            for key in keys:
                found[key] = entry
    return found


def used_models(conn, path):
    """Every weight name one item used: the base model, then the LoRA stack."""
    row = conn.execute("SELECT model FROM item WHERE path = ?", (path,)).fetchone()
    names = []
    if row and row["model"]:
        names.append((row["model"], None, "base"))
    for lora in conn.execute(
        "SELECT name, strength FROM item_lora WHERE path = ? ORDER BY ord", (path,)
    ):
        names.append((lora["name"], lora["strength"], "lora"))
    return names


def lookup(models, name):
    """Resolve one weight name against the model map, however it was written."""
    if not name:
        return None
    key = str(name).replace("\\", "/").lower()
    return models.get(key) or models.get(os.path.basename(key))


def verdict(conn, path, models, records, rules, facts, ship=True):
    """What the policy says about the models that made one item.

    Computed here and never stored, for the same reason `ship_ok` is absent
    from a ledger record. An unresolved model is reported as unresolved and is
    never treated as clean: a file klin cannot trace is exactly the case a ship
    gate exists to stop, and silence would invert its meaning.
    """
    by_id = dict((r["id"], r) for r in records)
    out = {"models": [], "findings": [], "unresolved": [], "ship": True}

    for name, strength, role in used_models(conn, path):
        entry = lookup(models, name)
        if entry is None or not entry.get("record"):
            out["unresolved"].append(
                {"name": name, "role": role, "strength": strength,
                 "why": "ambiguous" if entry else "no ledger record"}
            )
            out["ship"] = False
            continue
        record = by_id.get(entry["record"])
        if record is None:
            out["unresolved"].append(
                {"name": name, "role": role, "strength": strength,
                 "why": "record %s is not in this ledger" % entry["record"]}
            )
            out["ship"] = False
            continue
        # `declared` and `families` differ in a way worth carrying separately.
        # An adapter that reads a vendor's permission flags and finds nothing
        # restrictive writes `families: []`, which is a checked result. But
        # `policy.families` tests that list for truthiness, so an empty one
        # falls through to the identifier, and a `LicenseRef-` id classifies as
        # `unknown`. Reporting "unknown" for a licence somebody did resolve is
        # the one thing this tool must not do, so the raw declaration travels
        # alongside the classification and the caller can tell them apart.
        out["models"].append(
            {
                "name": name,
                "role": role,
                "strength": strength,
                "record": record["id"],
                "how": entry["how"],
                "licence": ledger.field(record, "licence.id"),
                "families": sorted(policy.families(record)),
                "declared": ledger.field(record, "licence.families"),
            }
        )

    used = [by_id[m["record"]] for m in out["models"] if m["record"] in by_id]
    if used:
        findings = policy.evaluate(used, rules, facts, ship=ship)
        out["findings"] = findings
        if policy.failed(findings):
            out["ship"] = False
    return out


def query(
    conn,
    project=None,
    unclaimed=False,
    model=None,
    lora=None,
    seed=None,
    prompt=None,
    workflow=None,
    since=None,
    kind=None,
    with_provenance=None,
    limit=None,
):
    """Rows matching the filters, newest file first."""
    where = []
    params = []
    if project:
        where.append("item.project = ?")
        params.append(project)
    if unclaimed:
        where.append("item.project IS NULL")
    if kind:
        where.append("item.kind = ?")
        params.append(kind)
    if model:
        where.append("LOWER(item.model) LIKE ?")
        params.append("%" + model.lower() + "%")
    if seed is not None:
        where.append("item.seed = ?")
        params.append(seed)
    if workflow:
        where.append("item.workflow LIKE ?")
        params.append(workflow + "%")
    if prompt:
        where.append("LOWER(item.prompt) LIKE ?")
        params.append("%" + prompt.lower() + "%")
    if since:
        where.append("item.mtime >= ?")
        params.append(_epoch(since))
    if with_provenance is True:
        where.append("item.model IS NOT NULL")
    elif with_provenance is False:
        where.append("item.model IS NULL")
    if lora:
        where.append(
            "EXISTS (SELECT 1 FROM item_lora WHERE item_lora.path = item.path "
            "AND LOWER(item_lora.name) LIKE ?)"
        )
        params.append("%" + lora.lower() + "%")

    sql = "SELECT * FROM item"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY item.mtime DESC"
    if limit:
        sql += " LIMIT %d" % int(limit)
    return list(conn.execute(sql, params))


def _epoch(value):
    """A `YYYY-MM-DD` date as a timestamp. Anything else is an error worth
    raising, because a silently-ignored filter reads as "nothing matched"."""
    try:
        return time.mktime(time.strptime(str(value)[:10], "%Y-%m-%d"))
    except ValueError:
        raise IndexingError("--since wants YYYY-MM-DD, got %r" % (value,))


def loras_of(conn, path):
    return list(
        conn.execute(
            "SELECT name, strength FROM item_lora WHERE path = ? ORDER BY ord",
            (path,),
        )
    )


def status(conn):
    """A summary of what the index holds."""
    one = lambda sql, *a: conn.execute(sql, a).fetchone()[0]  # noqa: E731
    return {
        "items": one("SELECT COUNT(*) FROM item"),
        "with_provenance": one("SELECT COUNT(*) FROM item WHERE model IS NOT NULL"),
        "unclaimed": one("SELECT COUNT(*) FROM item WHERE project IS NULL"),
        "unhashed": one("SELECT COUNT(*) FROM item WHERE sha256 IS NULL"),
        "workflows": one("SELECT COUNT(DISTINCT workflow) FROM item "
                         "WHERE workflow IS NOT NULL"),
        "projects": [
            (row["project"], row["n"])
            for row in conn.execute(
                "SELECT project, COUNT(*) AS n FROM item GROUP BY project "
                "ORDER BY n DESC"
            )
        ],
        "models": [
            (row["model"], row["n"])
            for row in conn.execute(
                "SELECT model, COUNT(*) AS n FROM item WHERE model IS NOT NULL "
                "GROUP BY model ORDER BY n DESC"
            )
        ],
        "roots": [
            (row["root"], row["n"])
            for row in conn.execute(
                "SELECT root, COUNT(*) AS n FROM item GROUP BY root ORDER BY n DESC"
            )
        ],
    }
