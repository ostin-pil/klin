"""Vendor adapters, and the registry that finds them.

`cli` promises that "adding one must never require touching this file's
structure". That promise is kept by discovery rather than by a list: every
module in this package that declares `NAME` and `configure` becomes a
subcommand of `klin fetch`. Vendor three is a new file in this directory and no
edit anywhere else, which is a claim the test suite checks rather than trusts.

What every adapter owes, beyond fetching bytes:

**Never guess a licence.** klin holds no opinions about licences; the consuming
project's policy document does. An adapter maps a vendor field only where the
mapping is exact, and where it is not, records the vendor's own words verbatim
and says so loudly. `policy.families` classifies an unmapped identifier as
`unknown`, which fails the stage rule visibly. An adapter that invented
`noncommercial` for `license:other` would be a worse bug than one that refuses,
because a refusal appears in the audit and an invention does not.

**Record what was mapped from what.** Where a vendor publishes permission flags
instead of an identifier, the derived families and the raw flags both go into
the record. A reader who disagrees with the mapping can then see the input, and
the sidecar keeps the whole response so a re-classification never needs a
re-download.
"""

import hashlib
import importlib
import io
import json
import os
import pkgutil
import time

from .. import ledger, manifest, net, policy, secrets

#: Where `--as` lands when a project wires the cache into a downstream tool.
#: Optional: without it a fetch still succeeds and reports the cache path.
MODELS_ENV = "KLIN_MODELS"


class FetchError(Exception):
    pass


def adapters():
    """Every adapter module in this package, by name.

    Discovery, not a list. A module that fails to import is a bug worth
    surfacing immediately rather than a vendor that silently goes missing.
    """
    found = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module("%s.%s" % (__name__, info.name))
        name = getattr(module, "NAME", None)
        if not name or not hasattr(module, "configure"):
            continue
        found[name] = module
    return found


def configure(parser):
    """Wire every discovered adapter in as a subcommand."""
    vendors = parser.add_subparsers(dest="vendor")
    for name in sorted(adapters()):
        module = adapters()[name]
        sub = vendors.add_parser(name, help=getattr(module, "HELP", None))
        sub.add_argument(
            "--as",
            dest="as_kind",
            default=None,
            help="subdirectory of the models tree to link the file into",
        )
        sub.add_argument(
            "--families",
            default=None,
            help="comma-separated licence families, set by hand; wins outright",
        )
        sub.add_argument("--dest", default=None, help="write here instead of the cache")
        sub.add_argument("--resume", action="store_true", help="continue a partial file")
        sub.add_argument(
            "--force",
            action="store_true",
            help="re-download even when a verified copy is already cached",
        )
        sub.add_argument(
            "--dry-run",
            action="store_true",
            help="resolve and classify, but download nothing",
        )
        sub.add_argument(
            "--adopt",
            default=None,
            metavar="PATH",
            help="record a file already on disk as this one, fetching nothing",
        )
        module.configure(sub)
        sub.set_defaults(func=_run_adapter, module=module)
    return parser


def _run_adapter(args, stream):
    module = args.module
    data = {}
    path = args.manifest or os.path.join(args.repo, manifest.DEFAULT_MANIFEST)
    if os.path.isfile(path):
        data = manifest.load(path)
    return module.run(args, Context(args, data, stream))


class Context(object):
    """What an adapter is handed: paths, credentials, and the output stream.

    Deliberately narrow. An adapter resolves metadata, classifies a licence and
    streams a file; it does not read the policy rules, and it does not decide
    whether a record passes a gate.
    """

    def __init__(self, args, data, stream):
        self.args = args
        self.manifest = data
        self.stream = stream
        self.repo = args.repo

    def say(self, text=""):
        self.stream.write(text + "\n")

    def cache_dir(self):
        return manifest.cache_dir(self.manifest, default=None)

    def models_dir(self):
        value = os.environ.get(MODELS_ENV) or self.manifest.get("models_dir")
        if not value:
            return None
        return os.path.normpath(os.path.expanduser(os.path.expandvars(str(value))))

    def token(self, name):
        """A credential, or None when the vendor does not need one.

        Resolution order is the secrets module's business, not the adapter's.
        A missing credential is not an error here: some vendors are open, and
        the download itself reports a 401 with a message that says what to do.
        """
        try:
            return secrets.resolve(name, manifest.secrets(self.manifest).get(name))
        except secrets.SecretError:
            return None

    def ledger_path(self):
        return manifest.resolve(self.manifest, "ledger", self.repo)


def target_path(ctx, vendor, ident, filename):
    """`<cache>/<vendor>/<id>/<filename>`, or `--dest` when given."""
    if ctx.args.dest:
        dest = os.path.expanduser(os.path.expandvars(ctx.args.dest))
        if os.path.isdir(dest):
            return os.path.join(dest, filename)
        return dest
    return os.path.join(ctx.cache_dir(), vendor, str(ident), filename)


def write_sidecar(path, payload):
    """The vendor's raw metadata, beside the file.

    This is what makes a later re-classification possible without fetching
    seventeen gigabytes again. It is also the evidence for the mapping: when
    somebody disputes a derived family, the input is on disk.
    """
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    sidecar = os.path.join(parent, "meta.json")
    with io.open(sidecar, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return sidecar


def link_into_models(ctx, path, kind):
    """Hardlink a cached file into the models tree, so it exists once.

    A copy would double the disk cost of every checkpoint, and a config file
    listing a second search root is not always available: a tree may already be
    reached through a junction, in which case adding it again registers the same
    directory twice under two names. A hardlink sidesteps both, and falls back
    to a copy when the cache and the tree are on different volumes.
    """
    if not kind:
        return None
    root = ctx.models_dir()
    if not root:
        ctx.say(
            "note: --as %s given but no models tree configured; set %s or a "
            "'models_dir' manifest key to have klin link it in" % (kind, MODELS_ENV)
        )
        return None
    target_dir = os.path.join(root, kind)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    target = os.path.join(target_dir, os.path.basename(path))
    if os.path.exists(target):
        if os.path.samefile(target, path):
            return target
        os.remove(target)
    try:
        os.link(path, target)
    except OSError:
        import shutil

        shutil.copy2(path, target)
        ctx.say("note: hardlink unavailable across volumes; copied instead")
    return target


def find_local(ctx, expected_size, expected_sha256):
    """A file already on this machine whose bytes are the vendor's, or None.

    Downloading something the machine already holds is the common case, not the
    exceptional one: a model tree is usually older than the tool recording it,
    and the same weight arrives under different names from different vendors.
    So the search runs before every transfer rather than behind a flag.

    **Size first, hash second.** The vendor publishes a size, so a stat over
    the tree reduces tens of thousands of files to the handful that could
    possibly be this one, and only those get read. Hashing a tree to find one
    file would cost more than downloading it.

    **A published hash is required.** A size match alone is not provenance:
    `sd_xl_base_1.0.safetensors` and `sd_xl_base_1.0_0.9vae.safetensors` are
    byte-identical in length and different models. Adopting on size would have
    recorded one as the other, silently, and the record would have been false.
    Explicit `--adopt` is allowed to proceed on size and header alone because a
    person named that file; this runs unattended and may not guess.
    """
    if not expected_size or not expected_sha256:
        return None

    roots = []
    for root in (ctx.models_dir(), _cache_root(ctx)):
        if root and os.path.isdir(root) and root not in roots:
            roots.append(root)
    if not roots:
        return None

    candidates = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                try:
                    if os.path.getsize(path) == int(expected_size):
                        candidates.append(path)
                except OSError:
                    continue
    if not candidates:
        return None

    wanted = str(expected_sha256).lower()
    for path in candidates:
        digest = hashlib.sha256()
        try:
            with io.open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
        except OSError:
            continue
        if digest.hexdigest() == wanted:
            return path

    ctx.say(
        "note: %d file(s) here are the right size and none has the vendor's "
        "hash, so this is a real download" % len(candidates)
    )
    return None


def _cache_root(ctx):
    try:
        return ctx.cache_dir()
    except Exception:
        return None


def adopt(ctx, path, expected_size=None, expected_sha256=None):
    """Record a file already on disk as the vendor's, transferring nothing.

    Every model tree predates the tool that would have recorded it. Barinn's
    held sixty-nine gigabytes of base models across five files, none of them
    fetched through klin, and every image they produced was therefore
    untraceable. The options were re-downloading all of it to learn what was
    already on the disk, or writing the records by hand and inventing the
    licences. Neither is acceptable, so this is the third one.

    **Adoption is a fetch minus the transfer, not a weaker fetch.** The guards
    that make a downloaded file trustworthy are all properties of the bytes
    rather than of how they arrived: the size matches what the vendor
    publishes, a `.safetensors` header parses, and the digest matches the
    vendor's own hash. Running those against a local file proves the same thing
    a download proves, which is that this is the vendor's file.

    So a mismatch is refused rather than noted. A record is an assertion about
    provenance, and writing one for a file that failed the vendor's own hash
    would assert something klin has just disproved. The two guards that cannot
    run are the transport's own (content type, and the partial-file check), and
    neither says anything about a file that is already complete.
    """
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    if not os.path.isfile(path):
        raise FetchError("--adopt %s is not a file" % path)

    size = os.path.getsize(path)
    if expected_size and int(size) != int(expected_size):
        raise FetchError(
            "%s is %d bytes and the vendor publishes %d. This is not that "
            "file, so klin will not record it as one. Check the revision, or "
            "the variant: repositories often hold several quantisations whose "
            "names differ by a few characters." % (path, size, int(expected_size))
        )

    if path.endswith(".safetensors"):
        net.check_safetensors(path)

    ctx.say("hashing %s (%.1f GiB)" % (path, size / float(1 << 30)))
    digest = hashlib.sha256()
    with io.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    got = digest.hexdigest()

    if expected_sha256:
        if got.lower() != str(expected_sha256).lower():
            raise FetchError(
                "sha256 mismatch: vendor publishes %s, %s hashes to %s. The "
                "bytes on disk are not the bytes the vendor serves, and a "
                "record saying otherwise would be false."
                % (expected_sha256, path, got)
            )
        ctx.say("sha256 matches the vendor's published hash")
    else:
        ctx.say(
            "note: this vendor publishes no hash for the file, so the size and "
            "the safetensors header are the whole of the check"
        )

    return {
        "path": path,
        "bytes": size,
        "sha256": got,
        "content_type": None,
        "final_url": None,
        "adopted": True,
    }


def write_sidecar_beside(path, payload):
    """The vendor's metadata for an adopted file, as `<file>.meta.json`.

    The cache gives every file its own directory, so a plain `meta.json` is
    unambiguous there. A weights tree does not: adopting two checkpoints from
    the same directory would have the second overwrite the first's metadata,
    silently and with a file that looks right.
    """
    sidecar = path + ".meta.json"
    with io.open(sidecar, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return sidecar


def classify(ctx, record, derived=None):
    """Settle a record's licence families, and say so when nobody could.

    Precedence, highest first: `--families` on the command line, then whatever
    the adapter derived from vendor fields, then `policy.families` reading the
    identifier. The first two write `licence.families` onto the record, which
    `policy.families` documents as winning outright.
    """
    override = ctx.args.families
    if override:
        record["licence"]["families"] = sorted(
            part.strip() for part in override.split(",") if part.strip()
        )
        return "hand", set(record["licence"]["families"])

    if derived is not None:
        record["licence"]["families"] = sorted(derived)
        return "derived", set(derived)

    got = policy.families(record)
    if got in ({"unknown"}, {"unlicensed"}):
        return "unknown", got
    return "identifier", got


def report_classification(ctx, record, how, found):
    """Print the classification, and demand a human where one is needed."""
    ident = ledger.field(record, "licence.id") or "(none recorded)"
    ctx.say("licence: %s -> %s (%s)" % (ident, ", ".join(sorted(found)) or "none", how))
    if how == "unknown":
        ctx.say("")
        ctx.say("  This licence maps to nothing klin understands, so it has been")
        ctx.say("  recorded verbatim and left unclassified. That is deliberate:")
        ctx.say("  klin does not guess licences, and an invented family would")
        ctx.say("  pass an audit that ought to stop. Classify it by hand:")
        ctx.say("")
        ctx.say("      klin fetch ... --families noncommercial")
        ctx.say("")
        ctx.say("  Families: %s" % ", ".join(policy.KNOWN_FAMILIES))
        ctx.say("")


def record_for(vendor, ident, filename):
    """A blank record with the adapter and retrieval date already filled in."""
    record = ledger.blank("%s-%s" % (vendor, ident), kind="model")
    record["source"]["adapter"] = vendor
    record["source"]["retrieved"] = time.strftime("%Y-%m-%d")
    return record


def finish(ctx, record, facts, linked=None):
    """Write the record and tell the user what to run next."""
    record["sha256"] = facts["sha256"]
    paths = [facts["path"]]
    if linked:
        paths.append(linked)
    record["paths"] = paths

    path = ctx.ledger_path()
    ledger.add(path, record, replace=True)

    ctx.say("")
    ctx.say("%s  %.2f GiB" % (facts["path"], facts["bytes"] / float(1 << 30)))
    if linked:
        ctx.say("linked  %s" % linked)
    ctx.say("sha256  %s" % facts["sha256"])
    ctx.say("recorded %s in %s" % (record["id"], os.path.relpath(path, ctx.repo)))
    ctx.say("")
    ctx.say("next: klin ledger audit")
    return 0
