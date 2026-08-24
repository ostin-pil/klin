"""Read a klin manifest: a markdown document with one fenced YAML block.

Same convention as lifecycle-kit's manifest, deliberately. The prose around the
block is for the human; the block is the only thing klin reads.
"""

import io
import os
import re

import yaml

FENCE = re.compile(r"^```ya?ml[ \t]*\r?$(.*?)^```[ \t]*\r?$", re.M | re.S)

DEFAULT_MANIFEST = os.path.join(".claude", "klin-manifest.md")


class ManifestError(Exception):
    pass


def load(path):
    """Return the manifest's YAML block as a dict."""
    if not os.path.isfile(path):
        raise ManifestError("no manifest at %s" % path)
    text = io.open(path, encoding="utf-8").read()
    match = FENCE.search(text)
    if match is None:
        raise ManifestError("%s: no fenced yaml block" % path)
    data = yaml.safe_load(match.group(1))
    if not isinstance(data, dict):
        raise ManifestError("%s: yaml block is not a mapping" % path)
    return data


def resolve(manifest, key, repo, default=None):
    """Resolve a repo-relative path key from the manifest to an absolute path."""
    value = manifest.get(key, default)
    if value is None:
        raise ManifestError("manifest has no '%s'" % key)
    return os.path.normpath(os.path.join(repo, value))


#: An environment override for the cache location. A manifest is committed and
#: describes the project; which volume on this machine has room for eighty
#: gigabytes of weights is not a property of the project and does not belong in
#: it. Same reasoning the lifecycle manifest already applies to a local engine
#: path.
CACHE_ENV = "KLIN_CACHE"

#: What an unexpanded variable looks like in either platform's syntax. Checked
#: in both forms regardless of host, because `os.path.expandvars` only knows the
#: local one: a Windows `%VAR%` passes straight through a Linux CI runner
#: untouched, and would otherwise be taken for a directory name.
UNEXPANDED = re.compile(r"%[A-Za-z_][A-Za-z0-9_]*%|\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")


def cache_dir(manifest, default=None):
    """Resolve `cache_dir` to an absolute path outside the repo.

    Deliberately not `resolve()`. That joins its value to the repo root, which
    is right for `ledger` and `staging_dir` and wrong here. The cache is defined
    as living outside every checkout, and its value carries environment
    variables on purpose so one committed manifest can describe several
    machines. Joining `%LOCALAPPDATA%/klin/cache` onto a repo root produces a
    literal `%LOCALAPPDATA%` directory inside the tree — wrong in both halves,
    and wrong quietly, because that is a perfectly legal directory name and the
    download would succeed into it.
    """
    value = os.environ.get(CACHE_ENV) or manifest.get("cache_dir", default)
    if value is None:
        raise ManifestError("manifest has no 'cache_dir', and %s is unset" % CACHE_ENV)
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    if UNEXPANDED.search(expanded):
        raise ManifestError(
            "cache_dir is %r after expansion, so a variable in it is not set on "
            "this machine. Set it, or set %s." % (expanded, CACHE_ENV)
        )
    if not os.path.isabs(expanded):
        raise ManifestError(
            "cache_dir resolved to %r, which is not absolute. The cache lives "
            "outside every checkout, so a repo-relative path cannot be what was "
            "meant; give an absolute path or set %s." % (expanded, CACHE_ENV)
        )
    return os.path.normpath(expanded)


def rules(manifest):
    got = manifest.get("rules") or []
    if not isinstance(got, list):
        raise ManifestError("manifest 'rules' is not a list")
    for rule in got:
        if not isinstance(rule, dict):
            raise ManifestError("a rule is not a mapping: %r" % (rule,))
        if "id" not in rule or "kind" not in rule:
            raise ManifestError("rule missing 'id' or 'kind': %r" % (rule,))
    return got


SECRET_KEYS = ("env", "ref", "description")


def secrets(manifest):
    """The secrets block: names and references, never values.

    A manifest is committed into the consuming project's repository, so it may
    say that an adapter needs a HuggingFace token and what the conventional
    environment variable for it is called. It may never say what the token is.
    Absence of the block is not an error; only adapters need one.
    """
    got = manifest.get("secrets") or {}
    if not isinstance(got, dict):
        raise ManifestError("manifest 'secrets' is not a mapping")
    for name, spec in got.items():
        if not isinstance(spec, dict):
            raise ManifestError("secret %r is not a mapping: %r" % (name, spec))
        for key, value in spec.items():
            if key not in SECRET_KEYS:
                raise ManifestError(
                    "secret %r has unknown key %r; expected one of %s"
                    % (name, key, ", ".join(SECRET_KEYS))
                )
            if not isinstance(value, str):
                raise ManifestError(
                    "secret %r: '%s' is not a string: %r" % (name, key, value)
                )
    return got


def build_facts(manifest):
    facts = manifest.get("build_facts") or {}
    if not isinstance(facts, dict):
        raise ManifestError("manifest 'build_facts' is not a mapping")
    return facts
