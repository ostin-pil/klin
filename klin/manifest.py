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


def build_facts(manifest):
    facts = manifest.get("build_facts") or {}
    if not isinstance(facts, dict):
        raise ManifestError("manifest 'build_facts' is not a mapping")
    return facts
