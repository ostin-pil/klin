"""Licence classification and rule evaluation.

The rules are not written here. They live in the consuming project's manifest,
transcribed from that project's own policy document, and every rule carries the
text it was transcribed from. This module applies them and quotes them back, so
a rule that has drifted from its source shows up in the audit output instead of
silently doing the wrong thing.
"""

import fnmatch

from . import ledger

#: Families a licence can belong to. A licence may be in several at once:
#: CC-BY-NC-SA is attribution, noncommercial *and* share-alike.
KNOWN_FAMILIES = (
    "public-domain",
    "permissive",
    "attribution",
    "share-alike",
    "noncommercial",
    "noderivatives",
    "copyleft",
    "editorial",
    "fan-art",
    "unknown",
    "unlicensed",
)

_PERMISSIVE = ("MIT*", "APACHE*", "BSD*", "ISC*", "ZLIB*")
_COPYLEFT = ("GPL*", "AGPL*", "LGPL*", "*-GPL-*")


def families(record):
    """Classify a record's licence into zero or more families.

    An explicit `licence.families` list on the record wins outright. That is the
    escape hatch for anything no identifier can express — unlicensed fan art of
    a trademarked property being the case this project cares about.
    """
    explicit = ledger.field(record, "licence.families")
    if explicit:
        return set(explicit)

    raw = ledger.field(record, "licence.id")
    if ledger.is_empty(raw):
        return {"unlicensed"}

    ident = str(raw).strip().upper().replace("_", "-").replace(" ", "-")
    found = set()

    if ident.startswith("CC0") or ident in ("UNLICENSE", "PUBLIC-DOMAIN", "PD"):
        found.add("public-domain")
    elif ident.startswith("CC-BY"):
        found.add("attribution")
        # Order matters for reading, not for logic: a licence can carry several.
        if "-SA" in ident:
            found.add("share-alike")
        if "-NC" in ident:
            found.add("noncommercial")
        if "-ND" in ident:
            found.add("noderivatives")
    if any(fnmatch.fnmatch(ident, pat) for pat in _PERMISSIVE):
        found.add("permissive")
    if any(fnmatch.fnmatch(ident, pat) for pat in _COPYLEFT):
        found.add("copyleft")
    if "EDITORIAL" in ident:
        found.add("editorial")

    return found or {"unknown"}


class Finding(object):
    """One rule firing against one record, or against the whole set."""

    def __init__(self, rule, level, summary, record_id=None):
        self.rule = rule
        self.level = level  # "fail" or "warn"
        self.summary = summary
        self.record_id = record_id

    @property
    def number(self):
        return self.rule.get("rule")

    @property
    def rule_id(self):
        return self.rule.get("id")

    @property
    def text(self):
        return " ".join((self.rule.get("text") or "").split())

    def __repr__(self):
        return "<Finding %s %s %s>" % (self.level, self.rule_id, self.record_id)


def _applies(rule, ship):
    when = rule.get("when", "always")
    if when == "always":
        return True
    if when == "ship":
        return ship
    if when == "never":
        return False
    raise ValueError("rule %r: unknown 'when' value %r" % (rule.get("id"), when))


def _require(rule, records):
    findings = []
    for name in rule.get("fields") or []:
        for record in records:
            if ledger.is_empty(ledger.field(record, name)):
                findings.append(
                    Finding(
                        rule,
                        "fail",
                        "%s is missing or empty" % name,
                        record_id=record["id"],
                    )
                )
    return findings


def _deny(rule, records):
    wanted = set(rule.get("families") or [])
    findings = []
    for record in records:
        hit = families(record) & wanted
        if hit:
            findings.append(
                Finding(
                    rule,
                    "fail",
                    "licence %s is %s"
                    % (
                        ledger.field(record, "licence.id") or "unrecorded",
                        ", ".join(sorted(hit)),
                    ),
                    record_id=record["id"],
                )
            )
    return findings


def _prefer(rule, records):
    wanted = set(rule.get("families") or [])
    findings = []
    for record in records:
        if not families(record) & wanted:
            findings.append(
                Finding(
                    rule,
                    "warn",
                    "licence %s is not %s"
                    % (
                        ledger.field(record, "licence.id") or "unrecorded",
                        " or ".join(sorted(wanted)),
                    ),
                    record_id=record["id"],
                )
            )
    return findings


def _assert(rule, records, facts):
    """A set-level rule: a condition over the whole ledger combined with a fact
    about the build. Rule 3 (CC-BY plus Steam's DRM wrapper) is this shape, and
    it is why build facts exist at all — there is no per-record field that could
    express it."""
    wanted = set(rule.get("if_any_family") or [])
    required = rule.get("and_fact") or {}

    for key, expected in required.items():
        if facts.get(key) != expected:
            return []

    matched = [r["id"] for r in records if families(r) & wanted]
    if not matched:
        return []

    condition = ", ".join("%s=%r" % (k, v) for k, v in sorted(required.items()))
    return [
        Finding(
            rule,
            "fail",
            "%s holds and %d record(s) are %s: %s"
            % (
                condition,
                len(matched),
                " or ".join(sorted(wanted)),
                ", ".join(sorted(matched)),
            ),
        )
    ]


_KINDS = {
    "require": lambda rule, records, facts: _require(rule, records),
    "deny": lambda rule, records, facts: _deny(rule, records),
    "prefer": lambda rule, records, facts: _prefer(rule, records),
    "assert": _assert,
}


def _waived_rules(record):
    waiver = record.get("waiver") or {}
    if not isinstance(waiver, dict):
        return set()
    return set(waiver.get("rules") or [])


def evaluate(records, rules, facts, ship=False):
    """Apply every rule in force and return findings, worst first.

    A waiver never removes a finding, it downgrades one. The record still
    appears in the audit, tagged `waived`, carrying the same rule text. An
    accepted risk that has stopped being visible has stopped being accepted.
    """
    by_id = dict((r["id"], r) for r in records)
    findings = []
    for rule in rules:
        if not _applies(rule, ship):
            continue
        kind = rule.get("kind")
        if kind not in _KINDS:
            raise ValueError("rule %r: unknown kind %r" % (rule.get("id"), kind))
        findings.extend(_KINDS[kind](rule, records, facts))

    for finding in findings:
        record = by_id.get(finding.record_id)
        if record is None:
            continue
        if finding.rule_id in _waived_rules(record) and finding.level == "fail":
            finding.level = "waived"

    order = {"fail": 0, "warn": 1, "waived": 2}
    findings.sort(
        key=lambda f: (order.get(f.level, 9), f.number or 0, f.record_id or "")
    )
    return findings


def failed(findings):
    return [f for f in findings if f.level == "fail"]
