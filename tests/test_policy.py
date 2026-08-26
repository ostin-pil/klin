import pytest

from klin import manifest, policy

from conftest import record


def rules_for(made):
    data = manifest.load(made.manifest)
    return data, manifest.rules(data), manifest.build_facts(data)


@pytest.mark.parametrize(
    "licence_id,expected",
    [
        ("CC0-1.0", {"public-domain"}),
        ("CC-BY-4.0", {"attribution"}),
        ("CC-BY-SA-4.0", {"attribution", "share-alike"}),
        ("CC-BY-NC-4.0", {"attribution", "noncommercial"}),
        ("CC-BY-NC-SA-4.0", {"attribution", "noncommercial", "share-alike"}),
        ("CC-BY-ND-4.0", {"attribution", "noderivatives"}),
        ("GPL-3.0", {"copyleft"}),
        ("MIT", {"permissive"}),
        ("Sketchfab-Editorial", {"editorial"}),
        ("Whatever-Custom", {"unknown"}),
    ],
)
def test_families_classification(licence_id, expected):
    assert policy.families(record("x", licence_id)) == expected


def test_missing_licence_is_unlicensed_not_unknown():
    assert policy.families(record("x", licence_id=None)) == {"unlicensed"}


def test_explicit_families_override_the_identifier():
    # The escape hatch for what no identifier can express: unlicensed fan art of
    # a trademarked property looks like CC0 until a human says otherwise.
    item = record("x", "CC0-1.0")
    item["licence"]["families"] = ["fan-art"]
    assert policy.families(item) == {"fan-art"}


def test_cc0_record_passes_the_ship_gate(repo):
    made = repo()
    made.write_records([record("clean", "CC0-1.0")])
    data, rules, facts = rules_for(made)
    findings = policy.evaluate(
        [record("clean", "CC0-1.0")], rules, facts, ship=True
    )
    assert policy.failed(findings) == []


def test_share_alike_fails_rule_two_and_quotes_it(repo):
    made = repo()
    data, rules, facts = rules_for(made)
    findings = policy.evaluate(
        [record("tavern-tools", "CC-BY-SA-4.0")], rules, facts, ship=True
    )
    fails = policy.failed(findings)
    assert [f.rule_id for f in fails] == ["no-share-alike"]
    assert fails[0].number == 2
    # The rule prints its own text, which is what makes drift from the prose
    # document visible in the output.
    assert "Don't take it." in fails[0].text


def test_noncommercial_fails_rule_four(repo):
    made = repo()
    data, rules, facts = rules_for(made)
    findings = policy.evaluate(
        [record("noncommercial-output", "CC-BY-NC-4.0")], rules, facts, ship=True
    )
    assert "excluded-licences" in [f.rule_id for f in policy.failed(findings)]


def test_ship_only_rules_do_not_fire_at_the_stage_gate(repo):
    made = repo()
    data, rules, facts = rules_for(made)
    items = [record("tavern-tools", "CC-BY-SA-4.0")]
    assert policy.failed(policy.evaluate(items, rules, facts, ship=False)) == []
    assert policy.failed(policy.evaluate(items, rules, facts, ship=True))


def test_the_stage_rule_fires_at_both_gates(repo):
    made = repo()
    data, rules, facts = rules_for(made)
    item = record("mystery", "CC0-1.0")
    item["source"]["url"] = None
    for ship in (False, True):
        fails = policy.failed(policy.evaluate([item], rules, facts, ship=ship))
        assert [f.rule_id for f in fails] == ["recorded"]


def test_missing_licence_text_fails_rule_five(repo):
    made = repo()
    data, rules, facts = rules_for(made)
    item = record("no-terms", "CC0-1.0")
    item["licence"]["text"] = "   "
    fails = policy.failed(policy.evaluate([item], rules, facts, ship=True))
    assert "storefront-is-not-a-licence" in [f.rule_id for f in fails]


def test_cc_by_warns_but_does_not_fail(repo):
    made = repo()
    data, rules, facts = rules_for(made)
    findings = policy.evaluate([record("attributed", "CC-BY-4.0")], rules, facts, ship=True)
    assert policy.failed(findings) == []
    assert "prefer-cc0" in [f.rule_id for f in findings if f.level == "warn"]


def test_set_level_rule_stays_quiet_when_the_fact_is_false(repo):
    made = repo(steam_drm=False)
    data, rules, facts = rules_for(made)
    findings = policy.evaluate([record("attributed", "CC-BY-4.0")], rules, facts, ship=True)
    assert "ccby-vs-steam-drm" not in [f.rule_id for f in policy.failed(findings)]


def test_set_level_rule_fires_on_the_whole_set(repo):
    made = repo(steam_drm=True)
    data, rules, facts = rules_for(made)
    items = [record("attributed", "CC-BY-4.0"), record("clean", "CC0-1.0")]
    fails = [f for f in policy.failed(policy.evaluate(items, rules, facts, ship=True))
             if f.rule_id == "ccby-vs-steam-drm"]
    assert len(fails) == 1
    # A set-level finding belongs to no single record, and names the ones that
    # triggered it.
    assert fails[0].record_id is None
    assert "attributed" in fails[0].summary
    assert "clean" not in fails[0].summary


def test_a_waiver_downgrades_but_never_hides(repo):
    made = repo()
    data, rules, facts = rules_for(made)
    item = record("tavern-tools", "CC-BY-SA-4.0")
    item["waiver"] = {"rules": ["no-share-alike"], "why": "test", "by": "K."}
    findings = policy.evaluate([item], rules, facts, ship=True)
    assert policy.failed(findings) == []
    waived = [f for f in findings if f.level == "waived"]
    assert [f.rule_id for f in waived] == ["no-share-alike"]
    # Still carries the rule text, so the accepted risk stays legible.
    assert "Don't take it." in waived[0].text


def test_a_waiver_for_one_rule_does_not_waive_another(repo):
    made = repo()
    data, rules, facts = rules_for(made)
    item = record("bad", "CC-BY-NC-SA-4.0")
    item["waiver"] = {"rules": ["no-share-alike"]}
    fails = policy.failed(policy.evaluate([item], rules, facts, ship=True))
    assert [f.rule_id for f in fails] == ["excluded-licences"]


# --------------------------- an empty family list is a result, not an absence


def test_a_derived_empty_list_is_a_classification_not_a_gap():
    """The conflation that made seven of Barinn's LoRAs read as unknown.

    A Civitai adapter reads permission flags rather than an identifier, and
    flags carrying no restriction derive to `[]`. Testing that list for
    truthiness could not tell it from a missing key, so it fell through to a
    `LicenseRef-` id and classified as `unknown`.
    """
    item = record("x", "LicenseRef-Civitai-648058")
    item["licence"]["families"] = []
    assert policy.families(item) == set()


def test_an_empty_list_without_an_identifier_is_still_unlicensed():
    """Nothing was recorded, so there was nothing to have checked."""
    item = record("x", licence_id=None)
    item["licence"]["families"] = []
    assert policy.families(item) == {"unlicensed"}


# ------------------------------------------------- the coverage finding


def test_the_ship_gate_stops_on_a_licence_it_could_not_classify(repo):
    made = repo()
    data, rules, facts = rules_for(made)
    records = [record("mystery", "LicenseRef-Nobody-Knows")]

    findings = policy.evaluate(records, rules, facts, ship=True)
    coverage = [f for f in findings if f.rule_id == "unclassified"]
    assert len(coverage) == 1
    assert coverage[0].level == "fail"
    assert coverage[0].record_id == "mystery"
    assert coverage[0].number is None


def test_the_stage_gate_does_not(repo):
    """A prototype takes anything; the obligation is only to write it down."""
    made = repo()
    data, rules, facts = rules_for(made)
    records = [record("mystery", "LicenseRef-Nobody-Knows")]
    findings = policy.evaluate(records, rules, facts, ship=False)
    assert [f for f in findings if f.rule_id == "unclassified"] == []


def test_classifying_it_by_hand_settles_it(repo):
    made = repo()
    data, rules, facts = rules_for(made)
    item = record("mystery", "LicenseRef-Nobody-Knows")
    item["licence"]["families"] = ["permissive"]
    findings = policy.evaluate([item], rules, facts, ship=True)
    assert [f for f in findings if f.rule_id == "unclassified"] == []


def test_a_waiver_downgrades_the_coverage_finding_like_any_other(repo):
    made = repo()
    data, rules, facts = rules_for(made)
    item = record("mystery", "LicenseRef-Nobody-Knows")
    item["waiver"] = {"rules": ["unclassified"], "why": "ours, sorting it out"}
    findings = policy.evaluate([item], rules, facts, ship=True)
    coverage = [f for f in findings if f.rule_id == "unclassified"]
    assert coverage[0].level == "waived"
    assert policy.failed(findings) == []


def test_a_record_with_no_licence_at_all_is_caught_too(repo):
    made = repo()
    data, rules, facts = rules_for(made)
    findings = policy.evaluate(
        [record("bare", licence_id=None)], rules, facts, ship=True
    )
    coverage = [f for f in findings if f.rule_id == "unclassified"]
    assert coverage and "unlicensed" in coverage[0].summary


def test_a_clean_ledger_gains_no_coverage_finding(repo):
    made = repo()
    data, rules, facts = rules_for(made)
    findings = policy.evaluate(
        [record("fine", "CC0-1.0")], rules, facts, ship=True
    )
    assert [f for f in findings if f.rule_id == "unclassified"] == []
