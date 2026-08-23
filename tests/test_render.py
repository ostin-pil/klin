import pytest

from klin import render

from conftest import record


def test_pipes_in_a_value_are_escaped():
    # A pipe splits a table cell even inside backticks. Left unescaped it
    # silently mangles the row rather than erroring.
    assert render.escape("forge: github | forgejo | none") == (
        "forge: github \\| forgejo \\| none"
    )


def test_newlines_collapse_so_a_cell_stays_one_row():
    assert render.escape("two\nlines") == "two lines"


def test_table_has_one_row_per_record_plus_a_header():
    body = render.table([record("a"), record("b")])
    rows = body.splitlines()
    assert len(rows) == 4
    assert rows[0].startswith("| Asset |")


def test_every_rendered_row_has_the_same_cell_count():
    item = record("piped")
    item["licence"]["id"] = "Weird|Licence"
    item["notes"] = "a note with a | pipe"
    rows = render.table([item]).splitlines()
    counts = set()
    for row in rows:
        stripped = row.replace("\\|", "")
        counts.add(stripped.count("|"))
    assert len(counts) == 1


def test_details_render_notes_and_modifications(repo):
    item = record("kaykit")
    item["notes"] = "203 byte-identical copies of one atlas."
    item["modifications"] = ["Pruned fbx/ and obj/ on import."]
    body = render.details([item])
    assert "203 byte-identical copies" in body
    assert "- Pruned fbx/ and obj/ on import." in body


def test_details_skip_records_with_nothing_extra():
    assert render.details([record("plain")]) == ""


def test_splice_replaces_only_the_marked_block(repo):
    made = repo()
    text = made.licenses()
    out = render.splice(text, "records", "GENERATED")
    assert "GENERATED" in out
    assert "Hand-written policy lives here" in out
    assert "Also hand-written, also untouched" in out


def test_splice_is_idempotent(repo):
    made = repo()
    once = render.splice(made.licenses(), "records", "GENERATED")
    twice = render.splice(once, "records", "GENERATED")
    assert once == twice


def test_check_detects_a_hand_edit_inside_the_block(repo):
    made = repo()
    text = render.splice(made.licenses(), "records", "GENERATED")
    assert render.check(text, "records", "GENERATED")
    tampered = text.replace("GENERATED", "SOMEONE EDITED THIS")
    assert not render.check(tampered, "records", "GENERATED")


def test_missing_markers_explain_how_to_add_them():
    with pytest.raises(render.RenderError) as caught:
        render.splice("# doc\n\nno markers here\n", "records", "X")
    message = str(caught.value)
    assert "<!-- klin:begin records -->" in message
    assert "<!-- klin:end records -->" in message


def test_markers_in_the_wrong_order_are_rejected():
    text = "<!-- klin:end records -->\n<!-- klin:begin records -->\n"
    with pytest.raises(render.RenderError):
        render.splice(text, "records", "X")


def test_an_empty_ledger_renders_a_statement_not_a_blank(repo):
    assert "No assets recorded yet" in render.body([])
