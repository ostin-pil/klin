"""find: browsing itch.io's catalogue and previewing one page, offline.

Every response here is canned from the shapes the live site returned on
2026-08-28, because the adapter's docstring makes factual claims about that
surface and the tests should pin the parsing of exactly those shapes. No test
touches the network.
"""

import io

import pytest

from klin import cli, find, net
from klin.find import itch


FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>Top free game assets tagged tavern</title>
<item>
  <plainTitle>Low_Poly Medieval Tavern Interior</plainTitle>
  <title>Low_Poly Medieval Tavern Interior [Free]</title>
  <price>$0.00</price>
  <currency>USD</currency>
  <link>https://battleroach.itch.io/low-poly-m</link>
  <updateDate>2025-05-18 12:00:00</updateDate>
</item>
<item>
  <plainTitle>Fancy Paid Tavern</plainTitle>
  <title>Fancy Paid Tavern [$5.00]</title>
  <price>$5.00</price>
  <currency>USD</currency>
  <link>https://somebody.itch.io/fancy-tavern</link>
  <pubDate>Sat, 17 May 2025 14:46:00 GMT</pubDate>
</item>
</channel></rss>
"""

DATA_JSON = {
    "title": "Low_Poly Medieval Tavern Interior",
    "authors": [{"name": "BattleRoach", "url": "https://battleroach.itch.io"}],
    "price": "$0.00",
    "tags": ["3d", "low-poly", "no-ai"],
    "id": 3568294,
}

PAGE_WITH_LICENCE = """
<html><body>
<div class="buy_row"><a class="button buy_btn">Download Now</a>
<span class="buy_message"><span class="sub">Name your own price</span></span></div>
<div class="game_info_panel_widget"><table>
<tr><td>Status</td><td>Released</td></tr>
<tr><td>Category</td><td>Assets</td></tr>
<tr><td>Asset license</td><td><a href="/game-assets/assets-cc0">Creative Commons Zero v1.0 Universal</a></td></tr>
</table></div>
</body></html>
"""

PAGE_WITHOUT_LICENCE = """
<html><body>
<div class="game_info_panel_widget"><table>
<tr><td>Status</td><td>Released</td></tr>
<tr><td>Author</td><td>BattleRoach</td></tr>
</table></div>
<h4>License:</h4><p>License is CC0.</p>
</body></html>
"""


def run(argv):
    stream = io.StringIO()
    code = cli.main(["--repo", "."] + argv, stream=stream)
    return code, stream.getvalue()


# ------------------------------------------------------------------ browse


def test_browse_builds_the_canonical_feed_url():
    class A(object):
        licence = "cc0"
        free = False
        sort = None
        tag = "tavern"
        page = 1

    assert itch.browse_url(A()) == "https://itch.io/game-assets/assets-cc0/tag-tavern.xml"


def test_browse_pages_past_the_first_carry_the_page_param():
    class A(object):
        licence = None
        free = True
        sort = None
        tag = None
        page = 3

    assert itch.browse_url(A()).endswith("free.xml?page=3")


def test_three_filter_segments_are_refused_with_the_reason():
    """Three segments trip a Cloudflare challenge two do not, so the command
    line that would 403 intermittently is refused deterministically instead."""
    class A(object):
        licence = "cc0"
        free = True
        sort = None
        tag = "tavern"
        page = 1

    with pytest.raises(find.FindError) as exc:
        itch.browse_url(A())
    assert "two filter segments" in str(exc.value)


def test_no_filters_at_all_is_refused():
    class A(object):
        licence = None
        free = False
        sort = None
        tag = None
        page = 1

    with pytest.raises(find.FindError):
        itch.browse_url(A())


def test_browse_prints_price_title_and_link(monkeypatch):
    monkeypatch.setattr(net, "get_text", lambda url, **kw: FEED)
    code, out = run(["find", "itch", "--tag", "tavern", "--free"])
    assert code == 0
    assert "free" in out and "$5.00" in out
    assert "https://battleroach.itch.io/low-poly-m" in out
    assert "no feed carries a licence" in out


# ----------------------------------------------------------------- preview


def test_preview_reads_the_licence_row_when_the_creator_set_one(monkeypatch):
    monkeypatch.setattr(net, "get_text", lambda url, **kw: PAGE_WITH_LICENCE)
    monkeypatch.setattr(net, "get_json", lambda url, **kw: DATA_JSON)
    code, out = run(["find", "itch", "battleroach/low-poly-m"])
    assert code == 0
    assert "Creative Commons Zero" in out
    assert "Name your own price" in out
    assert "creator's claim, not a grant" in out


def test_preview_says_plainly_when_the_metadata_declares_nothing(monkeypatch):
    """A page with no licence row is not a page with no licence; the words
    usually live in the description as prose, and prose is quoted by a person,
    never paraphrased by a finder."""
    monkeypatch.setattr(net, "get_text", lambda url, **kw: PAGE_WITHOUT_LICENCE)
    monkeypatch.setattr(net, "get_json", lambda url, **kw: DATA_JSON)
    code, out = run(["find", "itch", "https://battleroach.itch.io/low-poly-m"])
    assert code == 0
    assert "declares no licence" in out


def test_a_target_that_is_neither_url_nor_slug_is_refused():
    with pytest.raises(find.FindError) as exc:
        itch.page_url("what even is this?")
    assert "robots.txt" in str(exc.value)


# ---------------------------------------------------------------- registry


def test_find_discovers_its_adapters_the_way_fetch_does():
    assert "itch" in find.adapters()
