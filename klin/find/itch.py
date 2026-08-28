"""itch.io: browse the game-asset catalogue, and read one project page's facts.

What itch.io actually exposes, verified against the live site on 2026-08-28
(each fact shaped this adapter, so they are recorded here rather than in a
lost research note):

- **The official server-side API is about the key owner's own account** —
  uploads, purchases, download keys. No search, no browse, no reading an
  arbitrary public page. Their own docs point at the RSS feeds for the
  public catalogue, which is what this adapter uses.
- **Browse pages serve RSS by appending `.xml`**, with `plainTitle`, `price`,
  `currency`, `link`, `description`, `pubDate`/`updateDate` and platform
  flags per item. `?page=N` paginates; past the end the channel is empty.
  What the feed does not carry: the licence, the tags, the author as a
  field (the author is the subdomain of `link`).
- **Filters are path segments in a canonical order** — licence
  (`assets-cc0`), then `free` or a sort like `newest`, then `tag-<t>` —
  and **three filter segments trip a Cloudflare challenge** where two do
  not, URL-pattern-specific and independent of rate. So this adapter caps a
  browse at two segments and refuses the third with the reason, rather than
  producing a command line that intermittently 403s.
- **`/search` is disallowed by robots.txt** (browse and project pages are
  not), so there is deliberately no search flag here: narrowing past two
  filters is the reader's job, not a third request's.
- **A project page is fully server-rendered.** The info table (Status,
  Category, Author, Tags, and `Asset license` when the creator filled that
  field in) is in the initial HTML, and `<url>/data.json` returns clean
  json: title, authors, price, tags, id. "Name your own price" appears only
  in the page's buy row; feeds and data.json show it as `$0.00`.
- **The `Asset license` row exists only when the creator set it.** A page
  with no row is not a page with no licence — the words usually live in the
  description as prose. This adapter prints the row verbatim when present
  and otherwise says plainly that the metadata declares nothing, because a
  storefront is not a licence and paraphrasing prose into one would be the
  invention the fetch adapters' first rule forbids.
"""

import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

from .. import net
from . import FindError

NAME = "itch"
HELP = "itch.io game-asset listings: browse the catalogue, preview one page"

BROWSE_ROOT = "https://itch.io/game-assets"

#: Licence filters itch exposes as path segments. Only the ones useful to a
#: CC0-first pipeline are mapped; the vendor has more.
LICENCES = {"cc0": "assets-cc0", "cc-by": "assets-cc-by"}

SORTS = ("newest", "top-sellers", "top-rated")


def configure(parser):
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="a project page to preview (https://author.itch.io/slug or author/slug); "
        "omit it to browse",
    )
    parser.add_argument("--tag", default=None, help="a browse tag, e.g. tavern")
    parser.add_argument(
        "--free", action="store_true", help="only free listings (a price, not a licence)"
    )
    parser.add_argument(
        "--licence",
        choices=sorted(LICENCES),
        default=None,
        help="only listings whose creator set this asset licence",
    )
    parser.add_argument("--sort", choices=SORTS, default=None)
    parser.add_argument("--page", type=int, default=1)


def browse_url(args):
    """The feed URL for a browse, capped at two filter segments.

    The cap is not taste: three segments trip a Cloudflare challenge that two
    do not, so a three-segment command line would work never rather than
    sometimes. The canonical segment order (licence, then free or a sort,
    then tag) is itch's own; other orders redirect.
    """
    segments = []
    if args.licence:
        segments.append(LICENCES[args.licence])
    if args.free:
        segments.append("free")
    if args.sort:
        segments.append(args.sort)
    if args.tag:
        segments.append("tag-%s" % args.tag)
    if not segments:
        raise FindError(
            "browsing all of itch.io is not an answer anybody wants; "
            "give at least one of --tag, --free, --licence, --sort"
        )
    if len(segments) > 2:
        raise FindError(
            "itch.io answers two filter segments and challenges three "
            "(%s). Drop one and narrow by eye instead."
            % ", ".join(segments)
        )
    url = "%s/%s.xml" % (BROWSE_ROOT, "/".join(segments))
    if args.page and args.page > 1:
        url += "?page=%d" % args.page
    return url


def parse_feed(text):
    """The rows a browse feed carries, price included."""
    root = ET.fromstring(text)
    rows = []
    for item in root.iter("item"):
        def field(name):
            node = item.find(name)
            return (node.text or "").strip() if node is not None else ""

        rows.append({
            "title": field("plainTitle") or field("title"),
            "price": field("price"),
            "link": field("link"),
            "updated": field("updateDate") or field("pubDate"),
        })
    return rows


class _InfoTable(HTMLParser):
    """The rows of the server-rendered info table, and the buy row's message.

    The table sits in a div classed `game_info_panel_widget` and each row is
    two cells: a label and a value. "Name your own price" lives in a span
    classed `sub` inside the buy row, and nowhere in any feed.
    """

    def __init__(self):
        HTMLParser.__init__(self)
        self.in_panel = 0
        self.in_cell = False
        self.cells = []
        self.rows = []
        self.in_sub = False
        self.buy_message = ""

    def handle_starttag(self, tag, attrs):
        classes = dict(attrs).get("class", "")
        if tag == "div" and "game_info_panel_widget" in classes:
            self.in_panel = 1
        elif self.in_panel and tag == "div":
            self.in_panel += 1
        if self.in_panel and tag == "td":
            self.in_cell = True
            self.cells.append("")
        if tag == "span" and "sub" in classes.split():
            self.in_sub = True

    def handle_endtag(self, tag):
        if self.in_panel and tag == "div":
            self.in_panel -= 1
        if tag == "td":
            self.in_cell = False
        if tag == "tr" and len(self.cells) >= 2:
            self.rows.append((self.cells[0].strip(), " ".join(
                c.strip() for c in self.cells[1:] if c.strip())))
            self.cells = []
        if tag == "span":
            self.in_sub = False

    def handle_data(self, data):
        if self.in_cell and self.cells:
            self.cells[-1] += data
        if self.in_sub and data.strip():
            self.buy_message = data.strip()


def page_url(target):
    if re.match(r"^https://[a-z0-9_-]+\.itch\.io/[a-z0-9_-]+/?$", target, re.I):
        return target.rstrip("/")
    m = re.match(r"^([a-z0-9_-]+)/([a-z0-9_-]+)$", target, re.I)
    if m:
        return "https://%s.itch.io/%s" % (m.group(1), m.group(2))
    raise FindError(
        "%r is neither a project url nor author/slug. /search is off the "
        "table by the site's own robots.txt; browse with --tag instead." % target
    )


def run(args, stream):
    def say(text=""):
        stream.write(text + "\n")

    if args.target:
        url = page_url(args.target)
        data = net.get_json(url + "/data.json")
        html = net.get_text(url)
        table = _InfoTable()
        table.feed(html)

        say("%s" % (data.get("title") or url))
        for author in data.get("authors") or []:
            say("by       %s  %s" % (author.get("name", "?"), author.get("url", "")))
        price = data.get("price") or ""
        if table.buy_message:
            price = "%s (%s)" % (price, table.buy_message)
        say("price    %s" % price)
        if data.get("tags"):
            say("tags     %s" % ", ".join(data["tags"]))
        licence_row = None
        for label, value in table.rows:
            say("%-8s %s" % (label.lower()[:8], value))
            if "license" in label.lower():
                licence_row = value
        say("")
        if licence_row:
            say("the page's metadata names a licence: %s" % licence_row)
            say("read it with your own eyes before recording it; a storefront")
            say("is not a licence and the row is the creator's claim, not a grant.")
        else:
            say("the page's metadata declares no licence. If the description")
            say("names one, that sentence is what a ledger record quotes -")
            say("verbatim, typo included. %s" % url)
        return 0

    url = browse_url(args)
    rows = parse_feed(net.get_text(url))
    if not rows:
        say("nothing on this page of the catalogue. Fewer filters, or an earlier --page.")
        return 0
    for row in rows:
        price = row["price"] or "?"
        if price in ("$0.00", "0.00"):
            price = "free"
        say("%-10s %-48s %s" % (price, row["title"][:48], row["link"]))
    say("")
    say("%d listing(s). Prices are prices; no feed carries a licence -" % len(rows))
    say("preview one with `klin find itch <link>` before believing anything.")
    return 0
