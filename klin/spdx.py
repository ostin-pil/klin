"""The SPDX licence list: an identifier's canonical text.

klin refuses to derive a licence family from a string, and this does not
contradict that. An SPDX identifier is not a name that resembles a licence, it
is a key into a published register where each entry has exactly one text. So
looking `Apache-2.0` up and recording what SPDX says `Apache-2.0` is amounts to
a lookup, not an inference, and the failure mode of an inference is absent: an
identifier that is not in the register resolves to nothing at all.

The case that forced this. Barinn's rule 5 requires `licence.text` on anything
shipped, transcribed from a policy document whose reasoning is that a storefront
is not a licence and "Free" on itch.io is a price. Two Apache-2.0 models failed
it, not because their licensing was unclear but because Comfy-Org publishes no
`LICENSE` file in either repository and there was nothing for the field to hold.
The terms were never in doubt; the document was simply somewhere else.

Nothing here is cached to disk. The list is fetched once per process, which is
the right granularity for a tool that acquires a handful of models per run, and
a stale copy of a register that changes a few times a year is a worse problem
than one HTTP request.
"""

from . import net

LIST_URL = "https://spdx.org/licenses/licenses.json"
TEXT_URL = "https://spdx.org/licenses/%s.txt"

#: Fetched at most once per process. `None` until the first lookup, and a dict
#: afterwards, including an empty one when the register could not be reached.
_INDEX = None


def _load(get_json):
    """Lowercased identifier to canonical identifier, for the whole register."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    try:
        payload = get_json(LIST_URL)
    except Exception:
        # Offline is not an error here. The field stays empty, the project's
        # own rule reports it, and nothing has been invented.
        _INDEX = {}
        return _INDEX
    found = {}
    for entry in (payload or {}).get("licenses") or []:
        ident = entry.get("licenseId")
        if ident:
            found[str(ident).strip().lower()] = str(ident)
    _INDEX = found
    return _INDEX


def reset():
    """Forget the register. For tests, and for a long-lived process."""
    global _INDEX
    _INDEX = None


def canonical(ident, get_json=None):
    """The register's spelling of an identifier, or None if it is not in it.

    Vendors publish identifiers in whatever case suits them: the Hub's tag for
    Apache is `apache-2.0` and SPDX writes `Apache-2.0`. Matching case-blind and
    returning the register's own spelling is what makes the text URL below
    resolvable, and it is also the check for whether this is an SPDX identifier
    at all. `LicenseRef-Civitai-648058` is not, and gets nothing.
    """
    if not ident:
        return None
    index = _load(get_json or net.get_json)
    return index.get(str(ident).strip().lower())


def text(ident, get_json=None, get_text=None):
    """`(url, text)` for a recognised identifier, or `(None, None)`.

    An identifier outside the register returns nothing rather than a guess,
    which is the same line `policy.families` holds when it classifies an
    unrecognised identifier as `unknown`.
    """
    name = canonical(ident, get_json=get_json)
    if not name:
        return None, None
    url = TEXT_URL % name
    fetch_text = get_text or _default_text
    body = fetch_text(url)
    if not body or not body.strip():
        return None, None
    return url, body


def _default_text(url):
    from urllib.request import Request, urlopen

    try:
        request = Request(url, headers={"User-Agent": net.USER_AGENT})
        with urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8", "replace")
    except Exception:
        return None
