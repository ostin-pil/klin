"""HuggingFace: an identifier is published, so klin maps it and stops there.

The Hub carries a `license:` tag, and for the common cases that tag is an SPDX
identifier `policy.families` already understands. Those map straight through:
the adapter writes the identifier onto the record and lets the policy module
classify it, rather than deciding anything itself.

`license:other` is the case that matters, and it is not the end of the story.
A card setting `license: other` almost always sets `license_name` and
`license_link` beside it, and those carry the answer the tag withheld.
Flux.1-dev declares `flux-1-dev-non-commercial-license` and links the terms; the
Shakker ControlNet declares the same. Reading the tag and stopping meant klin
asked for a hand classification while the vendor's own answer sat two fields
below, which is worse than not asking at all, because it teaches the reader that
the question is unanswerable.

So all three fields are read now. What is still refused is the guess: a licence
*named* non-commercial is not thereby classified, because deriving a family from
a string that happens to contain a word is exactly the invention this module
exists to prevent. The name and the link go into the record, the terms are
fetched where the link is a document, and the operator is asked with the answer
in front of them rather than in the abstract.

One link shape is different in kind. A `bespoke-lora-trained-license` on the Hub
links to `multimodal.art/civitai-licenses?...`, whose query string is Civitai's
own permission flags, verbatim and machine-readable. That is not a name being
pattern-matched, it is the same structured data `civitai.py` already maps, so it
resolves through the same table and needs no human at all.
"""

import os
import re

from urllib.parse import parse_qsl, urlsplit
from urllib.request import Request, urlopen

from .. import net, spdx
from . import (
    adopt,
    classify,
    find_local,
    finish,
    link_into_models,
    record_for,
    report_classification,
    target_path,
    write_sidecar,
    write_sidecar_beside,
)
from .civitai import derive_families

NAME = "hf"
HELP = "a model repository on huggingface.co"

API = "https://huggingface.co/api/models/%s"
RESOLVE = "https://huggingface.co/%s/resolve/%s/%s"

#: Filenames a repository uses for its own licence text, used when the licence
#: link is not itself a document.
LICENCE_FILES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "LICENCE.md")

#: A licence link whose query string is Civitai's permission flags rather than
#: prose. Structured data, so it maps rather than needing a human.
CIVITAI_LICENCE_LINK = re.compile(r"civitai-licenses", re.I)

#: `license: other` is the Hub saying "not on our list", which is a statement
#: about the vocabulary and not about the licence.
UNINFORMATIVE = ("other", "unknown", "", None)


def configure(parser):
    parser.add_argument("repo_id", help="for example Comfy-Org/flux1-dev")
    parser.add_argument("--file", required=True, help="filename within the repository")
    parser.add_argument("--revision", default="main", help="branch, tag or commit")


def licence_fields(payload):
    """The identifier, and the name and link that `license: other` hides behind.

    Returns `(id, name, link)`. A card carrying a real SPDX identifier sets
    neither of the latter two and a card carrying `other` almost always sets
    both, so this is one lookup rather than two code paths.
    """
    ident = None
    for tag in payload.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            ident = tag.split(":", 1)[1]
            break

    card = payload.get("cardData") or {}
    if ident is None:
        value = card.get("license")
        if isinstance(value, list):
            value = value[0] if value else None
        ident = value

    return ident, card.get("license_name"), card.get("license_link")


def _as_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def flags_from_link(link):
    """Civitai's permission flags, when a licence link encodes them.

    The Hub's `bespoke-lora-trained-license` links to a page whose query string
    carries `allowCommercialUse`, `allowDerivatives` and `allowNoCredit`. Those
    are the fields the Civitai API publishes, so the same mapping applies and no
    judgment is made here that is not already made and documented in
    `civitai.py`.

    Returns a payload shaped like a Civitai model, or None when the link is
    ordinary prose.
    """
    if not link or not CIVITAI_LICENCE_LINK.search(link):
        return None
    pairs = parse_qsl(urlsplit(link).query, keep_blank_values=True)
    if not pairs:
        return None

    commercial = []
    payload = {}
    for key, value in pairs:
        if key == "allowCommercialUse":
            commercial.extend(part for part in value.split(",") if part)
        elif key in ("allowDerivatives", "allowNoCredit", "allowDifferentLicense"):
            payload[key] = _as_bool(value)
    payload["allowCommercialUse"] = commercial
    return payload


def _declared_sha256(payload, filename):
    """The digest the Hub publishes for an LFS file.

    Every weight worth fetching is stored in LFS, and the pointer carries the
    file's sha256. It is the strongest identity check available here and it
    costs nothing extra, because `?blobs=true` already returns it beside the
    size the size guard was reading anyway.

    The Hub spells it `sha256`. `oid` is the spelling in Git LFS's own pointer
    format and in some of the Hub's other responses, so both are read: a guard
    that finds neither reports nothing rather than a false pass, but silently
    checking nothing because of a key name is the failure this file already
    warns about for `files_metadata=true`.
    """
    for sibling in payload.get("siblings") or []:
        if sibling.get("rfilename") == filename:
            lfs = sibling.get("lfs") or {}
            got = lfs.get("sha256") or lfs.get("oid")
            return str(got) if got and len(str(got)) == 64 else None
    return None


def _declared_size(payload, filename):
    """The size the Hub publishes, which only `?blobs=true` fills in.

    `files_metadata=true` looks like the parameter for this and returns
    siblings with every size set to null, so a guard built on it silently
    degrades to checking nothing. Worth stating, because the failure mode is a
    guard that passes rather than one that errors.
    """
    for sibling in payload.get("siblings") or []:
        if sibling.get("rfilename") == filename:
            return sibling.get("size") or (sibling.get("lfs") or {}).get("size")
    return None


def _fetch_text(url):
    try:
        request = Request(url, headers={"User-Agent": net.USER_AGENT})
        with urlopen(request, timeout=60) as response:
            if "text/html" in (response.headers.get("Content-Type") or ""):
                # A licence *page* is not licence text. Recording markup would
                # satisfy rule 5 with something nobody can read.
                return None
            return response.read().decode("utf-8", "replace")
    except Exception:
        return None


def licence_text(repo_id, revision, payload, link):
    """The terms themselves, from the link if it is a document, else the repo.

    `license_link` usually points at a LICENSE file in whichever repository
    actually owns the terms, which for both FLUX derivatives is
    `black-forest-labs/FLUX.1-dev` rather than the repository being fetched. The
    fetched repository's own LICENSE is the fallback, not the first choice.
    """
    if link:
        # `blob` is the HTML view of a file; `resolve` is the file.
        raw = link.replace("/blob/", "/resolve/")
        text = _fetch_text(raw)
        if text:
            return raw, text

    names = {s.get("rfilename") for s in payload.get("siblings") or []}
    for candidate in LICENCE_FILES:
        if candidate in names:
            url = RESOLVE % (repo_id, revision, candidate)
            return url, _fetch_text(url)
    return None, None


def run(args, ctx):
    repo_id = args.repo_id
    filename = args.file
    token = ctx.token("huggingface")

    payload = net.get_json((API % repo_id) + "?blobs=true", token=token)

    gated = payload.get("gated")
    if gated not in (False, None) and not token:
        ctx.say(
            "note: %s is gated (%r) and no huggingface credential resolved. The "
            "download will fail with a 401 unless the repository is public to "
            "you; store one with `klin secret set huggingface`." % (repo_id, gated)
        )

    ident, name, link = licence_fields(payload)
    uninformative = ident in UNINFORMATIVE

    record = record_for(NAME, repo_id.replace("/", "--"), filename)
    record["author"]["name"] = payload.get("author") or repo_id.split("/")[0]
    record["author"]["url"] = "https://huggingface.co/%s" % repo_id.split("/")[0]
    record["source"]["url"] = "https://huggingface.co/%s" % repo_id
    record["source"]["upstream_version"] = payload.get("sha")
    record["licence"]["id"] = ident
    record["licence"]["name"] = name or ident
    record["licence"]["url"] = link or (
        "https://huggingface.co/%s/blob/%s/README.md" % (repo_id, args.revision)
    )

    derived = None
    flags = flags_from_link(link) if uninformative else None
    if flags is not None:
        derived, why = derive_families(flags)
        record["notes"] = (
            "families resolved from the permission flags in license_link: %s"
            % "; ".join(why)
            if why
            else "the permission flags in license_link grant commercial use"
        )
        ctx.say("license_link carries Civitai permission flags: %s" % flags)

    how, found = classify(ctx, record, derived=derived)

    if uninformative and derived is None:
        # The tag said nothing, so say what the card said instead. Asking for a
        # hand classification without this is asking a question whose answer is
        # already on the page.
        if name:
            ctx.say("license_name: %s" % name)
        if link:
            ctx.say("license_link: %s" % link)

    # The terms are recorded whatever the classification turned out to be.
    # Gating this on `unknown` tied the presence of a document to the outcome
    # of a lookup, which are unrelated: passing `--families` to settle an
    # OpenRAIL model by hand made klin stop recording the very text that
    # justified the decision, and a rule requiring `licence.text` then failed
    # the record for a file it had been holding one run earlier.
    where, text = licence_text(repo_id, args.revision, payload, link)
    if text:
        record["licence"]["text"] = text
        record["licence"]["url"] = where
        ctx.say("licence text: %d characters from %s" % (len(text), where))

    if not record["licence"]["text"]:
        # A recognised identifier has exactly one text in SPDX's register, so
        # recording what the register says it is amounts to a lookup. Reached
        # only when the repository itself publishes no licence document, which
        # is the ordinary case for a model card carrying a clean SPDX tag:
        # neither Comfy-Org repository Barinn adopted ships a LICENSE file, and
        # the terms were never in doubt, only somewhere else.
        where, text = spdx.text(ident)
        if text:
            record["licence"]["text"] = text
            ctx.say("licence text: %d characters from %s" % (len(text), where))

    report_classification(ctx, record, how, found)

    url = RESOLVE % (repo_id, args.revision, filename)
    dest = target_path(ctx, NAME, repo_id.replace("/", "--"), os.path.basename(filename))
    size = _declared_size(payload, filename)

    if args.dry_run:
        ctx.say("dry run: would fetch %s" % url)
        ctx.say("      to %s" % dest)
        ctx.say("  declared size: %s" % (size if size else "not published"))
        return 0

    published = _declared_sha256(payload, filename)
    here = args.adopt
    if not here and not args.force:
        here = find_local(ctx, size, published)
        if here:
            ctx.say("already on this machine, so nothing is downloaded:")
            ctx.say("  %s" % here)

    if here:
        facts = adopt(
            ctx,
            here,
            expected_size=size,
            expected_sha256=published,
        )
        write_sidecar_beside(facts["path"], payload)
        record["source"]["mirror_of"] = url
        record["notes"] = " ".join(
            filter(
                None,
                [
                    record.get("notes"),
                    "adopted from disk: the file predates klin and was verified "
                    "against the vendor's published size and hash rather than "
                    "re-downloaded.",
                ],
            )
        )
        return finish(ctx, record, facts)

    write_sidecar(dest, payload)
    ctx.say("fetching %s" % url)
    facts = net.download(
        url,
        dest,
        token=token,
        expected_size=size,
        resume=args.resume,
        stream=ctx.stream,
        force=args.force,
    )
    record["source"]["mirror_of"] = facts["final_url"]

    linked = link_into_models(ctx, facts["path"], args.as_kind)
    return finish(ctx, record, facts, linked)
