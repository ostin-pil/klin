"""HuggingFace: an identifier is published, so klin maps it and stops there.

The Hub carries a `license:` tag, and for the common cases that tag is an SPDX
identifier `policy.families` already understands. Those map straight through:
the adapter writes the identifier onto the record and lets the policy module
classify it, rather than deciding anything itself.

`license:other` is the case that matters. It is not a licence, it is the
absence of one that the tag vocabulary can express, and it covers Flux.1-dev
and the Shakker ControlNet — the two largest things this project fetches. For
those the adapter records the identifier verbatim, leaves the families unset so
`policy.families` reports `unknown`, downloads the licence text where the
repository ships one, and tells the operator to classify it by hand. Guessing
here would put `noncommercial` on some records and miss it on others, and the
misses are invisible.
"""

import os

from .. import net
from . import (
    classify,
    finish,
    link_into_models,
    record_for,
    report_classification,
    target_path,
    write_sidecar,
)

NAME = "hf"
HELP = "a model repository on huggingface.co"

API = "https://huggingface.co/api/models/%s"
RESOLVE = "https://huggingface.co/%s/resolve/%s/%s"

#: Filenames a repository uses for its own licence text. Fetched only when the
#: identifier is one klin cannot classify, which is exactly when the terms
#: themselves are the thing that matters.
LICENCE_FILES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "LICENCE.md")


def configure(parser):
    parser.add_argument("repo_id", help="for example Comfy-Org/flux1-dev")
    parser.add_argument("--file", required=True, help="filename within the repository")
    parser.add_argument("--revision", default="main", help="branch, tag or commit")


def _licence_id(payload):
    """The `license:` tag, or the card's own field, or nothing."""
    for tag in payload.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1]
    card = payload.get("cardData") or {}
    value = card.get("license")
    if isinstance(value, list):
        return value[0] if value else None
    return value


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


def _licence_text(repo_id, revision, payload):
    names = {s.get("rfilename") for s in payload.get("siblings") or []}
    for candidate in LICENCE_FILES:
        if candidate not in names:
            continue
        url = RESOLVE % (repo_id, revision, candidate)
        try:
            from urllib.request import Request, urlopen

            request = Request(url, headers={"User-Agent": net.USER_AGENT})
            with urlopen(request, timeout=60) as response:
                return candidate, response.read().decode("utf-8", "replace")
        except Exception:
            return candidate, None
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

    ident = _licence_id(payload)
    record = record_for(NAME, repo_id.replace("/", "--"), filename)
    record["author"]["name"] = payload.get("author") or repo_id.split("/")[0]
    record["author"]["url"] = "https://huggingface.co/%s" % repo_id.split("/")[0]
    record["source"]["url"] = "https://huggingface.co/%s" % repo_id
    record["source"]["upstream_version"] = payload.get("sha")
    record["licence"]["id"] = ident
    record["licence"]["name"] = ident
    record["licence"]["url"] = "https://huggingface.co/%s/blob/%s/README.md" % (
        repo_id,
        args.revision,
    )

    how, found = classify(ctx, record)

    # The terms are worth having on disk exactly when the identifier failed to
    # say anything, which is also when rule 5 ("a storefront is not a licence")
    # has real work to do.
    if how == "unknown":
        name, text = _licence_text(repo_id, args.revision, payload)
        if text:
            record["licence"]["text"] = text
            record["licence"]["url"] = RESOLVE % (repo_id, args.revision, name)
            ctx.say("licence text: %s, %d characters, recorded" % (name, len(text)))

    report_classification(ctx, record, how, found)

    url = RESOLVE % (repo_id, args.revision, filename)
    dest = target_path(ctx, NAME, repo_id.replace("/", "--"), os.path.basename(filename))
    size = _declared_size(payload, filename)

    if args.dry_run:
        ctx.say("dry run: would fetch %s" % url)
        ctx.say("      to %s" % dest)
        ctx.say("  declared size: %s" % (size if size else "not published"))
        return 0

    write_sidecar(dest, payload)
    ctx.say("fetching %s" % url)
    facts = net.download(
        url,
        dest,
        token=token,
        expected_size=size,
        resume=args.resume,
        stream=ctx.stream,
    )
    record["source"]["mirror_of"] = facts["final_url"]

    linked = link_into_models(ctx, facts["path"], args.as_kind)
    return finish(ctx, record, facts, linked)
