"""Civitai: no identifier is published, so the mapping is the judgment.

Civitai does not carry an SPDX identifier. It carries permission flags, and
translating those into klin's families is the piece of reasoning this adapter
exists to encode. The translation is arguable, so it is written down here, put
into the record's notes, and kept auditable by recording the raw flags beside
the derived families. A reader who disagrees can see exactly what was mapped
from what, and the sidecar holds the whole response.

The mapping:

===============================================  =========================
Vendor field                                     Consequence
===============================================  =========================
`allowCommercialUse` without `Image` or `Sell`   `noncommercial`
`allowCommercialUse` empty, or `[None]`          `noncommercial`
`allowDerivatives: false`                        add `noderivatives`
`allowNoCredit: false`                           add `attribution`
===============================================  =========================

The contested half is the first row. `Rent` and `RentCivit` grant a generation
service permission to run the model; they do not answer the question the
consuming project actually asks, which is whether an image made with it can be
sold. So a model offering only those is treated as noncommercial. That call is
what makes several rental-only LoRAs fail a ship gate automatically, instead of
failing whenever somebody remembers to check.

Because the licence has no identifier, the record carries a `LicenseRef-` id,
which is SPDX's own convention for terms that are not on its list. The families
are set explicitly from the flags, and `policy.families` documents an explicit
list as winning outright — so the `LicenseRef` is never classified, only
recorded.
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

NAME = "civitai"
HELP = "a model on civitai.com (needs the `civitai` secret)"

MODEL_API = "https://civitai.com/api/v1/models/%s"
DOWNLOAD = "https://civitai.com/api/download/models/%s"

#: Flags that answer "may an image made with this be sold". `Rent` and
#: `RentCivit` are deliberately absent: they permit a service to generate, which
#: is a different question.
COMMERCIAL = ("Image", "Sell")


def configure(parser):
    parser.add_argument("model_id", help="the numeric model id from the URL")
    parser.add_argument("--version", default=None, help="a specific modelVersion id")
    parser.add_argument(
        "--base-model",
        default=None,
        help="pick the version for this base model, e.g. 'Flux.1 D' or 'ZImageTurbo'",
    )
    parser.add_argument("--file", default=None, help="filename within the version")


def derive_families(payload):
    """Apply the table in this module's docstring. Returns (families, why)."""
    allowed = payload.get("allowCommercialUse") or []
    if not isinstance(allowed, list):
        allowed = [allowed]
    allowed = [a for a in allowed if a]

    families = set()
    why = []

    if not set(COMMERCIAL) & set(allowed):
        families.add("noncommercial")
        why.append(
            "allowCommercialUse is %s, which grants no permission to sell an "
            "image made with the model" % (allowed or "empty")
        )

    if payload.get("allowDerivatives") is False:
        families.add("noderivatives")
        why.append("allowDerivatives is false")

    if payload.get("allowNoCredit") is False:
        families.add("attribution")
        why.append("allowNoCredit is false, so credit is required")

    return families, why


def _pick_version(payload, args):
    versions = payload.get("modelVersions") or []
    if not versions:
        raise net.NetError("model %s publishes no versions" % args.model_id)

    if args.version:
        for version in versions:
            if str(version.get("id")) == str(args.version):
                return version
        raise net.NetError(
            "model %s has no version %s. Available: %s"
            % (
                args.model_id,
                args.version,
                ", ".join(
                    "%s (%s, %s)" % (v.get("id"), v.get("name"), v.get("baseModel"))
                    for v in versions
                ),
            )
        )

    if args.base_model:
        wanted = args.base_model.strip().lower()
        matches = [
            v for v in versions if (v.get("baseModel") or "").strip().lower() == wanted
        ]
        if not matches:
            raise net.NetError(
                "model %s has no version for base model %r. Available: %s"
                % (
                    args.model_id,
                    args.base_model,
                    ", ".join(sorted({str(v.get("baseModel")) for v in versions})),
                )
            )
        return matches[0]

    if len(versions) > 1:
        # Silence here would pick one of five variants and never say so.
        raise net.NetError(
            "model %s has %d versions; choose one with --version or --base-model.\n  %s"
            % (
                args.model_id,
                len(versions),
                "\n  ".join(
                    "--version %-9s %-14s %s"
                    % (v.get("id"), v.get("baseModel"), v.get("name"))
                    for v in versions
                ),
            )
        )
    return versions[0]


def _pick_file(version, args):
    files = version.get("files") or []
    if not files:
        raise net.NetError("version %s publishes no files" % version.get("id"))
    if args.file:
        for item in files:
            if item.get("name") == args.file:
                return item
        raise net.NetError(
            "version %s has no file %r. Available: %s"
            % (version.get("id"), args.file, ", ".join(f.get("name") for f in files))
        )
    primary = [f for f in files if f.get("primary")]
    if primary:
        return primary[0]
    weights = [f for f in files if str(f.get("name", "")).endswith(".safetensors")]
    return (weights or files)[0]


def run(args, ctx):
    token = ctx.token("civitai")
    if not token:
        ctx.say(
            "note: no `civitai` credential resolved. Downloads return 401; the "
            "metadata below still works. Store one with `klin secret set civitai`."
        )

    payload = net.get_json(MODEL_API % args.model_id, token=token)
    version = _pick_version(payload, args)
    item = _pick_file(version, args)

    families, why = derive_families(payload)

    record = record_for(NAME, version.get("id"), item.get("name"))
    record["author"]["name"] = (payload.get("creator") or {}).get("username")
    record["source"]["url"] = "https://civitai.com/models/%s" % args.model_id
    record["source"]["upstream_version"] = "%s (%s)" % (
        version.get("name"),
        version.get("baseModel"),
    )
    record["licence"]["id"] = "LicenseRef-Civitai-%s" % args.model_id
    record["licence"]["name"] = "%s — Civitai model terms" % payload.get("name")
    record["licence"]["url"] = record["source"]["url"]
    record["licence"]["text"] = _terms_text(payload)
    record["licence"]["attribution_required"] = payload.get("allowNoCredit") is False
    record["notes"] = "; ".join(why) if why else "vendor flags grant commercial use"

    how, found = classify(ctx, record, derived=families)
    ctx.say(
        "civitai flags: allowCommercialUse=%s allowDerivatives=%s allowNoCredit=%s"
        % (
            payload.get("allowCommercialUse"),
            payload.get("allowDerivatives"),
            payload.get("allowNoCredit"),
        )
    )
    report_classification(ctx, record, how, found)

    url = DOWNLOAD % version.get("id")
    dest = target_path(ctx, NAME, version.get("id"), item.get("name"))
    size = item.get("sizeKB")
    size = int(round(size * 1024)) if size else None

    if args.dry_run:
        ctx.say("dry run: would fetch %s" % url)
        ctx.say("      to %s" % dest)
        ctx.say("  declared size: %s" % (size if size else "not published"))
        return 0

    write_sidecar(dest, {"model": payload, "version": version, "file": item})
    ctx.say("fetching version %s, %s" % (version.get("id"), item.get("name")))
    facts = net.download(
        url,
        dest,
        token=token,
        expected_size=size,
        resume=args.resume,
        stream=ctx.stream,
        force=args.force,
    )

    # Civitai publishes a hash, so the computed digest can be checked against
    # the vendor's rather than merely recorded. A mismatch here means the bytes
    # differ from what the vendor believes it served.
    published = ((item.get("hashes") or {}).get("SHA256") or "").lower()
    if published and published != facts["sha256"]:
        os.remove(facts["path"])
        raise net.NetError(
            "sha256 mismatch: vendor publishes %s, download hashes to %s. The "
            "file has been deleted." % (published, facts["sha256"])
        )
    if published:
        ctx.say("sha256 matches the hash Civitai publishes")

    record["source"]["mirror_of"] = facts["final_url"]
    linked = link_into_models(ctx, facts["path"], args.as_kind)
    return finish(ctx, record, facts, linked)


def _terms_text(payload):
    """The permission flags as prose, since there is no licence document."""
    return (
        "Civitai model terms, recorded from the API rather than a licence "
        "document, because Civitai publishes permission flags and no identifier. "
        "allowCommercialUse=%s; allowDerivatives=%s; allowNoCredit=%s; "
        "allowDifferentLicense=%s."
        % (
            payload.get("allowCommercialUse"),
            payload.get("allowDerivatives"),
            payload.get("allowNoCredit"),
            payload.get("allowDifferentLicense"),
        )
    )
