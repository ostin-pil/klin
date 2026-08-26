---
name: klin-ledger
description: Record an asset's provenance and licence in the klin ledger, and run the licence gate. Use when adding a third-party or generated asset to a project, when asked whether an asset can ship, or before anything goes public.
---

# klin-ledger

klin records where every asset came from and what licence it carries, then
enforces the consuming project's own licence policy. The policy is not klin's;
it lives in the project's policy document and is transcribed into the project's
`.claude/klin-manifest.md`, rule by rule.

## Before anything else, read two files

1. `.claude/klin-manifest.md`, for the `rules` block, the `build_facts`, and the
   `policy_doc` and `ledger` paths.
2. The document `policy_doc` points at. **It is authoritative.** The manifest's
   rules are a transcription of its enforceable half. A rule in the manifest
   that contradicts the prose is a bug in the manifest.

Never invent a licence rule. If the policy document does not settle a case, say
so and ask, rather than guessing a verdict.

## Fetching from a vendor

When the asset comes from HuggingFace or Civitai, do not hand-write the record.
`klin fetch` resolves the metadata, classifies the licence, verifies the
download and writes the record itself:

```
klin fetch hf <repo-id> --file <filename> --as <subdirectory>
klin fetch civitai <model-id> [--version <id>] --as <subdirectory>
```

Add `--dry-run` first when the licence is what you are checking. It resolves and
classifies without downloading, which on a seventeen-gigabyte checkpoint is the
difference between a second and an hour.

**When klin says a licence is `unknown`, it means it.** That is not a failure to
look it up; it is klin refusing to guess. It will already have printed
`license_name`, `license_link` and the licence text where the vendor published
them, so read those, decide, and pass the decision back:

```
klin fetch hf <repo-id> --file <filename> --families noncommercial
```

A name like `flux-1-dev-non-commercial-license` is the vendor's own answer and
you should act on it. klin declines to read a family out of a substring, which
is a rule about what a tool may infer, not a suggestion that the question is
open.

Do not skip this and let the record stand unclassified. `unknown` fails the
**ship** gate, under klin's own `unclassified` finding, because no family rule a
project writes can match a licence klin could not classify, and a silent pass
would have meant "the rules never reached this one". The stage rule is the
weaker test and asks only that the thing is written down, so a prototype still
moves. An asset classified by guess, meanwhile, would pass a gate that ought to
stop it. If the terms do not settle the question, say so and ask rather than
choosing a family that makes the audit quiet.

An empty family list is a different thing from `unknown`, and it is a real
answer. It means the vendor's own permission flags were read and none of them
restricts anything klin tracks, which is the normal state of a permissively
licensed Civitai model. Leave it alone.

A Civitai model with several versions makes klin refuse and list them, because
picking one silently would fetch a Flux LoRA when the workflow wanted the
Z-Image variant. Choose with `--version` or `--base-model`.

## Adding an asset

```
klin ledger add --id <slug> --kind <mesh|texture|pack|lora|weights|audio>
```

That writes a blank record with every field present. Fill in:

- `source`, meaning adapter, url, retrieval date and upstream version.
  `source.url` is required at *every* gate: an unrecorded asset fails even in
  prototype stage.
- `licence`, meaning the identifier, the **verbatim licence text**, and whether
  attribution is required. "Free" on a storefront is a price, not a licence.
- `produced_by`, for generated assets only, naming the model and **the model's
  own licence**. This is the field that cannot be reconstructed later.
- `notes` and `modifications`, free text rendered verbatim into the policy
  document. Put the detail here that a schema would flatten.

Never set `ship_ok`. It does not exist in the schema; a ship verdict is computed
at audit time so a policy amendment cannot leave a stale verdict behind.

## The two gates

```
klin ledger audit           # stage gate: is it recorded at all?
klin ledger audit --ship    # ship gate: the full policy
```

The stage gate is the prototype rule: use what looks right, but write it down.
The ship gate runs before anything reaches a person outside the project: a
demo build, a store page, a playtest handed to strangers. It exits non-zero on
any failure and prints the verbatim text of every rule it applied.

**Report what the audit printed.** Do not paraphrase a rule and do not soften a
failure. The output quotes the policy on purpose, so that a rule which has
drifted from its source is visible rather than hidden.

## Keeping the policy document current

```
klin ledger render          # write the records into the marked block
klin ledger render --check  # fail if the block is stale
```

Only the block between `<!-- klin:begin records -->` and its `end` marker is
generated. **Everything outside those markers is hand-written and must stay
that way**, including the reasoning, the named traps and the stage rule.
Never edit inside the markers; edit the ledger and re-render.

Run `render --check` alongside the audit. A ledger that has moved on without
the document is the failure mode the markers exist to catch.

## What klin does not do

klin's job ends when an asset is recorded, validated and conformed. It does not
place props, size rooms or write scene files. If a task is about where an asset
goes in a level rather than where it came from, this is the wrong tool.
