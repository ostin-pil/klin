# klin

Asset acquisition, generation and provenance for game projects.

The core idea is small. Every asset a project uses carries a record of where it
came from, which model produced it if any, and what licence it arrived under.
That record is enough to answer the only question that really matters later:
can this ship?

klin does not hold opinions about licences. The consuming project writes its own
policy in its own words, transcribes the enforceable half into a manifest, and
klin applies it and quotes it back.

## Status

v0.1. The ledger, the policy engine and the renderer came first, because the
shared core is the reason this is one repo rather than several, so it exists
before anything depends on it. On top of it sit two vendor adapters under
`fetch`, and the index, which scans what a machine already holds and traces
each file back to the models that made it. No `gen` adapter yet.

## Install

```
uv tool install klin
```

Or add the plugin marketplace to Claude Code and install from it:

```
/plugin marketplace add ostin-pil/klin
/plugin install klin@ostin-pil-klin
```

## Use

Every project holds one `.claude/klin-manifest.md`. It names the paths, declares
facts about the shipped build, and carries the rule table.

```
klin ledger add --id kaykit-dungeon --kind pack
klin ledger audit              # the stage gate: is it recorded at all?
klin ledger audit --ship       # the full policy
klin ledger render             # write the records into the policy document
klin ledger render --check     # fail if that document has gone stale
```

`audit --ship` exits non-zero on any failure and prints the verbatim text of
every rule it applied.

```
klin audit: ship gate | 3 record(s) | policy transcribed from game/assets/LICENSES.md

FAIL  rule 2 (no-share-alike)  opengameart-tavern-tools
      licence CC-BY-SA-4.0 is share-alike
      Never take CC-BY-SA. It is share-alike, and Creative Commons does not
      resolve whether embedding an asset makes the whole game an adaptation.
      [...] It will look like a perfect find. Don't take it.
```

## Fetch

```
klin fetch hf Comfy-Org/flux1-dev --file flux1-dev-fp8.safetensors --as checkpoints
klin fetch civitai 980106 --as loras
klin fetch civitai 1041229 --version 2499170 --as loras
```

Each invocation resolves the vendor's metadata, classifies the licence, streams
the file while computing its sha256, verifies it, writes a ledger record, and
tells you to run `klin ledger audit`. Add `--dry-run` to stop after
classification, which is the cheap way to see what a licence resolves to before
committing to a download.

**klin never guesses a licence.** HuggingFace publishes an identifier, so the
adapter records it and lets `policy.py` classify it. `license:other` is not an
identifier, so it maps to nothing, `policy.py` reports `unknown`, and klin
prints an instruction to settle it by hand:

```
klin fetch hf Comfy-Org/flux1-dev --file flux1-dev-fp8.safetensors     --families noncommercial
```

An explicit `--families` list wins outright, which is the escape hatch
`policy.py` already documents. An adapter that quietly decided `noncommercial`
for `license:other` would be right about the repositories somebody checked and
wrong invisibly about every later one.

**Reading is not guessing, though.** `license: other` is the Hub saying "not on
our list", which is a statement about its own vocabulary rather than about the
licence, and a card that sets it almost always sets `license_name` and
`license_link` beside it. klin reads all three, fetches the terms from the link,
and prints the name and the link with its request for a decision, so the
question is asked with the answer in view:

```
license_name: flux-1-dev-non-commercial-license
license_link: https://huggingface.co/black-forest-labs/FLUX.1-dev/.../LICENSE.md
licence text: 18491 characters from ...
licence: other -> unknown (unknown)
```

The link matters more than it looks. It points at whichever repository owns the
terms, which for both FLUX derivatives is `black-forest-labs/FLUX.1-dev` rather
than the one being downloaded, so the fetched repository's own `LICENSE` file
would have been the wrong document or no document at all.

The classification still stays `unknown` until a human says otherwise. A licence
named non-commercial is very probably noncommercial, and deriving a family from
a string that happens to contain a word is still the invention this refuses.

One link shape is different in kind. A `bespoke-lora-trained-license` links to
`multimodal.art/civitai-licenses?allowCommercialUse=Image&...`, whose query
string is Civitai's permission flags verbatim. That is structured data rather
than a name, so it resolves through the same table below and needs no human.
A flag link carrying no flags is treated as malformed rather than as a grant of
nothing, because reading an empty query as `noncommercial` would be a guess, and
a confident one.

Civitai publishes no identifier at all, only permission flags, so the mapping is
a judgment and it is recorded as one. The derived families and the vendor's raw
flags both go into the record, and the whole API response is kept in a
`meta.json` sidecar beside the file, so a later re-classification never needs a
second download:

| Vendor field | Consequence |
| --- | --- |
| `allowCommercialUse` without `Image` or `Sell` | `noncommercial` |
| `allowCommercialUse` empty, or `[None]` | `noncommercial` |
| `allowDerivatives: false` | add `noderivatives` |
| `allowNoCredit: false` | add `attribution` |

The contested row is the first. `Rent` and `RentCivit` grant a generation
service permission to run the model, which does not answer whether an image made
with it may be sold, so a model offering only those is treated as
noncommercial. Records carry a `LicenseRef-` id, SPDX's own convention for terms
that are not on its list.

### Where files land

`cache_dir` in the manifest, overridden by `KLIN_CACHE`, which is where a
machine-specific path belongs rather than in a committed file. The layout is
`<cache>/<vendor>/<id>/<filename>`, with the `meta.json` sidecar alongside.

`--as <subdirectory>` additionally hardlinks the file into `models_dir`
(or `KLIN_MODELS`) so a downstream tool sees it without a second copy. A
hardlink rather than a copy because a checkpoint is seventeen gigabytes, and
rather than a search-path config file because a model tree may already be
reached through a junction, in which case adding it again registers one
directory under two names.

### What the guards catch

A vendor that refuses a request does not always say so with a status code. Four
guards run before any record is written, and a failure deletes the partial file:

1. The content type is on an allowlist, never merely off a denylist. Civitai's
   edge refuses an unrecognised client with seventeen bytes of `text/plain`, so
   a rule written against `text/html` alone waves it through to be saved as a
   model.
2. The byte count matches the vendor's declared size, where one is published.
3. A `.safetensors` file starts with a header length and a JSON header that
   parses.
4. sha256 is computed while streaming, and checked against the vendor's own
   hash where one is published.

Downloads land on a `.part` file and are renamed only once every guard passes,
so an interrupted fetch is never mistaken for a finished one. `--resume`
continues one over HTTP Range.

Every request carries a User-Agent naming klin. Without one, Civitai's edge
answers `403 error code: 1010`, which reads like a rejected credential and is
not. The block is on `Python-urllib` specifically rather than on non-browser
clients, so klin says what it is instead of claiming to be Chrome.

## Index

`fetch` answers where a model came from. The index answers the other half: what
is already on this machine, and which of those models made it.

```
klin index build          # scan the roots, read what each file says about itself
klin index status         # what the index holds
klin ls --lora psx --since 2026-08-20
klin ls --check           # exit non-zero if anything listed fails the ship gate
klin show psx-final-B-iso-101
```

A generated image is usually self-describing. ComfyUI writes its whole graph
into a PNG text chunk, so the base model, the LoRA stack with its strengths, the
seed, the sampler settings and both prompts are in the file. Recovering that is
a scan rather than an archaeology project, and it needs no dependency: chunk
framing is a length, a type, a payload and a CRC.

A reader walks the graph from the sampler backwards along its `model` input.
Collecting every node whose class name contains `Lora` is the obvious shortcut
and it is wrong twice over, because it loses the order the LoRAs were applied
in and it counts loaders sitting in the graph wired to nothing.

### The index is derived, the ledger is truth

A ledger record is a committed statement about an asset, written once and
reviewed by a person. The index is a cache of what a scan found, and deleting it
costs nothing but a rebuild. That is why the database lives beside the cache and
never in a repository: the rule that no weights and no raw generation output
enter a game repo covers a table describing them just as much.

It follows that **the licence verdict is computed at query time and never
stored**, for the same reason `ship_ok` is absent from a ledger record. The
index describes files that do not change, while the policy around them changes
constantly, so a cached verdict would be wrong the first time somebody
classified a licence by hand.

### Machine-scoped, project-tagged

One output directory serves every project on a machine. Splitting the index per
project would scan the same files twice and answer "what else made this" with
silence, so there is one database, and each row carries whichever project
claimed its path.

```yaml
index:
  roots:
    - "D:/ComfyUI/output"
  claim:
    - "mock/*"
```

Roots are scanned; claim patterns say which of the results belong to this
project. A file no project claims is indexed and counted, never dropped, because
an unclaimed file is the normal state of a corpus that predates the index. Two
projects claiming one path is reported as a conflict with the first claim left
standing, since letting the last scan win would make the owner depend on running
order.

### Models resolve by identity, not by name

Each model an image used is traced back to a ledger record through the weights
tree, and the match is on filesystem identity: `st_dev` and `st_ino`. `fetch
--as` hardlinks a cached file into that tree rather than copying it, so the
tree's entry and the path in the ledger are one file under two names.

Names cannot do this job. In the tree this was built against, fourteen of
fourteen recorded models resolved by identity and **not one** had the same
filename in both places, because the weights tree is exactly where files get
renamed into something readable. Two of them were one file linked under two
names, which a filename match would have reported as two different models. So a
name match is still offered, and it is labelled as one wherever it is used.

Where no record can be found, the item is marked `?` and reported as untraceable
rather than passing:

```
SHIP  FILE                              SIZE       SEED   MODEL
NO    psx-final-B-iso-101_00001_.png    640x368    101    flux1-dev-fp8 + ps1_style_flux_v1@0.7
?     cd_Pv_Np_face_00001_.png          1024x1280  8043   z_image_base_bf16

1 marked ? : a model klin cannot trace to a ledger record. That is not a pass.
```

Silence there would invert the meaning of the gate. A file whose origin is
unknown is the case a ship gate exists to catch.

## Three decisions worth knowing about

**The policy document stays authoritative.** A project's licence policy is
prose, written to persuade a reader, full of reasoning and named traps. That
does not survive being turned into a schema, so it is not turned into one. The
manifest carries a transcription of the enforceable half, each rule citing the
number it came from and the text klin prints when it fires. A rule that has
drifted from its source shows up in audit output rather than hiding.

**`ship_ok` is computed, never stored.** A verdict written into a record goes
stale the moment policy is amended, and a stale pass is worse than no pass at
all. Records persist only `reviewed_at` and an explicit waiver.

**A waiver downgrades a finding, it never removes one.** A waived record still
appears in the audit, tagged, carrying the same rule text. An accepted risk that
has stopped being visible has stopped being accepted.

## Set-level rules

Some rules are assertions over the whole set combined with a fact about the
build, and no per-record field can express them. The case that forced this:
CC BY 4.0 forbids applying technological measures that restrict the licensed
rights, so shipping any CC-BY asset alongside Steam's optional DRM wrapper is a
licence breach, while either one alone is fine.

```yaml
build_facts:
  steam_drm: false

rules:
  - rule: 3
    id: ccby-vs-steam-drm
    kind: assert
    when: ship
    if_any_family: [attribution]
    and_fact: {steam_drm: true}
    text: Do not enable Steam's DRM wrapper if any CC-BY asset is in the build.
```

Rule kinds are `require`, `deny`, `prefer` and `assert`. Licence families are
derived from the identifier, and an explicit `licence.families` list on a record
overrides that derivation, which is the escape hatch for anything an identifier
cannot express.

## Secrets

Adapters need credentials. The manifest names them and never holds them.

```yaml
secrets:
  civitai:
    env: CIVITAI_API_TOKEN
    description: read access for the Civitai fetch adapter
  huggingface:
    env: HF_TOKEN
```

```
klin secret set civitai        # prompts, or reads stdin when piped
klin secret list               # names and where each resolves from
klin secret doctor             # what is holding them, and what is missing
klin secret rm civitai
```

Values go to the operating system's credential store: Credential Manager on
Windows, Keychain on macOS, Secret Service on Linux. A lookup checks
`KLIN_SECRET_<NAME>` first, then whichever conventional variable the manifest
names, then the store. That order is why CI supplies credentials through the
environment and never reaches for a vault, and why an SSH session still works
on Windows, where Credential Manager is unreachable.

The store protects a credential from another user on the machine, from a disk
read outside the logon session, and from reaching a commit. It does not protect
it from code running as your own account, because on Windows nothing local
does. Keep tokens read-only and rotate them.

Treat the store as a cache as well. It does not survive a profile rebuild, so
the durable copy belongs in a password database, and the store is seeded from
that database by hand:

```
keepassxc-cli show -q -s -a Password ~/vault.kdbx klin/civitai | klin secret set civitai
```

One database per person rather than per project, with a `klin/` group for the
credentials klin reads. Seeding stays manual on purpose: automating it would
mean klin holding the database credential, which trades away the property the
database is there for. `research/2026-08-24_secrets-on-windows.md` has the
reasoning and the alternatives that were weighed.

## Layout

```
klin/                    the Python package
  ledger.py              JSONL records, one line per asset
  policy.py              licence families and rule evaluation
  render.py              the marked block inside a prose document
  manifest.py            the per-project manifest
  secrets.py             credential lookup, references only in the manifest
  net.py                 streaming downloads and the four guards
  fetch/                 vendor adapters, discovered rather than listed
    hf.py                huggingface.co
    civitai.py           civitai.com
  index/                 what is on this machine, and what made it
    png.py               chunk framing, stdlib only
    comfy.py             the graph a ComfyUI output carries about itself
  cli.py                 fetch | gen | conform | ledger | secret | index
plugin/                  the Claude Code plugin: skill and command
tests/
.claude-plugin/          the marketplace, so a second plugin is a directory
```

Roles are the CLI's verbs and vendors are adapters underneath them. Adding
Civitai, HuggingFace or ComfyUI support is a new adapter module, never a new
plugin and never a new repo. `fetch` enforces that by discovery: any module in
`klin/fetch/` declaring `NAME` and `configure` becomes a subcommand, so vendor
three is one new file and no edit to `cli.py`. There is a test that writes a
module into the package and checks it appears.

`index/` works the same way. A module there declaring `NAME` and `read` becomes
a provenance reader, so teaching klin to recognise another generator's output is
also one new file.

## Licence

MIT. See `LICENSE`.
