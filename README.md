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

v0.1, the core only. The ledger, the policy engine and the renderer are built
and tested. No vendor adapter exists yet. That order is deliberate: the shared
core is the reason this is one repo rather than several, so it exists before
anything depends on it.

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

Every request carries a browser User-Agent. Without one, Civitai's edge answers
`403 error code: 1010`, which reads like a rejected credential and is not.

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
  cli.py                 fetch | gen | conform | ledger | secret
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

## Licence

MIT. See `LICENSE`.
