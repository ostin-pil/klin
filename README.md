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
  cli.py                 fetch | gen | conform | ledger | secret
plugin/                  the Claude Code plugin: skill and command
tests/
.claude-plugin/          the marketplace, so a second plugin is a directory
```

Roles are the CLI's verbs and vendors are adapters underneath them. Adding
Civitai, HuggingFace or ComfyUI support is a new adapter module, never a new
plugin and never a new repo.

## Licence

MIT. See `LICENSE`.
