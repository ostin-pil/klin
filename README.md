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
`fetch`, the index, which scans what a machine already holds and traces each
file back to the models that made it, `gen`, which checks a workflow's licence
posture before the GPU starts, and `conform`, which turns a mesh somebody else
authored into one a project can ship.

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

Each invocation resolves the vendor's metadata, classifies the licence, looks
for the bytes on this machine before transferring any, streams the file while
computing its sha256 where it has to, verifies it, writes a ledger record, and
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

### Adoption, which is what happens by default

Every model directory predates the tool that would have recorded it. Barinn's
held sixty-nine gigabytes across five base models, none of them fetched through
klin, so every image they produced was untraceable: the index could name the
model behind a plate and then find no record to look up. Re-downloading all of
it to learn what was already on the disk is not an answer, and hand-writing the
records means inventing the licences.

So a fetch looks before it transfers. Every one searches the weights tree and
the cache for the bytes it is about to download, and adopts them where it finds
them:

```
klin fetch hf Comfy-Org/z_image_turbo --file split_files/.../z_image_turbo_bf16.safetensors

already on this machine, so nothing is downloaded:
  D:\comfy-models\diffusion_models\z_image_turbo_bf16.safetensors
sha256 matches the vendor's published hash
recorded hf-Comfy-Org--z_image_turbo
```

That took thirty-three seconds instead of twelve gigabytes. The search is size
first and hash second, because the vendor publishes a size and a stat over the
tree reduces the candidates to a handful; hashing a whole tree to find one file
would cost more than the download it saves.

**A published hash is required for this to happen unattended.** A size match is
not provenance: `sd_xl_base_1.0.safetensors` and
`sd_xl_base_1.0_0.9vae.safetensors` are identical in length and are different
models, so adopting on size would have recorded one as the other in silence.
Where the vendor publishes no hash, klin downloads. `--force` downloads anyway.

`--adopt PATH` names a file explicitly, which is the attended path and is
allowed to proceed on size and header alone, because a person chose that file:

```
klin fetch hf Comfy-Org/flux1-schnell --file flux1-schnell-fp8.safetensors \
    --adopt D:/comfy-models/checkpoints/flux1-schnell-fp8.safetensors
```

Either way it is a fetch minus the transfer. The guards that make a downloaded
file trustworthy are properties of the bytes and not of how they arrived: the
size matches what the vendor publishes, the safetensors header parses, and the
digest matches the vendor's own hash. Run against a local file they prove the
same thing.

So a mismatch is refused outright rather than noted, because a record is an
assertion about provenance and writing one for a file that failed the vendor's
own hash asserts what klin has just disproved:

```
klin: z_image_base_bf16.safetensors is 12309874112 bytes and the vendor
publishes 12309866400. This is not that file, so klin will not record it as one.
```

Three of Barinn's five adopted with their hashes matching the Hub's. Two refused
on exact byte counts, 7,712 and 40 bytes out, because no published file matches
them. Both were locally converted, so there is no upstream to verify them
against, and that is the answer rather than an obstacle.

### Where files land

`cache_dir` in the manifest, overridden by `KLIN_CACHE`, which is where a
machine-specific path belongs rather than in a committed file. The layout is
`<cache>/<vendor>/<id>/<filename>`, with the `meta.json` sidecar alongside.

That override has a failure mode worth knowing about, because it is silent by
construction. A variable set when the files were fetched and unset later
resolves somewhere else entirely, and nothing appears to break: the records
still point at real files, the audit still passes, and the next fetch downloads
seventeen gigabytes into a second tree. Barinn spent two sessions in that state,
every weight under `D:/klin-cache` and a manifest naming `%LOCALAPPDATA%`.

So `klin ledger audit` checks whether any recorded file lives under the cache in
force, which is a different question from whether the recorded files exist. That
second question stays answered "yes" throughout the whole failure.

```
note: cache_dir resolves to C:\Users\pilyu\AppData\Local\klin\cache,
      and none of the 14 recorded file(s) are under it. They are in:
        D:\klin-cache\civitai\1406637
      Set KLIN_CACHE to the tree that is actually in use, or correct
      cache_dir.
```

It is a note and never a failure. A project may legitimately keep assets outside
the cache, and klin cannot tell that apart from a variable that has gone
missing. What it can do is stop the discrepancy being invisible, which is the
only reason such a state persists.

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

## Gen

```
klin gen comfy --workflow flux_schnell_style_t2i.json     --prompt-file scene.txt --seed 101 --size 640x368     --lora ps1_style_flux_v1.safetensors@0.7 --check
```

Driving a sampler is not what klin is for, and a shell script does it in eighty
lines. What a script cannot do is answer the question that matters before the
work happens rather than months after it.

**The licence posture of a graph is knowable before it runs.** A workflow names
its checkpoint and its LoRA stack, the index resolves those names to ledger
records by filesystem identity, and the policy engine applies the project's
rules to records. So the whole chain already exists:

```
workflow design/mockups/workflows/flux_dev_style_t2i.json
  checkpoint flux1-dev-fp8.safetensors    LicenseRef-FLUX-1-dev-Non-Commercial [noncommercial]

output could ship: NO
  excluded-licences: licence LicenseRef-FLUX-1-dev-Non-Commercial is noncommercial
```

`--check` refuses to queue on that, and exits non-zero. Without it klin says so
and generates anyway, because a prototype plate from a non-commercial model is a
legitimate thing to want. This inverts how it normally goes: the alternative is
what this project did, which was generate 1,384 plates over several weeks and
then discover that 66 came from FLUX.1-dev. The models were the same either way.
Only the moment of finding out changed.

Outputs go into the index as they land, so a generation cannot accumulate into
an unattributed pile.

**Raw output does not enter the ledger.** A record per generated image would put
thousands of rows describing throwaway plates into a file meant to describe
shipped assets. A record appears when an image becomes an asset, which is a
different verb.

### Roles are located by traversal, not by node title

The driver this was ported from found the positive prompt by looking for a
`CLIPTextEncode` titled `POSITIVE`, which works until somebody renames a node.
The index already locates every role by walking the graph from the sampler,
because it had to read graphs nobody wrote for it, and the same walk works for
writing. So a template needs no special titles, and the reader and the writer
cannot disagree about which node is which.

Two things that walk found, both in real templates and neither in a synthetic
one:

A `ControlNetApplyAdvanced` takes a positive *and* a negative conditioning, so a
walk that always tried `positive` first followed the wrong edge for the negative
and put both prompts in one node. The walk now follows the branch it came down.

Flux and Z-Image templates derive the negative from the positive through a
`ConditioningZeroOut`, so the graph has one encoder and no negative prompt to
write. `--negative` there overwrote the prompt, leaving a plate whose record
said one thing and whose pixels came from another. It is refused now, by name.

An unused LoRA slot is zeroed rather than unwired, because rewiring a graph by
hand is where a sweep quietly starts producing something other than what its
record says it produced.

### TRELLIS.2, and a licence that turns on one import

```
klin gen trellis --patch check
klin gen trellis --image stove.png --out meshes/
```

TRELLIS.2 lifts an image into a mesh. Its own code and its weights are MIT, and
the non-commercial part is two optional NVIDIA dependencies reached through a
single module-scope import, which loads them on every run whether or not
anything wants what they do. A project that discards generated textures never
wants the texture bake they serve, and was loading them anyway.

So this adapter also patches the install: the import moves into the bake block
that uses it, and a `bake_texture=False` exit returns geometry, normals and uvs
above it. Generation and the patch live together because splitting them would
leave a generator whose licence status depended on a step somebody remembered
to run.

The proof that a run touched none of it is a removal rather than a reading,
since a transitive import three modules deep does not appear in a diff. The
harness takes the restricted modules out of the import system for the whole
process and generates anyway, so a mesh coming out the other side is the
evidence. It refuses the imports rather than stubbing them, because a stub
would let the pipeline believe it had a rasterizer and leave open what it did
with it.

Two conditions travel with the result. The clean path is the non-default
argument, so this makes a clean path available rather than making the tool
clean by default. And the install is not a git repository, so a reinstall
reverts the patch silently, which is why `--patch check` exists and why
applying it twice is safe.

## Conform

A mesh from anywhere becomes a project's own by pointing its uvs at the
project's texture atlas.

```
klin conform blender counter.blend --from battleroach-tavern
```

That is a narrower operation than it sounds, and the narrowness is the useful
part. A project whose atlas is a grid of flat colour swatches does not author
a texture per asset; it points each material at one texel. So conforming is
setting every loop of every face whose slot is named `wood_mid` to the single
coordinate that names the wood_mid cell, then collapsing the slots into one
shared material and triangulating. A mesh from any source comes out
style-native and nobody painted anything.

Pinning to a coordinate rather than to a colour is what lets a project re-grade
every asset at once by editing one image. An asset carrying its own baked
colour would stay behind on the old palette, so the export is made to reference
the shared atlas rather than to embed a copy of it, and that is checked rather
than hoped for.

klin holds no opinion about which swatches exist or how many triangles is too
many. The table is the project's file and the budget is the project's number,
both named in its manifest:

```yaml
staging_dir: assets/_staging
conform:
  swatches: assets/swatches.json
  max_triangles: 900
  blender: "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe"
  gltf_validator: null
```

### What is checked, and where it lands

Blender writes to a temporary file. klin runs every gate against that file and
moves it into `staging_dir` only once they all pass, so a file that failed one
is never something a later step can pick up by mistake. Four gates: every
material slot names a swatch the table knows, every uv in the exported file is
one of the texels it was meant to be, the triangle count is inside the
project's budget, and the Khronos validator has nothing to report.

The uv gate is a measurement rather than an argument. glTF puts the uv origin
at the top left and Blender at the bottom left, so a pin that looks right in
the scene can be upside down in the file, and both look equally clean from
inside Blender. The conformer therefore re-imports what it just exported and
reports the coordinates that actually shipped. They are compared as a set,
because collapsing the slots leaves a re-imported mesh with nothing to say
which face used to be which slot, and because the set is the stronger claim
anyway: it catches a uv that never moved, one that moved to the wrong cell,
and a whole layout written inverted.

The validator is the one soft gate. It is a separate binary that many machines
will not have, so its absence is loud, reaches the record as a sentence, and
becomes a failure under `--strict`.

Geometry checks live in the verb and not in the rule engine. Every rule kind in
`policy.py` is a pure function of records, which is why an audit runs on a
checkout holding no assets at all. A triangle budget would have to open a file
or keep a second copy of a number that already exists in one, and a second copy
of a measured number is its own kind of bug.

### The record it writes

A conformed asset is a derivative, so its licence is its source's, copied whole
from the record named by `--from` rather than invented. `source.derived_from`
holds that record's id. With no `--from` the licence stays empty and klin says
so, because a conform that quietly wrote one would be inventing provenance.

A re-conform keeps what a person wrote. `notes`, `used_for`, `reviewed_at` and
any hand-written modifications survive; only the fields a machine owns are
replaced.

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

## The one rule klin brings itself

Every rule above is the consuming project's, transcribed from its own policy
document, because klin holds no opinions about licences. There is one
exception, and it holds no opinion either.

A licence klin cannot classify falls into the `unknown` family, and no family
rule can match it. A project's table denies share-alike, or noncommercial, or
whatever it has decided about; nothing in it denies "we could not tell". So an
unclassifiable record passed the ship gate silently, and a pass that meant "the
rules never reached this one" printed identically to a pass that meant "the
rules applied and were satisfied". That is a statement about the audit rather
than about any licence, which is why it belongs here.

```
FAIL  klin (unclassified)  barinn-authored-props
      licence LicenseRef-Barinn-Own classified as unknown
      klin could not classify this licence, so no family rule above can have
      applied to it. That is a gap in the audit rather than a verdict on the
      licence. Settle it by setting licence.families on the record [...]
```

It carries no rule number, because it was not transcribed from anybody's policy
document and citing a number it never had would misattribute it. It fires only
at the ship gate, since a prototype takes anything and the stage rule's whole
content is that the thing gets written down. The escape hatch is the one already
documented: set `licence.families` by hand. A waiver works too, and downgrades
rather than removes, like any other.

**An empty family list is a result, not an absence.** A vendor publishing
permission flags instead of an identifier is read by its adapter, and flags
carrying no restriction klin tracks derive to `[]`. Testing that list for
truthiness could not tell "checked, nothing applies" from "nobody said", so it
fell through to the identifier and a `LicenseRef-` id classified as `unknown`.
Seven of Barinn's Civitai LoRAs sat in that state with `allowCommercialUse:
[Image, Sell]` in their sidecars. Reporting a resolved licence as unresolved is
the mirror of inventing one, and with the gate above now stopping on `unknown`,
the conflation would have become a false failure rather than a quiet one.

### Narrowing a rule to what klin could not read

`require` takes `unless_classified: true`, which skips records whose licence
klin was able to classify. The case it exists for is a rule demanding
`licence.text`, written because a storefront is not a licence and "Free" on
itch.io is a price. That reasoning is about terms nobody can look up, and it
does not reach `Apache-2.0`, where the identifier is the licence.

```yaml
  - rule: 5
    kind: require
    when: ship
    fields: [licence.text]
    unless_classified: true
```

It narrows in the safe direction, exempting exactly the records klin could read
and never the ones it could not. An unclassified record is the one the rule was
written about, and it still has to carry its terms.

The other half of that fix is `spdx.py`. A recognised identifier has exactly one
text in a published register, so recording what the register says `Apache-2.0`
is amounts to a lookup rather than an inference, and an identifier outside the
register fills nothing at all. It runs only where the repository itself
publishes no licence document, which is the ordinary case for a model card
carrying a clean tag: neither Comfy-Org repository ships a `LICENSE` file, and
the terms were never in doubt, only somewhere else.

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
  spdx.py                the licence register, for an identifier's own text
  render.py              the marked block inside a prose document
  manifest.py            the per-project manifest
  secrets.py             credential lookup, references only in the manifest
  net.py                 streaming downloads and the four guards
  fetch/                 vendor adapters, discovered rather than listed
    hf.py                huggingface.co
    civitai.py           civitai.com
  gen/                   generators, discovered rather than listed
    comfy.py             a local ComfyUI, patched by role
    trellis.py           TRELLIS.2, and the patch that keeps it clean
    _inside_trellis.py   runs in TRELLIS's interpreter, never in klin's
  conform/               conformers, discovered rather than listed
    blender.py           headless Blender
    _inside_blender.py   runs in Blender's interpreter, never in klin's
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
