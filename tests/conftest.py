import io
import json
import os

import pytest

MANIFEST = """# klin manifest — fixture

Prose above the block is ignored by the loader, which is the point of the
convention: the document explains itself to a person, and only the fenced block
is read.

```yaml
product_name: Fixture

policy_doc: assets/LICENSES.md
ledger: assets/ledger.jsonl
render_target: assets/LICENSES.md
render_marker: records

stage: prototype

build_facts:
  steam_drm: {steam_drm}
{secrets}
rules:
  - rule: 0
    id: recorded
    kind: require
    when: always
    fields: [source.url, licence.id]
    text: Stage rule - use whatever looks right, but write it down.

  - rule: 1
    id: prefer-cc0
    kind: prefer
    when: ship
    families: [public-domain]
    text: CC0 is the default. Prefer it over CC-BY wherever a usable option exists.

  - rule: 2
    id: no-share-alike
    kind: deny
    when: ship
    families: [share-alike]
    text: >-
      Never take CC-BY-SA. OpenGameArt's "Tavern Tools and Furniture" is exactly
      the right content and is CC-BY-SA 4.0. It will look like a perfect find.
      Don't take it.

  - rule: 3
    id: ccby-vs-steam-drm
    kind: assert
    when: ship
    if_any_family: [attribution]
    and_fact: {{steam_drm: true}}
    text: Do not enable Steam's DRM wrapper if any CC-BY asset is in the build.

  - rule: 4
    id: excluded-licences
    kind: deny
    when: ship
    families: [noncommercial, noderivatives, editorial, copyleft, fan-art]
    text: Exclude outright NonCommercial, NoDerivatives, editorial and GPL-tagged art.

  - rule: 5
    id: storefront-is-not-a-licence
    kind: require
    when: ship
    fields: [licence.text]
    text: A storefront is not a licence. "Free" on itch.io is a price.
```
"""

SECRETS = """
secrets:
  civitai:
    env: CIVITAI_API_TOKEN
    description: read access for the Civitai fetch adapter
  huggingface:
    env: HF_TOKEN
"""

LICENSES = """# Asset provenance

Hand-written policy lives here and stays here.

## Current contents

<!-- klin:begin records -->
<!-- klin:end records -->

## Rules for anything added later

Also hand-written, also untouched by the renderer.
"""


def record(record_id, licence_id="CC0-1.0", **over):
    base = {
        "id": record_id,
        "kind": "mesh",
        "paths": ["assets/%s.glb" % record_id],
        "sha256": None,
        "author": {"name": "Someone", "url": None},
        "source": {
            "adapter": "manual",
            "url": "https://example.invalid/%s" % record_id,
            "retrieved": "2026-08-23",
        },
        "licence": {
            "id": licence_id,
            "name": licence_id,
            "url": "https://example.invalid/licence",
            "text": "Licence text, verbatim.",
            "attribution_required": bool(licence_id)
            and licence_id.upper().startswith("CC-BY"),
        },
        "produced_by": None,
        "modifications": [],
        "notes": None,
        "used_for": None,
        "reviewed_at": None,
        "waiver": None,
    }
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value
    return base


class Repo(object):
    def __init__(self, root):
        self.root = str(root)

    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def write_records(self, records):
        target = self.path("assets", "ledger.jsonl")
        with io.open(target, "w", encoding="utf-8", newline="\n") as handle:
            for item in records:
                handle.write(json.dumps(item, sort_keys=True) + "\n")

    def licenses(self):
        return io.open(self.path("assets", "LICENSES.md"), encoding="utf-8").read()

    @property
    def manifest(self):
        return self.path(".claude", "klin-manifest.md")


@pytest.fixture(autouse=True)
def never_the_real_cache(monkeypatch):
    """The developer's machine may name a real cache; the tests must not see it.

    `cache_dir` resolves the environment before the manifest, so a KLIN_CACHE
    set at the user level would leak into every fixture repo and make drift
    behaviour depend on whose machine the suite runs on. The suite happened to
    pass with the variable set when this guard was added — luck, not design.
    A test that wants a cache sets one explicitly.
    """
    monkeypatch.delenv("KLIN_CACHE", raising=False)


@pytest.fixture(autouse=True)
def never_a_real_vault(monkeypatch):
    """Belt and braces.

    Every test that reaches the credential store substitutes an in-memory one,
    but a mistake in that substitution would otherwise write to the developer's
    own keychain. The null backend makes such a mistake fail rather than
    succeed quietly.
    """
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")
    monkeypatch.delenv("KLIN_SECRET_BACKEND", raising=False)


@pytest.fixture
def repo(tmp_path):
    def build(steam_drm=False, secrets_block=False):
        os.makedirs(str(tmp_path / ".claude"), exist_ok=True)
        os.makedirs(str(tmp_path / "assets"), exist_ok=True)
        io.open(str(tmp_path / ".claude" / "klin-manifest.md"), "w", encoding="utf-8").write(
            MANIFEST.format(
                steam_drm=str(bool(steam_drm)).lower(),
                secrets=SECRETS if secrets_block else "",
            )
        )
        io.open(str(tmp_path / "assets" / "LICENSES.md"), "w", encoding="utf-8").write(
            LICENSES
        )
        made = Repo(tmp_path)
        made.write_records([])
        return made

    return build
