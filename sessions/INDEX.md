# Session index — klin

klin's sessions are its own. Work here is never part of a consuming project's
session: the decision to extract this tooling belongs in the project that made
it, and how the tooling got built is logged here.

| # | Date | Log | What happened |
|---|---|---|---|
| 1 | 2026-08-23 | [session 1](2026-08-23_session_1.md) | Repo birth. The ledger, policy engine, renderer, CLI and plugin, with no vendor adapter by design. 46 tests. Verified against a real consuming project's policy document. |
| 2 | 2026-08-23 | [session 2](2026-08-23_session_2.md) | CI: pytest across 3.11-3.13 and the prose gate. The prose action is a cross-repo reference to the sibling marketplace, which is a coupling worth knowing about. |
| 3 | 2026-08-24 | [session 3](2026-08-24_session_3.md) | Secrets. Windows has no per-application ACL and Credential Manager dies with the profile, so the design is two layers: a KeePassXC database as system of record, the OS vault as a cache klin reads. Ships `klin secret`, a `secrets:` manifest block, and a Windows CI runner. |
| 4 | 2026-08-24 | [session 4](2026-08-24_session_4.md) | The fetch adapters. `klin fetch hf` and `klin fetch civitai`, discovered rather than listed, so vendor three is one new file. The Civitai permission flags are mapped to licence families and the contested row is written down as a judgment. Validating the mapping against the live API corrected two of the eight models it was specified from. |
| 5 | 2026-08-24 | [session 5](2026-08-24_session_5.md) | Reading the licence the vendor published. `license: other` is a statement about the Hub's vocabulary, and the answer sits in the two fields beside it; klin read the tag and stopped, so the two FLUX models were reported unclassifiable while their terms were one link away. A bespoke-lora licence link turns out to carry Civitai's flags as query parameters, so it resolves through the table that already exists. The second follow-up filed against klin was not a defect and is recorded rather than fixed. |
