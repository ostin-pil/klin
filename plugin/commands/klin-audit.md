---
description: Run the klin licence gate over this project's asset ledger and report what failed.
---

Run the klin ship gate and report the result.

1. Read `.claude/klin-manifest.md` for the `policy_doc` and `ledger` paths. If
   there is no manifest, say so and stop; klin needs one per project.
2. Run `klin ledger audit --ship`.
3. Run `klin ledger render --check`.

Then report:

- Every failure, with the rule number, the record it fired on, and **the rule
  text exactly as the audit printed it**. Do not paraphrase and do not soften.
- Warnings, more briefly.
- Whether the policy document is up to date with the ledger.

If the audit exits non-zero, the project cannot ship as it stands. Say that
plainly and name the assets to replace. If a failure looks wrong, the fix is a
correction to the transcribed rule in the manifest checked against the
authoritative policy document, never a change to the record to dodge the rule.

$ARGUMENTS
