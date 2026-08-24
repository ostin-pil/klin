# Handover: the KeePassXC resolver

Written 2026-08-24, at the end of the session that built klin's secrets layer.
This picks up where `2026-08-24_secrets-on-windows.md` stops. Read that first
for why any of this is shaped the way it is; this file is state, decisions and
next actions only.

The job: make step three of klin's secret lookup real, so a declared
`ref:` resolves out of KeePassXC without the passphrase ever reaching klin.

## Where things stand

Shipped in PR #2 (`feature/session-3-secrets`), all checks green, unmerged:

- `klin/secrets.py`, the lookup and the store wrapper. Windows persistence
  pinned to the local machine, oversize values refused before `CredWrite`, and
  klin's own index of stored names because keyring cannot enumerate a vault.
- `klin secret set | get | list | rm | doctor`.
- `manifest.secrets()`, validating a `secrets:` block of names and references.
- `ledger.sanitise()`, stripping credential-shaped query parameters from record
  URLs on `add`.
- 88 tests, including a `windows-latest` CI entry.

The lookup order is `KLIN_SECRET_<NAME>`, then the manifest's declared `env`
alias, then a `ref`, then the keyring store. **Step three currently raises**
(`klin/secrets.py:243`), naming `klin secret set` as the thing to do instead.
That is deliberate: a reference the tool cannot follow is a broken promise, and
quietly serving a cached value instead would be the silent fallback the research
document objects to everywhere else. Nothing declares a `ref` yet.

Outside klin:

- `C:\Users\pilyu\vault.kdbx` exists. KDBX 4.0, Argon2d, 64 MiB, 80 iterations,
  4 lanes. Passphrase set. Groups `klin/` and `Barinn/`. Synced to
  `gdrive:vault/` and restore-drilled for integrity.
- Barinn declares `civitai` and `huggingface` in `.claude/klin-manifest.md`,
  `env` only, no `ref`.
- **KeePassXC browser integration is off.** `keepassxc.ini` has no `[Browser]`
  section. Nothing in any playbook covers it.

## The decision that blocks the design

The browser protocol is **URL-keyed**. `get_logins("https://example.invalid")`
matches entries by their URL field. There is no message that fetches the entry
at a group path. The reference shape this session validated and committed is a
group path:

```yaml
ref: keepassxc://klin/civitai
```

Those do not meet. Pick one before writing code, because the schema is already
validated in `manifest.secrets()` and shipped.

1. **Give entries a synthetic URL.** Put `klin://civitai` in the entry's URL
   field and keep the manifest shape as it is. The manifest stays readable, and
   the coupling moves into a convention about how entries are filled in, which
   the playbook would have to state and a human would have to honour.
2. **Change the reference to carry a URL.** `ref: keepassxc://klin%2Fcivitai`
   or an explicit `url:` key. Honest about what the protocol does, uglier in the
   manifest, and it means a schema change to something already committed.
3. **Resolve by path through a different mechanism** and keep the browser
   protocol for something else. This reopens the passphrase problem, so it is
   listed only to be ruled out on the record.

Leaning towards 1, because it keeps the committed schema and the manifest
legible, and because the entry convention has to be documented either way. But
it makes the database contents load-bearing for klin's behaviour, which is worth
weighing rather than waving through.

## Decisions still open

- **The client library.** `keepassxc-proxy-client` on PyPI is small and does
  associate plus `get_logins`. `varjolintu/keepassxc-browser-client` is closer
  to the protocol. Implementing the protocol directly needs libsodium bindings,
  which is a compiled dependency and against the pure-Python install story, so
  it should be an optional extra whichever way it goes.
- **Where the association key lives.** Credential Manager, under klin's own
  service, is the obvious answer: it is a key, not a passphrase, and the cache
  is exactly where keys belong. That makes klin's index gain an entry that is
  not a secret, which the `list` output should probably distinguish.
- **What happens when KeePassXC is not running or is locked.** The current
  `ref` behaviour raises. Once a resolver exists, "cannot reach the app" and
  "the app says no" are different from "no resolver", and at least the first
  should say so plainly. Falling through to the cache silently is still the
  wrong answer.
- **Whether `KLIN_KEEPASS_DB` is still needed.** The research document specified
  it as the machine-level setting for the database path. The browser protocol
  talks to the running application rather than a file, so a path may be
  unnecessary. The protocol does expose a database-hash message, so the useful
  form of that setting may be an expected hash to pin which database klin is
  willing to talk to, rather than a path to open. Check the protocol document
  before deciding.

## First steps, in order

1. Enable browser integration in KeePassXC (Tools, Settings, Browser
   Integration). No browser needs to be installed for a direct pipe client;
   confirm that rather than assuming it.
2. Before touching klin, prove the mechanism with a throwaway script:
   associate, approve the dialog, read one entry back. If that does not work by
   hand it will not work behind an abstraction, and the failure will be much
   harder to read once it is three layers down.
3. Settle the URL question above against what that script actually needed.
4. Then write the resolver.

## Gotchas already paid for

- **It needs KeePassXC running and unlocked.** Useless in CI and headless runs.
  That is what the environment-variable step exists for, and the resolver must
  not become a reason to weaken it.
- **Association is interactive.** A human approves it in the KeePassXC window.
  So no test may require a real KeePassXC. Follow the existing pattern: tests
  monkeypatch `secrets._store` with an in-memory store, and the resolver wants
  the same seam so a fake connection can stand in.
- **The Windows named-pipe namespace is global.** KeePassXC now includes the
  username in the pipe name after that was reported as an issue. Worth a note
  in whatever documents this, and a reason to require a current KeePassXC.
- **`keepassxc-cli` is not on `PATH` by default.** Added by hand on this machine
  2026-08-24. Anything running as a scheduled task still needs the full path.

## Invariants that must survive

These are the point of the design. If a change breaks one, the change is wrong.

- The database passphrase never reaches klin, an agent, a command string, or a
  transcript. This is why the resolver uses the protocol rather than the CLI.
- No key file as a substitute. It turns the root credential into a file readable
  by anything running as the user, which is the Windows gap the whole
  investigation is about.
- The environment comes first in the lookup order, always.
- Values never enter a manifest, the ledger, or the rendered block.
- A reference klin cannot follow raises rather than falling through.

## Loose ends from this session, unrelated to the resolver

- PR #2 is unmerged. Barinn's `feature/session-6-secrets` is unmerged and has no
  session log; that is `session-end`'s to write.
- Barinn's `playbooks/secrets.md` exposure table is deliberately unchanged. The
  database is a destination, and nothing has moved into it. The table gets
  rewritten by the session that moves the crypt passwords in and encrypts
  `rclone.conf`, which is step 2 of that playbook.
- The restore drill proved transport and integrity. Opening a restored copy with
  the passphrase has not been done and is still owed.
- Argon2id instead of Argon2d is one dropdown away and optional. Recorded in the
  playbook as a modest improvement on something already strong.
- klin is not installed as a tool on this machine, so `klin` is not on `PATH`
  for other sessions. `uv tool install klin` once PR #2 merges.
