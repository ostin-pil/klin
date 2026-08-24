# Secrets on Windows

klin holds no secrets today. It will. The roadmap puts vendor adapters under
`fetch` and `gen`, and a Civitai or HuggingFace adapter needs an API key sitting
on the user's machine. klin runs on developer machines rather than servers, and
the machine this was written on is Windows, so the question is what Windows
offers in place of the macOS Keychain, where each option falls short, and what
can be done about each shortfall.

The short answer is two layers. A KeePassXC database, synced off the machine, is
the system of record: it is the only option that both locks behind a prompt and
survives the hardware. Windows Credential Manager, reached through `keyring`, is
the runtime cache klin actually reads, because an audit in CI cannot stop to
prompt. Do not build a Keychain equivalent. The reason is in the next section,
and it is not a matter of effort.

## What the Keychain gives you that Windows does not

macOS attaches an access control list to each Keychain item. `securityd` decides
per process, using code signing and entitlements, whether that process may read
a given item, and prompts the user when an application is not on the item's
list. Two properties fall out of that: a secret can be bound to one application,
and reading it can require a human to say yes.

Windows has neither for unsigned desktop applications. Anything running as the
user can read any credential that user has stored, which is catalogued as MITRE
ATT&CK T1555.004 and is the assumption every credential-stealing toolkit already
makes. There is no ACL to attach and no daemon adjudicating callers.

This is an operating system property. No library closes it, and neither would
anything klin wrote. So every option below gets judged against a boundary that
stops at three things: another user on the same machine, a stolen disk, and a
credential accidentally committed to a repository. The only Windows mechanism
that reaches past that boundary is a Windows Hello prompt, and what it buys is
user consent rather than process isolation.

Worth saying plainly, because it is the honest frame for everything after it.

## The options

### Windows Credential Manager

What it is. The operating system's credential vault, reached through `CredWrite`
and `CredRead`, encrypting each blob with the user's logon session key using
DPAPI underneath. Microsoft names it the preferred approach for new desktop work
once passwordless options are off the table. The user can see what is in it
through `cmdkey /list` and the Control Panel applet, which matters more than it
sounds: a secret store nobody can inspect is a secret store nobody trusts.

Where it falls short.

- No per-application ACL and no prompt on access, as above.
- The credential blob caps out around 2560 bytes, which is roughly 1280
  characters once `keyring` encodes to UTF-16. Going over surfaces as
  `CredWrite` error 1783, "The stub received bad data", which tells the user
  nothing.
- It is unavailable over a network or SSH logon session. git-credential-manager
  documents this as a hard limitation of the store.
- `keyring` writes with `CRED_PERSIST_ENTERPRISE`, so on a domain-joined machine
  with roaming profiles the credential follows the user to other machines.

What mitigates it.

- Scope what goes in rather than trying to harden the vault. A read-only,
  per-machine, short-lived vendor token with a recorded rotation date is worth
  little to malware running as the user, and that is the realistic threat here.
  This is the mitigation that moves the needle; the rest are hygiene.
- Check length before writing and fail with a message a human can act on. For
  anything genuinely larger than the cap, store a wrapping key in the vault and
  the ciphertext in a file next to it.
- Pin persistence to `CRED_PERSIST_LOCAL_MACHINE` through the backend's
  `persist` attribute, so credentials stay on the machine that made them.
- Cover the SSH and network-session case with an environment variable override,
  which is the situation such an override exists for.

### DPAPI directly

What it is. `CryptProtectData` encrypts a byte string against the user account
or the machine, reachable from pure Python through `ctypes` and
`windll.crypt32`. It is what Credential Manager uses internally, and what
git-credential-manager offers as its `dpapi_store` option.

Where it falls short.

- The same absence of per-application ACL, since it is the same machinery.
- You now own the file, its location, its rotation and its deletion.
- The optional entropy parameter has to live somewhere itself, which puts you
  back where you started.
- Ciphertext is bound to that user on that machine and does not survive a
  profile rebuild.
- A private file under `%APPDATA%` appears in no tool the user already knows.

What mitigates it.

- Derive the entropy from a passphrase prompted at read time. That gives you
  something malware cannot read silently, at the price of a prompt on every run.
- Reserve `CRYPTPROTECT_LOCAL_MACHINE` for service accounts that need it, since
  it widens the boundary to every user on the box.
- Publish the storage path through a doctor command so the file is visible.
- Mostly, the mitigation is to not choose this. Credential Manager reaches the
  same boundary using tooling the user already has.

### Windows Hello and KeyCredentialManager

What it is. A TPM-backed RSA key gated by PIN, face or fingerprint. The one
Windows mechanism that prompts a human before releasing access.

Where it falls short.

- It signs challenges rather than storing secrets, so the envelope encryption
  around it is yours to build.
- The WinRT surface is awkward and heavy to reach from Python.
- Per-machine, with no roaming and no recovery once the TPM or the enrolment
  goes.

What mitigates it.

- The envelope pattern is settled, and KeePassXC ships a reference
  implementation of exactly this shape: sign a random challenge with the
  Hello-protected key, SHA-256 the signature, use that as the key encrypting the
  stored credential. Borrow the pattern rather than inventing one.
- Treat recovery as reauthentication rather than key escrow. For a read-scoped
  API token, the answer to a dead TPM is to issue a new token, which costs
  nothing.
- Gate a session rather than each read, so the prompt is paid once.
- Keep the WinRT dependency an opt-in extra, since reaching it needs compiled
  packages a base install should not carry.

### KeePassXC driven through keepassxc-cli

What it is. A KDBX database plus a scriptable CLI that ships on Windows, macOS
and Linux. `keepassxc-cli show -q -s -a Password <db> <entry>` prints one
attribute; `-k` takes a key file, `--no-password` drops the passphrase factor,
and `-y` takes a YubiKey slot for challenge-response. KDBX4 uses Argon2, the
file is portable and syncs like any other file, and 2.7 added Windows Hello
quick unlock.

Where it falls short.

- The user has to install and maintain KeePassXC.
- Every invocation wants the database credential, which moves the bootstrap
  problem up one level rather than solving it.
- A locked database means an interactive prompt in the middle of a script.
- The Secret Service integration is Linux-only, so `python-keyring` cannot reach
  a KDBX on Windows, and the KDBX record layout differs from what the
  SecretService backend expects even where the daemon does exist.

What mitigates it.

- Put the database credential in Windows Credential Manager and let the tool
  unlock the KDBX with it. You keep a portable, inspectable, syncable database,
  and exactly one bootstrap secret sits in the OS vault.
- Or drop the passphrase entirely: a key file plus YubiKey challenge-response
  leaves no password to steal.
- Hold an unlocked session with `keepassxc-cli open` for a batch of reads
  instead of unlocking per call.
- Turn on Windows Hello quick unlock so the interactive prompt is a fingerprint
  rather than a passphrase.

Worth noting what that combination adds up to: a store that locks, and that
prompts a human to unlock it. That is the closest thing on Windows to the
Keychain behaviour that prompted this research, and it arrives by composing two
tools rather than by finding one.

### KeePass databases read directly with pykeepass

What it is. A Python library that opens KDBX3 and KDBX4 without any external
binary.

Where it falls short. It pulls `argon2-cffi`, `pycryptodomex`, `lxml` and
`construct`, every one of which ships compiled wheels, against a klin tree whose
only runtime dependency is `pyyaml` and which installs through
`uv tool install`. The tool would also hold the master password in its own
address space, and writing to a database a GUI has open invites corruption.

What mitigates it. Keep it behind an optional extra so the base install stays
pure Python. Open read-only and never write, which removes the corruption risk
and leaves the GUI as the only writer. Combine with the Credential Manager
bootstrap above so no script ever prompts for the master credential.

### KeePass 2.x with the KeePassWinHello plugin

What it is. The original .NET KeePass, with a plugin gating unlock behind
Windows Hello.

Where it falls short. Windows-only, .NET-bound, and scripting it means the
separate KPScript plugin rather than a first-class CLI.

What mitigates it. Prefer KeePassXC for anything a tool drives. For a user who
already lives in KeePass 2.x, the KDBX file is the interoperability point, and
`keepassxc-cli` or `pykeepass` reads it without anyone changing apps.

### PowerShell SecretManagement and SecretStore

What it is. A cross-platform local vault kept as a password-protected file under
`%LOCALAPPDATA%\Microsoft\PowerShell\secretmanagement\localstore\`, encrypted
with .NET crypto, behind a cmdlet interface that also fronts Azure Key Vault and
others.

Where it falls short. Requires PowerShell 7 and shelling out to it from Python.
Microsoft has declared SecretManagement feature complete, with security fixes
only.

What mitigates it. Feature complete still receives security fixes, so the risk
is stagnation rather than rot. Its password-protected mode with an unlock
timeout is closer to Keychain's locking behaviour than Credential Manager
manages. Treat it as one optional source behind a common lookup interface, so
users with vaults already registered get them without the tool depending on
PowerShell.

### External managers: op, bw, pass

What it is. Real per-item access control, audit trails, sharing and rotation,
addressed through secret references such as `op://vault/item/field`.

Where it falls short. It requires users to install and log into a product the
tool does not control. Supply-chain exposure is not theoretical: a trojanised
Bitwarden CLI reached the npm registry in April 2026 and was live for about
ninety minutes.

What mitigates it. Never require it. Support it as a reference declared in the
manifest, resolved by shelling out only when a reference of that shape is
present. Install from the vendor's signed package rather than a language
registry, and check the signature. Use service accounts and short-lived session
tokens for automation rather than a full interactive session.

### Plaintext file

What it is. What `huggingface_hub` does with `~/.cache/huggingface/token`, what
the AWS CLI does, and what git-credential-manager offers as a last resort under
the warning that it "is NOT secure".

Where it falls short. It offers no protection of any kind.

What mitigates it. Restrict the file to the current user and refuse to read one
whose permissions are broader. Make it explicit opt-in with a loud warning, and
never a silent fallback, which is the trap `keyrings.alt` sets by supplying an
insecure backend automatically when nothing better is found. Keep it out of the
repository, which klin has to guarantee for `ledger.jsonl` and the rendered
block anyway.

### Cloud vaults

What it is. Azure Key Vault, HashiCorp Vault and friends: server-side secret
management with policy, rotation and audit.

Where it falls short. Wrong shape for a local CLI operating on a checkout, and
bootstrapping still needs a local credential.

What mitigates it. They are reachable through the SecretManagement interface
above for anyone who needs them, so klin does not have to know about them
directly.

## The constraint the security comparison misses

Everything above weighs the options by what can read a secret. There is a second
axis, and on this machine it is the one that decides the answer: what happens
when the machine goes.

Credential Manager blobs are encrypted with the user's logon session key. DPAPI
ciphertext is bound to the user on that machine. Neither survives a profile
rebuild, and neither survives a board replacement. That is not a hypothetical
here. Barinn's `playbooks/backup-and-restore.md` records 15+ bugchecks since July
2026 attributed to suspected Arrow Lake-HX degradation, with board replacement on
the table, and it exists because the project needs to survive that event.

So a store that is only Credential Manager is a store that has to be rebuilt by
hand after a hardware failure, from secrets that must therefore exist somewhere
else anyway. The question is where that somewhere else is, and the answer cannot
be another DPAPI-bound location on the same disk.

## Build or wire

Wire two layers, and leave a seam between them.

Building a DPAPI file store means reimplementing Credential Manager, owning the
crypto plumbing, and arriving at the same security boundary with worse
discoverability. Building a Keychain equivalent is not achievable on Windows at
all, for the reason in the second section. Neither is a good trade for a tool
whose job is licence provenance.

The durable layer is a KeePassXC database. It is a file, so it survives the
machine by being copied off it; it is Argon2-encrypted, so copying it off is
safe; it locks, and Windows Hello quick unlock makes unlocking it a fingerprint.
That combination is the closest thing Windows has to the Keychain, and the
portability is what makes it the system of record rather than a curiosity.

The runtime layer is `keyring` 25.7.0, which on Windows means Credential Manager.
It is what klin actually reads, because a licence audit in CI or a fetch adapter
in a script cannot stop to prompt. On Windows `keyring`'s only dependency is
`pywin32-ctypes`, a pure-Python `ctypes` wrapper, so the pure-Python install
story holds and `uv tool install` keeps working. The same API covers macOS
Keychain and Linux Secret Service, which matters because klin's CI is Linux and
its users will not all be on Windows.

The relationship between the two is the whole design: the database is the system
of record, the OS vault is a cache seeded from it. Losing the cache costs one
reseed. Losing the database is the event worth engineering against, and it is
the one that off-site sync addresses.

Four gaps stay klin's problem, and all four have been seen in the wild:

1. No backend available at all, which is CI and headless Linux. An environment
   variable step ahead of the vault means CI never reaches `keyring`.
2. `keyrings.alt` silently supplying an insecure backend. A doctor command names
   the active backend and flags the non-recommended ones.
3. The blob size limit. Check before writing.
4. Enterprise persistence. Pin it to the local machine.

The seam is the lookup order. Environment variable first, then an adapter's
conventional variable such as `HF_TOKEN`, then an external reference declared in
the manifest, then the `keyring` store. Step three is where the KDBX resolver
lands, and where `op` or `bw` could land instead for anyone who prefers them.

Seeding runs the other way and stays a human action: read a value out of
KeePassXC, put it in the OS vault with `klin secret set`. After a board swap the
whole recovery is unlock the database, reseed, carry on. Automating that
seeding would mean klin holding the database credential, which trades the one
property the database is there for.

## Where the database lives

The KDBX has to be reachable without the machine it was made on, and the
existing rclone setup in Barinn already provides the transport. One trap is
worth naming before anyone wires it.

The crypt passwords for `barinn-crypt:` are exactly the kind of secret this
database exists to hold. Putting the database inside `barinn-crypt:` therefore
locks the key inside the box: restoring the backup needs the passwords, and
getting the passwords needs the backup. `mirror.ps1` mirrors a bare git repo
rather than arbitrary files, so the KDBX would not land there by accident, but
the reasoning matters more than the mechanism.

The database goes somewhere reachable with nothing but a Google login: a plain
`gdrive:` path, and the external drive as the second copy. It is already
encrypted with Argon2, so an unencrypted transport is the correct choice rather
than a compromise. Two rules follow. The key file, if the database uses one,
never syncs with the database, since together they are the whole credential.
And the database passphrase is the one secret that lives only in a human head,
because it is the root of the chain.

## What the result protects against

Worth writing down, because a secret store that overstates itself is worse than
one that does not exist.

It protects against another user on the machine, against a disk read outside the
logon session, against a credential landing in a git commit, and against the
machine dying with the only copy of something. The last one is the addition the
database makes, and on current hardware it is the one that pays for itself.

It does not protect against code running as you. On Windows nothing local does,
short of a Windows Hello prompt on every read, and that is a trade worth making
only for something more valuable than a read-scoped download token. Note that
the cache inherits the weaker of the two boundaries: anything seeded into
Credential Manager is readable by anything running as you, whatever the database
it came from is protected by.

Which points at the real control: keep the tokens small. Read-only scope,
per-machine, rotated, so what the cache exposes is worth little. The vault is the
second line and the database is the third.

## Sources

- [Handling Passwords, Win32 apps](https://learn.microsoft.com/en-us/windows/win32/secbp/handling-passwords)
- [CryptProtectData function (dpapi.h)](https://learn.microsoft.com/en-us/windows/win32/api/dpapi/nf-dpapi-cryptprotectdata)
- [Credentials from Password Stores: Windows Credential Manager, T1555.004](https://attack.mitre.org/techniques/T1555/004/)
- [git-credential-manager: credential stores](https://github.com/git-ecosystem/git-credential-manager/blob/main/docs/credstores.md)
- [keyring on PyPI](https://pypi.org/project/keyring/) and its [Windows backend](https://github.com/jaraco/keyring/blob/main/keyring/backends/Windows.py)
- [keyring issue 540: support longer passwords in Windows](https://github.com/jaraco/keyring/issues/540)
- [keyring documentation](https://keyring.readthedocs.io/en/stable/)
- [Windows Hello for developers](https://learn.microsoft.com/en-us/windows/apps/develop/security/windows-hello)
- [KeePassXC 2.7.0: Windows Hello quick unlock](https://github.com/keepassxreboot/keepassxc/pull/7384)
- [keepassxc-cli manual](https://man.archlinux.org/man/keepassxc-cli.1.en)
- [Using python-keyring with KeePassXC](https://github.com/jaraco/keyring/issues/448)
- [pykeepass on PyPI](https://pypi.org/project/pykeepass/)
- [SecretManagement and SecretStore overview](https://learn.microsoft.com/en-us/powershell/utility-modules/secretmanagement/overview?view=ps-modules)
- [1Password CLI secret references](https://www.1password.dev/cli/secret-references)
- [Bitwarden CLI trojanised on npm, April 2026](https://www.csoonline.com/article/4162865/bitwarden-cli-password-manager-trojanized-in-supply-chain-attack.html)
- [Hugging Face Hub environment variables](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)
