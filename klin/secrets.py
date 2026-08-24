"""Secrets: where a vendor credential lives, and what klin is allowed to know.

The manifest names a secret. The value lives in the operating system's
credential store, or in the environment for CI. Nothing secret ever enters the
repository, which is the whole design and the reason this module and `manifest`
are separate: one holds references, the other holds values.

The store is a cache rather than a system of record. On Windows a credential is
encrypted with the user's logon session key and does not survive a profile
rebuild or a board replacement, so anything kept here has to exist somewhere
durable as well. research/2026-08-24_secrets-on-windows.md has the reasoning,
the alternatives, and a plain statement of what this does and does not protect
against.

Nothing here talks to a vendor either. Adapters ask for a name and get a value.
"""

import json
import os
import re

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SERVICE = "klin"

# keyring cannot enumerate what is in a vault: the backends expose get, set and
# delete against a known name and nothing else. So klin keeps its own index of
# the names it has stored, in the vault, under a name no secret may use. Losing
# it costs the listing, never a value.
INDEX = "__index__"

NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
REF = re.compile(r"^([a-z][a-z0-9+.-]*)://(.+)$")

# Query parameters that carry a credential rather than a request. Stripped from
# any URL that reaches the ledger, because the ledger is committed.
SECRET_PARAMS = re.compile(
    r"^(x-amz-.*|.*token.*|.*api[-_]?key.*|key|sig|signature|secret|password|"
    r"passwd|pwd|auth.*|credential.*|session.*|code)$",
    re.I,
)

REDACTED = "[redacted]"

# Backends that actually protect the value. Anything else is either a degenerate
# backend (null, fail) or one of the keyrings.alt stores, which include plain
# and obfuscated files, and which keyring will select silently when nothing
# better is installed.
RECOMMENDED = (
    "keyring.backends.Windows.WinVaultKeyring",
    "keyring.backends.macOS.Keyring",
    "keyring.backends.SecretService.Keyring",
    "keyring.backends.kwallet.DBusKeyring",
    "keyring.backends.libsecret.Keyring",
)


class SecretError(Exception):
    pass


def env_name(name):
    """The environment variable that overrides a given secret."""
    return "KLIN_SECRET_%s" % normalise(name).upper().replace("-", "_").replace(
        ".", "_"
    )


def normalise(name):
    name = (name or "").strip().lower()
    if name == INDEX:
        raise SecretError("%r is reserved for klin's own index" % INDEX)
    if not NAME.match(name):
        raise SecretError(
            "%r is not a usable secret name; use lower-case letters, digits, "
            "dot, dash and underscore" % name
        )
    return name


def ref_scheme(ref):
    """The scheme of an external reference, validated but not resolved."""
    match = REF.match((ref or "").strip())
    if match is None:
        raise SecretError(
            "%r is not a usable reference; expected something like "
            "keepassxc://database-entry/field" % ref
        )
    return match.group(1)


class KeyringStore(object):
    """The operating system's credential store, reached through keyring."""

    def __init__(self):
        self.keyring = _import_keyring()
        self.backend = self.keyring.get_keyring()
        self.max_chars = _pin_windows(self.backend)

    @property
    def backend_name(self):
        kind = type(self.backend)
        return "%s.%s" % (kind.__module__, kind.__name__)

    def get(self, name):
        try:
            return self.keyring.get_password(SERVICE, name)
        except Exception as exc:
            raise SecretError(_unavailable(exc))

    def set(self, name, value):
        try:
            self.keyring.set_password(SERVICE, name, value)
        except Exception as exc:
            raise SecretError(_unavailable(exc))

    def delete(self, name):
        try:
            self.keyring.delete_password(SERVICE, name)
            return True
        except Exception as exc:
            if "not found" in str(exc).lower() or type(exc).__name__ in (
                "PasswordDeleteError",
            ):
                return False
            raise SecretError(_unavailable(exc))


def _import_keyring():
    try:
        import keyring
    except ImportError:
        raise SecretError(
            "keyring is not installed, so klin cannot reach the credential "
            "store; reinstall klin, or set the value in the environment as %s"
            % env_name("example")
        )
    return keyring


def _unavailable(exc):
    return (
        "the credential store is unavailable (%s: %s); set the value in the "
        "environment instead, or run `klin secret doctor`"
        % (type(exc).__name__, exc)
    )


def _pin_windows(backend):
    """Keep Windows credentials on the machine that made them, and report the
    blob limit that machine imposes.

    keyring writes with CRED_PERSIST_ENTERPRISE, which on a domain-joined
    machine roams the credential to every other machine the user logs into.
    Local machine is the honest default for a cache that is meant to be
    reseeded rather than carried.
    """
    if "WinVault" not in type(backend).__name__:
        return None
    try:
        from win32ctypes.pywin32 import win32cred

        backend.persist = win32cred.CRED_PERSIST_LOCAL_MACHINE
    except Exception:
        # Persistence is a hardening detail. Failing to pin it is worth
        # reporting in doctor, never worth refusing to store a credential over.
        pass
    # CRED_MAX_CREDENTIAL_BLOB_SIZE is 5 * 512 bytes and the backend encodes
    # UTF-16, so anything longer fails inside CredWrite as error 1783, "The
    # stub received bad data", with nothing to act on.
    return 1280


_STORE = None


def _store():
    """The credential store. Built late, so no klin command pays for keyring
    unless it is actually reaching for a secret."""
    global _STORE
    if _STORE is None:
        _STORE = KeyringStore()
    return _STORE


def _index_get(store):
    raw = store.get(INDEX)
    if not raw:
        return []
    try:
        got = json.loads(raw)
    except ValueError:
        return []
    return sorted(str(name) for name in got if isinstance(name, str))


def _index_set(store, names):
    names = sorted(set(names))
    if names:
        store.set(INDEX, json.dumps(names))
    else:
        store.delete(INDEX)


def use_store():
    """True when the lookup is allowed past the environment.

    KLIN_SECRET_BACKEND=env pins every lookup to the environment, for machines
    where the vault is unreachable, unwanted, or being deliberately bypassed.
    """
    return (os.environ.get("KLIN_SECRET_BACKEND") or "").strip().lower() != "env"


def lookup(name, spec=None):
    """Where a secret comes from and what it is, in precedence order.

    Returns (source, value), or (None, None) when nothing has it. The order is
    environment, then the adapter's own conventional variable, then an external
    reference, then the store. The environment comes first so CI never reaches
    for a vault, and so an SSH session, where Credential Manager is unreachable,
    still works.
    """
    name = normalise(name)
    spec = spec or {}

    value = os.environ.get(env_name(name))
    if value:
        return "env", value

    alias = spec.get("env")
    if alias:
        value = os.environ.get(alias)
        if value:
            return "env:%s" % alias, value

    if not use_store():
        return None, None

    ref = spec.get("ref")
    if ref:
        scheme = ref_scheme(ref)
        raise SecretError(
            "%s declares the %s reference %r, and klin has no resolver for "
            "that scheme yet; set the value with `klin secret set %s` instead"
            % (name, scheme, ref, name)
        )

    value = _store().get(name)
    if value:
        return "store", value
    return None, None


def resolve(name, spec=None):
    """The value, or an error that says what to do about its absence."""
    source, value = lookup(name, spec)
    if value:
        return value
    raise SecretError(
        "no value for %s; set it with `klin secret set %s`, or put it in the "
        "environment as %s" % (normalise(name), normalise(name), env_name(name))
    )


def status(name, spec=None):
    """The source of a secret without its value, for listing and diagnosis."""
    try:
        source, value = lookup(name, spec)
    except SecretError as exc:
        return {"name": normalise(name), "source": "error", "detail": str(exc)}
    return {
        "name": normalise(name),
        "source": source or "unset",
        "detail": env_name(name) if source == "env" else "",
    }


def store_secret(name, value):
    name = normalise(name)
    if not value:
        raise SecretError("refusing to store an empty value for %s" % name)
    store = _store()
    limit = getattr(store, "max_chars", None)
    if limit and len(value) > limit:
        raise SecretError(
            "%s is %d characters and this credential store holds %d; keep the "
            "value in a password database and store a shorter token here"
            % (name, len(value), limit)
        )
    store.set(name, value)
    _index_set(store, _index_get(store) + [name])
    return name


def delete_secret(name):
    name = normalise(name)
    store = _store()
    existed = store.delete(name)
    _index_set(store, [n for n in _index_get(store) if n != name])
    return existed


def stored_names():
    """Every name klin has stored, from its own index."""
    return _index_get(_store())


def backend_report():
    """What is actually holding the credentials, and whether that is a store
    worth trusting."""
    if not use_store():
        return {
            "backend": "environment only",
            "recommended": True,
            "note": "KLIN_SECRET_BACKEND=env, so the credential store is not "
            "consulted",
        }
    try:
        store = _store()
    except SecretError as exc:
        return {"backend": "unavailable", "recommended": False, "note": str(exc)}
    name = store.backend_name
    recommended = name in RECOMMENDED
    note = ""
    if not recommended:
        note = (
            "this backend is not one of the recommended ones; keyring falls "
            "back to keyrings.alt file stores when no system vault is "
            "available, and those are not secure"
        )
    return {"backend": name, "recommended": recommended, "note": note}


def scrub_url(url):
    """Drop credential-shaped query parameters from a URL.

    A fetch adapter that follows a presigned or tokenised download link would
    otherwise write that link into the ledger, and the ledger is committed. The
    rest of the URL is the provenance and is worth keeping.
    """
    if not url or not isinstance(url, str):
        return url
    split = urlsplit(url)
    if not split.query:
        return url
    kept = [
        (key, value)
        for key, value in parse_qsl(split.query, keep_blank_values=True)
        if not SECRET_PARAMS.match(key)
    ]
    if len(kept) == len(parse_qsl(split.query, keep_blank_values=True)):
        return url
    return urlunsplit(
        (split.scheme, split.netloc, split.path, urlencode(kept), split.fragment)
    )


def scrub(text, values):
    """Replace known secret values wherever they appear in a string."""
    if not text or not isinstance(text, str):
        return text
    for value in values:
        if value and isinstance(value, str) and len(value) >= 8 and value in text:
            text = text.replace(value, REDACTED)
    return text
