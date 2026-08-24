"""Streaming downloads, and the guards that decide a file is what it claims.

A vendor that refuses a request does not always say so with a status code. The
failure this module exists to prevent is a refusal served with a 200 and a body
of markup, written to disk as `.safetensors`, and discovered hours later when a
sampler chokes on it. By then the download is long finished and the error names
the sampler rather than the cause.

So nothing here trusts the status line alone. Four guards run against every
download, in order of how early they can fire:

1. The content type is on an allowlist, never merely off a denylist. Civitai's
   edge refuses an unrecognised client with `text/plain`, not `text/html`, so a
   rule written against markup alone waves it through.
2. The byte count matches the vendor's declared size, where one is published.
3. A `.safetensors` file starts with a header length and a JSON header that
   parses. This is the only guard that reads the format rather than the
   envelope.
4. sha256, computed while streaming rather than by re-reading, goes into the
   record, and an adapter with a published hash to compare against checks it.

A guard that fires deletes the partial file. A half-written model that survives
a failure is the same trap one download later.
"""

import hashlib
import io
import json
import os
import re

from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from . import secrets

#: Cloudflare blocks urllib's default agent outright: Civitai answers
#: `403 error code: 1010`, which is the edge refusing the client and not the
#: vendor refusing the credential. The distinction is invisible in the status
#: code, so a real agent goes on every request rather than being discovered
#: once per vendor.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

CHUNK = 1024 * 1024

#: Types a model file may arrive as. An allowlist, because the interesting
#: failures are the ones nobody predicted: `text/plain` was on no denylist
#: until an edge server used it to say no.
BINARY_TYPES = (
    "application/octet-stream",
    "binary/octet-stream",
    "application/x-safetensors",
    "application/zip",
)

CLOUDFLARE_BLOCK = re.compile(r"error code:\s*(\d+)")


class NetError(Exception):
    """A download that must not be treated as a file."""


def _host(url):
    return urlsplit(url).netloc or url


def _headers(token=None):
    head = {"User-Agent": USER_AGENT}
    if token:
        head["Authorization"] = "Bearer " + token
    return head


def _describe_http_error(exc, url):
    """Turn an HTTPError into something naming the cause, not the symptom."""
    body = b""
    try:
        body = exc.read(400)
    except Exception:
        pass
    text = body.decode("utf-8", "replace").strip()

    if exc.code in (401, 403):
        match = CLOUDFLARE_BLOCK.search(text)
        if match:
            return NetError(
                "%s refused the client, not the credential (edge block %s). Such "
                "a request needs a browser User-Agent; klin sends one, so a fresh "
                "occurrence means the edge rules changed. Body: %s"
                % (_host(url), match.group(1), text)
            )
        if exc.code == 401:
            return NetError(
                "%s requires a credential (401). Store one with `klin secret "
                "set <name>`, then run this again. Body: %s" % (_host(url), text)
            )
        return NetError("%s refused the request (403). Body: %s" % (_host(url), text))

    if exc.code == 416:
        return NetError(
            "%s rejected the resume range (416). The partial file is longer than "
            "the remote one; delete it and fetch again." % _host(url)
        )
    return NetError("%s returned HTTP %s. Body: %s" % (_host(url), exc.code, text))


def check_content_type(value, url):
    """Guard 1. An allowlist, so an unforeseen refusal fails rather than lands."""
    base = (value or "").split(";")[0].strip().lower()
    if not base:
        # No type at all is unusual but not proof of a refusal, and some CDNs
        # omit it on a range response. The later guards still have to pass.
        return
    if base in BINARY_TYPES:
        return
    raise NetError(
        "%s served Content-Type %r, which is not a model file. This is how a "
        "login wall or an edge block arrives with a success status."
        % (_host(url), base)
    )


def check_size(got, expected, url):
    """Guard 2. Only meaningful when the vendor publishes a size."""
    if expected in (None, 0):
        return
    if int(got) != int(expected):
        raise NetError(
            "%s: expected %d bytes, received %d. A short read is a truncated "
            "download, not a smaller file." % (_host(url), int(expected), int(got))
        )


def check_safetensors(path):
    """Guard 3. The only guard that reads the format rather than the envelope.

    A safetensors file opens with an unsigned 64-bit little-endian header
    length, followed by that many bytes of JSON. Both have to hold: a header
    length on its own is eight bytes of anything at all.
    """
    size = os.path.getsize(path)
    if size < 8:
        raise NetError("%s is %d bytes, too short to be safetensors" % (path, size))
    with io.open(path, "rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        if length <= 0 or length + 8 > size:
            raise NetError(
                "%s declares a %d-byte safetensors header inside a %d-byte file, "
                "which cannot be right" % (path, length, size)
            )
        raw = handle.read(length)
        try:
            header = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise NetError(
                "%s: safetensors header is not valid JSON (%s)" % (path, exc)
            )
    if not isinstance(header, dict):
        raise NetError("%s: safetensors header is not an object" % path)
    return header


def verify_existing(dest, expected_size=None):
    """Facts about a file already on disk, or None when it cannot be trusted.

    A cache is only a cache if a second run is cheap. Re-fetching seventeen
    gigabytes to rediscover bytes that are already correct is the difference
    between a session that starts and one that waits, so a file that passes the
    same guards a fresh download would pass is accepted as it stands.

    Every guard still runs. The size must match where one is declared, the
    format must parse, and the digest is recomputed rather than remembered,
    because a remembered digest proves the download succeeded once and says
    nothing about the file that is there now.
    """
    if not os.path.isfile(dest):
        return None
    size = os.path.getsize(dest)
    if expected_size and int(size) != int(expected_size):
        return None
    if dest.endswith(".safetensors"):
        try:
            check_safetensors(dest)
        except NetError:
            return None
    digest = hashlib.sha256()
    with io.open(dest, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return {
        "path": dest,
        "bytes": size,
        "sha256": digest.hexdigest(),
        "content_type": None,
        "final_url": None,
        "reused": True,
    }


def download(
    url,
    dest,
    token=None,
    expected_size=None,
    resume=False,
    stream=None,
    force=False,
):
    """Stream `url` to `dest`, verifying as it goes. Returns a facts dict.

    The file is written to `<dest>.part` and renamed only once every guard has
    passed, so an interrupted or refused download can never be mistaken for a
    finished one by anything that looks at the directory.
    """
    if not force:
        already = verify_existing(dest, expected_size)
        if already is not None:
            if stream is not None:
                stream.write(
                    "already cached and verified: %s\n" % os.path.basename(dest)
                )
            return already

    parent = os.path.dirname(dest)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)

    part = dest + ".part"
    digest = hashlib.sha256()
    start = 0

    if resume and os.path.isfile(part):
        # The hash has to cover the bytes already on disk, and they are not in
        # memory any more, so they are re-read once. Cheaper than re-fetching
        # seventeen gigabytes to recompute a prefix.
        start = os.path.getsize(part)
        with io.open(part, "rb") as handle:
            for block in iter(lambda: handle.read(CHUNK), b""):
                digest.update(block)
        if stream is not None:
            stream.write("resuming at %d bytes\n" % start)
    elif os.path.isfile(part):
        os.remove(part)

    head = _headers(token)
    if start:
        head["Range"] = "bytes=%d-" % start

    request = Request(url, headers=head, method="GET")
    try:
        response = urlopen(request, timeout=120)
    except HTTPError as exc:
        raise _describe_http_error(exc, url)
    except URLError as exc:
        raise NetError("could not reach %s: %s" % (_host(url), exc.reason))

    with response:
        check_content_type(response.headers.get("Content-Type"), url)

        if start and getattr(response, "status", None) != 206:
            # The server ignored the range and is sending the whole file.
            # Starting over is the only safe move: appending would concatenate
            # a prefix onto a complete file and corrupt it silently.
            start = 0
            digest = hashlib.sha256()
            if os.path.isfile(part):
                os.remove(part)

        mode = "ab" if start else "wb"
        written = start
        with io.open(part, mode) as handle:
            for block in iter(lambda: response.read(CHUNK), b""):
                handle.write(block)
                digest.update(block)
                written += len(block)

        declared = expected_size
        if declared is None:
            span = response.headers.get("Content-Range")
            if span and "/" in span:
                tail = span.rsplit("/", 1)[-1].strip()
                if tail.isdigit():
                    declared = int(tail)
            elif response.headers.get("Content-Length") and not start:
                declared = int(response.headers["Content-Length"])

        final_url = response.geturl()
        content_type = response.headers.get("Content-Type")

    try:
        check_size(written, declared, url)
        if dest.endswith(".safetensors"):
            check_safetensors(part)
    except NetError:
        os.remove(part)
        raise

    if os.path.exists(dest):
        os.remove(dest)
    os.rename(part, dest)

    return {
        "path": dest,
        "bytes": written,
        "sha256": digest.hexdigest(),
        "content_type": content_type,
        "reused": False,
        # Scrubbed here rather than left to `ledger.sanitise`, so a signed CDN
        # URL never sits in a record even briefly.
        "final_url": secrets.scrub_url(final_url),
    }


def get_json(url, token=None, timeout=60):
    """Fetch and parse a vendor metadata document."""
    request = Request(url, headers=_headers(token), method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise _describe_http_error(exc, url)
    except URLError as exc:
        raise NetError("could not reach %s: %s" % (_host(url), exc.reason))
    except ValueError as exc:
        raise NetError("%s did not return JSON (%s)" % (_host(url), exc))
