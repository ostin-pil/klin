"""PNG framing: the header, and the text chunks before the pixels.

Separate from any one reader because these are facts about the container, not
about whichever tool wrote it. `dimensions` answers what a file actually is for
every PNG in a scan, including the contact sheets and masks that carry no
generation metadata at all, and the ComfyUI reader is only one consumer of
`text_chunks`.

**Stdlib only, deliberately.** Pillow reads all of this in three lines and is
the obvious dependency to reach for. klin ships with two dependencies and this
adds none: chunk framing is a length, a four-byte type, the payload and a CRC,
and that is the whole format needed here. A scanner walking a hundred thousand
files also has no business decoding pixels, which is what opening an image
would do.
"""

import struct
import zlib

#: The eight bytes every PNG starts with.
MAGIC = b"\x89PNG\r\n\x1a\n"

#: Where the pixel data begins. Every chunk klin reads is written before it, so
#: the reader stops here rather than streaming a twelve-megabyte image.
STOP = b"IDAT"

#: How many chunks to walk before giving up. A PNG that puts its metadata after
#: five hundred other chunks is malformed in a way worth failing on.
MAX_CHUNKS = 512


def _inflate(data):
    return zlib.decompress(data)


def _itxt_value(rest):
    """An iTXt payload after its keyword: flag, method, language, translated
    keyword, then the text. Compressed only when the flag byte is 1."""
    if len(rest) < 2:
        return None
    compressed = rest[0]
    parts = rest[2:].split(b"\x00", 2)
    if len(parts) < 3:
        return None
    text = parts[2]
    if compressed == 1:
        try:
            return _inflate(text)
        except zlib.error:
            return None
    return text


def _walk(handle):
    """Yield `(type, payload)` for each chunk up to the pixel data."""
    if handle.read(8) != MAGIC:
        return
    for _ in range(MAX_CHUNKS):
        head = handle.read(8)
        if len(head) < 8:
            return
        length, kind = struct.unpack(">I4s", head)
        payload = handle.read(length)
        handle.read(4)  # CRC, unchecked: a corrupt payload fails to parse
        if kind == STOP:  # downstream, and is reported there instead.
            return
        yield kind, payload


def dimensions(path):
    """`(width, height)` from the header, or None if this is not a PNG.

    IHDR is the first chunk of every valid PNG, so this reads about thirty
    bytes. These are the output's real dimensions, which is what somebody
    browsing a corpus wants, and they are correct even when an upscale sat
    between the sampler and the save.
    """
    with open(path, "rb") as handle:
        for kind, payload in _walk(handle):
            if kind == b"IHDR" and len(payload) >= 8:
                width, height = struct.unpack(">II", payload[:8])
                return width, height
            break
    return None


def text_chunks(path):
    """Every text chunk before the pixel data, as name -> bytes.

    `zTXt` and `iTXt` are decoded too. ComfyUI writes plain `tEXt` today, but
    the compressed variants are the same metadata under a different chunk type
    and cost a few lines to support. A scanner that skipped them would report a
    file unreadable for a reason having nothing to do with its provenance.
    """
    found = {}
    with open(path, "rb") as handle:
        for kind, payload in _walk(handle):
            if kind == b"tEXt":
                name, _, value = payload.partition(b"\x00")
                found[name.decode("latin-1")] = value
            elif kind == b"zTXt":
                name, _, rest = payload.partition(b"\x00")
                if rest[:1] == b"\x00":
                    try:
                        found[name.decode("latin-1")] = _inflate(rest[1:])
                    except zlib.error:
                        pass
            elif kind == b"iTXt":
                name, _, rest = payload.partition(b"\x00")
                value = _itxt_value(rest)
                if value is not None:
                    found[name.decode("latin-1")] = value
    return found
