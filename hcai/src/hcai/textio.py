"""Encoding-robust text reading. Windows PowerShell 5 `>` writes UTF-16 LE with a BOM, Notepad
may add a UTF-8 BOM, and legacy exports use cp1252; all of them must load without a crash."""

from __future__ import annotations

from pathlib import Path


def decode_bytes(raw: bytes) -> str:
    """Decode bytes from UTF-8 (+/- BOM), UTF-16 (BOM or NUL pattern) or cp1252."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")  # BOM selects endianness and is stripped
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    if len(raw) >= 4 and raw[1:2] == b"\x00" and raw[3:4] == b"\x00":  # BOM-less UTF-16 LE
        return raw.decode("utf-16-le")
    for enc in ("utf-8", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_text(path: Path) -> str:
    return decode_bytes(Path(path).read_bytes())
