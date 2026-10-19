"""Numeric release versions shared by the Windows packaging tools."""
from __future__ import annotations

import re


_RELEASE_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+){2,3}")


def parse_release_version(version: str) -> tuple[int, ...]:
    """Return three or four unsigned 16-bit components, or raise ValueError."""
    if not isinstance(version, str) or _RELEASE_VERSION.fullmatch(version) is None:
        raise ValueError("Release version requires three or four numeric parts")
    parts = tuple(int(part) for part in version.split("."))
    if any(part > 65535 for part in parts):
        raise ValueError("Release version components must be between 0 and 65535")
    return parts


def windows_version_tuple(version: str) -> tuple[int, int, int, int]:
    """Pad a three-part release with revision zero; retain a fourth component."""
    parts = parse_release_version(version)
    return parts + (0,) * (4 - len(parts))


def windows_file_version(version: str) -> str:
    """Return the same four-part numeric form for PE and NSIS metadata."""
    return ".".join(str(part) for part in windows_version_tuple(version))
