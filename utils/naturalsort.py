"""Natural ("human") sort: img2 < img10, Episode 9 < Episode 10, case-insensitive. Use it everywhere files are listed."""
import os
import re
from typing import Iterable, List

_CHUNK = re.compile(r"(\d+)")


def natural_key(text: str):
    """sorted(names, key=natural_key)"""
    return [int(t) if t.isdigit() else t.casefold() for t in _CHUNK.split(os.fspath(text))]


def path_key(path: str):
    """Sort full paths folder-by-folder so siblings stay together and numbers inside any component count."""
    return [natural_key(part) for part in os.path.normpath(os.fspath(path)).split(os.sep)]


def natural_sorted(items: Iterable[str]) -> List[str]:
    return sorted(items, key=path_key)
