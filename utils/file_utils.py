"""File helpers"""
import os
from typing import List

from config import MEDIA_EXTENSIONS
from utils.naturalsort import natural_key, path_key  # noqa: F401  (natural_key re-exported for callers)


def is_media_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in MEDIA_EXTENSIONS


def collect_media_files(paths: List[str]) -> List[str]:
    """Expand files/folders (recursively) into a naturally sorted list of media files"""
    found = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for fn in files:
                    full = os.path.join(root, fn)
                    if is_media_file(full):
                        found.append(full)
        elif os.path.isfile(p) and is_media_file(p):
            found.append(p)
    found.sort(key=path_key)
    return found


def format_duration(seconds) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:
        return "--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
