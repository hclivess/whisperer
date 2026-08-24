"""
The extra models the Model dropdown offers: a built-in list, overridden by models.json
next to the app so a model nobody has heard of yet can be added without touching the code.
"""
import json
import os
import sys
from typing import Dict, List

from config import MODEL_CATALOG, MODEL_CATALOG_FILE


def _app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def catalog_path() -> str:
    return os.path.join(_app_dir(), MODEL_CATALOG_FILE)


def _clean(entries) -> List[Dict]:
    """Accepts ["owner/repo", ...] or [{"id": ..., "label": ..., "language": ...}, ...]"""
    out = []
    for entry in entries or []:
        if isinstance(entry, str):
            entry = {"id": entry}
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id", "")).strip()
        if not model_id:
            continue
        out.append({
            "id": model_id,
            "label": str(entry.get("label", "") or model_id).strip(),
            "language": str(entry.get("language", "") or "").strip(),
        })
    return out


def write_default_catalog(path: str = "") -> str:
    """Put the built-in list on disk as an editable file; returns the path (empty if it failed)"""
    path = path or catalog_path()
    data = {
        "_readme": [
            "Extra faster-whisper models for the Model dropdown. Any CTranslate2 Whisper model on "
            "the Hugging Face Hub works: give its repo id (owner/name), or a full path to a local "
            "folder holding model.bin. label is what the tooltip shows; language (optional) is the "
            "language selected for you when you pick the model. Delete entries you do not want.",
        ],
        "models": MODEL_CATALOG,
    }
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        return path
    except Exception:
        return ""


def load_catalog() -> List[Dict]:
    """models.json if the user has one, otherwise the built-in list (written out on first run)"""
    path = catalog_path()
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            entries = data.get("models") if isinstance(data, dict) else data
            cleaned = _clean(entries)
            if cleaned:
                return cleaned
        except Exception:
            pass          # a broken or emptied file falls back to the built-in list
    else:
        write_default_catalog(path)
    return _clean(MODEL_CATALOG)
