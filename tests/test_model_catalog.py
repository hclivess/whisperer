import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_CATALOG  # noqa: E402
from modules import model_catalog  # noqa: E402


def _use(tmp_path, monkeypatch):
    path = str(tmp_path / "models.json")
    monkeypatch.setattr(model_catalog, "catalog_path", lambda: path)
    return path


def test_builtin_catalog_is_well_formed():
    for entry in MODEL_CATALOG:
        assert "/" in entry["id"], entry           # a Hub repo id, not a size name
        assert entry["label"] and entry["language"]


def test_first_run_writes_the_file_and_returns_the_builtins(tmp_path, monkeypatch):
    path = _use(tmp_path, monkeypatch)
    assert [e["id"] for e in model_catalog.load_catalog()] == [e["id"] for e in MODEL_CATALOG]
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh)["models"][0]["id"] == MODEL_CATALOG[0]["id"]


def test_the_file_wins_and_bare_strings_are_accepted(tmp_path, monkeypatch):
    path = _use(tmp_path, monkeypatch)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"models": ["owner/only-an-id",
                              {"id": " spaced/id ", "label": "Mine", "language": "cs"},
                              {"label": "no id at all"}]}, fh)
    assert model_catalog.load_catalog() == [
        {"id": "owner/only-an-id", "label": "owner/only-an-id", "language": ""},
        {"id": "spaced/id", "label": "Mine", "language": "cs"},
    ]


def test_a_bare_list_is_accepted_too(tmp_path, monkeypatch):
    path = _use(tmp_path, monkeypatch)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(["owner/one"], fh)
    assert [e["id"] for e in model_catalog.load_catalog()] == ["owner/one"]


def test_a_broken_or_emptied_file_falls_back_to_the_builtins(tmp_path, monkeypatch):
    path = _use(tmp_path, monkeypatch)
    for content in ("{ not json", '{"models": []}'):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        assert [e["id"] for e in model_catalog.load_catalog()] == [e["id"] for e in MODEL_CATALOG]
