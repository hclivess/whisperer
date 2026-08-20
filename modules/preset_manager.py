"""Presets: save/load/import/export settings as JSON (presets/ next to the app) + user defaults"""
import json
import os
import sys
from typing import Dict, Optional

from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from config import QUALITY_PRESETS, DEFAULT_SETTINGS


def _app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PresetManager:
    def __init__(self, main_window):
        self.main_window = main_window
        self.presets_dir = os.path.join(_app_dir(), "presets")
        os.makedirs(self.presets_dir, exist_ok=True)
        self.defaults_path = os.path.join(self.presets_dir, "defaults.json")

    # -- helpers -------------------------------------------------------------
    @property
    def ui(self):
        return self.main_window.ui_manager

    def apply_settings(self, settings: Dict):
        self.ui.apply_settings(settings)

    def _read(self, path: str) -> Optional[Dict]:
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            QMessageBox.warning(self.main_window, "Preset", f"Could not read {path}:\n{exc}")
            return None

    def _write(self, path: str, data: Dict) -> bool:
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            return True
        except Exception as exc:
            QMessageBox.warning(self.main_window, "Preset", f"Could not write {path}:\n{exc}")
            return False

    def _list_presets(self):
        return sorted(f[:-5] for f in os.listdir(self.presets_dir)
                      if f.endswith(".json") and f != "defaults.json")

    # -- built-in ------------------------------------------------------------
    def apply_preset(self, key: str):
        preset = QUALITY_PRESETS.get(key)
        if preset:
            self.apply_settings(preset["settings"])
            self.ui.update_status(f"Preset applied: {preset['name']}")

    # -- user presets --------------------------------------------------------
    def save_preset(self):
        name, ok = QInputDialog.getText(self.main_window, "Save Preset", "Preset name:")
        if ok and name.strip():
            path = os.path.join(self.presets_dir, name.strip() + ".json")
            if self._write(path, self.ui.get_current_settings()):
                self.ui.update_status(f"Preset saved: {name.strip()}")

    def load_preset(self):
        names = self._list_presets()
        if not names:
            QMessageBox.information(self.main_window, "Load Preset", "No saved presets yet.")
            return
        name, ok = QInputDialog.getItem(self.main_window, "Load Preset", "Preset:", names, 0, False)
        if ok and name:
            data = self._read(os.path.join(self.presets_dir, name + ".json"))
            if data:
                self.apply_settings(data)
                self.ui.update_status(f"Preset loaded: {name}")

    def delete_preset(self):
        names = self._list_presets()
        if not names:
            QMessageBox.information(self.main_window, "Delete Preset", "No saved presets.")
            return
        name, ok = QInputDialog.getItem(self.main_window, "Delete Preset", "Preset:", names, 0, False)
        if ok and name:
            os.remove(os.path.join(self.presets_dir, name + ".json"))
            self.ui.update_status(f"Preset deleted: {name}")

    def import_preset(self):
        path, _ = QFileDialog.getOpenFileName(self.main_window, "Import Preset", "", "JSON (*.json)")
        if path:
            data = self._read(path)
            if data:
                self.apply_settings(data)
                dest = os.path.join(self.presets_dir, os.path.basename(path))
                if not os.path.exists(dest):
                    self._write(dest, data)
                self.ui.update_status(f"Preset imported: {os.path.basename(path)}")

    def export_preset(self):
        path, _ = QFileDialog.getSaveFileName(self.main_window, "Export Preset", "preset.json", "JSON (*.json)")
        if path:
            if self._write(path, self.ui.get_current_settings()):
                self.ui.update_status(f"Preset exported: {path}")

    # -- defaults ------------------------------------------------------------
    def load_defaults(self) -> Optional[Dict]:
        if os.path.exists(self.defaults_path):
            try:
                with open(self.defaults_path, encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                return None
        return None

    def save_as_defaults(self):
        if self._write(self.defaults_path, self.ui.get_current_settings()):
            self.ui.update_status("Current settings saved as defaults")

    def reset_defaults(self):
        if os.path.exists(self.defaults_path):
            os.remove(self.defaults_path)
        self.apply_settings(DEFAULT_SETTINGS)
        self.ui.update_status("Factory defaults restored")
