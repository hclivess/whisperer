#!/usr/bin/env python3
"""
whisperer - batch subtitle generator GUI (Whisper via faster-whisper / whisper.cpp)
"""
import os
import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PySide6.QtCore import QSettings
from PySide6.QtGui import QIcon

from modules.ui_manager import UIManager
from modules.file_manager import FileManager
from modules.process_manager import ProcessManager
from modules.preset_manager import PresetManager
from utils.ffmpeg_utils import check_ffmpeg_status
from config import APP_NAME, APP_VERSION, WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setAcceptDrops(True)
        self.settings = QSettings(APP_NAME, "Settings")
        self.setMinimumWidth(WINDOW_MIN_WIDTH)
        self.setMinimumHeight(WINDOW_MIN_HEIGHT)

        self.file_manager = FileManager()
        self.process_manager = ProcessManager(self)
        self.preset_manager = PresetManager(self)
        self.ui_manager = UIManager(self)
        self.ui_manager.setup_ui()
        self.connect_signals()
        self.load_settings()
        self.check_dependencies()

    def connect_signals(self):
        fm, pm, ui = self.file_manager, self.process_manager, self.ui_manager
        fm.files_updated.connect(ui.update_file_list)
        fm.file_count_changed.connect(ui.update_file_count)
        fm.files_updated.connect(self._on_files_updated_during_processing)

        pm.progress_updated.connect(ui.update_progress)
        pm.file_progress_updated.connect(ui.update_file_progress)
        pm.status_updated.connect(ui.update_status)
        pm.stats_updated.connect(ui.update_stats)
        pm.file_state_changed.connect(ui.set_file_state)
        pm.segment_ready.connect(ui.append_segment)
        pm.file_started.connect(ui.on_file_started)
        pm.paused_state_changed.connect(ui.set_paused_state)
        pm.processing_finished.connect(self.on_processing_finished)

        ui.start_processing.connect(self.start_processing)
        ui.stop_processing.connect(self.stop_processing)
        ui.pause_clicked.connect(self.toggle_pause)
        ui.files_added.connect(fm.add_files)
        ui.files_removed.connect(self._on_files_removed)
        ui.queue_cleared.connect(fm.clear_queue)
        ui.files_reordered.connect(fm.move_file)

    def check_dependencies(self):
        available = check_ffmpeg_status()
        self.ui_manager.update_ffmpeg_status(available)
        if not available:
            QMessageBox.warning(self, "FFmpeg Not Found",
                                "FFmpeg was not found in your PATH or next to the application.\n"
                                "faster-whisper can still decode most files on its own, but whisper.cpp, "
                                "subtitle embedding and duration/ETA display need FFmpeg.")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()

    def dropEvent(self, event):
        self.file_manager.add_files([u.toLocalFile() for u in event.mimeData().urls()])

    def start_processing(self):
        if not self.file_manager.has_files():
            QMessageBox.warning(self, "No Files", "Please add video or audio files first.")
            return
        settings = self.ui_manager.get_current_settings()
        issues = self.process_manager.validate_settings(settings)
        if issues:
            reply = QMessageBox.warning(
                self, "Check Settings",
                "The current settings have potential problems:\n\n• " + "\n• ".join(issues)
                + "\n\nStart anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.process_manager.start_processing(self.file_manager.get_queue(), settings)
        self.ui_manager.set_processing_state(True)

    def stop_processing(self):
        self.process_manager.stop_processing()
        self.ui_manager.set_processing_state(False)

    def toggle_pause(self):
        if self.process_manager.is_paused():
            self.process_manager.resume_processing()
        else:
            self.process_manager.pause_processing()

    def _on_files_updated_during_processing(self, files):
        if self.process_manager.is_processing():
            self.process_manager.sync_pending(files)

    def _on_files_removed(self, indices):
        if self.process_manager.is_processing():
            current = self.process_manager.current_file_index
            safe = [i for i in indices if i > current]
            if len(safe) != len(indices):
                QMessageBox.warning(self, "Cannot Remove",
                                    f"{len(indices) - len(safe)} file(s) already processed or in progress.")
            if safe:
                self.file_manager.remove_files(safe)
        else:
            self.file_manager.remove_files(indices)

    def on_processing_finished(self, success_count, total_count):
        self.ui_manager.set_processing_state(False)
        self.ui_manager.update_status(f"Finished: {success_count} of {total_count} file(s) subtitled")
        QMessageBox.information(self, "Done", f"Subtitles generated for {success_count} of {total_count} file(s).")

    def load_settings(self):
        user_defaults = self.preset_manager.load_defaults()
        if user_defaults:
            self.preset_manager.apply_settings(user_defaults)
        self.ui_manager.load_settings(self.settings)

    def save_settings(self):
        self.ui_manager.save_settings(self.settings)

    def closeEvent(self, event):
        if self.process_manager.is_processing():
            reply = QMessageBox.question(self, "Transcription in Progress",
                                         "A transcription is still running. Quit anyway?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self.process_manager.stop_processing()
        self.save_settings()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    window = MainWindow()
    window.show()
    selftest = os.environ.get("WHISPERER_SELFTEST")
    if selftest:
        _run_selftest(window, selftest)
    sys.exit(app.exec())


def _run_selftest(window: "MainWindow", path: str):
    """Headless smoke test used by CI / packaging: transcribe one file with tiny.en and exit (0 = ok)"""
    from PySide6.QtCore import QTimer
    window.preset_manager.apply_settings({"engine": "faster_whisper", "model": "tiny.en", "device": "cpu",
                                          "formats": ["srt"], "overwrite": True, "beam_size": 1})
    window.file_manager.add_files([path])
    window.process_manager.status_updated.connect(lambda m: print("selftest:", m, flush=True))

    def finished(ok, total):
        print(f"selftest: finished {ok}/{total}", flush=True)
        QTimer.singleShot(0, lambda: sys.exit(0 if ok == total else 1))
    window.process_manager.processing_finished.disconnect(window.on_processing_finished)
    window.process_manager.processing_finished.connect(finished)
    QTimer.singleShot(200, lambda: window.process_manager.start_processing(
        window.file_manager.get_queue(), window.ui_manager.get_current_settings()))


if __name__ == "__main__":
    main()
