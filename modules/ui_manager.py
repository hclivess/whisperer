"""
UI Manager for whisperer — builds the window (queue + progress on the left, settings tabs +
live transcript on the right) and translates between widgets and the settings dict.
"""
import os
from typing import Any, Dict, List

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton,
                               QLineEdit, QFileDialog, QGroupBox, QListWidget, QProgressBar, QSplitter,
                               QListWidgetItem, QTabWidget, QSpinBox, QDoubleSpinBox, QComboBox,
                               QGridLayout, QFrame, QSizePolicy, QMessageBox, QPlainTextEdit, QTextEdit,
                               QFormLayout, QScrollArea)
from PySide6.QtCore import Qt, Signal, QSettings
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QColor, QBrush, QDesktopServices
from PySide6.QtCore import QUrl

from config import (APP_NAME, APP_VERSION, ENGINES, MODEL_SIZES, DEVICES, COMPUTE_TYPES, LANGUAGES,
                    TASKS, SUBTITLE_FORMATS, MUX_CONTAINERS, DEFAULT_SETTINGS, QUALITY_PRESETS,
                    MEDIA_EXTENSIONS, SYNC_MODES)
from modules.backends import cuda_available, faster_whisper_available, is_model_cached, model_repo_id
from utils.cuda_utils import cuda_status, download_cuda_libraries, default_cuda_dir
from PySide6.QtCore import QThread
from utils.file_utils import format_duration


class FileListWidget(QListWidget):
    """List with external drag & drop of files/folders and internal reordering"""
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setAlternatingRowColors(False)
        self.setStyleSheet("""
            QListWidget { border: 2px solid #aaa; border-radius: 5px; padding: 5px; background-color: #f9f9f9; }
            QListWidget::item { border-bottom: 1px solid #eee; padding: 8px; margin: 2px; border-radius: 3px; }
            QListWidget::item:selected { background-color: #0078D7; color: white; }
            QListWidget::item:hover { background-color: #e3f2fd; }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            event.accept()
            self.files_dropped.emit([u.toLocalFile() for u in event.mimeData().urls()])
        else:
            super().dropEvent(event)


_PILL = "QLabel { background: %s; color: %s; border-radius: 8px; padding: 1px 8px; font-size: 11px; font-weight: bold; }"
_BAR_STYLE = """
    QProgressBar { border: 1px solid #c8c8c8; border-radius: 4px; background: #f0f0f0;
                   text-align: center; height: 18px; font-weight: bold; color: #333; }
    QProgressBar::chunk { background-color: %s; border-radius: 3px; }
"""
_BTN_STYLE = """
    QPushButton { font-size: 14px; font-weight: bold; padding: 8px; background-color: %s; color: %s; border-radius: 5px; }
    QPushButton:hover { background-color: %s; }
    QPushButton:disabled { background-color: #cccccc; color: #666; }
"""


class UIManager(QWidget):
    start_processing = Signal()
    stop_processing = Signal()
    pause_clicked = Signal()
    files_added = Signal(list)
    files_removed = Signal(list)
    queue_cleared = Signal()
    files_reordered = Signal(int, int)

    _FILE_STATE_COLORS = {
        "pending": None,
        "extracting": QColor("#fff3cd"),
        "processing": QColor("#cce5ff"),
        "done": QColor("#d4edda"),
        "error": QColor("#f8d7da"),
        "skipped": QColor("#e2e3e5"),
    }

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.controls: Dict[str, Any] = {}
        self._files = []
        self._processing_active = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def setup_ui(self):
        self._create_menu_bar()
        self._create_main_layout()
        self._create_status_bar()
        self._on_engine_changed()
        self._on_output_mode_changed()

    def _add_action(self, menu, text, shortcut, slot):
        action = QAction(text, self.main_window)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _create_menu_bar(self):
        menubar = self.main_window.menuBar()
        file_menu = menubar.addMenu("File")
        self._add_action(file_menu, "Add Files", "Ctrl+O", self._on_add_files)
        self._add_action(file_menu, "Add Folder", "Ctrl+Shift+O", self._on_add_folder)
        file_menu.addSeparator()
        self._add_action(file_menu, "Clear Queue", None, self._on_clear_queue)
        file_menu.addSeparator()
        self._add_action(file_menu, "Exit", "Ctrl+Q", self.main_window.close)

        pm = self.main_window.preset_manager
        presets_menu = menubar.addMenu("Presets")
        self._add_action(presets_menu, "Save Current Settings", None, pm.save_preset)
        self._add_action(presets_menu, "Load Preset", None, pm.load_preset)
        presets_menu.addSeparator()
        for key, preset in QUALITY_PRESETS.items():
            self._add_action(presets_menu, preset["name"], None, lambda checked=False, k=key: pm.apply_preset(k))
        presets_menu.addSeparator()
        self._add_action(presets_menu, "Import Preset…", None, pm.import_preset)
        self._add_action(presets_menu, "Export Preset…", None, pm.export_preset)
        self._add_action(presets_menu, "Delete Preset…", None, pm.delete_preset)
        presets_menu.addSeparator()
        self._add_action(presets_menu, "Save Current as Defaults", None, pm.save_as_defaults)
        self._add_action(presets_menu, "Reset to Factory Defaults", None, pm.reset_defaults)

        help_menu = menubar.addMenu("Help")
        self._add_action(help_menu, "Model download / cache info", None, self._show_model_info)
        self._add_action(help_menu, "About", None, self._show_about)

    def _create_main_layout(self):
        main_widget = QWidget()
        self.main_window.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([480, 720])

    # -- left: queue + progress + buttons ----------------------------------
    def _create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        files_group = QGroupBox("Input Files (video or audio)")
        files_layout = QVBoxLayout()
        self.file_count_label = QLabel("0 files in queue")
        files_layout.addWidget(self.file_count_label)
        self.file_list = FileListWidget()
        self.file_list.files_dropped.connect(self.files_added)
        self.file_list.model().rowsMoved.connect(self._on_rows_moved)
        self.file_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        files_layout.addWidget(self.file_list)

        file_controls = QHBoxLayout()
        for key, text, slot in (("add_files", "Add Files", self._on_add_files),
                                ("add_folder", "Add Folder", self._on_add_folder),
                                ("remove_files", "Remove", self._on_remove_files),
                                ("clear_files", "Clear All", self._on_clear_queue)):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            self.controls[key] = btn
            file_controls.addWidget(btn)
        files_layout.addLayout(file_controls)
        files_group.setLayout(files_layout)
        layout.addWidget(files_group, 1)

        layout.addWidget(self._create_progress_group())

        buttons = QHBoxLayout()
        self.controls["start"] = QPushButton("Start Transcribing")
        self.controls["start"].clicked.connect(self.start_processing)
        self.controls["start"].setStyleSheet(_BTN_STYLE % ("#28a745", "white", "#218838"))
        self.controls["pause"] = QPushButton("Pause")
        self.controls["pause"].clicked.connect(self.pause_clicked)
        self.controls["pause"].setEnabled(False)
        self.controls["pause"].setStyleSheet(_BTN_STYLE % ("#ffc107", "#333", "#e0a800"))
        self.controls["stop"] = QPushButton("Stop")
        self.controls["stop"].clicked.connect(self.stop_processing)
        self.controls["stop"].setEnabled(False)
        self.controls["stop"].setStyleSheet(_BTN_STYLE % ("#dc3545", "white", "#c82333"))
        for k in ("start", "pause", "stop"):
            buttons.addWidget(self.controls[k])
        layout.addLayout(buttons)
        return panel

    def _make_stat_tile(self, title: str) -> QLabel:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet("QFrame { background: #fafafa; border: 1px solid #e0e0e0; border-radius: 4px; }")
        box = QVBoxLayout(frame)
        box.setContentsMargins(8, 4, 8, 4)
        box.setSpacing(0)
        caption = QLabel(title)
        caption.setStyleSheet("color: #888; font-size: 10px; border: none;")
        value = QLabel("--")
        value.setStyleSheet("font-weight: bold; font-size: 13px; border: none;")
        box.addWidget(caption)
        box.addWidget(value)
        value.tile = frame
        return value

    def _create_progress_group(self) -> QGroupBox:
        group = QGroupBox("Progress")
        layout = QVBoxLayout()
        layout.setSpacing(6)

        header = QHBoxLayout()
        self.phase_label = QLabel("Idle")
        self.phase_label.setStyleSheet(_PILL % ("#e0e0e0", "#444"))
        self.current_file_label = QLabel("Ready — add files and press Start")
        self.current_file_label.setStyleSheet("font-weight: bold;")
        self.current_file_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.counter_label = QLabel("")
        self.counter_label.setStyleSheet("color: #666;")
        header.addWidget(self.phase_label)
        header.addWidget(self.current_file_label, 1)
        header.addWidget(self.counter_label)
        layout.addLayout(header)

        self.file_progress_bar = QProgressBar()
        self.file_progress_bar.setRange(0, 1000)
        self.file_progress_bar.setFormat("%p%")
        self.file_progress_bar.setStyleSheet(_BAR_STYLE % "#4a90d9")
        layout.addWidget(self.file_progress_bar)
        file_line = QHBoxLayout()
        self.file_eta_label = QLabel("File ETA: --")
        self.file_elapsed_label = QLabel("Elapsed: --")
        self.file_elapsed_label.setStyleSheet("color: #666;")
        file_line.addWidget(self.file_eta_label)
        file_line.addStretch()
        file_line.addWidget(self.file_elapsed_label)
        layout.addLayout(file_line)

        overall_caption = QLabel("Overall")
        overall_caption.setStyleSheet("color: #666; font-size: 11px; margin-top: 4px;")
        layout.addWidget(overall_caption)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet(_BAR_STYLE % "#28a745")
        layout.addWidget(self.progress_bar)
        total_line = QHBoxLayout()
        self.total_elapsed_label = QLabel("Total elapsed: --")
        self.total_elapsed_label.setStyleSheet("color: #666;")
        total_line.addStretch()
        total_line.addWidget(self.total_elapsed_label)
        layout.addLayout(total_line)

        tiles = QHBoxLayout()
        tiles.setSpacing(6)
        self.stat_speed = self._make_stat_tile("Speed")
        self.stat_position = self._make_stat_tile("Position")
        self.stat_duration = self._make_stat_tile("Duration")
        self.stat_segments = self._make_stat_tile("Segments")
        self.stat_device = self._make_stat_tile("Device")
        for t in (self.stat_speed, self.stat_position, self.stat_duration, self.stat_segments, self.stat_device):
            tiles.addWidget(t.tile, 1)
        layout.addLayout(tiles)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #555; font-size: 11px;")
        self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.status_label)
        group.setLayout(layout)
        return group

    # -- right: settings tabs + transcript ----------------------------------
    def _create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.tabs = QTabWidget()
        for widget, title in ((self._create_model_tab(), "Model"),
                              (self._create_transcription_tab(), "Transcription"),
                              (self._create_sync_tab(), "Sync"),
                              (self._create_subtitles_tab(), "Subtitles"),
                              (self._create_advanced_tab(), "Advanced")):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setWidget(widget)
            self.tabs.addTab(scroll, title)
        self._fix_field_sizing()
        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.addWidget(self.tabs)

        transcript_group = QGroupBox("Live Transcript")
        tl = QVBoxLayout()
        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setStyleSheet("QTextEdit { font-family: monospace; font-size: 12px; background: #fcfcfc; }")
        tl.addWidget(self.transcript)
        trow = QHBoxLayout()
        self.transcript_info = QLabel("")
        self.transcript_info.setStyleSheet("color: #666; font-size: 11px;")
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.transcript.clear)
        trow.addWidget(self.transcript_info, 1)
        trow.addWidget(clear_btn)
        tl.addLayout(trow)
        transcript_group.setLayout(tl)
        self.transcript.setMinimumHeight(60)
        vsplit.addWidget(transcript_group)
        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 1)
        vsplit.setSizes([520, 200])
        layout.addWidget(vsplit)
        return panel

    def _fix_field_sizing(self):
        """Let combos / line edits shrink with the window instead of forcing the width of their longest entry,
        and keep them comfortably tall on any DPI"""
        for combo in self.tabs.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            combo.setMinimumHeight(28)
        for edit in self.tabs.findChildren(QLineEdit):
            edit.setMinimumWidth(80)
            edit.setMinimumHeight(28)
        for spin in self.tabs.findChildren(QSpinBox) + self.tabs.findChildren(QDoubleSpinBox):
            spin.setMinimumHeight(28)
            spin.setMinimumWidth(90)
        for grid in self.tabs.findChildren(QGridLayout):
            grid.setColumnStretch(1, 1)
        for label in self.tabs.findChildren(QLabel):
            if label.wordWrap():
                # wrapped labels must not dictate the minimum width of the panel
                label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
                label.setMinimumWidth(0)
        for btn in self.tabs.findChildren(QPushButton):
            btn.setMinimumHeight(28)

    def _browse_dir(self, line_edit: QLineEdit, title: str):
        d = QFileDialog.getExistingDirectory(self.main_window, title, line_edit.text() or os.path.expanduser("~"))
        if d:
            line_edit.setText(d)

    def _browse_file(self, line_edit: QLineEdit, title: str, filt: str = "All files (*)"):
        f, _ = QFileDialog.getOpenFileName(self.main_window, title, line_edit.text() or "", filt)
        if f:
            line_edit.setText(f)

    def _path_row(self, line_edit: QLineEdit, slot) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(line_edit, 1)
        btn = QPushButton("Browse…")
        btn.clicked.connect(slot)
        row.addWidget(btn)
        return row

    def _create_model_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        engine_group = QGroupBox("Engine")
        form = QFormLayout()
        self.controls["engine"] = QComboBox()
        for key, label in ENGINES.items():
            self.controls["engine"].addItem(label, key)
        self.controls["engine"].currentIndexChanged.connect(self._on_engine_changed)
        form.addRow("Engine:", self.controls["engine"])
        self.engine_hint = QLabel("")
        self.engine_hint.setWordWrap(True)
        self.engine_hint.setStyleSheet("color: #666; font-size: 11px;")
        form.addRow("", self.engine_hint)
        engine_group.setLayout(form)
        layout.addWidget(engine_group)

        model_group = QGroupBox("Model")
        grid = QGridLayout()
        self.controls["model"] = QComboBox()
        self.controls["model"].setEditable(True)
        self.controls["model"].addItems(MODEL_SIZES)
        self.controls["model"].setCurrentText(DEFAULT_SETTINGS["model"])
        self.controls["model"].setToolTip("Model size. *.en models are English-only and a bit faster/more accurate for English.\n"
                                          "faster-whisper: HuggingFace repo id also accepted.\n"
                                          "whisper.cpp: size name (looks for ggml-<size>.bin in the model folder) or a full path to a .bin file.")
        grid.addWidget(QLabel("Model:"), 0, 0)
        grid.addWidget(self.controls["model"], 0, 1)
        self.controls["device"] = QComboBox()
        self.controls["device"].addItems(DEVICES)
        self.controls["device"].setToolTip("auto = CUDA if available, otherwise CPU")
        grid.addWidget(QLabel("Device:"), 1, 0)
        grid.addWidget(self.controls["device"], 1, 1)
        self.controls["compute_type"] = QComboBox()
        self.controls["compute_type"].addItems(COMPUTE_TYPES)
        self.controls["compute_type"].setToolTip("default = float16 on CUDA, int8 on CPU")
        grid.addWidget(QLabel("Compute type:"), 2, 0)
        grid.addWidget(self.controls["compute_type"], 2, 1)
        self.controls["cpu_threads"] = QSpinBox()
        self.controls["cpu_threads"].setRange(0, 256)
        self.controls["cpu_threads"].setSpecialValueText("auto")
        grid.addWidget(QLabel("CPU threads:"), 3, 0)
        grid.addWidget(self.controls["cpu_threads"], 3, 1)
        self.controls["model_dir"] = QLineEdit()
        self.controls["model_dir"].setPlaceholderText("default cache / folder with ggml-*.bin files")
        grid.addWidget(QLabel("Model folder:"), 4, 0)
        grid.addLayout(self._path_row(self.controls["model_dir"],
                                      lambda: self._browse_dir(self.controls["model_dir"], "Model folder")), 4, 1)
        # model download / cache row
        dl_row = QHBoxLayout()
        self.model_cache_label = QLabel("")
        self.model_cache_label.setStyleSheet("color: #666; font-size: 11px;")
        self.model_cache_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.controls["download_model"] = QPushButton("Download / check model")
        self.controls["download_model"].setToolTip("Fetch the selected faster-whisper model now instead of at the first transcription")
        self.controls["download_model"].clicked.connect(self._on_download_model)
        dl_row.addWidget(self.model_cache_label, 1)
        dl_row.addWidget(self.controls["download_model"])
        grid.addLayout(dl_row, 5, 0, 1, 2)
        model_group.setLayout(grid)
        layout.addWidget(model_group)
        self.controls["model"].currentTextChanged.connect(self._refresh_model_cache_label)
        self.controls["model_dir"].editingFinished.connect(self._refresh_model_cache_label)

        gpu_group = QGroupBox("GPU acceleration (NVIDIA CUDA)")
        gl = QGridLayout()
        self.hw_label = QLabel("")
        self.hw_label.setWordWrap(True)
        self.hw_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        gl.addWidget(self.hw_label, 0, 0, 1, 2)
        self.controls["cuda_lib_dir"] = QLineEdit()
        self.controls["cuda_lib_dir"].setPlaceholderText("auto (pip packages, CUDA_PATH, 'cuda' folder next to the app)")
        self.controls["cuda_lib_dir"].setToolTip("Folder containing cuBLAS 12 and cuDNN 9 libraries "
                                                 "(cublas64_12.dll, cublasLt64_12.dll, cudnn64_9.dll / libcublas.so.12, libcudnn.so.9)")
        self.controls["cuda_lib_dir"].editingFinished.connect(self._refresh_gpu_status)
        gl.addWidget(QLabel("CUDA libraries folder:"), 1, 0)
        gl.addLayout(self._path_row(self.controls["cuda_lib_dir"],
                                    lambda: (self._browse_dir(self.controls["cuda_lib_dir"], "CUDA libraries folder"),
                                             self._refresh_gpu_status())), 1, 1)
        btn_row = QHBoxLayout()
        self.cuda_dl_label = QLabel("")
        self.cuda_dl_label.setStyleSheet("color: #666; font-size: 11px;")
        self.cuda_dl_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.controls["download_cuda"] = QPushButton("Download CUDA libraries")
        self.controls["download_cuda"].setToolTip("Fetch cuBLAS 12 + cuDNN 9 from PyPI (official NVIDIA wheels, ~1 GB) "
                                                  "into a 'cuda' folder next to the app")
        self.controls["download_cuda"].clicked.connect(self._on_download_cuda)
        recheck = QPushButton("Re-check GPU")
        recheck.clicked.connect(self._refresh_gpu_status)
        btn_row.addWidget(self.cuda_dl_label, 1)
        btn_row.addWidget(self.controls["download_cuda"])
        btn_row.addWidget(recheck)
        gl.addLayout(btn_row, 2, 0, 1, 2)
        gpu_group.setLayout(gl)
        layout.addWidget(gpu_group)
        layout.addStretch()
        return tab

    def _create_transcription_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        lang_group = QGroupBox("Language")
        form = QFormLayout()
        self.controls["language"] = QComboBox()
        for code, name in LANGUAGES.items():
            self.controls["language"].addItem(f"{name} ({code})", code)
        form.addRow("Spoken language:", self.controls["language"])
        self.controls["task"] = QComboBox()
        for key, label in TASKS.items():
            self.controls["task"].addItem(label, key)
        form.addRow("Task:", self.controls["task"])
        lang_group.setLayout(form)
        layout.addWidget(lang_group)

        dec_group = QGroupBox("Decoding")
        grid = QGridLayout()
        self.controls["beam_size"] = QSpinBox()
        self.controls["beam_size"].setRange(1, 10)
        self.controls["beam_size"].setValue(5)
        self.controls["beam_size"].setToolTip("1 = greedy (fastest), 5 = default quality")
        grid.addWidget(QLabel("Beam size:"), 0, 0)
        grid.addWidget(self.controls["beam_size"], 0, 1)
        self.controls["vad_filter"] = QCheckBox("VAD filter (skip silence)")
        self.controls["vad_filter"].setChecked(True)
        grid.addWidget(self.controls["vad_filter"], 1, 0, 1, 2)
        self.controls["vad_min_silence_ms"] = QSpinBox()
        self.controls["vad_min_silence_ms"].setRange(50, 10000)
        self.controls["vad_min_silence_ms"].setSingleStep(50)
        self.controls["vad_min_silence_ms"].setSuffix(" ms")
        self.controls["vad_min_silence_ms"].setValue(500)
        grid.addWidget(QLabel("VAD min. silence:"), 2, 0)
        grid.addWidget(self.controls["vad_min_silence_ms"], 2, 1)
        self.controls["word_timestamps"] = QCheckBox("Word timestamps")
        self.controls["word_timestamps"].setToolTip("Per-word timing: more precise cue splitting, a bit slower")
        grid.addWidget(self.controls["word_timestamps"], 3, 0, 1, 2)
        self.controls["condition_on_previous_text"] = QCheckBox("Condition on previous text")
        self.controls["condition_on_previous_text"].setToolTip("Use the previous segment as context: better consistency, "
                                                               "but can get stuck repeating on bad audio")
        self.controls["condition_on_previous_text"].setChecked(True)
        grid.addWidget(self.controls["condition_on_previous_text"], 4, 0, 1, 2)
        self.controls["second_pass"] = QCheckBox("Second pass: transcribe twice and keep what both agree on")
        self.controls["second_pass"].setChecked(True)
        self.controls["second_pass"].setToolTip(
            "Whisper invents differently every time it is asked, but real speech comes back the same, so a "
            "second decode is evidence about the first. Cues both passes produce are kept, a cue the second "
            "pass heard nothing under (and the VAD finds no speech in) is dropped as hallucinated, and speech "
            "the first pass skipped is recovered. Timing is never touched. Costs one more decode of the file.")
        grid.addWidget(self.controls["second_pass"], 5, 0, 1, 2)
        dec_group.setLayout(grid)
        layout.addWidget(dec_group)

        prompt_group = QGroupBox("Initial prompt (names, jargon, punctuation style)")
        pl = QVBoxLayout()
        self.controls["initial_prompt"] = QPlainTextEdit()
        self.controls["initial_prompt"].setPlaceholderText("e.g. Hello, welcome to the Videer podcast with Jan Kučera.")
        self.controls["initial_prompt"].setMaximumHeight(80)
        pl.addWidget(self.controls["initial_prompt"])
        prompt_group.setLayout(pl)
        layout.addWidget(prompt_group)
        layout.addStretch()
        return tab

    def _create_sync_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        mode_group = QGroupBox("What to do")
        mg = QVBoxLayout()
        self.controls["sync_mode"] = QComboBox()
        for key, label in SYNC_MODES.items():
            self.controls["sync_mode"].addItem(label, key)
        self.controls["sync_mode"].currentIndexChanged.connect(self._on_sync_mode_changed)
        mg.addWidget(self.controls["sync_mode"])
        self.resync_widget = QWidget()
        rg = QGridLayout(self.resync_widget)
        rg.setContentsMargins(0, 0, 0, 0)
        self.controls["resync_file"] = QLineEdit()
        self.controls["resync_file"].setPlaceholderText("auto: <video>.srt / .vtt next to the file")
        self.controls["resync_file"].setToolTip("Subtitle file to align. Leave empty to pick the .srt/.vtt with the "
                                                "video's name. The result is written as <video>.synced.srt.")
        rg.addWidget(QLabel("Subtitle file:"), 0, 0)
        rg.addLayout(self._path_row(self.controls["resync_file"],
                                    lambda: self._browse_file(self.controls["resync_file"], "Subtitle file to resync",
                                                              "Subtitles (*.srt *.vtt);;All files (*)")), 0, 1)
        self.controls["resync_fit_speed"] = QCheckBox("Also fit speed (frame-rate drift, 23.976 ↔ 25 fps)")
        self.controls["resync_fit_speed"].setChecked(True)
        self.controls["resync_fit_speed"].setToolTip("Off: a single constant offset is fitted. On: offset and a speed "
                                                     "factor, as SubSync does; needed when subtitles drift over time.")
        rg.addWidget(self.controls["resync_fit_speed"], 1, 0, 1, 2)
        mg.addWidget(self.resync_widget)
        hint = QLabel("Resync transcribes the audio, matches the words of the existing subtitles against it, fits "
                      "offset + speed robustly and rewrites every cue — the text is never changed.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 11px;")
        mg.addWidget(hint)
        mode_group.setLayout(mg)
        layout.addWidget(mode_group)

        snap_group = QGroupBox("Timing")
        grid = QGridLayout()
        self.controls["snap_to_speech"] = QCheckBox("Snap cues to detected speech (recommended)")
        self.controls["snap_to_speech"].setChecked(True)
        self.controls["snap_to_speech"].setToolTip(
            "Runs a voice-activity detector over the audio and moves every cue's start and end onto the nearest "
            "speech onset / offset. Whisper's own timestamps are 20 ms decoder guesses that drift at segment edges; "
            "the detector sees the actual waveform. Also turns on word timestamps for the engine.")
        grid.addWidget(self.controls["snap_to_speech"], 0, 0, 1, 2)
        self.controls["snap_max_shift_ms"] = QSpinBox()
        self.controls["snap_max_shift_ms"].setRange(50, 3000)
        self.controls["snap_max_shift_ms"].setSingleStep(50)
        self.controls["snap_max_shift_ms"].setSuffix(" ms")
        self.controls["snap_max_shift_ms"].setValue(600)
        self.controls["snap_max_shift_ms"].setToolTip("How far a cue edge may be moved to reach speech")
        grid.addWidget(QLabel("Max shift:"), 1, 0)
        grid.addWidget(self.controls["snap_max_shift_ms"], 1, 1)
        self.controls["end_padding_ms"] = QSpinBox()
        self.controls["end_padding_ms"].setRange(0, 2000)
        self.controls["end_padding_ms"].setSingleStep(50)
        self.controls["end_padding_ms"].setSuffix(" ms")
        self.controls["end_padding_ms"].setValue(200)
        self.controls["end_padding_ms"].setToolTip("Keep the cue visible this long after speech ends (never into the next cue)")
        grid.addWidget(QLabel("Hold after speech:"), 2, 0)
        grid.addWidget(self.controls["end_padding_ms"], 2, 1)
        self.controls["min_cue_ms"] = QSpinBox()
        self.controls["min_cue_ms"].setRange(100, 5000)
        self.controls["min_cue_ms"].setSingleStep(100)
        self.controls["min_cue_ms"].setSuffix(" ms")
        self.controls["min_cue_ms"].setValue(800)
        self.controls["min_cue_ms"].setToolTip(
            "No cue is written shorter than this while a neighbour still has time to spare - it is stretched "
            "into the free time around it and, failing that, borrows the rest from the cue before or after it. "
            "Applies in every mode, resync included, and needs no text to be rewritten.")
        grid.addWidget(QLabel("Min cue duration:"), 3, 0)
        grid.addWidget(self.controls["min_cue_ms"], 3, 1)
        self.controls["merge_short_cues"] = QCheckBox("Merge cues that stay too short into their neighbour")
        self.controls["merge_short_cues"].setChecked(True)
        self.controls["merge_short_cues"].setToolTip(
            "A cue that cannot reach the minimum duration in the free time around it is glued to the cue next to "
            "it instead of being extended over the next line's speech - Whisper emits segments as short as 10 ms, "
            "which flash on screen unreadably. Applies to resynced subtitles too. If the joined text is over the cue "
            "layout, the cues are merged anyway and the text takes an extra line - the merged cue spans exactly "
            "the two it replaces, so nothing moves out of sync. Merging is refused only past the maximum cue "
            "duration or when the two cues are far apart; that cue then borrows time from a neighbour instead.")
        grid.addWidget(self.controls["merge_short_cues"], 4, 0, 1, 2)
        self.controls["min_gap_ms"] = QSpinBox()
        self.controls["min_gap_ms"].setRange(0, 1000)
        self.controls["min_gap_ms"].setSingleStep(10)
        self.controls["min_gap_ms"].setSuffix(" ms")
        self.controls["min_gap_ms"].setValue(80)
        grid.addWidget(QLabel("Min gap between cues:"), 5, 0)
        grid.addWidget(self.controls["min_gap_ms"], 5, 1)
        self.controls["global_offset_ms"] = QSpinBox()
        self.controls["global_offset_ms"].setRange(-600000, 600000)
        self.controls["global_offset_ms"].setSingleStep(50)
        self.controls["global_offset_ms"].setSuffix(" ms")
        self.controls["global_offset_ms"].setToolTip("Added to every cue at the very end. Negative = earlier.")
        grid.addWidget(QLabel("Global offset:"), 6, 0)
        grid.addWidget(self.controls["global_offset_ms"], 6, 1)
        snap_group.setLayout(grid)
        layout.addWidget(snap_group)
        hint2 = QLabel("Symptoms → fix: cues appear before people speak or linger over silence → snapping (on by "
                       "default). Everything is early/late by the same amount → global offset. Subtitles from "
                       "elsewhere drift apart over the film → Resync with speed fitting.")
        hint2.setWordWrap(True)
        hint2.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(hint2)
        layout.addStretch()
        self._on_sync_mode_changed()
        return tab

    def _on_sync_mode_changed(self, *_):
        resync = self.controls["sync_mode"].currentData() == "resync"
        self.resync_widget.setVisible(resync)

    def _create_subtitles_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        fmt_group = QGroupBox("Output formats")
        fl = QHBoxLayout()
        self.format_checks = {}
        for fmt in SUBTITLE_FORMATS:
            cb = QCheckBox(fmt.upper())
            cb.setChecked(fmt in DEFAULT_SETTINGS["formats"])
            self.format_checks[fmt] = cb
            fl.addWidget(cb)
        fl.addStretch()
        fmt_group.setLayout(fl)
        layout.addWidget(fmt_group)

        layout_group = QGroupBox("Cue layout")
        grid = QGridLayout()
        self.controls["max_line_chars"] = QSpinBox()
        self.controls["max_line_chars"].setRange(10, 200)
        self.controls["max_line_chars"].setValue(42)
        grid.addWidget(QLabel("Max characters per line:"), 0, 0)
        grid.addWidget(self.controls["max_line_chars"], 0, 1)
        self.controls["max_lines"] = QSpinBox()
        self.controls["max_lines"].setRange(1, 5)
        self.controls["max_lines"].setValue(2)
        grid.addWidget(QLabel("Max lines per cue:"), 1, 0)
        grid.addWidget(self.controls["max_lines"], 1, 1)
        self.controls["max_segment_seconds"] = QDoubleSpinBox()
        self.controls["max_segment_seconds"].setRange(0, 60)
        self.controls["max_segment_seconds"].setDecimals(1)
        self.controls["max_segment_seconds"].setSingleStep(0.5)
        self.controls["max_segment_seconds"].setSuffix(" s")
        self.controls["max_segment_seconds"].setSpecialValueText("unlimited")
        self.controls["max_segment_seconds"].setValue(7.0)
        grid.addWidget(QLabel("Max cue duration:"), 2, 0)
        grid.addWidget(self.controls["max_segment_seconds"], 2, 1)
        self.controls["capitalize_sentences"] = QCheckBox("Capitalise sentence starts")
        self.controls["capitalize_sentences"].setChecked(True)
        self.controls["capitalize_sentences"].setToolTip("Whisper sometimes emits the first word of a file or of a "
                                                         "VAD chunk in lower case; this upper-cases the first letter "
                                                         "of the first cue and of every cue after a full stop.")
        grid.addWidget(self.controls["capitalize_sentences"], 3, 0, 1, 2)
        self.controls["strip_foreign_script"] = QCheckBox("Remove stray foreign-script characters")
        self.controls["strip_foreign_script"].setChecked(True)
        self.controls["strip_foreign_script"].setToolTip("Whisper sometimes hallucinates a single Chinese / Korean / "
                                                         "Cyrillic character inside the text (\"Bar标\"). Letters of a "
                                                         "script that makes up under 5 % of the transcript are removed; "
                                                         "genuinely bilingual files are left alone.")
        grid.addWidget(self.controls["strip_foreign_script"], 4, 0, 1, 2)
        self.controls["repair_sentence_breaks"] = QCheckBox("Remove full stops the speaker never made")
        self.controls["repair_sentence_breaks"].setChecked(True)
        self.controls["repair_sentence_breaks"].setToolTip(
            "Whisper punctuates by language model, not by ear, and sometimes ends a sentence in the middle of a "
            "phrase (\"When someone is. First about to embark…\"). People pause between sentences, so a full stop "
            "with less silence around it than below is the model's invention: it is removed and the next word goes "
            "back to lower case. Names and \"I\" keep their capital. Needs word timestamps (on automatically).")
        grid.addWidget(self.controls["repair_sentence_breaks"], 5, 0, 1, 2)
        self.controls["sentence_pause_ms"] = QSpinBox()
        self.controls["sentence_pause_ms"].setRange(0, 2000)
        self.controls["sentence_pause_ms"].setSingleStep(50)
        self.controls["sentence_pause_ms"].setSuffix(" ms")
        self.controls["sentence_pause_ms"].setValue(250)
        self.controls["sentence_pause_ms"].setToolTip("Silence a full stop needs to be believed. Raise it for a "
                                                      "speaker who runs sentences together, lower it if real "
                                                      "sentence ends are being joined.")
        grid.addWidget(QLabel("Shortest pause between sentences:"), 6, 0)
        grid.addWidget(self.controls["sentence_pause_ms"], 6, 1)
        layout_group.setLayout(grid)
        layout.addWidget(layout_group)

        out_group = QGroupBox("Output location")
        og = QGridLayout()
        self.controls["output_mode"] = QComboBox()
        self.controls["output_mode"].addItem("Next to the source file", "same")
        self.controls["output_mode"].addItem("Custom folder", "custom")
        self.controls["output_mode"].currentIndexChanged.connect(self._on_output_mode_changed)
        og.addWidget(QLabel("Save to:"), 0, 0)
        og.addWidget(self.controls["output_mode"], 0, 1)
        self.controls["output_dir"] = QLineEdit()
        og.addWidget(QLabel("Folder:"), 1, 0)
        og.addLayout(self._path_row(self.controls["output_dir"],
                                    lambda: self._browse_dir(self.controls["output_dir"], "Output folder")), 1, 1)
        self.controls["suffix"] = QLineEdit()
        self.controls["suffix"].setPlaceholderText("optional, e.g. _whisper")
        og.addWidget(QLabel("File name suffix:"), 2, 0)
        og.addWidget(self.controls["suffix"], 2, 1)
        self.controls["language_suffix"] = QCheckBox("Language code in file name (video.en.srt)")
        self.controls["language_suffix"].setChecked(True)
        og.addWidget(self.controls["language_suffix"], 3, 0, 1, 2)
        self.controls["overwrite"] = QCheckBox("Overwrite existing subtitle files")
        og.addWidget(self.controls["overwrite"], 4, 0, 1, 2)
        out_group.setLayout(og)
        layout.addWidget(out_group)

        mux_group = QGroupBox("Embed into video (stream copy, no re-encode)")
        ml = QHBoxLayout()
        self.controls["mux_subtitles"] = QCheckBox("Write a copy of the video with a soft subtitle track")
        ml.addWidget(self.controls["mux_subtitles"])
        self.controls["mux_container"] = QComboBox()
        self.controls["mux_container"].addItems(list(MUX_CONTAINERS.keys()))
        ml.addWidget(QLabel("Container:"))
        ml.addWidget(self.controls["mux_container"])
        ml.addStretch()
        mux_group.setLayout(ml)
        layout.addWidget(mux_group)
        layout.addStretch()
        return tab

    def _create_advanced_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        cpp_group = QGroupBox("whisper.cpp")
        grid = QGridLayout()
        self.controls["whisper_cli_path"] = QLineEdit()
        self.controls["whisper_cli_path"].setPlaceholderText("auto (PATH or next to the app)")
        grid.addWidget(QLabel("whisper-cli executable:"), 0, 0)
        grid.addLayout(self._path_row(self.controls["whisper_cli_path"],
                                      lambda: self._browse_file(self.controls["whisper_cli_path"], "whisper-cli executable")), 0, 1)
        self.controls["extra_args"] = QLineEdit()
        self.controls["extra_args"].setPlaceholderText("extra whisper-cli arguments, e.g. -ml 60 -sow")
        grid.addWidget(QLabel("Extra arguments:"), 1, 0)
        grid.addWidget(self.controls["extra_args"], 1, 1)
        cpp_group.setLayout(grid)
        layout.addWidget(cpp_group)

        info = QLabel(
            "<b>Notes</b><br>"
            "• faster-whisper downloads models from Hugging Face on first use into the model folder "
            "(or ~/.cache/huggingface). large-v3 is ~3 GB, small.en ~0.5 GB.<br>"
            "• whisper.cpp needs GGML models (ggml-small.en.bin …) from "
            "<a href='https://huggingface.co/ggerganov/whisper.cpp/tree/main'>huggingface.co/ggerganov/whisper.cpp</a>.<br>"
            "• Audio is extracted with FFmpeg to 16 kHz mono WAV in a temp folder and deleted afterwards.<br>"
            "• GPU: use <i>Model → Download CUDA libraries</i> once (cuBLAS 12 + cuDNN 9, ~1 GB), "
            "or pip install nvidia-cublas-cu12 nvidia-cudnn-cu12.")
        info.setWordWrap(True)
        info.setOpenExternalLinks(True)
        info.setStyleSheet("color: #555;")
        layout.addWidget(info)
        layout.addStretch()
        return tab

    def _create_status_bar(self):
        sb = self.main_window.statusBar()
        self.ffmpeg_status = QLabel("FFmpeg: checking…")
        self.engine_status = QLabel("")
        sb.addPermanentWidget(self.engine_status)
        sb.addPermanentWidget(self.ffmpeg_status)
        sb.showMessage(f"{APP_NAME} v{APP_VERSION}")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_engine_changed(self, *_):
        engine = self.controls["engine"].currentData()
        if engine == "faster_whisper":
            ok = faster_whisper_available()
            self.engine_hint.setText("Python CTranslate2 implementation, bundled with the app. CPU or NVIDIA CUDA. "
                                     "Models are downloaded automatically." if ok else
                                     "faster-whisper is NOT installed: pip install faster-whisper")
            self._refresh_gpu_status()
            self._refresh_model_cache_label()
            self.engine_status.setText("Engine: faster-whisper ✓" if ok else "Engine: faster-whisper ✗")
            self.engine_status.setStyleSheet("color: green;" if ok else "color: red;")
        else:
            from modules.backends import find_whisper_cli
            exe = find_whisper_cli(self.controls["whisper_cli_path"].text() if "whisper_cli_path" in self.controls else "")
            self.engine_hint.setText("External C/C++ whisper.cpp binary (CPU, CUDA, Vulkan, Metal builds). "
                                     "Set the executable on the Advanced tab and a folder with ggml models above.")
            self.hw_label.setText(f"whisper-cli: {exe or 'not found'} — GPU use depends on how whisper.cpp was built "
                                  "(CUDA / Vulkan / Metal builds use the GPU automatically; Device=cpu passes -ng).")
            self.model_cache_label.setText("whisper.cpp: put ggml-<model>.bin files into the model folder")
            self.engine_status.setText("Engine: whisper.cpp ✓" if exe else "Engine: whisper.cpp ✗")
            self.engine_status.setStyleSheet("color: green;" if exe else "color: red;")
        self.stat_device.setText("--")

    def _refresh_gpu_status(self, *_):
        if self.controls["engine"].currentData() != "faster_whisper":
            return
        st = cuda_status(self.controls["cuda_lib_dir"].text().strip())
        self.hw_label.setText(str(st["text"]))
        self.hw_label.setStyleSheet("color: %s; font-size: 11px;" % ("#155724" if st["ready"] else "#856404"))
        self._cuda_ready = bool(st["ready"])

    def _on_download_cuda(self):
        import sys as _sys
        if _sys.platform == "darwin":
            QMessageBox.information(self.main_window, "CUDA", "CUDA is not available on macOS.")
            return
        dest = self.controls["cuda_lib_dir"].text().strip() or default_cuda_dir()
        reply = QMessageBox.question(
            self.main_window, "Download CUDA libraries",
            f"Download cuBLAS 12 and cuDNN 9 (official NVIDIA wheels from PyPI, about 1 GB) into:\n{dest}\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        class _Dl(QThread):
            progress = Signal(str, int, int)
            done = Signal(list, str)
            stop = False

            def run(self_inner):
                try:
                    files = download_cuda_libraries(dest, progress=lambda m, d, t: self_inner.progress.emit(m, d, t),
                                                    should_stop=lambda: self_inner.stop)
                    self_inner.done.emit(files, "")
                except Exception as exc:  # noqa: BLE001
                    self_inner.done.emit([], str(exc))

        self.controls["download_cuda"].setEnabled(False)
        self.controls["download_cuda"].setText("Downloading…")
        self._cuda_thread = _Dl()

        def on_progress(msg, done, total):
            if total:
                self.cuda_dl_label.setText(f"{msg} — {done / 1048576:.0f} / {total / 1048576:.0f} MB")
            else:
                self.cuda_dl_label.setText(msg)

        def finished(files, err):
            self.controls["download_cuda"].setEnabled(True)
            self.controls["download_cuda"].setText("Download CUDA libraries")
            if err:
                self.cuda_dl_label.setText("Download failed")
                QMessageBox.warning(self.main_window, "Download failed", err)
            else:
                self.cuda_dl_label.setText(f"{len(files)} libraries installed in {dest}")
                if not self.controls["cuda_lib_dir"].text().strip():
                    self.controls["cuda_lib_dir"].setText(dest)
            self._refresh_gpu_status()
        self._cuda_thread.progress.connect(on_progress)
        self._cuda_thread.done.connect(finished)
        self._cuda_thread.start()

    def _refresh_model_cache_label(self, *_):
        if self.controls["engine"].currentData() != "faster_whisper":
            return
        model = self.controls["model"].currentText().strip()
        if not model:
            self.model_cache_label.setText("")
            return
        cached = is_model_cached(model, self.controls["model_dir"].text().strip())
        self.model_cache_label.setText(f"{model_repo_id(model)} — {'✓ downloaded' if cached else 'not downloaded yet (fetched on first use)'}")
        self.model_cache_label.setStyleSheet("color: %s; font-size: 11px;" % ("#155724" if cached else "#666"))

    def _on_download_model(self):
        if self.controls["engine"].currentData() != "faster_whisper":
            QMessageBox.information(self.main_window, "whisper.cpp models",
                                    "Download GGML models from huggingface.co/ggerganov/whisper.cpp and put them in the model folder.")
            return
        model = self.controls["model"].currentText().strip()
        model_dir = self.controls["model_dir"].text().strip()
        if not model:
            return
        from modules.backends import download_model

        class _Dl(QThread):
            done = Signal(str, str)

            def run(self_inner):
                try:
                    path = download_model(model, model_dir)
                    self_inner.done.emit(path, "")
                except Exception as exc:  # noqa: BLE001
                    self_inner.done.emit("", str(exc))

        self.controls["download_model"].setEnabled(False)
        self.controls["download_model"].setText("Downloading…")
        self.update_status(f"Downloading model {model} ({model_repo_id(model)})…")
        self._dl_thread = _Dl()

        def finished(path, err):
            self.controls["download_model"].setEnabled(True)
            self.controls["download_model"].setText("Download / check model")
            if err:
                QMessageBox.warning(self.main_window, "Download failed", err)
                self.update_status("Model download failed")
            else:
                self.update_status(f"Model ready: {path}")
            self._refresh_model_cache_label()
        self._dl_thread.done.connect(finished)
        self._dl_thread.start()

    def _on_output_mode_changed(self, *_):
        custom = self.controls["output_mode"].currentData() == "custom"
        self.controls["output_dir"].setEnabled(custom)

    def _on_rows_moved(self, _parent, start, _end, _dest, row):
        to = row - 1 if row > start else row
        self.files_reordered.emit(start, to)

    def _on_add_files(self):
        exts = " ".join(f"*{e}" for e in MEDIA_EXTENSIONS)
        files, _ = QFileDialog.getOpenFileNames(self.main_window, "Add media files", "",
                                                f"Media files ({exts});;All files (*)")
        if files:
            self.files_added.emit(files)

    def _on_add_folder(self):
        folder = QFileDialog.getExistingDirectory(self.main_window, "Add folder (recursive)")
        if folder:
            self.files_added.emit([folder])

    def _on_remove_files(self):
        rows = sorted({i.row() for i in self.file_list.selectedIndexes()})
        if rows:
            self.files_removed.emit(rows)

    def _on_clear_queue(self):
        self.queue_cleared.emit()

    def _on_item_double_clicked(self, item: QListWidgetItem):
        row = self.file_list.row(item)
        if 0 <= row < len(self._files):
            f = self._files[row]
            target = f.outputs[0] if f.outputs else f.path
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(target)))

    def _show_about(self):
        QMessageBox.about(self.main_window, f"About {APP_NAME}",
                          f"<h3>{APP_NAME} v{APP_VERSION}</h3>"
                          "<p>Batch subtitle generator for video and audio files, powered by OpenAI Whisper "
                          "models via <b>faster-whisper</b> (CTranslate2) or <b>whisper.cpp</b>.</p>"
                          "<p>Drop files, pick a model, press Start — SRT/VTT/TXT/JSON subtitles are written "
                          "next to your videos, optionally embedded as a soft subtitle track.</p>"
                          "<p>MIT License · <a href='https://github.com/hclivess/whisperer'>github.com/hclivess/whisperer</a></p>")

    def _show_model_info(self):
        QMessageBox.information(self.main_window, "Models",
                                "faster-whisper models are fetched from Hugging Face (Systran/faster-whisper-*) on first "
                                "use and cached in the model folder (default ~/.cache/huggingface/hub).\n\n"
                                "Approximate sizes: tiny 75 MB · base 145 MB · small 480 MB · medium 1.5 GB · "
                                "large-v3 3 GB · large-v3-turbo 1.6 GB · distil-large-v3 1.5 GB.\n\n"
                                "For English content the *.en variants are recommended up to 'medium'; "
                                "large-v3-turbo is the best speed/quality trade-off on a GPU.")

    # ------------------------------------------------------------------
    # Updates from managers
    # ------------------------------------------------------------------
    def update_file_list(self, files):
        self._files = files
        self.file_list.clear()
        for f in files:
            item = QListWidgetItem(self._item_text(f))
            item.setToolTip(f.path)
            color = self._FILE_STATE_COLORS.get(f.status)
            if color:
                item.setBackground(QBrush(color))
            self.file_list.addItem(item)

    @staticmethod
    def _item_text(f) -> str:
        text = f.name
        if f.duration:
            text += f"  [{format_duration(f.duration)}]"
        if f.status == "done" and f.outputs:
            text += f"  →  {', '.join(os.path.basename(o) for o in f.outputs)}"
        elif f.status == "error":
            text += f"  ✗ {f.message}"
        elif f.status == "skipped":
            text += "  (stopped)"
        return text

    def update_file_count(self, count):
        self.file_count_label.setText(f"{count} file{'s' if count != 1 else ''} in queue")

    def update_progress(self, value, maximum):
        self.progress_bar.setRange(0, maximum)
        self.progress_bar.setValue(value)

    def update_file_progress(self, value):
        self.file_progress_bar.setRange(0, 1000)
        self.file_progress_bar.setValue(value)

    def update_status(self, message):
        self.status_label.setText(message)
        self.main_window.statusBar().showMessage(message, 8000)

    def on_file_started(self, index: int, name: str):
        self.current_file_label.setText(name)
        self.counter_label.setText(f"{index + 1} / {len(self._files)}")
        self.phase_label.setText("Working")
        self.phase_label.setStyleSheet(_PILL % ("#cce5ff", "#004085"))
        self.file_progress_bar.setValue(0)
        for t in (self.stat_speed, self.stat_position, self.stat_duration, self.stat_segments):
            t.setText("--")
        self.transcript.append(f"<b>— {name} —</b>")

    def update_stats(self, snap: Dict[str, Any]):
        if "speed" in snap:
            self.stat_speed.setText(f"{snap['speed']:.1f}× realtime" if snap["speed"] else "--")
        if "position" in snap:
            self.stat_position.setText(format_duration(snap["position"]))
        if "duration" in snap and snap["duration"]:
            self.stat_duration.setText(format_duration(snap["duration"]))
        if "segments" in snap:
            self.stat_segments.setText(str(snap["segments"]))
        if "file_eta" in snap:
            self.file_eta_label.setText(f"File ETA: {format_duration(snap['file_eta'])}")
        if "file_elapsed" in snap:
            self.file_elapsed_label.setText(f"Elapsed: {format_duration(snap['file_elapsed'])}")
        if "total_elapsed" in snap:
            self.total_elapsed_label.setText(f"Total elapsed: {format_duration(snap['total_elapsed'])}")
        if "device" in snap:
            self.stat_device.setText(snap["device"])

    def append_segment(self, _index: int, seg: Dict):
        self.transcript.append(f"<span style='color:#888'>[{format_duration(seg['start'])} → "
                               f"{format_duration(seg['end'])}]</span> {seg['text']}")
        sb = self.transcript.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_file_state(self, index: int, state: str, message: str = ""):
        if 0 <= index < len(self._files):
            self._files[index].status = state
            self._files[index].message = message
            item = self.file_list.item(index)
            if item:
                item.setText(self._item_text(self._files[index]))
                color = self._FILE_STATE_COLORS.get(state)
                item.setBackground(QBrush(color) if color else QBrush())
        if state == "extracting":
            self.phase_label.setText("Extracting")
            self.phase_label.setStyleSheet(_PILL % ("#fff3cd", "#856404"))
        elif state == "processing":
            self.phase_label.setText("Transcribing")
            self.phase_label.setStyleSheet(_PILL % ("#cce5ff", "#004085"))

    def update_ffmpeg_status(self, available):
        self.ffmpeg_status.setText("FFmpeg: ✓ Found" if available else "FFmpeg: ✗ Not Found")
        self.ffmpeg_status.setStyleSheet("color: green;" if available else "color: red;")

    def set_paused_state(self, paused: bool):
        self.controls["pause"].setText("Resume" if paused else "Pause")
        if paused:
            self.phase_label.setText("Paused")
            self.phase_label.setStyleSheet(_PILL % ("#fff3cd", "#856404"))
        else:
            self.phase_label.setText("Transcribing")
            self.phase_label.setStyleSheet(_PILL % ("#cce5ff", "#004085"))

    def set_processing_state(self, is_processing):
        self._processing_active = is_processing
        if is_processing:
            self.update_status("Starting…")
            self.progress_bar.setValue(0)
            self.file_progress_bar.setValue(0)
            self.stat_device.setText(self._device_text())
        else:
            self.phase_label.setText("Idle")
            self.phase_label.setStyleSheet(_PILL % ("#e0e0e0", "#444"))
            self.file_eta_label.setText("File ETA: --")
            self.counter_label.setText("")
        self.controls["start"].setEnabled(not is_processing)
        self.controls["pause"].setEnabled(is_processing)
        self.controls["pause"].setText("Pause")
        self.controls["stop"].setEnabled(is_processing)
        self.controls["clear_files"].setEnabled(not is_processing)
        self.tabs.setEnabled(not is_processing)
        self.file_list.setDragDropMode(QListWidget.DragDropMode.NoDragDrop if is_processing
                                       else QListWidget.DragDropMode.InternalMove)
        self.file_list.setAcceptDrops(True)

    def _device_text(self) -> str:
        if self.controls["engine"].currentData() == "whisper_cpp":
            return "whisper.cpp"
        dev = self.controls["device"].currentText()
        if dev == "auto":
            dev = "cuda" if cuda_available(self.controls["cuda_lib_dir"].text().strip()) else "cpu"
        return dev.upper()

    # ------------------------------------------------------------------
    # Settings <-> widgets
    # ------------------------------------------------------------------
    def get_current_settings(self) -> Dict[str, Any]:
        c = self.controls
        return {
            "engine": c["engine"].currentData(),
            "model": c["model"].currentText().strip(),
            "device": c["device"].currentText(),
            "compute_type": c["compute_type"].currentText(),
            "model_dir": c["model_dir"].text().strip(),
            "whisper_cli_path": c["whisper_cli_path"].text().strip(),
            "cpu_threads": c["cpu_threads"].value(),
            "cuda_lib_dir": c["cuda_lib_dir"].text().strip(),
            "language": c["language"].currentData(),
            "task": c["task"].currentData(),
            "vad_filter": c["vad_filter"].isChecked(),
            "vad_min_silence_ms": c["vad_min_silence_ms"].value(),
            "beam_size": c["beam_size"].value(),
            "word_timestamps": c["word_timestamps"].isChecked(),
            "condition_on_previous_text": c["condition_on_previous_text"].isChecked(),
            "second_pass": c["second_pass"].isChecked(),
            "initial_prompt": c["initial_prompt"].toPlainText().strip(),
            "extra_args": c["extra_args"].text().strip(),
            "sync_mode": c["sync_mode"].currentData(),
            "snap_to_speech": c["snap_to_speech"].isChecked(),
            "snap_max_shift_ms": c["snap_max_shift_ms"].value(),
            "end_padding_ms": c["end_padding_ms"].value(),
            "min_cue_ms": c["min_cue_ms"].value(),
            "merge_short_cues": c["merge_short_cues"].isChecked(),
            "min_gap_ms": c["min_gap_ms"].value(),
            "global_offset_ms": c["global_offset_ms"].value(),
            "resync_file": c["resync_file"].text().strip(),
            "resync_fit_speed": c["resync_fit_speed"].isChecked(),
            "formats": [f for f, cb in self.format_checks.items() if cb.isChecked()],
            "max_line_chars": c["max_line_chars"].value(),
            "max_lines": c["max_lines"].value(),
            "max_segment_seconds": c["max_segment_seconds"].value(),
            "capitalize_sentences": c["capitalize_sentences"].isChecked(),
            "strip_foreign_script": c["strip_foreign_script"].isChecked(),
            "repair_sentence_breaks": c["repair_sentence_breaks"].isChecked(),
            "sentence_pause_ms": c["sentence_pause_ms"].value(),
            "output_mode": c["output_mode"].currentData(),
            "output_dir": c["output_dir"].text().strip(),
            "suffix": c["suffix"].text().strip(),
            "language_suffix": c["language_suffix"].isChecked(),
            "overwrite": c["overwrite"].isChecked(),
            "mux_subtitles": c["mux_subtitles"].isChecked(),
            "mux_container": c["mux_container"].currentText(),
        }

    @staticmethod
    def _set_combo_data(combo: QComboBox, data):
        idx = combo.findData(data)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    @staticmethod
    def _set_combo_text(combo: QComboBox, text):
        idx = combo.findText(str(text))
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif combo.isEditable():
            combo.setCurrentText(str(text))

    def apply_settings(self, s: Dict[str, Any]):
        c = self.controls
        for key, val in s.items():
            if key == "formats":
                for f, cb in self.format_checks.items():
                    cb.setChecked(f in val)
            elif key in ("engine", "language", "task", "output_mode", "sync_mode"):
                self._set_combo_data(c[key], val)
            elif key in ("model", "device", "compute_type", "mux_container"):
                self._set_combo_text(c[key], val)
            elif key == "initial_prompt":
                c[key].setPlainText(str(val))
            elif key in c:
                w = c[key]
                if isinstance(w, QCheckBox):
                    w.setChecked(bool(val))
                elif isinstance(w, (QSpinBox,)):
                    w.setValue(int(val))
                elif isinstance(w, QDoubleSpinBox):
                    w.setValue(float(val))
                elif isinstance(w, QLineEdit):
                    w.setText(str(val))
        self._refresh_gpu_status()
        self._refresh_model_cache_label()

    def load_settings(self, qsettings: QSettings):
        s = {}
        for key, default in DEFAULT_SETTINGS.items():
            if not qsettings.contains(key):
                continue
            val = qsettings.value(key)
            if isinstance(default, bool):
                val = str(val).lower() in ("true", "1", "yes")
            elif isinstance(default, int):
                val = int(val)
            elif isinstance(default, float):
                val = float(val)
            elif isinstance(default, list):
                val = list(val) if isinstance(val, (list, tuple)) else ([val] if val else [])
            s[key] = val
        if s:
            self.apply_settings(s)
        geo = qsettings.value("window_geometry")
        if geo:
            self.main_window.restoreGeometry(geo)

    def save_settings(self, qsettings: QSettings):
        for key, val in self.get_current_settings().items():
            qsettings.setValue(key, val)
        qsettings.setValue("window_geometry", self.main_window.saveGeometry())
