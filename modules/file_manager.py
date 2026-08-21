"""Queue of input files"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

from PySide6.QtCore import QObject, Signal

from utils.file_utils import collect_media_files


@dataclass
class QueueFile:
    path: str
    status: str = "pending"          # pending | extracting | processing | done | error | skipped
    duration: Optional[float] = None
    outputs: List[str] = field(default_factory=list)
    message: str = ""

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


class FileManager(QObject):
    files_updated = Signal(list)
    file_count_changed = Signal(int)
    duplicates_skipped = Signal(int)

    def __init__(self):
        super().__init__()
        self._files: List[QueueFile] = []

    def _emit(self):
        self.files_updated.emit(self._files)
        self.file_count_changed.emit(len(self._files))

    @staticmethod
    def _norm(p: str) -> str:
        return os.path.normcase(os.path.abspath(p))

    def add_files(self, paths: List[str]) -> int:
        existing = {self._norm(f.path) for f in self._files}
        added = 0
        self.last_duplicates = 0
        for p in collect_media_files(paths):
            key = self._norm(p)
            if key in existing:
                self.last_duplicates += 1
                continue
            self._files.append(QueueFile(p))
            existing.add(key)
            added += 1
        if added:
            self._emit()
        if self.last_duplicates:
            self.duplicates_skipped.emit(self.last_duplicates)
        return added

    def remove_files(self, indices: List[int]):
        for i in sorted(set(indices), reverse=True):
            if 0 <= i < len(self._files):
                del self._files[i]
        self._emit()

    def clear_queue(self):
        self._files.clear()
        self._emit()

    def move_file(self, from_index: int, to_index: int):
        if 0 <= from_index < len(self._files) and 0 <= to_index < len(self._files):
            self._files.insert(to_index, self._files.pop(from_index))
            self._emit()

    def get_queue(self) -> List[QueueFile]:
        return list(self._files)

    def has_files(self) -> bool:
        return bool(self._files)
