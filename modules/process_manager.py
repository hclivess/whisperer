"""
Runs the transcription queue on a worker thread and reports progress to the UI.
"""
import os
import tempfile
import threading
import time
import traceback
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QThread, Signal

from config import MUX_CONTAINERS
from modules.backends import BACKENDS, TranscribeCallbacks, StoppedError, find_whisper_cli, faster_whisper_available
from utils.ffmpeg_utils import find_ffmpeg, probe_duration, extract_audio, mux_subtitles
from utils.subtitle_utils import (capitalize_sentences, enforce_min_duration, merge_short_cues,
                                  repair_sentence_breaks, split_segments, strip_foreign_script,
                                  write_subtitles)
from utils import childproc
from utils.sync_utils import parse_subtitles, resync_cues, shift_cues, snap_to_speech, speech_regions

try:
    import psutil
except Exception:  # psutil is optional (used to suspend whisper-cli on pause)
    psutil = None


class _Worker(QThread):
    progress = Signal(int, int)                 # overall done, total (per mille)
    file_progress = Signal(int)                 # 0..1000 for the current file
    status = Signal(str)
    stats = Signal(dict)
    file_state = Signal(int, str, str)          # index, state, message
    segment = Signal(int, dict)                 # index, segment dict
    file_started = Signal(int, str)             # index, name
    finished_all = Signal(int, int)             # success, total
    paused_changed = Signal(bool)

    def __init__(self, files, settings: Dict):
        super().__init__()
        self._files = list(files)
        self._settings = settings
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._proc = None
        self.current_index = -1
        self._pending_replacement: Optional[list] = None

    # -- control -----------------------------------------------------------
    def request_stop(self):
        self._stop.set()
        self._pause.clear()
        with self._lock:
            proc = self._proc
        if proc:
            try:
                if psutil and proc.poll() is None:
                    psutil.Process(proc.pid).resume()
            except Exception:
                pass
            childproc.kill(proc)

    def set_paused(self, paused: bool):
        if paused:
            self._pause.set()
        else:
            self._pause.clear()
        with self._lock:
            proc = self._proc
        if proc and psutil:
            try:
                p = psutil.Process(proc.pid)
                p.suspend() if paused else p.resume()
            except Exception:
                pass
        self.paused_changed.emit(paused)

    def is_paused(self) -> bool:
        return self._pause.is_set()

    def sync_pending(self, files):
        """Replace the not-yet-started tail of the queue (live queue editing)"""
        with self._lock:
            self._pending_replacement = list(files)

    def _wait_if_paused(self):
        while self._pause.is_set() and not self._stop.is_set():
            time.sleep(0.2)

    # -- work --------------------------------------------------------------
    def run(self):
        s = self._settings
        success = 0
        total_duration_done = 0.0
        run_start = time.time()
        index = 0
        while True:
            with self._lock:
                if self._pending_replacement is not None:
                    # keep processed prefix, take the new tail
                    new = self._pending_replacement
                    self._pending_replacement = None
                    if len(new) >= index:
                        self._files = self._files[:index] + new[index:]
            if index >= len(self._files) or self._stop.is_set():
                break
            qf = self._files[index]
            self.current_index = index
            self.file_started.emit(index, qf.name)
            file_start = time.time()
            try:
                outputs = self._process_one(index, qf, s, run_start, total_duration_done)
                qf.outputs = outputs
                self.file_state.emit(index, "done", f"{len(outputs)} file(s) written")
                success += 1
            except StoppedError:
                self.file_state.emit(index, "skipped", "stopped")
                break
            except InterruptedError:
                self.file_state.emit(index, "skipped", "stopped")
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc) or exc.__class__.__name__
                traceback.print_exc()
                self.file_state.emit(index, "error", msg)
                self.status.emit(f"Error: {qf.name}: {msg}")
            if qf.duration:
                total_duration_done += qf.duration
            self.stats.emit({"file_elapsed": time.time() - file_start})
            index += 1
        self.finished_all.emit(success, len(self._files))

    def _process_one(self, index: int, qf, s: Dict, run_start: float, duration_done_before: float) -> List[str]:
        src = qf.path
        engine = s["engine"]
        if engine not in BACKENDS:
            raise RuntimeError(f"Unknown engine {engine}")
        if engine == "faster_whisper" and not faster_whisper_available():
            raise RuntimeError("faster-whisper is not installed (pip install faster-whisper)")

        # where do outputs go?
        out_dir = s["output_dir"] if s["output_mode"] == "custom" and s["output_dir"] else os.path.dirname(src)
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(src))[0] + (s.get("suffix") or "")
        if s.get("sync_mode") == "resync" and not s.get("suffix"):
            stem += ".synced"
        if s.get("language_suffix"):
            lang = s["language"]
            if lang != "auto":
                stem += f".{lang}" if s["task"] == "transcribe" else ".en"
        base_path = os.path.join(out_dir, stem)
        for fmt in s["formats"]:
            p = f"{base_path}.{fmt}"
            if os.path.exists(p) and not s["overwrite"]:
                raise FileExistsError(f"Output exists (enable Overwrite on the Subtitles tab): {p}")

        qf.duration = probe_duration(src)
        self.file_state.emit(index, "extracting", "")
        self.status.emit(f"Extracting audio: {qf.name}")
        self.file_progress.emit(0)

        tmpdir = tempfile.mkdtemp(prefix="whisperer-")
        wav = os.path.join(tmpdir, "audio.wav")
        audio_for_engine = src
        try:
            if find_ffmpeg():
                extract_audio(src, wav, stop_check=self._stop.is_set)
                audio_for_engine = wav
            elif engine == "whisper_cpp":
                raise RuntimeError("FFmpeg is required to feed whisper.cpp (16 kHz WAV)")
            if self._stop.is_set():
                raise StoppedError()

            self.file_state.emit(index, "processing", "")
            seg_count = [0]
            t0 = time.time()

            def on_progress(done, total):
                total_len = total or qf.duration or 0
                if total_len:
                    frac = min(1.0, max(0.0, done / total_len))
                    self.file_progress.emit(int(frac * 1000))
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0
                remaining = (total_len - done) / speed if speed > 0 and total_len else None
                self.stats.emit({
                    "speed": speed, "position": done, "duration": total_len,
                    "segments": seg_count[0], "file_eta": remaining,
                    "file_elapsed": elapsed,
                    "total_elapsed": time.time() - run_start,
                })
                self._emit_overall(index, done, total_len)

            def on_segment(seg):
                seg_count[0] += 1
                self.segment.emit(index, seg)

            cb = TranscribeCallbacks(
                progress=on_progress, segment=on_segment, status=self.status.emit,
                should_stop=self._stop.is_set, wait_if_paused=self._wait_if_paused,
                set_process=self._set_process, extra={"duration": qf.duration})
            engine_settings = dict(s)
            if s.get("snap_to_speech") or s.get("sync_mode") == "resync" or s.get("repair_sentence_breaks"):
                engine_settings["word_timestamps"] = True      # word-level alignment is what we snap / match
            segments, meta = BACKENDS[engine](audio_for_engine, engine_settings, cb)
            if self._stop.is_set():
                raise StoppedError()

            if s.get("sync_mode") == "resync":
                sub_path = self._resync_source(src, s)
                self.status.emit(f"Aligning {os.path.basename(sub_path)} to the audio…")
                with open(sub_path, encoding="utf-8", errors="replace") as fh:
                    cues = parse_subtitles(fh.read())
                if not cues:
                    raise RuntimeError(f"No cues found in {sub_path}")
                segments, report = resync_cues(cues, segments, bool(s.get("resync_fit_speed", True)))
                meta["resync"] = {"source": os.path.basename(sub_path), **report}
                self.status.emit(f"Resync: offset {report['offset']:+.3f} s, speed {report['speed']:.5f} "
                                 f"({report['drift_per_hour']:+.1f} s/h), {report['inliers']}/{report['matches']} words agree")
            else:
                if s.get("repair_sentence_breaks", True):
                    # before splitting and snapping: the cue text is still Whisper's own, word for word
                    segments = repair_sentence_breaks(segments, int(s.get("sentence_pause_ms", 250)) / 1000.0)
                segments = split_segments(segments, int(s["max_line_chars"]), int(s["max_lines"]),
                                          float(s["max_segment_seconds"]))

            if s.get("snap_to_speech"):
                self.status.emit(f"Snapping cues to speech: {qf.name}")
                regions = speech_regions(audio_for_engine)
                if self._stop.is_set():
                    raise StoppedError()
                segments = snap_to_speech(
                    segments, regions, max_shift=int(s.get("snap_max_shift_ms", 600)) / 1000.0,
                    end_padding=int(s.get("end_padding_ms", 200)) / 1000.0,
                    min_duration=int(s.get("min_cue_ms", 800)) / 1000.0, min_gap=int(s.get("min_gap_ms", 80)) / 1000.0,
                    max_duration=float(s["max_segment_seconds"]) if s.get("sync_mode") != "resync" else 0.0)
                meta["speech_regions"] = len(regions)
            if s.get("merge_short_cues", True) and s.get("sync_mode") != "resync":
                before = len(segments)
                segments = merge_short_cues(
                    segments, min_duration=int(s.get("min_cue_ms", 800)) / 1000.0,
                    min_gap=int(s.get("min_gap_ms", 80)) / 1000.0,
                    max_duration=float(s["max_segment_seconds"]),
                    limit_chars=int(s["max_line_chars"]) * int(s["max_lines"]),
                    capitalize=bool(s.get("capitalize_sentences", True)))
                if len(segments) != before:
                    self.status.emit(f"Merged {before - len(segments)} cue(s) that were too short to read")
            if s.get("strip_foreign_script", True) and s.get("sync_mode") != "resync":
                before = len(segments)
                segments = strip_foreign_script(segments)
                if len(segments) != before:
                    self.status.emit(f"Removed {before - len(segments)} cue(s) that were only stray foreign characters")
            if s.get("capitalize_sentences", True) and s.get("sync_mode") != "resync":
                segments = capitalize_sentences(segments)
            if int(s.get("min_cue_ms", 800)) > 0:
                # the floor, in every mode: whatever snapping, resyncing or a refused merge left behind,
                # no cue goes into the file too short to read while a neighbour has time to spare
                flashes = sum(1 for c in segments
                              if c["end"] - c["start"] < int(s["min_cue_ms"]) / 1000.0 - 1e-6)
                segments = enforce_min_duration(
                    segments, min_duration=int(s["min_cue_ms"]) / 1000.0,
                    min_gap=int(s.get("min_gap_ms", 80)) / 1000.0,
                    max_duration=float(s["max_segment_seconds"]) if s.get("sync_mode") != "resync" else 0.0)
                left = sum(1 for c in segments
                           if c["end"] - c["start"] < int(s["min_cue_ms"]) / 1000.0 - 1e-6)
                if flashes:
                    self.status.emit(f"Gave {flashes - left} of {flashes} too-short cue(s) their reading time")
            if int(s.get("global_offset_ms", 0)):
                segments = shift_cues(segments, int(s["global_offset_ms"]) / 1000.0)
            meta["source"] = os.path.basename(src)
            self.status.emit(f"Writing subtitles: {qf.name}")
            outputs = write_subtitles(segments, base_path, list(s["formats"]), int(s["max_line_chars"]),
                                      int(s["max_lines"]), meta, bool(s["overwrite"]))

            if s.get("mux_subtitles"):
                srt = f"{base_path}.srt"
                if "srt" not in s["formats"]:
                    # need an srt for muxing; write a temporary one
                    srt = os.path.join(tmpdir, "subs.srt")
                    write_subtitles(segments, os.path.splitext(srt)[0], ["srt"], int(s["max_line_chars"]),
                                    int(s["max_lines"]), meta, True)
                container = s.get("mux_container", "mkv")
                if container not in MUX_CONTAINERS:
                    container = "mkv"
                muxed = os.path.join(out_dir, os.path.splitext(os.path.basename(src))[0] + f".subbed.{container}")
                if os.path.exists(muxed) and not s["overwrite"]:
                    raise FileExistsError(f"Output exists (enable Overwrite): {muxed}")
                self.status.emit(f"Muxing subtitles into {os.path.basename(muxed)}")
                lang = meta.get("language") or s["language"]
                if s["task"] == "translate":
                    lang = "en"
                mux_subtitles(src, srt, muxed, container, lang, stop_check=self._stop.is_set)
                outputs.append(muxed)

            self.file_progress.emit(1000)
            self._emit_overall(index, qf.duration or 0, qf.duration or 0)
            return outputs
        finally:
            try:
                if os.path.exists(wav):
                    os.remove(wav)
                for fn in os.listdir(tmpdir):
                    os.remove(os.path.join(tmpdir, fn))
                os.rmdir(tmpdir)
            except OSError:
                pass

    @staticmethod
    def _resync_source(src: str, s: Dict) -> str:
        explicit = (s.get("resync_file") or "").strip()
        if explicit:
            if os.path.isfile(explicit):
                return explicit
            raise FileNotFoundError(f"Subtitle file to resync not found: {explicit}")
        stem = os.path.splitext(src)[0]
        folder = os.path.dirname(src) or "."
        base = os.path.basename(stem)
        candidates = [f"{stem}.srt", f"{stem}.vtt"]
        for fn in sorted(os.listdir(folder)):
            if fn.startswith(base + ".") and fn.lower().endswith((".srt", ".vtt")) and not fn.endswith(".synced.srt"):
                candidates.append(os.path.join(folder, fn))
        for c in candidates:
            if os.path.isfile(c):
                return c
        raise FileNotFoundError(f"No .srt / .vtt next to {os.path.basename(src)} to resync "
                                "(set the file on the Sync tab)")

    def _set_process(self, proc):
        with self._lock:
            self._proc = proc

    def _emit_overall(self, index: int, done: float, total_len: float):
        n = len(self._files)
        if n == 0:
            return
        frac_file = (done / total_len) if total_len else 0.0
        overall = (index + min(1.0, frac_file)) / n
        self.progress.emit(int(overall * 1000), 1000)


class ProcessManager(QObject):
    progress_updated = Signal(int, int)
    file_progress_updated = Signal(int)
    status_updated = Signal(str)
    stats_updated = Signal(dict)
    file_state_changed = Signal(int, str, str)
    segment_ready = Signal(int, dict)
    file_started = Signal(int, str)
    processing_finished = Signal(int, int)
    paused_state_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[_Worker] = None

    @property
    def current_file_index(self) -> int:
        return self._worker.current_index if self._worker else -1

    def is_processing(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def is_paused(self) -> bool:
        return bool(self._worker and self._worker.is_paused())

    def validate_settings(self, s: Dict) -> List[str]:
        issues = []
        if not s["formats"] and not s.get("mux_subtitles"):
            issues.append("No output format selected (Subtitles tab).")
        if s["engine"] == "faster_whisper" and not faster_whisper_available():
            issues.append("faster-whisper is not installed: pip install faster-whisper")
        if s["engine"] == "whisper_cpp" and not find_whisper_cli(s.get("whisper_cli_path", "")):
            issues.append("whisper-cli executable not found (Advanced tab).")
        if s["engine"] == "whisper_cpp" and not find_ffmpeg():
            issues.append("FFmpeg is required for whisper.cpp (audio must be converted to 16 kHz WAV).")
        if s["engine"] == "faster_whisper" and s["device"] == "cuda":
            from utils.cuda_utils import cuda_status
            st = cuda_status(s.get("cuda_lib_dir", ""))
            if not st["ready"]:
                issues.append("Device is set to CUDA but the GPU is not usable: " + str(st["text"]).split("\n")[0])
        if s["engine"] == "faster_whisper" and s.get("extra_args"):
            issues.append("Extra whisper-cli arguments are ignored by the faster-whisper engine.")
        if s.get("mux_subtitles") and not find_ffmpeg():
            issues.append("FFmpeg is required to mux subtitles into the video.")
        if s["task"] == "translate" and s["model"].endswith(".en"):
            issues.append("English-only models (*.en) cannot translate; pick a multilingual model.")
        if s["language"] not in ("en", "auto") and s["model"].endswith(".en"):
            issues.append(f"Model {s['model']} is English-only but language is set to {s['language']}.")
        if s["output_mode"] == "custom" and not s["output_dir"]:
            issues.append("Custom output folder is enabled but empty; files will be written next to the sources.")
        return issues

    def start_processing(self, files, settings: Dict):
        if self.is_processing():
            return
        w = _Worker(files, settings)
        w.progress.connect(self.progress_updated)
        w.file_progress.connect(self.file_progress_updated)
        w.status.connect(self.status_updated)
        w.stats.connect(self.stats_updated)
        w.file_state.connect(self.file_state_changed)
        w.segment.connect(self.segment_ready)
        w.file_started.connect(self.file_started)
        w.finished_all.connect(self._on_finished)
        w.paused_changed.connect(self.paused_state_changed)
        self._worker = w
        w.start()

    def _on_finished(self, ok: int, total: int):
        self.processing_finished.emit(ok, total)

    def stop_processing(self):
        if self._worker:
            self._worker.request_stop()
            self._worker.wait(5000)

    def pause_processing(self):
        if self._worker:
            self._worker.set_paused(True)

    def resume_processing(self):
        if self._worker:
            self._worker.set_paused(False)

    def sync_pending(self, files):
        if self._worker:
            self._worker.sync_pending(files)
