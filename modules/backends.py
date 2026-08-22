"""
Transcription engines.

Each backend exposes:
    transcribe(audio_path, settings, callbacks) -> (segments, info)
where callbacks is a TranscribeCallbacks with:
    progress(done_seconds, total_seconds)   # called as segments arrive
    segment(segment_dict)                   # live transcript
    status(message)                         # human readable phase text
    should_stop() -> bool
    wait_if_paused()                        # blocks while paused
"""
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from utils.ffmpeg_utils import find_binary
from utils.cuda_utils import setup_cuda
from utils import childproc

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@dataclass
class TranscribeCallbacks:
    progress: Callable[[float, float], None] = lambda d, t: None
    segment: Callable[[Dict], None] = lambda s: None
    status: Callable[[str], None] = lambda m: None
    should_stop: Callable[[], bool] = lambda: False
    wait_if_paused: Callable[[], None] = lambda: None
    set_process: Callable[[Optional[subprocess.Popen]], None] = lambda p: None
    extra: Dict = field(default_factory=dict)


class StoppedError(Exception):
    pass


# ----------------------------------------------------------------------------
# faster-whisper (CTranslate2)
# ----------------------------------------------------------------------------
_model_cache = {}
_model_lock = threading.Lock()


def faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def cuda_available(cuda_lib_dir: str = "") -> bool:
    try:
        setup_cuda(cuda_lib_dir)
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _resolve_device(device: str, cuda_lib_dir: str = "") -> str:
    if device == "auto":
        return "cuda" if cuda_available(cuda_lib_dir) else "cpu"
    if device == "cuda":
        setup_cuda(cuda_lib_dir)
    return device


def model_repo_id(model: str) -> str:
    """Size name -> Hugging Face repo id used by faster-whisper (passes repo ids through)"""
    try:
        from faster_whisper.utils import _MODELS
        return _MODELS.get(model, model)
    except Exception:
        return model


def is_model_cached(model: str, model_dir: str = "") -> bool:
    """True if the faster-whisper model is already in the local cache"""
    if os.path.isdir(model) and os.path.isfile(os.path.join(model, "model.bin")):
        return True
    repo = model_repo_id(model)
    try:
        from huggingface_hub import try_to_load_from_cache
        res = try_to_load_from_cache(repo, "model.bin", cache_dir=model_dir or None)
        return isinstance(res, str) and os.path.isfile(res)
    except Exception:
        return False


def download_model(model: str, model_dir: str = "") -> str:
    """Fetch a faster-whisper model into the cache; returns the local folder"""
    from faster_whisper import download_model as _dl
    return _dl(model, cache_dir=model_dir or None)


def _load_model(settings: Dict, cb: TranscribeCallbacks):
    from faster_whisper import WhisperModel

    device = _resolve_device(settings["device"], settings.get("cuda_lib_dir", ""))
    compute = settings["compute_type"]
    if compute == "default":
        compute = "float16" if device == "cuda" else "int8"
    key = (settings["model"], device, compute, settings.get("model_dir", ""), int(settings.get("cpu_threads", 0)))
    with _model_lock:
        model = _model_cache.get(key)
        if model is None:
            _model_cache.clear()  # keep at most one model resident
            cb.status(f"Loading model {settings['model']} on {device} ({compute}) — downloading on first use…")
            kwargs = dict(device=device, compute_type=compute)
            if settings.get("model_dir"):
                kwargs["download_root"] = settings["model_dir"]
            if int(settings.get("cpu_threads", 0)) > 0:
                kwargs["cpu_threads"] = int(settings["cpu_threads"])
            model = WhisperModel(settings["model"], **kwargs)
            _model_cache[key] = model
    return model, device, compute


def transcribe_faster_whisper(audio_path: str, settings: Dict, cb: TranscribeCallbacks) -> Tuple[List[Dict], Dict]:
    model, device, compute = _load_model(settings, cb)
    if cb.should_stop():
        raise StoppedError()

    language = None if settings["language"] == "auto" else settings["language"]
    kwargs = dict(
        language=language,
        task=settings["task"],
        beam_size=int(settings["beam_size"]),
        vad_filter=bool(settings["vad_filter"]),
        word_timestamps=bool(settings["word_timestamps"]),
        condition_on_previous_text=bool(settings["condition_on_previous_text"]),
    )
    if settings["vad_filter"]:
        kwargs["vad_parameters"] = dict(min_silence_duration_ms=int(settings["vad_min_silence_ms"]))
    if settings.get("initial_prompt"):
        kwargs["initial_prompt"] = settings["initial_prompt"]

    cb.status("Transcribing…")
    segments_iter, info = model.transcribe(audio_path, **kwargs)
    total = float(info.duration or 0)
    segments = []
    for seg in segments_iter:
        cb.wait_if_paused()
        if cb.should_stop():
            raise StoppedError()
        d = {"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()}
        # what the decoder thought of its own output: the second pass and the hallucination checks need it
        for field in ("avg_logprob", "no_speech_prob", "compression_ratio", "temperature"):
            value = getattr(seg, field, None)
            if value is not None:
                d[field] = round(float(value), 4)
        if seg.words:
            d["words"] = [{"start": float(w.start), "end": float(w.end), "word": w.word} for w in seg.words]
        segments.append(d)
        cb.segment(d)
        cb.progress(d["end"], total)
    meta = {"engine": "faster-whisper", "model": settings["model"], "device": device,
            "compute_type": compute, "language": info.language,
            "language_probability": round(float(info.language_probability or 0), 3),
            "duration": total}
    return segments, meta


# ----------------------------------------------------------------------------
# whisper.cpp (external whisper-cli)
# ----------------------------------------------------------------------------
_WCPP_LINE = re.compile(r"^\[(\d+):(\d+):(\d+)[.,](\d+)\s*-->\s*(\d+):(\d+):(\d+)[.,](\d+)\]\s*(.*)$")
_WCPP_PROGRESS = re.compile(r"progress\s*=\s*(\d+)%")


def find_whisper_cli(override: str = "") -> Optional[str]:
    for name in ("whisper-cli", "whisper-cpp", "whisper", "main"):
        found = find_binary(name, override)
        if found and name != "main":
            return found
    return find_binary("whisper-cli", override)


def _wcpp_model_path(settings: Dict) -> str:
    """Resolve ggml model file: explicit path in 'model' or <model_dir>/ggml-<size>.bin"""
    model = settings["model"]
    if os.path.isfile(model):
        return model
    candidates = []
    model_dir = settings.get("model_dir") or ""
    for d in filter(None, [model_dir, os.getcwd(), os.path.dirname(os.path.abspath(sys.argv[0])),
                           os.path.join(os.path.expanduser("~"), ".cache", "whisper.cpp")]):
        candidates += [os.path.join(d, f"ggml-{model}.bin"), os.path.join(d, f"{model}.bin"),
                       os.path.join(d, "models", f"ggml-{model}.bin")]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"whisper.cpp model 'ggml-{model}.bin' not found. Set the model folder on the Model tab "
        f"(download models with whisper.cpp's models/download-ggml-model.sh) or enter a full path to the .bin file.")


def _ts_to_seconds(h, m, s, frac) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(frac) / (10 ** len(frac))


def transcribe_whisper_cpp(audio_path: str, settings: Dict, cb: TranscribeCallbacks) -> Tuple[List[Dict], Dict]:
    exe = find_whisper_cli(settings.get("whisper_cli_path", ""))
    if not exe:
        raise FileNotFoundError("whisper-cli executable not found. Set its path on the Advanced tab.")
    model_path = _wcpp_model_path(settings)
    total = float(cb.extra.get("duration") or 0)

    cmd = [exe, "-m", model_path, "-f", audio_path, "-pp", "-np"]
    # -np suppresses banner/timings on stderr; timestamps stay on stdout
    cmd += ["-l", "auto" if settings["language"] == "auto" else settings["language"]]
    if settings["task"] == "translate":
        cmd.append("-tr")
    if int(settings["beam_size"]) > 1:
        cmd += ["-bs", str(int(settings["beam_size"]))]
    if int(settings.get("cpu_threads", 0)) > 0:
        cmd += ["-t", str(int(settings["cpu_threads"]))]
    if settings.get("initial_prompt"):
        cmd += ["--prompt", settings["initial_prompt"]]
    if settings.get("device") == "cpu":
        cmd.append("-ng")
    if settings.get("extra_args"):
        cmd += shlex.split(settings["extra_args"])

    cb.status("Transcribing with whisper.cpp…")
    proc = childproc.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                           encoding="utf-8", errors="replace", bufsize=1)
    cb.set_process(proc)
    stderr_lines = []

    def _drain_err():
        for line in proc.stderr:
            stderr_lines.append(line)
            m = _WCPP_PROGRESS.search(line)
            if m and total:
                cb.progress(total * int(m.group(1)) / 100.0, total)
    t = threading.Thread(target=_drain_err, daemon=True)
    t.start()

    segments = []
    try:
        for line in proc.stdout:
            if cb.should_stop():
                childproc.kill(proc)
                raise StoppedError()
            m = _WCPP_LINE.match(line.strip())
            if not m:
                continue
            start = _ts_to_seconds(*m.groups()[0:4])
            end = _ts_to_seconds(*m.groups()[4:8])
            text = m.group(9).strip()
            if not text:
                continue
            d = {"start": start, "end": end, "text": text}
            segments.append(d)
            cb.segment(d)
            if total:
                cb.progress(end, total)
        proc.wait()
    finally:
        childproc.forget(proc)
        cb.set_process(None)
        t.join(timeout=2)
    if cb.should_stop():
        raise StoppedError()
    if proc.returncode != 0:
        tail = "".join(stderr_lines[-15:]).strip()
        raise RuntimeError(f"whisper-cli exited with code {proc.returncode}:\n{tail}")
    meta = {"engine": "whisper.cpp", "model": os.path.basename(model_path),
            "language": settings["language"], "duration": total}
    return segments, meta


BACKENDS = {
    "faster_whisper": transcribe_faster_whisper,
    "whisper_cpp": transcribe_whisper_cpp,
}
