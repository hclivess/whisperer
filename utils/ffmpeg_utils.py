"""FFmpeg / ffprobe helpers: locating binaries, probing duration, extracting audio, muxing subtitles"""
import json
import os
import shutil
import subprocess
import sys
from typing import Optional

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _app_dir() -> str:
    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_binary(name: str, override: str = "") -> Optional[str]:
    """Look for an executable: explicit override, next to the app, then PATH"""
    exe = name + (".exe" if sys.platform == "win32" else "")
    if override:
        if os.path.isfile(override):
            return override
        which = shutil.which(override)
        if which:
            return which
    local = os.path.join(_app_dir(), exe)
    if os.path.isfile(local):
        return local
    return shutil.which(name)


def find_ffmpeg() -> Optional[str]:
    return find_binary("ffmpeg")


def find_ffprobe() -> Optional[str]:
    return find_binary("ffprobe")


def check_ffmpeg_status() -> bool:
    return find_ffmpeg() is not None


def probe_duration(path: str) -> Optional[float]:
    """Media duration in seconds via ffprobe (None if unavailable)"""
    ffprobe = find_ffprobe()
    if not ffprobe:
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=60, creationflags=_CREATE_NO_WINDOW)
        data = json.loads(out.stdout or "{}")
        return float(data["format"]["duration"])
    except Exception:
        return None


def extract_audio(src: str, dst_wav: str, stop_check=None) -> None:
    """Decode any media file to 16 kHz mono PCM WAV (what Whisper models expect)"""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found")
    # first_pts=0 keeps an audio stream that starts after the video (common in MKV remuxes / captures) in place by
    # padding silence, and async=1 fills timestamp gaps, so second N of the WAV is second N of the video.
    # Without it every cue is early by the audio start time and drifts over dropped packets.
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
           "-i", src, "-vn", "-sn", "-dn", "-map", "0:a:0",
           "-af", "aresample=async=1:first_pts=0", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", dst_wav]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            creationflags=_CREATE_NO_WINDOW)
    while True:
        try:
            _, err = proc.communicate(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if stop_check and stop_check():
                proc.kill()
                proc.wait()
                raise InterruptedError("Audio extraction stopped")
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to extract audio: {err.decode(errors='replace').strip()}")


def mux_subtitles(video: str, subtitle: str, output: str, container: str, language: str) -> None:
    """Copy the video streams and add the subtitle file as a soft subtitle track"""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found")
    codec = {"mkv": "srt", "mp4": "mov_text"}.get(container, "srt")
    lang = language if language and language != "auto" else "und"
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
           "-i", video, "-i", subtitle,
           "-map", "0", "-map", "1:0", "-c", "copy", "-c:s", codec,
           "-metadata:s:s:0", f"language={lang}", "-disposition:s:0", "default",
           output]
    proc = subprocess.run(cmd, capture_output=True, creationflags=_CREATE_NO_WINDOW)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to mux subtitles: {proc.stderr.decode(errors='replace').strip()}")
