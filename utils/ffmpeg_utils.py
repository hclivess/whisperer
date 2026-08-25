"""FFmpeg / ffprobe helpers: locating binaries, probing duration, extracting audio, muxing and burning in subtitles"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Optional

from utils import childproc

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
        out = childproc.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            text=True, timeout=60)
        data = json.loads(out.stdout or "{}")
        return float(data["format"]["duration"])
    except Exception:
        return None


def has_video(path: str) -> bool:
    """True if the file carries a real video stream. Cover art in an MP3 is a video stream to ffprobe,
    so a single attached picture does not count — there is nothing there to burn subtitles into."""
    ffprobe = find_ffprobe()
    if not ffprobe:
        return False
    try:
        out = childproc.run(
            [ffprobe, "-v", "error", "-select_streams", "v",
             "-show_entries", "stream=codec_name,disposition", "-of", "json", path],
            text=True, timeout=60)
        for stream in json.loads(out.stdout or "{}").get("streams", []):
            disp = stream.get("disposition") or {}
            if not disp.get("attached_pic") and stream.get("codec_name") not in ("mjpeg", "png", "bmp", "gif"):
                return True
        return False
    except Exception:
        return False


def probe_fps(path: str) -> Optional[float]:
    """Frame rate of the first video stream (None when there is none, or ffprobe is missing).
    MicroDVD subtitles are written in frames, so they need it to mean anything."""
    ffprobe = find_ffprobe()
    if not ffprobe:
        return None
    try:
        out = childproc.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
             "-of", "json", path],
            text=True, timeout=60)
        streams = json.loads(out.stdout or "{}").get("streams") or []
        num, _, den = (streams[0].get("r_frame_rate") or "").partition("/")
        fps = float(num) / float(den or 1)
        return fps if 1.0 <= fps <= 1000.0 else None
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
    proc = childproc.popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    while True:
        try:
            _, err = proc.communicate(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if stop_check and stop_check():
                childproc.kill(proc)
                raise InterruptedError("Audio extraction stopped")
    childproc.forget(proc)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to extract audio: {err.decode(errors='replace').strip()}")


def mux_subtitles(video: str, subtitle: str, output: str, container: str, language: str, stop_check=None) -> None:
    """Copy the video streams and add the subtitle file as a soft subtitle track.
    Writes to a temporary name and renames on success, so a stopped or failed mux never leaves a partial video."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found")
    codec = {"mkv": "srt", "mp4": "mov_text"}.get(container, "srt")
    lang = language if language and language != "auto" else "und"
    tmp = f"{os.path.splitext(output)[0]}.part.{container}"
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
           "-i", video, "-i", subtitle,
           "-map", "0", "-map", "1:0", "-c", "copy", "-c:s", codec,
           "-metadata:s:s:0", f"language={lang}", "-disposition:s:0", "default",
           "-f", {"mkv": "matroska", "mp4": "mp4"}.get(container, "matroska"), tmp]
    proc = childproc.popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        while True:
            try:
                _, err = proc.communicate(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                if stop_check and stop_check():
                    childproc.kill(proc)
                    raise InterruptedError("Muxing stopped")
        childproc.forget(proc)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed to mux subtitles: {err.decode(errors='replace').strip()}")
        os.replace(tmp, output)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _run_with_progress(cmd, cwd=None, stop_check=None, progress=None, what="FFmpeg") -> None:
    """Run ffmpeg with -progress on stdout, reporting seconds encoded, and killable at any moment.
    stderr is drained by a thread: -loglevel error is quiet, but a file that warns on every frame
    would otherwise fill the pipe and deadlock the encode."""
    proc = childproc.popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    errors = []
    drain = threading.Thread(target=lambda: errors.append(proc.stderr.read()), daemon=True)
    drain.start()
    try:
        for raw in proc.stdout:
            if stop_check and stop_check():
                childproc.kill(proc)
                raise InterruptedError(f"{what} stopped")
            line = raw.decode(errors="replace").strip()
            if progress and line.startswith("out_time_us="):
                try:
                    progress(int(line.split("=", 1)[1]) / 1_000_000.0)
                except ValueError:
                    pass                      # "N/A" until the first frame is written
        proc.wait()
        drain.join(timeout=5)
        if proc.returncode != 0:
            err = (errors[0] if errors else b"").decode(errors="replace").strip()
            raise RuntimeError(f"{what} failed: {err or 'exit code %d' % proc.returncode}")
    finally:
        childproc.forget(proc)
        for pipe in (proc.stdout, proc.stderr):
            try:
                pipe.close()
            except Exception:
                pass


def burn_subtitles(video: str, subtitle: str, output: str, container: str = "mkv", crf: int = 20,
                   preset: str = "medium", stop_check=None, progress=None) -> None:
    """Burn the subtitles into the picture (hardcoding). The video is re-encoded — this is the slow one,
    and the only one that shows subtitles on a player that can display none.

    The subtitle file is copied to a temp folder as subs.srt and ffmpeg is run from there: the filter
    argument is parsed by ffmpeg, not the shell, so a real path's colons, backslashes, commas and
    brackets would each break it in their own way. A plain name in the working directory has none.

    Writes to a temporary name and renames on success, so a stopped or failed encode never leaves a
    half-written video where a whole one should be.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found")
    # ffmpeg is run from the temp folder for the filter's sake, so every other path must be absolute:
    # a relative one would be resolved against that folder and nothing would be found there
    video, subtitle, output = (os.path.abspath(p) for p in (video, subtitle, output))
    if not has_video(video):
        raise RuntimeError(f"{os.path.basename(video)} has no video stream — nothing to burn subtitles into. "
                           "Use 'Subtitle file' or 'Embed in the video' for audio files.")
    workdir = tempfile.mkdtemp(prefix="whisperer-burn-")
    tmp = f"{os.path.splitext(output)[0]}.part.{container}"
    try:
        shutil.copyfile(subtitle, os.path.join(workdir, "subs.srt"))
        base = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-nostats",
                "-progress", "pipe:1", "-i", video, "-vf", "subtitles=subs.srt",
                "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-crf", str(int(crf)),
                "-preset", preset, "-pix_fmt", "yuv420p"]
        tail = ["-f", {"mkv": "matroska", "mp4": "mp4"}.get(container, "matroska"), tmp]
        try:
            _run_with_progress(base + ["-c:a", "copy"] + tail, cwd=workdir, stop_check=stop_check,
                               progress=progress, what="Hardcoding")
        except RuntimeError:
            # audio the container will not carry (FLAC or PCM in an MP4, say) fails the stream copy and
            # nothing else does: re-encode the sound rather than lose the whole encode over it
            _run_with_progress(base + ["-c:a", "aac", "-b:a", "192k"] + tail, cwd=workdir,
                               stop_check=stop_check, progress=progress, what="Hardcoding")
        os.replace(tmp, output)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
