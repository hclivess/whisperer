"""
Subtitle synchronisation.

Two tools, both driven by the audio itself:

* snap_to_speech()  — move every cue's start / end onto the nearest speech onset / offset found by Silero VAD,
                      trim cues that hang over silence, enforce minimum duration, minimum gap and no overlaps.
                      Whisper's own timestamps are decoder guesses quantised to 20 ms and drift at segment edges;
                      the VAD sees the waveform.
* resync_cues()     — align an existing subtitle file to a fresh transcript of the same audio: match the words,
                      robust-fit a linear time model (offset + speed, the SubSync model) and apply it to every cue.

A cue is a plain dict {"start": float, "end": float, "text": str, "words": [...]?}.
"""
import bisect
import difflib
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

Region = Tuple[float, float]


# ---------------------------------------------------------------- speech detection
def speech_regions(audio_path: str, threshold: float = 0.5, min_silence_ms: int = 100,
                   min_speech_ms: int = 60, pad_ms: int = 30) -> List[Region]:
    """Speech regions (seconds) from Silero VAD, bundled with faster-whisper."""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = decode_audio(audio_path, sampling_rate=16000)
    opts = VadOptions(threshold=threshold, min_silence_duration_ms=min_silence_ms,
                      min_speech_duration_ms=min_speech_ms, speech_pad_ms=pad_ms)
    ts = get_speech_timestamps(audio, opts, sampling_rate=16000)
    regions = [(t["start"] / 16000.0, t["end"] / 16000.0) for t in ts]
    return _merge(regions)


def _merge(regions: List[Region], gap: float = 0.0) -> List[Region]:
    out: List[Region] = []
    for s, e in sorted(regions):
        if out and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _nearest(values: Sequence[float], t: float, max_dist: float) -> Optional[float]:
    if not values:
        return None
    i = bisect.bisect_left(values, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(values) and abs(values[j] - t) <= max_dist:
            if best is None or abs(values[j] - t) < abs(best - t):
                best = values[j]
    return best


def _region_at(regions: List[Region], starts: Sequence[float], t: float) -> Optional[Region]:
    i = bisect.bisect_right(starts, t) - 1
    if i >= 0 and regions[i][0] <= t <= regions[i][1]:
        return regions[i]
    return None


# ---------------------------------------------------------------- snapping
def snap_to_speech(cues: List[Dict], regions: List[Region], max_shift: float = 0.6, end_padding: float = 0.2,
                   min_duration: float = 0.8, min_gap: float = 0.08, max_duration: float = 0.0) -> List[Dict]:
    """
    Align cue boundaries with detected speech.

    start: nearest speech onset within max_shift; if the start sits in silence and no onset is near, it is
           pulled forward to the next onset when that one is within max_shift.
    end:   nearest speech offset within max_shift; a cue that runs over silence is cut at the end of the last
           speech region it covers. end_padding is added afterwards (never into the next cue).
    """
    if not cues:
        return []
    regions = _merge(regions)
    starts = [r[0] for r in regions]
    ends = [r[1] for r in regions]
    out: List[Dict] = []
    for cue in cues:
        s, e = float(cue["start"]), float(cue["end"])
        orig_s, orig_e = s, e
        if regions:
            on = _nearest(starts, s, max_shift)
            if on is not None:
                s = on
            elif _region_at(regions, starts, s) is None:
                i = bisect.bisect_left(starts, s)
                if i < len(starts) and starts[i] - s <= max_shift:
                    s = starts[i]
            off = _nearest(ends, e, max_shift)
            if off is not None and off > s:
                e = off
            else:
                # cut at the last speech end inside the cue, if the cue overhangs silence
                i = bisect.bisect_right(ends, e) - 1
                if i >= 0 and ends[i] > s and _region_at(regions, starts, e) is None:
                    e = ends[i]
        if e - s < 0.05:                      # snapping collapsed it: keep the original timing
            s, e = orig_s, orig_e
        out.append({**cue, "start": s, "end": e})

    # padding, minimum duration, overlaps, gaps
    for i, cue in enumerate(out):
        nxt = out[i + 1]["start"] if i + 1 < len(out) else None
        e = cue["end"] + end_padding
        if cue["end"] - cue["start"] < min_duration:
            e = max(e, cue["start"] + min_duration)
        if max_duration > 0:
            e = min(e, cue["start"] + max_duration)
        if nxt is not None:
            e = min(e, nxt - min_gap)
        cue["end"] = max(e, cue["start"] + 0.05)
    # a later cue that still starts before the previous one ends (snapped onto the same onset)
    for i in range(1, len(out)):
        if out[i]["start"] < out[i - 1]["end"] + min_gap:
            out[i]["start"] = out[i - 1]["end"] + min_gap
            if out[i]["end"] < out[i]["start"] + 0.05:
                out[i]["end"] = out[i]["start"] + max(0.05, min_duration)
    return out


def shift_cues(cues: List[Dict], offset: float) -> List[Dict]:
    if not offset:
        return cues
    out = []
    for c in cues:
        s = max(0.0, c["start"] + offset)
        e = max(s + 0.05, c["end"] + offset)
        d = {**c, "start": s, "end": e}
        if c.get("words"):
            d["words"] = [{**w, "start": max(0.0, w["start"] + offset), "end": max(0.0, w["end"] + offset)}
                          for w in c["words"]]
        out.append(d)
    return out


# ---------------------------------------------------------------- SRT / VTT reading
_TS = re.compile(r"(\d+):(\d{1,2}):(\d{1,2})[.,](\d{1,3})")
_TAG = re.compile(r"<[^>]+>|\{\\[^}]*\}")


def _ts(m) -> float:
    h, mi, s, frac = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(frac.ljust(3, "0")) / 1000.0


def parse_subtitles(text: str) -> List[Dict]:
    """Parse SRT or WebVTT text into cues. Tags are stripped; cue settings after the timing are ignored."""
    text = text.lstrip("﻿")
    cues: List[Dict] = []
    block: List[str] = []

    def flush():
        if not block:
            return
        for i, line in enumerate(block):
            times = _TS.findall(line)
            if "-->" in line and len(times) >= 2:
                ms = list(_TS.finditer(line))
                start, end = _ts(ms[0]), _ts(ms[1])
                body = "\n".join(block[i + 1:]).strip()
                body = _TAG.sub("", body)
                if body:
                    cues.append({"start": start, "end": end, "text": body})
                return

    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if line.strip():
            block.append(line)
        else:
            flush()
            block = []
    flush()
    cues.sort(key=lambda c: c["start"])
    return cues


# ---------------------------------------------------------------- resync (SubSync model)
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text)]


def reference_words(segments: List[Dict]) -> List[Tuple[float, str]]:
    """(time, token) anchors from a transcript: real word timestamps when present, else interpolated."""
    anchors: List[Tuple[float, str]] = []
    for seg in segments:
        words = seg.get("words") or []
        if words:
            for w in words:
                for tok in _tokens(w["word"]):
                    anchors.append((float(w["start"]), tok))
        else:
            toks = _tokens(seg.get("text", ""))
            if not toks:
                continue
            dur = float(seg["end"]) - float(seg["start"])
            for i, tok in enumerate(toks):
                anchors.append((float(seg["start"]) + dur * i / len(toks), tok))
    return anchors


def subtitle_words(cues: List[Dict]) -> List[Tuple[float, str]]:
    anchors: List[Tuple[float, str]] = []
    for c in cues:
        toks = _tokens(c["text"])
        if not toks:
            continue
        dur = float(c["end"]) - float(c["start"])
        for i, tok in enumerate(toks):
            anchors.append((float(c["start"]) + dur * i / len(toks), tok))
    return anchors


def match_points(sub_anchors: List[Tuple[float, str]], ref_anchors: List[Tuple[float, str]],
                 min_block: int = 3) -> List[Tuple[float, float]]:
    """(subtitle time, audio time) pairs for words that match in runs of at least min_block."""
    a = [t for _, t in sub_anchors]
    b = [t for _, t in ref_anchors]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    pts: List[Tuple[float, float]] = []
    for i, j, n in sm.get_matching_blocks():
        if n >= min_block:
            for k in range(n):
                pts.append((sub_anchors[i + k][0], ref_anchors[j + k][0]))
    return pts


def fit_linear(points: List[Tuple[float, float]], fit_speed: bool = True, tolerance: float = 0.7,
               seed: int = 7) -> Tuple[float, float, int]:
    """
    Robust fit of audio_time = speed * sub_time + offset. Returns (speed, offset, inliers).
    Theil–Sen style median slope over random pairs, then least squares on the inliers.
    """
    if len(points) < 2:
        return 1.0, (points[0][1] - points[0][0]) if points else 0.0, len(points)
    rnd = random.Random(seed)
    span = max(x for x, _ in points) - min(x for x, _ in points)
    if span < 120.0:                             # speed cannot be measured on a short clip
        fit_speed = False
    if fit_speed:
        slopes = []
        for _ in range(min(4000, len(points) * 20)):
            (x1, y1), (x2, y2) = rnd.sample(points, 2)
            if abs(x2 - x1) > 20.0:              # far apart pairs give a stable slope
                slopes.append((y2 - y1) / (x2 - x1))
        speed = sorted(slopes)[len(slopes) // 2] if slopes else 1.0
        if not 0.9 <= speed <= 1.1:              # only film-rate conversions are plausible (23.976/25 = 0.959)
            speed = 1.0
    else:
        speed = 1.0
    offsets = sorted(y - speed * x for x, y in points)
    offset = offsets[len(offsets) // 2]
    inliers = [(x, y) for x, y in points if abs(y - (speed * x + offset)) <= tolerance]
    if len(inliers) >= 2 and fit_speed:
        n = len(inliers)
        mx = sum(x for x, _ in inliers) / n
        my = sum(y for _, y in inliers) / n
        sxx = sum((x - mx) ** 2 for x, _ in inliers)
        if sxx > 0:
            sp = sum((x - mx) * (y - my) for x, y in inliers) / sxx
            if 0.9 <= sp <= 1.1:
                speed = sp
                offset = my - speed * mx
    elif inliers:
        offset = sum(y - speed * x for x, y in inliers) / len(inliers)
    return speed, offset, len(inliers)


def apply_linear(cues: List[Dict], speed: float, offset: float) -> List[Dict]:
    out = []
    for c in cues:
        s = max(0.0, speed * c["start"] + offset)
        e = max(s + 0.05, speed * c["end"] + offset)
        out.append({**c, "start": s, "end": e})
    return out


def resync_cues(cues: List[Dict], transcript: List[Dict], fit_speed: bool = True) -> Tuple[List[Dict], Dict]:
    """Align existing cues to a transcript of the same audio. Returns (new cues, report)."""
    pts = match_points(subtitle_words(cues), reference_words(transcript))
    if len(pts) < 5:
        raise RuntimeError(f"Could not match the subtitles to the audio ({len(pts)} word matches) — "
                           "wrong language, wrong file, or the transcript is too poor.")
    speed, offset, inliers = fit_linear(pts, fit_speed)
    report = {"matches": len(pts), "inliers": inliers, "offset": round(offset, 3), "speed": round(speed, 6),
              "drift_per_hour": round((speed - 1.0) * 3600, 2)}
    if inliers < 5 or inliers < 0.3 * len(pts):
        raise RuntimeError(f"Subtitles do not fit a consistent offset ({inliers} of {len(pts)} matches agree) — "
                           "they may belong to a different cut of the video.")
    return apply_linear(cues, speed, offset), report
