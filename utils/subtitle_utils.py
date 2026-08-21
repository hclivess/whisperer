"""
Subtitle post-processing and writers.

A Segment is a plain dict: {"start": float, "end": float, "text": str, "words": [ {start,end,word}, ... ]}
"""
import json
import os
import re
import textwrap
from typing import Dict, List


def _ts_srt(t: float) -> str:
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(t: float) -> str:
    return _ts_srt(t).replace(",", ".")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_segments(segments: List[Dict], max_line_chars: int = 42, max_lines: int = 2,
                   max_seconds: float = 7.0) -> List[Dict]:
    """
    Break segments that are too long (in characters or time) into smaller cues.
    Uses word timestamps when present, otherwise splits proportionally by character count.
    """
    limit_chars = max(1, max_line_chars) * max(1, max_lines)
    out = []
    for seg in segments:
        text = _clean(seg.get("text", ""))
        if not text:
            continue
        duration = seg["end"] - seg["start"]
        if len(text) <= limit_chars and (max_seconds <= 0 or duration <= max_seconds):
            out.append({**seg, "text": text})
            continue

        words = seg.get("words") or []
        if words and all("start" in w and "end" in w for w in words):
            out.extend(_split_by_words(words, limit_chars, max_seconds))
        else:
            out.extend(_split_proportional(seg["start"], seg["end"], text, limit_chars, max_seconds))
    return out


def _split_by_words(words: List[Dict], limit_chars: int, max_seconds: float) -> List[Dict]:
    cues, cur, cur_len = [], [], 0
    for w in words:
        token = w["word"].strip()
        if not token:
            continue
        add_len = len(token) + (1 if cur else 0)
        too_long = cur and (cur_len + add_len > limit_chars or
                            (max_seconds > 0 and w["end"] - cur[0]["start"] > max_seconds))
        if too_long:
            cues.append(cur)
            cur, cur_len = [], 0
            add_len = len(token)
        cur.append(w)
        cur_len += add_len
    if cur:
        cues.append(cur)
    return [{"start": c[0]["start"], "end": c[-1]["end"],
             "text": _clean(" ".join(w["word"] for w in c)), "words": c} for c in cues]


def _split_proportional(start: float, end: float, text: str, limit_chars: int,
                        max_seconds: float) -> List[Dict]:
    duration = end - start
    # how many pieces do we need?
    pieces = max(1, -(-len(text) // limit_chars))
    if max_seconds > 0:
        pieces = max(pieces, int(-(-duration // max_seconds)))
    target = max(1, -(-len(text) // pieces))
    chunks = textwrap.wrap(text, width=max(target, 1), break_long_words=False, break_on_hyphens=False) or [text]
    # make sure no chunk exceeds the char limit (wrap again if needed)
    final = []
    for c in chunks:
        final.extend(textwrap.wrap(c, width=limit_chars, break_long_words=True) or [c])
    total = sum(len(c) for c in final) or 1
    out, t = [], start
    for c in final:
        d = duration * len(c) / total
        out.append({"start": t, "end": t + d, "text": c})
        t += d
    if out:
        out[-1]["end"] = end
    return out


_SENTENCE_END = re.compile(r"""[.!?…。！？]["'”’»)\]]*$""")
_FIRST_LETTER = re.compile(r"""^([^\w]*)(\w)""", re.UNICODE)


def capitalize_sentences(segments: List[Dict]) -> List[Dict]:
    """
    Upper-case the first letter of the first cue and of every cue that follows a finished sentence.
    Whisper often emits a lower-case first word when the audio starts cold or after a VAD cut; cues that merely
    continue a sentence are left alone.
    """
    out = []
    new_sentence = True
    for seg in segments:
        text = seg.get("text", "")
        if new_sentence:
            m = _FIRST_LETTER.match(text)
            if m and m.group(2).islower():
                text = text[:m.start(2)] + m.group(2).upper() + text[m.end(2):]
        out.append({**seg, "text": text} if text != seg.get("text", "") else seg)
        stripped = text.rstrip()
        if stripped:
            new_sentence = bool(_SENTENCE_END.search(stripped))
    return out


def wrap_lines(text: str, max_line_chars: int, max_lines: int) -> str:
    """Balance a cue's text over as few lines as possible (no orphan words on the last line)"""
    if max_line_chars <= 0 or len(text) <= max_line_chars:
        return text
    needed = -(-len(text) // max_line_chars)
    if max_lines > 0:
        needed = min(needed, max_lines)
    # start from an even split and widen until the text fits in `needed` lines
    width = max(1, -(-len(text) // needed))
    while True:
        lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
        if len(lines) <= needed and all(len(l) <= max(width, max_line_chars) for l in lines):
            break
        width += 1
        if width > len(text):
            lines = [text]
            break
    return "\n".join(lines)


def to_srt(segments: List[Dict], max_line_chars: int, max_lines: int) -> str:
    parts = []
    for i, s in enumerate(segments, 1):
        parts.append(f"{i}\n{_ts_srt(s['start'])} --> {_ts_srt(s['end'])}\n"
                     f"{wrap_lines(s['text'], max_line_chars, max_lines)}\n")
    return "\n".join(parts)


def to_vtt(segments: List[Dict], max_line_chars: int, max_lines: int) -> str:
    parts = ["WEBVTT", ""]
    for s in segments:
        parts.append(f"{_ts_vtt(s['start'])} --> {_ts_vtt(s['end'])}\n"
                     f"{wrap_lines(s['text'], max_line_chars, max_lines)}\n")
    return "\n".join(parts)


def to_txt(segments: List[Dict]) -> str:
    return "\n".join(s["text"] for s in segments) + "\n"


def to_json(segments: List[Dict], meta: Dict) -> str:
    payload = {"meta": meta, "segments": [
        {"start": round(s["start"], 3), "end": round(s["end"], 3), "text": s["text"],
         **({"words": [{"start": round(w["start"], 3), "end": round(w["end"], 3), "word": w["word"]}
                       for w in s["words"]]} if s.get("words") else {})}
        for s in segments]}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def write_subtitles(segments: List[Dict], base_path: str, formats: List[str],
                    max_line_chars: int, max_lines: int, meta: Dict, overwrite: bool) -> List[str]:
    """Write every requested format next to base_path (without extension). Returns written paths."""
    written = []
    for fmt in formats:
        path = f"{base_path}.{fmt}"
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"Output exists (enable Overwrite): {path}")
        if fmt == "srt":
            data = to_srt(segments, max_line_chars, max_lines)
        elif fmt == "vtt":
            data = to_vtt(segments, max_line_chars, max_lines)
        elif fmt == "txt":
            data = to_txt(segments)
        elif fmt == "json":
            data = to_json(segments, meta)
        else:
            continue
        tmp = path + ".part"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(data)
            os.replace(tmp, path)          # never leave a half-written subtitle under the real name
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        written.append(path)
    return written
