"""
Subtitle post-processing and writers.

A Segment is a plain dict: {"start": float, "end": float, "text": str, "words": [ {start,end,word}, ... ]}
"""
import json
import os
import re
import textwrap
from typing import Dict, List, Optional


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


# Words that cannot be the last word of a sentence, whatever the speaker meant. Used only when there are no
# word timestamps to measure the pause with, so it stays tiny: prepositions are NOT here, because English strands
# them at the end of a sentence all the time ("things to attend to.", "what you're looking at.").
_CANNOT_END_SENTENCE = {"a", "an", "the"}
# Real abbreviations: their full stop is part of the word, never a sentence end.
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "st", "vs", "etc", "e.g", "i.e", "no", "fig", "approx", "inc"}
_DOT = re.compile(r"\.(?=\s|$)")


def _proper_nouns(segments: List[Dict]) -> set:
    """Words seen capitalised in mid-sentence: they keep their capital when a false sentence break is repaired."""
    proper = set()
    for seg in segments:
        words = seg.get("text", "").split()
        sentence_start = True
        for i, w in enumerate(words):
            core = w.strip("\"'([{)]}").rstrip(",;:.!?…")
            if not sentence_start and core[:1].isupper() and not core.isupper():
                proper.add(core.lower())
            sentence_start = bool(_SENTENCE_END.search(w))
    return proper


def _pauses(segments: List[Dict], i: int) -> Optional[List[Optional[float]]]:
    """
    Silence after each full stop of segment i, in text order, or None when it cannot be measured.

    The k-th token ending in "." is the k-th "<dot><space>" of the text, so the two streams line up; if they
    do not (a backend whose word list does not spell the text), we do not guess.
    """
    words = segments[i].get("words") or []
    if not words or not all("start" in w and "end" in w for w in words):
        return None
    out: List[Optional[float]] = []
    for k, w in enumerate(words):
        if not w["word"].strip().endswith("."):
            continue
        if k + 1 < len(words):
            out.append(words[k + 1]["start"] - w["end"])
        else:                                   # last word: the pause runs into the next segment
            nxt = segments[i + 1] if i + 1 < len(segments) else None
            nxt_words = (nxt.get("words") if nxt else None) or []
            start = nxt_words[0]["start"] if nxt_words else (nxt["start"] if nxt else None)
            out.append(None if start is None else start - w["end"])
    return out if len(out) == len(_DOT.findall(segments[i].get("text", ""))) else None


def _spurious(text: str, pos: int, pause: Optional[float], min_pause: float) -> bool:
    """Is the full stop at `pos` one the speaker never made?"""
    if pos and text[pos - 1] == ".":                       # "..." trails off on purpose
        return False
    before = text[:pos].split()
    word = before[-1].lower().strip("\"'([{)]}") if before else ""
    if len(word) == 1 or word in _ABBREVIATIONS or "." in word[:-1]:   # "J. R. R." / "Dr." / "U.S."
        return False
    if word[-1:].isdigit():                                # "3." of a decimal or a list number
        return False
    if pause is None:
        return word in _CANNOT_END_SENTENCE
    return pause < min_pause


def _lower_first(text: str, proper: set) -> str:
    """Lower-case the word that used to start a sentence, unless it is "I" or a name."""
    m = _FIRST_LETTER.match(text)
    if not m or not m.group(2).isupper():
        return text
    word = text[m.start(2):].split()[0].strip("\"'([{)]}").rstrip(",;:.!?")
    if word == "I" or word.startswith("I'") or word.isupper() or word.lower() in proper:
        return text
    return text[:m.start(2)] + m.group(2).lower() + text[m.end(2):]


def repair_sentence_breaks(segments: List[Dict], min_pause: float = 0.25) -> List[Dict]:
    """
    Remove full stops the speaker never made.

    Whisper punctuates by language model, not by ear, and regularly ends a sentence in the middle of a phrase -
    "When someone is. First about to embark on a minor task" - which then reads as two broken sentences and gets
    a capital letter from capitalize_sentences on top. People pause between sentences, so a full stop with less
    than `min_pause` of silence around it is the model's invention: it is dropped and the next word goes back to
    lower case (names and "I" keep their capital). Without word timestamps only full stops after a word that
    cannot end a sentence ("of.", "the.", "and.") are removed.
    """
    if not segments or min_pause <= 0:
        return segments
    out = [dict(s) for s in segments]
    proper = _proper_nouns(out)
    for i, seg in enumerate(out):
        text = seg.get("text", "")
        pauses = _pauses(out, i)
        positions = [m.start() for m in _DOT.finditer(text)]
        # the k-th full stop of the text is the k-th word token ending in "." - keep both in step
        dot_tokens = [k for k, w in enumerate(seg.get("words") or []) if w["word"].strip().endswith(".")]
        aligned = len(dot_tokens) == len(positions)
        words = None
        for k in range(len(positions) - 1, -1, -1):        # backwards: earlier positions stay valid
            pos = positions[k]
            if not _spurious(text, pos, pauses[k] if pauses else None, min_pause):
                continue
            head, tail = text[:pos], text[pos + 1:]
            if tail.strip():                               # the sentence continues inside this cue
                text = head + tail[:len(tail) - len(tail.lstrip())] + _lower_first(tail.lstrip(), proper)
            elif i + 1 < len(out) and pauses:               # ... or in the next one, but only on measured silence
                out[i + 1] = {**out[i + 1], "text": _lower_first(out[i + 1].get("text", ""), proper)}
                text = head
            else:
                continue
            if aligned:
                words = words if words is not None else [dict(w) for w in seg["words"]]
                t = words[dot_tokens[k]]["word"]
                words[dot_tokens[k]]["word"] = t.rstrip()[:-1] + t[len(t.rstrip()):]
        if text != seg.get("text", ""):
            seg["text"] = text
            if words is not None:
                seg["words"] = words
    return out


def _merged(a: Dict, b: Dict, capitalize: bool = True) -> Dict:
    head, tail = a.get("text", ""), b.get("text", "")
    if capitalize and _SENTENCE_END.search(head.rstrip()):
        m = _FIRST_LETTER.match(tail)                 # "Yeah." + "so what" -> "Yeah. So what"
        if m and m.group(2).islower():
            tail = tail[:m.start(2)] + m.group(2).upper() + tail[m.end(2):]
    out = {**a, "start": min(a["start"], b["start"]), "end": max(a["end"], b["end"]),
           "text": _clean(f"{head} {tail}")}
    words = (a.get("words") or []) + (b.get("words") or [])
    if words:
        out["words"] = words
    return out


def _grow(cues: List[Dict], i: int, min_duration: float, min_gap: float, max_duration: float,
          earlier: bool = False) -> None:
    """
    Stretch cue i towards min_duration using only the free time around it (never over a neighbour).

    The end is extended first; `earlier` also allows the start to be pulled back into the silence before the
    cue, which puts the text on screen ahead of its speech - only worth doing once merging has been ruled out.
    """
    cue = cues[i]
    if cue["end"] - cue["start"] >= min_duration:
        return
    ceiling = cues[i + 1]["start"] - min_gap if i + 1 < len(cues) else float("inf")
    if max_duration > 0:
        ceiling = min(ceiling, cue["start"] + max_duration)
    cue["end"] = max(cue["end"], min(cue["start"] + min_duration, ceiling))
    if not earlier or cue["end"] - cue["start"] >= min_duration:
        return
    floor = cues[i - 1]["end"] + min_gap if i > 0 else 0.0
    cue["start"] = min(cue["start"], max(cue["end"] - min_duration, floor))


def merge_short_cues(segments: List[Dict], min_duration: float = 0.8, min_gap: float = 0.08,
                     max_duration: float = 7.0, limit_chars: int = 84,
                     max_merge_gap: float = 1.5, capitalize: bool = True) -> List[Dict]:
    """
    Give every cue enough time on screen to be read.

    Whisper emits segments as short as 10 ms, and snapping can squeeze a cue against its neighbour; both flash
    a line of text for a frame or two. Simply extending the end would push the cue over the next one's speech,
    so the cue is first stretched into whatever free time surrounds it and, when there is none, glued to the
    neighbouring cue - two short lines shown together read fine, a delayed line does not.

    A merge is refused when the joined text would not fit the cue layout (limit_chars), when the result would
    run longer than max_duration, or when the two cues are more than max_merge_gap apart (they belong to
    different moments). Cues that cannot be fixed either way are left as they are.
    """
    if min_duration <= 0 or not segments:
        return segments
    cues = [dict(s) for s in segments]

    def short(c: Dict) -> bool:
        return c["end"] - c["start"] < min_duration - 1e-6

    def joinable(a: Dict, b: Dict) -> bool:
        if max(0.0, b["start"] - a["end"]) > max_merge_gap:
            return False
        if max_duration > 0 and b["end"] - a["start"] > max_duration:
            return False
        return limit_chars <= 0 or len(a.get("text", "")) + 1 + len(b.get("text", "")) <= limit_chars

    for i in range(len(cues)):
        _grow(cues, i, min_duration, min_gap, max_duration)

    i = 0
    while i < len(cues):
        if not short(cues[i]):
            i += 1
            continue
        merged_at = i
        fwd = i + 1 < len(cues) and joinable(cues[i], cues[i + 1])
        bwd = i > 0 and joinable(cues[i - 1], cues[i])
        if fwd and bwd:
            # keep sentences whole: never glue a cue onto a finished sentence when it can go the other way
            if _SENTENCE_END.search(cues[i - 1].get("text", "").rstrip()):
                bwd = False
            elif _SENTENCE_END.search(cues[i].get("text", "").rstrip()):
                fwd = False
            elif cues[i]["start"] - cues[i - 1]["end"] < cues[i + 1]["start"] - cues[i]["end"]:
                fwd = False
        if fwd:
            cues[i:i + 2] = [_merged(cues[i], cues[i + 1], capitalize)]
        elif bwd:
            cues[i - 1:i + 1] = [_merged(cues[i - 1], cues[i], capitalize)]
            i -= 1
            merged_at = i
        else:
            # nothing to glue it to: show it a little early rather than leaving a flash on screen
            _grow(cues, i, min_duration, min_gap, max_duration, earlier=True)
            i += 1
            continue
        _grow(cues, merged_at, min_duration, min_gap, max_duration)   # the merged cue may still be short

    return cues


import unicodedata  # noqa: E402


def _script(ch: str) -> str:
    """Coarse script class of a letter: latin, cyrillic, greek, cjk, kana, hangul, arabic, hebrew, thai, devanagari, other"""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "other"
    for key, script in (("LATIN", "latin"), ("CYRILLIC", "cyrillic"), ("GREEK", "greek"), ("CJK", "cjk"),
                        ("HIRAGANA", "kana"), ("KATAKANA", "kana"), ("HANGUL", "hangul"), ("ARABIC", "arabic"),
                        ("HEBREW", "hebrew"), ("THAI", "thai"), ("DEVANAGARI", "devanagari")):
        if key in name:
            return script
    return "other"


def strip_foreign_script(segments: List[Dict], max_share: float = 0.05) -> List[Dict]:
    """
    Remove letters from scripts that are foreign to the transcript.

    Whisper occasionally hallucinates a stray CJK / Hangul / Cyrillic character inside otherwise English text
    ("Bar标"). The dominant script of the whole transcript is determined first; any other script whose letters
    make up less than `max_share` of all letters is removed. Cues that become empty are dropped, so a genuinely
    bilingual transcript (where the second script is common) is left untouched.
    """
    counts: Dict[str, int] = {}
    for seg in segments:
        for ch in seg.get("text", ""):
            if ch.isalpha():
                sc = _script(ch)
                counts[sc] = counts.get(sc, 0) + 1
    total = sum(counts.values())
    if not total:
        return segments
    dominant = max(counts, key=counts.get)
    foreign = {sc for sc, n in counts.items() if sc != dominant and sc != "other" and n / total < max_share}
    if not foreign:
        return segments
    out = []
    for seg in segments:
        text = seg.get("text", "")
        cleaned = "".join(ch for ch in text if not (ch.isalpha() and _script(ch) in foreign))
        if cleaned != text:
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if not cleaned:
                continue
            seg = {**seg, "text": cleaned}
        out.append(seg)
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
