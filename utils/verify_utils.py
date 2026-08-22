"""
Second-pass verification: two independent decodes of the same audio, kept where they agree.

A hallucination is text with no audio under it, so no amount of reading the text can prove it is one - only
the audio can. Whisper invents differently every time it is asked (different beam, no context carried over),
while real speech comes back the same, so the second decode is evidence about the first: text both passes
produce is trusted, text only one pass produced is decided by what the VAD heard in that span.

Nothing here rewrites words. A cue is confirmed, replaced by the other pass's decode of the same span,
dropped, or a span both passes agree on but the first one missed is added - and every one of those is
counted in the report.
"""
import difflib
from typing import Dict, List, Optional, Sequence, Tuple

from utils.sync_utils import _tokens

Region = Tuple[float, float]

# A cue whose words the second pass reproduces at least this well is the same sentence, punctuation and
# hesitations aside. Below it the two decodes genuinely disagree about what was said.
AGREE = 0.6
# How much of a span the VAD must find speech in before text over it is believable.
MIN_SPEECH = 0.35


def _words(segments: Sequence[Dict]) -> List[Dict]:
    """Flat (start, end, token, word) stream; segments without word timestamps are interpolated."""
    out: List[Dict] = []
    for seg in segments:
        words = seg.get("words") or []
        if words:
            for w in words:
                for tok in _tokens(w["word"]):
                    out.append({"start": float(w["start"]), "end": float(w["end"]),
                                "tok": tok, "word": w["word"], "raw": w})
        else:
            toks = _tokens(seg.get("text", ""))
            if not toks:
                continue
            start, end = float(seg["start"]), float(seg["end"])
            step = (end - start) / len(toks)
            for i, tok in enumerate(toks):
                out.append({"start": start + i * step, "end": start + (i + 1) * step,
                            "tok": tok, "word": tok, "raw": None})
    return out


def _in_span(words: List[Dict], start: float, end: float, pad: float = 0.4) -> List[Dict]:
    """Words of the other pass around a cue's span - padded, because the two passes cut segments differently."""
    pad = min(pad, max(0.05, (end - start) / 4))
    return [w for w in words if w["end"] > start - pad and w["start"] < end + pad]


def _inside(words: List[Dict], start: float, end: float) -> List[Dict]:
    """Words whose middle falls in the span. What replacement text is built from: no word twice in the file."""
    return [w for w in words if start <= (w["start"] + w["end"]) / 2 <= end]


def speech_fraction(regions: Optional[Sequence[Region]], start: float, end: float) -> Optional[float]:
    """How much of [start, end] the VAD found speech in, or None when there is nothing to ask."""
    if regions is None or end <= start:
        return None
    covered = sum(max(0.0, min(end, r_end) - max(start, r_start)) for r_start, r_end in regions)
    return covered / (end - start)


def _text_of(words: Sequence[Dict]) -> str:
    return " ".join(w["word"].strip() for w in words if w["word"].strip()).strip()


def _quality(seg: Dict, speech: Optional[float], n_tokens: int) -> float:
    """
    How much this decode of a span is worth believing. Higher is better.

    Whisper's own numbers first (avg_logprob is how sure it was, compression_ratio betrays a repetition
    loop), then how much text was claimed for the amount of speech actually there.
    """
    score = float(seg.get("avg_logprob", -0.5))
    if float(seg.get("no_speech_prob", 0.0)) > 0.5:
        score -= 1.0
    if float(seg.get("compression_ratio", 1.0)) > 2.4:            # the same phrase over and over
        score -= 1.0
    duration = max(1e-3, float(seg["end"]) - float(seg["start"]))
    speech_seconds = duration * (speech if speech is not None else 1.0)
    if n_tokens and speech_seconds < n_tokens * 0.12:             # more words than there was speech for
        score -= 1.0
    return score


def verify_second_pass(first: List[Dict], second: List[Dict], regions: Optional[Sequence[Region]] = None,
                       agree: float = AGREE, min_speech: float = MIN_SPEECH) -> Tuple[List[Dict], Dict]:
    """
    Adjudicate a first transcript against a second decode of the same audio. Returns (segments, report).

    Timing is the first pass's throughout: a replaced cue keeps its span and only its text changes, so
    everything downstream (snapping, resync, the cue layout) sees the timeline it always saw.
    """
    report = {"checked": len(first), "confirmed": 0, "replaced": 0, "dropped": 0, "added": 0, "removed": []}
    if not first:
        return first, report
    if not second:
        report["note"] = "second pass produced nothing - the first pass was kept as it is"
        return first, report

    second_words = _words(second)
    by_time = sorted(second, key=lambda s: float(s["start"]))
    out: List[Dict] = []
    for seg in first:
        start, end = float(seg["start"]), float(seg["end"])
        mine = _tokens(seg.get("text", ""))
        theirs = _in_span(second_words, start, end)
        speech = speech_fraction(regions, start, end)
        ratio = difflib.SequenceMatcher(None, mine, [w["tok"] for w in theirs], autojunk=False).ratio()

        if not theirs:
            # the second pass heard nothing here: hallucinated, unless the VAD insists there was speech
            if speech is None or speech < min_speech:
                report["dropped"] += 1
                if len(report["removed"]) < 20:
                    report["removed"].append({"start": round(start, 2), "end": round(end, 2),
                                              "text": seg.get("text", "")[:120]})
                continue
            out.append(seg)
            continue

        if ratio >= agree:
            report["confirmed"] += 1
            out.append(seg)
            continue

        # the two decodes disagree: believe the one the audio supports
        other = _nearest_segment(by_time, start, end)
        if other is not None and _quality(other, speech, len(theirs)) > _quality(seg, speech, len(mine)) + 0.25:
            own = _inside(second_words, start, end)               # only what belongs to this cue
            text = _text_of(own)
            if text:
                report["replaced"] += 1
                new = {**seg, "text": text}
                raws = [w["raw"] for w in own if w["raw"]]
                if raws:
                    new["words"] = [dict(w) for w in raws]
                else:
                    new.pop("words", None)
                for k in ("avg_logprob", "no_speech_prob", "compression_ratio", "temperature"):
                    if k in other:
                        new[k] = other[k]
                out.append(new)
                continue
        out.append(seg)

    added = _missed_spans(out, second, regions, min_speech)
    if added:
        report["added"] = len(added)
        out = sorted(out + added, key=lambda s: float(s["start"]))
    return out, report


def _nearest_segment(by_time: List[Dict], start: float, end: float) -> Optional[Dict]:
    """The second pass's segment that overlaps [start, end] the most."""
    best, best_overlap = None, 0.0
    for seg in by_time:
        if float(seg["start"]) >= end:
            break
        overlap = min(end, float(seg["end"])) - max(start, float(seg["start"]))
        if overlap > best_overlap:
            best, best_overlap = seg, overlap
    return best


def _missed_spans(kept: List[Dict], second: List[Dict],
                  regions: Optional[Sequence[Region]], min_speech: float) -> List[Dict]:
    """
    Speech the first pass skipped altogether but the second pass and the VAD both found.

    Only added on the VAD's word: without speech regions there is no independent witness, and one pass
    inventing a line is exactly what this whole function exists to catch.
    """
    if regions is None:
        return []
    out = []
    for seg in second:
        start, end = float(seg["start"]), float(seg["end"])
        if any(min(end, float(k["end"])) - max(start, float(k["start"])) > 0.1 for k in kept):
            continue
        speech = speech_fraction(regions, start, end)
        if speech is None or speech < max(min_speech, 0.5):
            continue
        if float(seg.get("no_speech_prob", 0.0)) > 0.5 or float(seg.get("compression_ratio", 1.0)) > 2.4:
            continue
        if not _tokens(seg.get("text", "")):
            continue
        out.append(dict(seg))
    return out


def vocabulary_prompt(segments: Sequence[Dict], base: str = "", limit: int = 30, max_chars: int = 200) -> str:
    """
    The names the first pass kept using, as a prompt for the second one.

    Whisper spells an unfamiliar name differently every time it meets it; handing back the spellings it
    settled on makes the second decode consistent with itself. A word qualifies when it was capitalised in
    mid-sentence at least once - which a plain sentence start never is - and turns up more than once.
    """
    counts: Dict[str, int] = {}
    mid_sentence = set()
    for seg in segments:
        words = seg.get("text", "").split()
        sentence_start = True
        for w in words:
            core = w.strip("\"'([{)]}").rstrip(",;:.!?…")
            if len(core) > 2 and core[:1].isupper() and not core.isupper():
                counts[core] = counts.get(core, 0) + 1
                if not sentence_start:
                    mid_sentence.add(core)
            sentence_start = bool(core[-1:] in ".!?…")
    names = [w for w, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
             if n > 1 and w in mid_sentence][:limit]
    if not names:
        return base
    listing = ", ".join(names)
    if len(listing) > max_chars:
        listing = listing[:max_chars].rsplit(",", 1)[0]
    return f"{base.strip()} {listing}".strip() if base.strip() else listing
