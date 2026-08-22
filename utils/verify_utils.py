"""
Multi-pass verification: decode the same audio more than once and keep what the decodes agree on.

A hallucination is text with no audio under it, so no amount of reading the text can prove it is one - only
the audio can. Whisper invents differently every time it is asked (different beam, no context carried over,
a little sampling temperature) while real speech comes back the same, so every extra decode is evidence
about the first one.

Agreement decides, not confidence: invented text is often fluent and scores well, but two independent
decodes hardly ever invent the *same* words. A span where nothing agrees with anything is handed back to
the caller as unresolved, so it can be decoded again; when the passes run out, the score breaks the tie and
the span is reported as unresolved rather than silently trusted.

Nothing here rewrites words, and the first pass owns the timeline: a cue keeps its span whatever happens to
its text.
"""
import difflib
from typing import Dict, List, Optional, Sequence, Tuple

from utils.sync_utils import _tokens

Region = Tuple[float, float]

# Two decodes of the same span that reproduce each other this well said the same sentence, punctuation and
# hesitations aside. Below it they genuinely disagree about the words.
AGREE = 0.6
# How much of a span the VAD must find speech in before text over it is believable (used for scoring).
MIN_SPEECH = 0.35
# What the VAD has to find before it may vote a cue away: near enough to nothing, by both measures. A long
# cue with a short utterance in it has a low speech *fraction* and real words - that must never be a vote.
SILENT_FRACTION = 0.15
SILENT_SECONDS = 0.3
# How far a cue's claim on the other decodes' words may reach past its own span, when it is being replaced.
TERRITORY = 2.0
# Quality margin a rival decode must beat the first pass by before it may replace it on score alone.
MARGIN = 0.25
# Agreeing passes that still worded a cue differently below this are worth a line in the review list: a
# meaning-changing mis-hearing ("mating game" heard as "marriage") is a partial disagreement, not a fight.
WORDING = 0.9


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
    """Words of another pass around a cue's span - padded, because the passes cut segments differently."""
    pad = min(pad, max(0.05, (end - start) / 4))
    return [w for w in words if w["end"] > start - pad and w["start"] < end + pad]


def _inside(words: List[Dict], start: float, end: float) -> List[Dict]:
    """Words whose middle falls in the span."""
    return [w for w in words if start <= (w["start"] + w["end"]) / 2 <= end]


def _territories(cues: List[Dict], reach: float = TERRITORY) -> List[Region]:
    """
    The stretch of audio each cue owns: out to the middle of the gap to its neighbour, at most `reach` past
    its own span.

    Replacement text is taken from a cue's territory rather than its exact span. The passes cut segments in
    slightly different places, so a word sitting just outside the cue's own start or end still belongs to
    somebody - taking only the exact span dropped those words from the file, and taking a padded span put
    them in twice. Territories touch and never overlap, so every word of another decode lands in exactly
    one cue.
    """
    out: List[Region] = []
    for i, cue in enumerate(cues):
        start, end = float(cue["start"]), float(cue["end"])
        if i > 0:
            start = max(start - reach, (float(cues[i - 1]["end"]) + start) / 2)
        else:
            start = start - reach
        if i + 1 < len(cues):
            end = min(end + reach, (end + float(cues[i + 1]["start"])) / 2)
        else:
            end = end + reach
        out.append((start, end))
    return out


def speech_fraction(regions: Optional[Sequence[Region]], start: float, end: float) -> Optional[float]:
    """How much of [start, end] the VAD found speech in, or None when there is nothing to ask."""
    if regions is None or end <= start:
        return None
    covered = sum(max(0.0, min(end, r_end) - max(start, r_start)) for r_start, r_end in regions)
    return covered / (end - start)


def _covers(decode: Dict, start: float, end: float) -> bool:
    """Did this decode look at this span at all? A pass aimed at a few windows must not vote outside them."""
    covers = decode.get("covers")
    if not covers:
        return True
    return any(min(end, c_end) - max(start, c_start) > 0.1 for c_start, c_end in covers)


def _fully_seen(decode: Dict, start: float, end: float) -> bool:
    """
    Did this decode hear the whole stretch, or only part of it?

    A pass aimed at a few windows has hard edges: a cue lying across one of them was decoded up to the cut
    and no further. Such a decode may still vote on the cue - it heard most of it - but its text must never
    replace the cue's, because the words past the cut were never transcribed and would be lost.
    """
    covers = decode.get("covers")
    if not covers:
        return True
    return any(c_start <= start and end <= c_end for c_start, c_end in covers)


def _text_of(words: Sequence[Dict]) -> str:
    return " ".join(w["word"].strip() for w in words if w["word"].strip()).strip()


def _quality(seg: Optional[Dict], speech: Optional[float], n_tokens: int, span: float) -> float:
    """
    How much a decode of a span is worth believing, when nothing agrees and something has to decide.

    Whisper's own numbers first (avg_logprob is how sure it was, compression_ratio betrays a repetition
    loop), then how much text was claimed for the amount of speech actually there.
    """
    seg = seg or {}
    score = float(seg.get("avg_logprob", -0.5))
    if float(seg.get("no_speech_prob", 0.0)) > 0.5:
        score -= 1.0
    if float(seg.get("compression_ratio", 1.0)) > 2.4:            # the same phrase over and over
        score -= 1.0
    speech_seconds = max(1e-3, span) * (speech if speech is not None else 1.0)
    if n_tokens and speech_seconds < n_tokens * 0.12:             # more words than there was speech for
        score -= 1.0
    return score


def _nearest_segment(segments: List[Dict], start: float, end: float) -> Optional[Dict]:
    """The segment of another pass that overlaps [start, end] the most."""
    best, best_overlap = None, 0.0
    for seg in segments:
        overlap = min(end, float(seg["end"])) - max(start, float(seg["start"]))
        if overlap > best_overlap:
            best, best_overlap = seg, overlap
    return best


def _support(candidates: List[Dict], agree: float) -> List[int]:
    """For each candidate text, how many of the others say the same thing."""
    votes = []
    for i, a in enumerate(candidates):
        n = 0
        for j, b in enumerate(candidates):
            if i == j:
                continue
            if difflib.SequenceMatcher(None, a["tokens"], b["tokens"], autojunk=False).ratio() >= agree:
                n += 1
        votes.append(n)
    return votes


def resolve_passes(decodes: List[Dict], regions: Optional[Sequence[Region]] = None,
                   agree: float = AGREE, min_speech: float = MIN_SPEECH,
                   margin: float = MARGIN) -> Tuple[List[Dict], Dict, List[Region]]:
    """
    Adjudicate several decodes of the same audio. Returns (segments, report, spans still unresolved).

    `decodes` is a list of {"segments": [...], "covers": [(start, end), ...] or None}; the first one is the
    transcript being verified and owns the timing, and a decode only votes on the spans it actually looked
    at. Spans where no two decodes agree come back in the third value so the caller can spend another pass
    on them - the score decides them meanwhile, and they are counted as unresolved either way.
    """
    report = {"passes": len(decodes), "checked": 0, "confirmed": 0, "majority": 0, "scored": 0,
              "dropped": 0, "added": 0, "unresolved": 0, "removed": [], "review": []}
    if not decodes or not decodes[0].get("segments"):
        return (decodes[0].get("segments") if decodes else []), report, []
    first = decodes[0]["segments"]
    others = [d for d in decodes[1:] if d.get("segments") is not None]
    if not others:
        return first, report, []

    prepared = [{"decode": d, "words": _words(d.get("segments") or []),
                 "segments": d.get("segments") or []} for d in others]
    territories = _territories(first)
    out: List[Dict] = []
    unresolved: List[Region] = []

    for index, seg in enumerate(first):
        start, end = float(seg["start"]), float(seg["end"])
        speech = speech_fraction(regions, start, end)
        voters = [p for p in prepared if _covers(p["decode"], start, end)]
        if not voters:
            out.append(seg)
            continue
        report["checked"] += 1

        candidates = [{"tokens": _tokens(seg.get("text", "")), "seg": seg, "words": None,
                       "pass": None, "base": True, "text": seg.get("text", "")}]
        empty_votes = 0
        for p in voters:
            words = _in_span(p["words"], start, end)
            if not words:
                empty_votes += 1
                continue
            candidates.append({"tokens": [w["tok"] for w in words], "words": words, "base": False,
                               "seg": _nearest_segment(p["segments"], start, end), "pass": p,
                               # compared on the padded span, but shown as what belongs to this cue: a
                               # review line must not seem to hold a word from the cue next door
                               "text": _text_of(_inside(p["words"], *territories[index]))})
        if speech is not None and speech < SILENT_FRACTION and speech * (end - start) < SILENT_SECONDS:
            empty_votes += 1                      # the VAD gets a vote, and it can only vote for silence

        votes = _support(candidates, agree)
        best = max(votes) + 1 if votes else 1     # size of the biggest group that says the same thing

        if empty_votes and empty_votes > best:
            # more of the evidence says there was nothing here than says there was something
            report["dropped"] += 1
            if len(report["removed"]) < 20:
                report["removed"].append({"start": round(start, 2), "end": round(end, 2),
                                          "text": seg.get("text", "")[:120]})
            _note(report, start, end, "dropped: nothing was heard here", "", candidates)
            continue

        if best >= 2:
            winners = [c for c, v in zip(candidates, votes) if v + 1 == best]
            if any(c["base"] for c in winners):
                report["confirmed"] += 1          # the first pass is in the majority: keep it word for word
                # the furthest reading, not the nearest: one pass hearing something else entirely is the
                # interesting part, even when the others agree and the cue is settled
                furthest = min((difflib.SequenceMatcher(None, candidates[0]["tokens"], c["tokens"],
                                                        autojunk=False).ratio()
                                for c in candidates[1:] if _seen_whole(c, territories[index])), default=1.0)
                if furthest < WORDING:
                    _note(report, start, end, "the passes agreed but worded it differently",
                          seg.get("text", ""), candidates)
                out.append(seg)
                continue
            pick = max(winners, key=lambda c: _quality(c["seg"], speech, len(c["tokens"]), end - start))
            replaced = _replace(seg, pick, territories[index]) if _seen_whole(pick, territories[index]) else None
            if replaced is not None:
                report["majority"] += 1
                _note(report, start, end, "the other passes agreed against the first one",
                      replaced["text"], candidates)
                out.append(replaced)
                continue
            out.append(seg)
            continue

        # nothing agrees with anything: this span wants another pass, and the score decides meanwhile
        report["unresolved"] += 1
        unresolved.append((start, end))
        base_score = _quality(seg, speech, len(candidates[0]["tokens"]), end - start)
        rivals = [c for c in candidates if not c["base"]]
        pick = max(rivals, key=lambda c: _quality(c["seg"], speech, len(c["tokens"]), end - start),
                   default=None)
        if (pick is not None and _seen_whole(pick, territories[index])
                and _quality(pick["seg"], speech, len(pick["tokens"]), end - start) > base_score + margin):
            replaced = _replace(seg, pick, territories[index])
            if replaced is not None:
                report["scored"] += 1
                _note(report, start, end, "no two passes agreed - picked on confidence",
                      replaced["text"], candidates)
                out.append(replaced)
                continue
        _note(report, start, end, "no two passes agreed - the first pass was kept",
              seg.get("text", ""), candidates)
        out.append(seg)

    added = _missed_spans(out, prepared, regions, min_speech)
    if added:
        report["added"] = len(added)
        out = sorted(out + added, key=lambda s: float(s["start"]))
    return out, report, unresolved


def _seen_whole(pick: Dict, territory: Region) -> bool:
    """May this candidate's text stand in for the cue? Only if its pass heard the cue's whole territory."""
    return not pick.get("pass") or _fully_seen(pick["pass"]["decode"], *territory)


def _note(report: Dict, start: float, end: float, why: str, chosen: str, candidates: List[Dict]) -> None:
    """One line for the review list: where the passes did not simply agree, and what each of them said."""
    if len(report["review"]) >= 500:
        return
    report["review"].append({
        "start": round(start, 2), "end": round(end, 2), "why": why, "chosen": chosen,
        "said": [c.get("text", "") for c in candidates],
    })


def _replace(seg: Dict, pick: Dict, territory: Region) -> Optional[Dict]:
    """The first pass's cue with another decode's words in it. Timing, and everything else, stays."""
    own = _inside(pick["pass"]["words"], *territory) if pick.get("pass") else (pick.get("words") or [])
    text = _text_of(own)
    if not text:
        return None
    new = {**seg, "text": text}
    raws = [w["raw"] for w in own if w["raw"]]
    if raws:
        new["words"] = [dict(w) for w in raws]
    else:
        new.pop("words", None)
    for key in ("avg_logprob", "no_speech_prob", "compression_ratio", "temperature"):
        if pick.get("seg") and key in pick["seg"]:
            new[key] = pick["seg"][key]
    return new


def _missed_spans(kept: List[Dict], prepared: List[Dict], regions: Optional[Sequence[Region]],
                  min_speech: float) -> List[Dict]:
    """
    Speech the first pass skipped altogether but a later pass and the VAD both found.

    Only added on the VAD's word: without speech regions there is no independent witness, and one pass
    inventing a line is exactly what this module exists to catch.
    """
    if regions is None:
        return []
    out: List[Dict] = []
    for p in prepared:
        for seg in p["segments"]:
            start, end = float(seg["start"]), float(seg["end"])
            if any(min(end, float(k["end"])) - max(start, float(k["start"])) > 0.1 for k in kept + out):
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


def verify_second_pass(first: List[Dict], second: List[Dict], regions: Optional[Sequence[Region]] = None,
                       agree: float = AGREE, min_speech: float = MIN_SPEECH) -> Tuple[List[Dict], Dict]:
    """Two decodes: the plain case of resolve_passes, kept under its own name."""
    if not second:
        report = {"passes": 1, "checked": 0, "confirmed": 0, "majority": 0, "scored": 0, "dropped": 0,
                  "added": 0, "unresolved": 0, "replaced": 0, "removed": [],
                  "note": "the extra pass produced nothing - the first pass was kept as it is"}
        return first, report
    segments, report, _unresolved = resolve_passes(
        [{"segments": first, "covers": None}, {"segments": second, "covers": None}],
        regions, agree=agree, min_speech=min_speech)
    report["replaced"] = report["majority"] + report["scored"]
    return segments, report


def merge_windows(spans: Sequence[Region], window: float = 30.0, duration: float = 0.0,
                  pad: float = 1.0) -> List[Region]:
    """
    The audio a follow-up pass has to decode in order to settle `spans`.

    Whisper's encoder always processes 30 seconds, so a four second disagreement costs the same as the
    window it sits in: spans are snapped out to whole windows and overlapping ones merged, which makes a run
    of neighbouring disagreements one decode instead of ten.
    """
    out: List[Region] = []
    for span_start, span_end in sorted(spans):
        start = max(0.0, (span_start - pad) // window * window)
        end = ((span_end + pad) // window + 1) * window
        if duration > 0:
            end = min(end, duration)
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        elif end > start:
            out.append((start, end))
    return out


def vocabulary_prompt(segments: Sequence[Dict], base: str = "", limit: int = 30, max_chars: int = 200) -> str:
    """
    The names the first pass kept using, as a prompt for the next one.

    Whisper spells an unfamiliar name differently every time it meets it; handing back the spellings it
    settled on makes the later decodes consistent with it. A word qualifies when it was capitalised in
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


def review_lines(report: Dict, segments: Sequence[Dict], source: str = "",
                 low_confidence: float = -0.9) -> List[str]:
    """
    The places worth a human eye: where the passes disagreed, and where the decoder was least sure.

    A consistent mis-hearing - every pass confidently saying the same wrong word - cannot be caught by
    agreement, which is exactly why this list exists rather than another automatic repair.
    """
    lines = [f"whisperer review list{' - ' + source if source else ''}",
             f"{report.get('passes', 1)} passes over {report.get('checked', 0)} cue(s): "
             f"{report.get('confirmed', 0)} agreed, {report.get('majority', 0)} settled by majority, "
             f"{report.get('scored', 0)} picked on confidence, {report.get('dropped', 0)} dropped, "
             f"{report.get('added', 0)} recovered, {report.get('unresolved', 0)} left unresolved", ""]
    for item in report.get("review", []):
        lines.append(f"{_timecode(item['start'])} - {_timecode(item['end'])}  {item['why']}")
        if item.get("chosen"):
            lines.append(f"    kept  : {item['chosen']}")
        for n, said in enumerate(item.get("said", []), 1):
            lines.append(f"    pass {n}: {said or '(nothing)'}")
        lines.append("")
    weak = [s for s in segments if float(s.get("avg_logprob", 0.0)) < low_confidence]
    if weak:
        lines.append(f"Cues the decoder was least sure of (avg_logprob under {low_confidence}):")
        for s in weak[:200]:
            lines.append(f"{_timecode(float(s['start']))}  {float(s.get('avg_logprob', 0)):.2f}  "
                         f"{s.get('text', '')}")
        lines.append("")
    if len(lines) == 3:
        lines.append("Every cue was agreed on by at least two passes, and no cue was flagged as weak.")
    return lines


def _timecode(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600):02d}:{int(t // 60) % 60:02d}:{t % 60:06.3f}"
