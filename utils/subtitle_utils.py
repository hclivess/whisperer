"""
Subtitle post-processing and writers.

A Segment is a plain dict: {"start": float, "end": float, "text": str, "words": [ {start,end,word}, ... ]}
"""
import difflib
import json
import os
import re
import textwrap
from typing import Dict, List, Optional, Tuple


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


_POSSESSIVE = re.compile(r"['\u2019]s$")


def _title_cased(words: List[str]) -> bool:
    """
    Is this line Capitalised On Every Word?

    Such a stretch is a decode that came apart, not evidence about how the file writes a word - counting it
    turned one Title-Cased line into a licence to capitalise an ordinary word everywhere else.
    """
    real = [w for w in words if any(ch.isalnum() for ch in w)]
    if len(real) < 4:
        return False
    return sum(1 for w in real[1:] if w[:1].isupper()) / (len(real) - 1) > 0.6


def _name_key(core: str) -> str:
    """A word and its possessive are the same name: Halloran / Halloran's."""
    return _POSSESSIVE.sub("", core.lower())


def _case_profile(segments: List[Dict]) -> Dict[str, List[int]]:
    """
    How this transcript writes each word: [capitalised, lower case, capitalised as a possessive].

    Whisper is not consistent about names - the same lecture gives "Halloran" in one window and "halloran"
    in the next - so which way a word goes is decided by the whole file rather than by any one sighting.
    Sentence-initial words are not counted at all: their capital says nothing about the word.

    The third count is the exception that lets a name be recognised from a single sighting. One capital on
    its own is weak evidence - Whisper capitalises ordinary words mid-phrase all the time, which is what the
    sentence-start repair is for - but "somebody\'s" is a possessive, and a possessive is somebody.
    """
    counts: Dict[str, List[int]] = {}
    for seg in segments:
        words = seg.get("text", "").split()
        if _title_cased(words):
            continue                    # a decode that went to Title Case says nothing about any word
        sentence_start = True
        for w in words:
            core = w.strip("\"'([{)]}").rstrip(",;:.!?…")
            if core and not sentence_start and core[:1].isalpha():
                tally = counts.setdefault(_name_key(core), [0, 0, 0])
                if core[:1].isupper():
                    tally[0] += 1
                    if _POSSESSIVE.search(core):
                        tally[2] += 1
                else:
                    tally[1] += 1
            sentence_start = bool(core[-1:] in ".!?…")
    return counts


def _mostly_lower(counts: Dict[str, List[int]], core: str) -> bool:
    """Does this transcript write the word in lower case at least as often as not?"""
    tally = counts.get(_name_key(core), [0, 0, 0])
    upper, lower = tally[0], tally[1]
    return lower >= upper and lower > 0


def _word_stream(segments: List[Dict]) -> List[Optional[Tuple[int, int, Dict, bool]]]:
    """
    (segment, index, word, editable) for every word, with None where a segment carries no timings.

    A segment whose word list does not spell its text cannot be edited through its words without losing
    something, so it is marked not editable - but its words still carry real timings, so the pauses around
    it stay measurable. This used to give up on the whole file at the first such segment, and one "[MUSIC]"
    or one backend oddity in a fifty minute lecture was enough to silence every repair in it.
    """
    stream: List[Optional[Tuple[int, int, Dict, bool]]] = []
    for i, seg in enumerate(segments):
        words = seg.get("words") or []
        if not words or not all("start" in w and "end" in w for w in words):
            stream.append(None)                        # nothing measurable here: breaks the chain
            continue
        editable = _clean("".join(w["word"] for w in words)) == _clean(seg.get("text", ""))
        for k, w in enumerate(words):
            stream.append((i, k, w, editable))
    return stream


_TRAILING_PUNCT = re.compile(r"""[,;:.!?…—–-]["'”’»)\]]*$""")
# Words a sentence is not plausibly finished on, so no full stop is invented after them. Whisper cuts its
# windows at silences, which is why a capital so often follows a real pause without the sentence having
# ended - "we drove to [pause] A quiet village" must not become "we drove to. A quiet village". This is the
# mirror of the 1.3.7 rule and points the same way: when in doubt, leave the text alone.
_WEAK_SENTENCE_END = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "from", "by", "as",
    "that", "which", "who", "whose", "is", "was", "are", "were", "be", "been", "am", "if", "when", "while",
    "because", "so", "than", "then", "my", "your", "his", "her", "their", "our", "its", "this", "these",
    "those", "into", "onto", "about", "over", "under", "very", "just", "not", "no", "some", "any",
}


def _stands_alone(prev_token: str, nxt: Optional[Tuple[int, int, Dict, bool]]) -> bool:
    """
    Is this capital on its own in lower-case prose, or is it part of a run of them?

    "the book Long Way Home ends with a chapter called The Long Way Home" is full of capitals that
    belong to titles and names, and every one of them sits next to another capital. A capital with lower
    case on both sides - "by the door She's waiting", "they packed the car And then they left" - belongs to
    nobody, and that is the only kind this lowers.
    """
    prev_core = prev_token.strip("\"'([{)]}").rstrip(",;:.!?…")
    if prev_core[:1].isupper():
        return False
    if nxt is None:
        return True
    next_core = nxt[2]["word"].strip().strip("\"'([{)]}").rstrip(",;:.!?…")
    return not next_core[:1].isupper()


def repair_sentence_starts(segments: List[Dict], min_pause: float = 0.25,
                           sentence_pause: float = 0.0) -> List[Dict]:
    """
    Settle every capital that has no full stop in front of it.

    Whisper decodes in 30 second windows and starts each one as if it were a fresh utterance, so it writes
    capitals in the middle of a phrase - "we drove to A quiet village" - and, the other way round, ends a
    window without the full stop the sentence needed. The word timestamps say which of the two happened:
    people pause between sentences and do not pause inside them.

      the speaker did stop (a long enough pause, after a word a sentence can end on) -> add the full stop
      anything else, when the capital stands alone in lower-case prose                -> lower-case it

    A capital with another capital beside it is left alone whatever the pause says: those are titles and
    names - "a chapter called The Long Way Home" - and they are not this repair's business.

    There is no third outcome. A capital with nothing in front of it is an error either way, so leaving it
    standing on a pause too short to prove a sentence break only preserves the error - and lower case is
    what the surrounding prose reads as.

    Adding is held to a higher standard than lowering, because Whisper cuts its windows at silences: a
    capital after a real pause is ambiguous, so no full stop is invented after a word a sentence does not
    plausibly end on (_WEAK_SENTENCE_END) or over punctuation that is already there. A word this transcript
    never writes in lower case may be a name and keeps its capital either way. Without word timestamps
    nothing is measured and nothing is changed.
    """
    if not segments or min_pause <= 0:
        return segments
    sentence_pause = sentence_pause or max(0.5, min_pause * 2)
    out = [dict(s) for s in segments]
    for seg in out:
        if seg.get("words"):
            seg["words"] = [dict(w) for w in seg["words"]]
    stream = _word_stream(out)
    if not any(entry is not None for entry in stream):
        return segments
    common = _case_profile(out)
    changed = set()

    for n in range(1, len(stream)):
        here, before = stream[n], stream[n - 1]
        if here is None or before is None:             # a segment without timings: measure nothing across it
            continue
        seg_i, k, word, editable = here
        prev_seg_i, prev_k, prev, prev_editable = before
        token = word["word"].strip()
        core = token.strip("\"'([{)]}").rstrip(",;:.!?…")
        if not core or not core[:1].isupper() or (len(core) > 1 and core.isupper()):
            continue                                   # lower case already, or an acronym
        if len(core) == 1 and token.rstrip("\"'”’»)]").endswith("."):
            continue                                   # "J. R. R.": an initial, not a word
        if core == "I" or core.startswith("I'") or not _mostly_lower(common, core):
            continue                                   # "I", or a word this file mostly capitalises: a name
        prev_token = prev["word"].strip()
        if _SENTENCE_END.search(prev_token):
            continue                                   # the sentence is already closed: nothing to settle
        pause = float(word["start"]) - float(prev["end"])
        prev_core = prev_token.strip("\"'([{)]}").lower()
        # no full stop over a comma, or after "to" / "the" / "and", which a sentence does not end on
        can_close = (prev_editable and not _TRAILING_PUNCT.search(prev_token)
                     and prev_core not in _WEAK_SENTENCE_END and len(prev_core) >= 2)
        if pause >= sentence_pause and can_close:
            out[prev_seg_i]["words"][prev_k]["word"] = prev["word"].rstrip() + "."
            changed.add(prev_seg_i)
        elif editable and _stands_alone(prev_token, stream[n + 1] if n + 1 < len(stream) else None):
            i = token.index(core[0])
            out[seg_i]["words"][k]["word"] = word["word"].replace(token, token[:i] + core[0].lower() + token[i + 1:], 1)
            changed.add(seg_i)

    for i in changed:
        out[i]["text"] = _clean("".join(w["word"] for w in out[i]["words"]))
    return out

def _plain(word: str) -> str:
    return word.strip().strip("\"'([{)]}").rstrip(",;:.!?…").lower()


def drop_repeated_text(segments: List[Dict], min_words: int = 6, ratio: float = 0.85,
                       max_speed: float = 0.6) -> List[Dict]:
    """
    Remove a run of words a cue repeats verbatim from the cue before it.

    After a window boundary Whisper sometimes writes the sentence it has just written again and then
    carries on - "the bridge was rebuilt after the floods of december" / "But the city was destroyed by the
    floods of december the road stayed closed until the spring came back". The second copy is not
    speech: it sits on top of the words that were actually spoken there, which is why such a cue holds far
    more words than its span can hold.

    Only the repeated run is removed, never the cue: what follows it is the real continuation.

    A speaker repeating himself is not this. Saying a phrase twice takes roughly the same time both times,
    while a re-emission is squeezed into whatever room is left beside the words that were really spoken, so
    the copy is compared with the original by the clock: it is only removed when it runs at under max_speed
    of the time the same words took before. Together with a minimum length (short repetitions - "that's,
    it was the same" - are him, not the model) and a near-verbatim match, that keeps the speaker's own
    repetitions, however fast he talks, and takes only the model's.
    """
    if not segments or min_words < 2:
        return segments
    out = [dict(s) for s in segments]
    for seg in out:
        if seg.get("words"):
            seg["words"] = [dict(w) for w in seg["words"]]
    kept: List[Dict] = []
    for seg in out:
        words = seg.get("words") or []
        aligned = bool(words) and _clean("".join(w["word"] for w in words)) == _clean(seg.get("text", ""))
        if kept and aligned:
            before = [_plain(w["word"]) for w in (kept[-1].get("words") or [])] or \
                     [_plain(w) for w in kept[-1].get("text", "").split()]
            here = [_plain(w["word"]) for w in words]
            before_words = kept[-1].get("words") or []
            longest = min(len(before), len(here))
            for k in range(longest, min_words - 1, -1):
                if difflib.SequenceMatcher(None, before[-k:], here[:k], autojunk=False).ratio() < ratio:
                    continue
                if len(before_words) >= k:
                    said = float(before_words[-1]["end"]) - float(before_words[-k]["start"])
                    again = float(words[k - 1]["end"]) - float(words[0]["start"])
                    if said > 0 and again >= said * max_speed:
                        break                                     # said twice, at the speed of speech: his
                words = words[k:]
                if words:
                    seg["words"] = words
                    seg["text"] = _clean("".join(w["word"] for w in words))
                    seg["start"] = float(words[0]["start"])       # the cue begins where its real words do
                break
            if not words:
                continue                                          # the whole cue was the repeat
        kept.append(seg)
    return kept

_I_FORMS = {"i", "i'm", "i've", "i'll", "i'd"}
# Words that are never somebody's name, however often the decoder capitalises them. They open an utterance
# constantly - "okay so", "right, and" - and every one of those capitals that lands mid-phrase would
# otherwise be evidence that the word is a name.
_NEVER_A_NAME = {"okay", "ok", "yeah", "yes", "no", "well", "right", "sure", "hey", "oh", "so", "anyway",
                 "actually", "maybe", "please", "thanks", "hello", "goodbye", "alright"}


def unify_word_case(segments: List[Dict], english: bool = True, majority: float = 0.6,
                    min_sightings: int = 2) -> List[Dict]:
    """
    Give a word the case this transcript mostly gives it.

    Whisper is not consistent about names: the same lecture writes "Halloran" in one window and "halloran"
    in the next, "Long Way Home" here and "long way home" there. Nothing in the audio decides it - the
    rest of the file does. A word capitalised in most of its mid-sentence sightings is capitalised in the
    others too; a word the file mostly writes in lower case is left alone, which keeps ordinary words that
    happen to start a sentence somewhere ("will", "mark", "rose") out of it.

    In English the pronoun "I" is also restored, since a lower-case "i" is never right - and only in
    English, where "i" is not a word of its own.
    """
    if not segments:
        return segments
    out = [dict(s) for s in segments]
    for seg in out:
        if seg.get("words"):
            seg["words"] = [dict(w) for w in seg["words"]]
    counts = _case_profile(out)
    # a possessive is somebody: one sighting of "somebody\'s" is worth the two an ordinary word needs
    names = {word for word, (upper, lower, possessive) in counts.items()
             if word not in _NEVER_A_NAME
             and upper >= majority * (upper + lower) and (upper >= min_sightings or (upper and possessive))}
    if not names and not english:
        return segments

    def fixed(token: str) -> str:
        core = token.strip("\"'([{)]}").rstrip(",;:.!?…")
        if not core or not core[:1].islower():
            return token
        if english and core.lower() in _I_FORMS:
            return token.replace(core, core[0].upper() + core[1:], 1)
        if _name_key(core) in names:
            return token.replace(core, core[0].upper() + core[1:], 1)
        return token

    for seg in out:
        words = seg.get("words") or []
        aligned = bool(words) and _clean("".join(w["word"] for w in words)) == _clean(seg.get("text", ""))
        if aligned:
            new_words = []
            for w in words:
                lead = w["word"][:len(w["word"]) - len(w["word"].lstrip())]
                new_words.append({**w, "word": lead + fixed(w["word"].strip())})
            if any(a["word"] != b["word"] for a, b in zip(words, new_words)):
                seg["words"] = new_words
                seg["text"] = _clean("".join(w["word"] for w in new_words))
        else:
            text = re.sub(r"\S+", lambda m: fixed(m.group(0)), seg.get("text", ""))
            if text != seg.get("text", ""):
                seg["text"] = text          # line breaks and spacing stay exactly as they were
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


def _borrow(cues: List[Dict], i: int, min_duration: float, min_gap: float) -> None:
    """
    Last resort for a cue that can be neither grown nor merged: take screen time from a neighbour that has
    it to spare. A donor never drops below min_duration itself, so nothing is fixed by breaking something.

    Time is taken from the earlier cue first - our line then comes up a beat before its speech, which reads
    fine, while pushing the next cue back would put that line on screen after its own words were spoken.
    """
    cue = cues[i]
    need = min_duration - (cue["end"] - cue["start"])
    if need <= 1e-6:
        return
    if i > 0:
        prev = cues[i - 1]
        take = min(need, (prev["end"] - prev["start"]) - min_duration)
        if take > 1e-6:
            cue["start"] -= take
            prev["end"] = min(prev["end"], cue["start"] - min_gap)
            need -= take
    if need > 1e-6 and i + 1 < len(cues):
        nxt = cues[i + 1]
        take = min(need, (nxt["end"] - nxt["start"]) - min_duration)
        if take > 1e-6:
            cue["end"] += take
            nxt["start"] = max(nxt["start"], cue["end"] + min_gap)


def merge_short_cues(segments: List[Dict], min_duration: float = 0.8, min_gap: float = 0.08,
                     max_duration: float = 7.0, limit_chars: int = 84,
                     max_merge_gap: float = 1.5, capitalize: bool = True) -> List[Dict]:
    """
    Give every cue enough time on screen to be read.

    Whisper emits segments as short as 10 ms, and snapping can squeeze a cue against its neighbour; both flash
    a line of text for a frame or two. Simply extending the end would push the cue over the next one's speech,
    so the cue is first stretched into whatever free time surrounds it and, when there is none, glued to the
    neighbouring cue. The merged cue spans both, so no cue boundary moves and nothing loses sync - the text
    only comes on screen earlier and stays longer.

    limit_chars (the cue layout) is a preference, not a veto: a merge that fits it is chosen first, but when
    the only alternative is a flash, the cues are merged anyway and the text takes an extra line - lines stay
    within the line length, there are just more of them. A merge is refused only when the result would run
    longer than max_duration or the two cues are more than max_merge_gap apart (different moments); such a cue
    borrows the missing time from whichever neighbour has time to spare.
    """
    if min_duration <= 0 or not segments:
        return segments
    cues = [dict(s) for s in segments]

    def short(c: Dict) -> bool:
        return c["end"] - c["start"] < min_duration - 1e-6

    def joinable(a: Dict, b: Dict, layout: bool) -> bool:
        if max(0.0, b["start"] - a["end"]) > max_merge_gap:
            return False
        if max_duration > 0 and b["end"] - a["start"] > max_duration:
            return False
        if not layout or limit_chars <= 0:
            return True
        return len(a.get("text", "")) + 1 + len(b.get("text", "")) <= limit_chars

    def pick(i: int, layout: bool) -> str:
        """Which neighbour cue i should be glued to, '' for neither."""
        fwd = i + 1 < len(cues) and joinable(cues[i], cues[i + 1], layout)
        bwd = i > 0 and joinable(cues[i - 1], cues[i], layout)
        if fwd and bwd:
            # keep sentences whole: never glue a cue onto a finished sentence when it can go the other way
            if _SENTENCE_END.search(cues[i - 1].get("text", "").rstrip()):
                bwd = False
            elif _SENTENCE_END.search(cues[i].get("text", "").rstrip()):
                fwd = False
            elif cues[i]["start"] - cues[i - 1]["end"] < cues[i + 1]["start"] - cues[i]["end"]:
                fwd = False
        return "fwd" if fwd else ("bwd" if bwd else "")

    for i in range(len(cues)):
        _grow(cues, i, min_duration, min_gap, max_duration)

    i = 0
    while i < len(cues):
        if not short(cues[i]):
            i += 1
            continue
        way = pick(i, layout=True)
        if not way:
            # free time around the cue keeps the two cues separate, which reads better than one longer cue
            _grow(cues, i, min_duration, min_gap, max_duration, earlier=True)
            if not short(cues[i]):
                i += 1
                continue
            way = pick(i, layout=False)          # an extra line beats a line nobody can read
        if not way:
            _borrow(cues, i, min_duration, min_gap)
            i += 1
            continue
        merged_at = i
        if way == "fwd":
            cues[i:i + 2] = [_merged(cues[i], cues[i + 1], capitalize)]
        else:
            cues[i - 1:i + 1] = [_merged(cues[i - 1], cues[i], capitalize)]
            i -= 1
            merged_at = i
        _grow(cues, merged_at, min_duration, min_gap, max_duration)   # the merged cue may still be short

    for i in range(len(cues)):                    # a cue merging could not fix still gets its reading time
        if short(cues[i]):
            _grow(cues, i, min_duration, min_gap, max_duration, earlier=True)
            _borrow(cues, i, min_duration, min_gap)

    return cues


def enforce_min_duration(segments: List[Dict], min_duration: float = 0.8, min_gap: float = 0.08,
                         max_duration: float = 0.0) -> List[Dict]:
    """
    The floor every cue leaves the pipeline with: min_duration on screen whenever the time can be found.

    This is timing only - no text is joined, no cue disappears - so it also runs over a resynced subtitle,
    where merging cues would rewrite someone else's file, and over cues that merging had to refuse. Free
    time around the cue is used first; what is still missing is borrowed from a neighbour that can spare it.
    """
    if min_duration <= 0 or not segments:
        return segments
    cues = [dict(s) for s in segments]
    for i in range(len(cues)):
        if cues[i]["end"] - cues[i]["start"] < min_duration - 1e-6:
            _grow(cues, i, min_duration, min_gap, max_duration, earlier=True)
            _borrow(cues, i, min_duration, min_gap)
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
    # The line length is the hard limit, the line count is the target: text that does not fit max_lines
    # takes another line instead of being crammed into over-wide ones. Cues out of split_segments always
    # fit; merged cues and imported (resynced) ones are what can need the extra line.
    needed = -(-len(text) // max_line_chars)      # max_lines (unused here) is the target
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
