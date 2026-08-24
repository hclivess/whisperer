import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.subtitle_utils import wrap_lines  # noqa: E402


def test_wrap_lines_never_exceeds_limit():
    # a cue whose text has no 2-line split at word boundaries used to come out
    # with a 45-char first line; the limit is hard, the text takes a third line
    text = "several thousand years. So those are serpents back to back, by the way, the yin and"
    for line in wrap_lines(text, 42, 2).split("\n"):
        assert len(line) <= 42, line


def test_wrap_lines_balances_when_it_fits():
    text = "hello world how are you doing today my friend"
    lines = wrap_lines(text, 42, 2).split("\n")
    assert len(lines) == 2
    assert all(len(l) <= 42 for l in lines)
    # balanced: neither line is an orphan word
    assert min(len(l.split()) for l in lines) >= 2


def test_wrap_lines_short_text_untouched():
    assert wrap_lines("short line", 42, 2) == "short line"


def test_wrap_lines_unbreakable_word():
    # a single word longer than the limit is the only thing allowed to exceed it
    word = "Supercalifragilisticexpialidociousandthensomemore"
    out = wrap_lines(f"{word} is a word", 42, 2)
    lines = out.split("\n")
    assert word in lines[0]
    assert all(len(l) <= max(42, len(word)) for l in lines)


def test_wrap_lines_degenerate_width():
    assert wrap_lines("a b", 1, 2) == "a\nb"


def _seg(start, end, text):
    words = text.split()
    step = (end - start) / max(1, len(words))
    return {"start": start, "end": end, "text": text,
            "words": [{"start": start + i * step, "end": start + (i + 1) * step,
                       "word": (" " if i else "") + w} for i, w in enumerate(words)]}


def test_repair_sentence_breaks_keeps_words_in_step():
    from utils.subtitle_utils import repair_sentence_breaks, split_segments
    # a spurious full stop inside the cue: the dot goes, the capital lowers - in text AND words,
    # or the capital comes back the moment the text is rebuilt from the words
    seg = _seg(0.0, 2.0, "You have to focus. You focus your attention.")
    # no pause around the dot: 0.25s threshold, words are contiguous
    out = repair_sentence_breaks([seg], 0.25)
    assert out[0]["text"] == "You have to focus you focus your attention."
    joined = "".join(w["word"] for w in out[0]["words"]).strip()
    assert joined == out[0]["text"]
    # and a rebuild from words (what split_segments does) must not resurrect the capital
    split = split_segments(out, 22, 2, 7.0)
    assert not any(" You focus" in c["text"] or c["text"].startswith("You focus") for c in split)


def test_repair_sentence_breaks_next_segment_words_lowered():
    from utils.subtitle_utils import repair_sentence_breaks
    a = _seg(0.0, 1.0, "we drove on and on.")
    b = _seg(1.02, 2.0, "Then we stopped for a while.")
    out = repair_sentence_breaks([a, b], 0.25)
    assert out[1]["text"].startswith("then ")
    assert out[1]["words"][0]["word"].strip() == "then"


def test_drop_looped_text_finds_loop_inside_long_segment():
    from utils.subtitle_utils import drop_looped_text
    # 20s of ordinary speech, then the decoder loops "and it's a problem" in one second:
    # the cue-level rate looks human, the looped words' own rate does not
    normal = "we were driving along the road and everything was going fine until".split()
    loop = "and it's a problem" .split() * 4
    words, t = [], 0.0
    for w in normal:
        words.append({"start": t, "end": t + 1.0, "word": (" " if words else "") + w})
        t += 1.0
    for w in loop:
        words.append({"start": t, "end": t + 0.06, "word": " " + w})
        t += 0.06
    text = " ".join(normal + loop)
    seg = {"start": 0.0, "end": t, "text": text, "words": words}
    out = drop_looped_text([seg])
    assert out[0]["text"].count("and it's a problem") == 1
    assert out[0]["text"].startswith("we were driving")


def test_drop_looped_text_keeps_spoken_repetition_in_long_segment():
    from utils.subtitle_utils import drop_looped_text
    # the same repetition at the speed of speech is the speaker's own and stays
    phrase = "and it's a problem".split() * 3
    words, t = [], 0.0
    for w in phrase:
        words.append({"start": t, "end": t + 0.3, "word": (" " if words else "") + w})
        t += 0.3
    seg = {"start": 0.0, "end": t, "text": " ".join(phrase), "words": words}
    out = drop_looped_text([seg])
    assert out[0]["text"] == " ".join(phrase)


def test_repair_sentence_breaks_vad_overrides_fictitious_pause():
    from utils.subtitle_utils import repair_sentence_breaks
    # the decoder claims a 0.5s gap around the dot, but the VAD heard continuous speech:
    # the full stop is the model's invention and goes
    a = _seg(0.0, 1.0, "folk categories.")
    b = _seg(1.5, 2.5, "Are ancient ways of thinking.")
    regions = [(0.0, 2.5)]                       # one unbroken stretch of speech
    out = repair_sentence_breaks([a, b], 0.25, regions=regions)
    assert out[0]["text"] == "folk categories"
    assert out[1]["text"].startswith("are ")


def test_repair_sentence_breaks_vad_confirms_real_pause():
    from utils.subtitle_utils import repair_sentence_breaks
    a = _seg(0.0, 1.0, "but ultimately failed.")
    b = _seg(1.5, 2.5, "Now for something else.")
    regions = [(0.0, 1.0), (1.5, 2.5)]           # the VAD heard the silence too
    out = repair_sentence_breaks([a, b], 0.25, regions=regions)
    assert out[0]["text"] == "but ultimately failed."
    assert out[1]["text"].startswith("Now ")


def test_repair_sentence_breaks_article_dot_goes_despite_real_pause():
    from utils.subtitle_utils import repair_sentence_breaks
    # a speaker's dramatic pause after "the" does not end a sentence - it cannot
    a = _seg(0.0, 1.0, "aligned with the.")
    b = _seg(1.6, 2.6, "Harmony of multiplicity.")
    regions = [(0.0, 1.0), (1.6, 2.6)]
    out = repair_sentence_breaks([a, b], 0.25, regions=regions)
    assert out[0]["text"] == "aligned with the"
    assert out[1]["text"].startswith("harmony ")


def test_repair_sentence_breaks_initials_untouched():
    from utils.subtitle_utils import repair_sentence_breaks
    seg = _seg(0.0, 2.0, "the author J. R. R. Tolkien wrote it.")
    out = repair_sentence_breaks([seg], 0.25)
    assert out[0]["text"] == "the author J. R. R. Tolkien wrote it."
