import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sync_utils import (apply_linear, fit_linear, match_points, parse_subtitles, reference_words,  # noqa: E402
                              resync_cues, shift_cues, snap_to_speech, subtitle_words)


def test_snap_moves_edges_onto_speech():
    regions = [(1.00, 2.50), (3.00, 4.20)]
    cues = [{"start": 0.7, "end": 2.9, "text": "a"}, {"start": 3.3, "end": 5.0, "text": "b"}]
    out = snap_to_speech(cues, regions, max_shift=0.6, end_padding=0.0)
    assert abs(out[0]["start"] - 1.0) < 1e-6 and abs(out[0]["end"] - 2.5) < 1e-6
    assert abs(out[1]["start"] - 3.0) < 1e-6 and abs(out[1]["end"] - 4.2) < 1e-6


def test_snap_trims_overhang_and_keeps_gap():
    regions = [(0.0, 1.0), (5.0, 6.0)]
    cues = [{"start": 0.0, "end": 3.0, "text": "a"}, {"start": 5.0, "end": 6.0, "text": "b"}]
    out = snap_to_speech(cues, regions, max_shift=0.3, end_padding=0.2, min_gap=0.1)
    assert abs(out[0]["end"] - 1.2) < 1e-6           # cut at speech end + padding
    assert out[0]["end"] <= out[1]["start"] - 0.1


def test_snap_no_overlap_and_min_duration():
    regions = [(1.0, 1.2), (1.3, 3.0)]
    cues = [{"start": 1.0, "end": 1.2, "text": "hi"}, {"start": 1.25, "end": 3.0, "text": "there"}]
    out = snap_to_speech(cues, regions, max_shift=0.2, end_padding=0.0, min_duration=0.8, min_gap=0.08)
    assert out[0]["end"] - out[0]["start"] >= 0.05
    assert out[1]["start"] >= out[0]["end"] + 0.08 - 1e-9
    assert out[1]["end"] > out[1]["start"]


def test_shift_clamps_at_zero():
    out = shift_cues([{"start": 0.1, "end": 1.0, "text": "x"}], -0.5)
    assert out[0]["start"] == 0.0 and out[0]["end"] == 0.5


def test_parse_srt_and_vtt():
    srt = "1\n00:00:01,000 --> 00:00:02,500\n<i>Hello</i> world\n\n2\n00:01:00,000 --> 00:01:02,000\nBye\n"
    cues = parse_subtitles(srt)
    assert [c["text"] for c in cues] == ["Hello world", "Bye"]
    assert cues[1]["start"] == 60.0
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.500 line:0\nHi\n"
    assert parse_subtitles(vtt)[0]["text"] == "Hi"


def _transcript(words, t0=0.0, step=0.4):
    return [{"start": t0, "end": t0 + step * len(words),
             "words": [{"start": t0 + i * step, "end": t0 + (i + 1) * step, "word": w} for i, w in enumerate(words)]}]


def test_resync_recovers_offset_and_speed():
    base = ("the quick brown fox jumps over the lazy dog and then runs far away into the deep dark forest "
            "where nobody can ever find it again until the morning comes").split()
    words = [f"{w}{i // len(base)}" for i, w in enumerate(base * 20)]     # ~560 distinct-ish words, 280 s
    transcript = _transcript(words, t0=10.0, step=0.5)
    speed_true = 25 / 23.976
    cues = []
    for i in range(0, len(words), 5):
        t = 10.0 + i * 0.5
        cues.append({"start": (t + 3.0) / speed_true, "end": (t + 2.5 + 3.0) / speed_true,
                     "text": " ".join(words[i:i + 5])})
    fixed, report = resync_cues(cues, transcript, fit_speed=True)
    assert abs(fixed[0]["start"] - 10.0) < 0.15
    assert abs(fixed[-1]["start"] - (10.0 + (len(words) - 5) * 0.5)) < 0.3
    assert abs(report["speed"] - speed_true) < 0.002


def test_resync_short_clip_fits_offset_only():
    words = "one two three four five six seven eight nine ten eleven twelve".split()
    transcript = _transcript(words, t0=5.0, step=0.5)
    cues = [{"start": 3.0 + i * 0.5, "end": 3.4 + i * 0.5, "text": w} for i, w in enumerate(words)]
    fixed, report = resync_cues(cues, transcript, fit_speed=True)
    assert report["speed"] == 1.0
    assert abs(report["offset"] - 2.0) < 0.05


def test_resync_rejects_unrelated_text():
    transcript = _transcript("completely different words spoken here today".split())
    cues = [{"start": 0, "end": 2, "text": "lorem ipsum dolor sit amet"}]
    try:
        resync_cues(cues, transcript)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_capitalize_sentences():
    from utils.subtitle_utils import capitalize_sentences
    cues = [{"text": "the quick brown fox"}, {"text": "jumps over the dog."}, {"text": "¿qué tal?"},
            {"text": "\"fine,\" he said."}, {"text": "and then"}]
    out = [c["text"] for c in capitalize_sentences(cues)]
    assert out == ["The quick brown fox", "jumps over the dog.", "¿Qué tal?", "\"Fine,\" he said.", "And then"]


def test_strip_foreign_script():
    from utils.subtitle_utils import strip_foreign_script
    cues = [{"text": "In the sense that it has an operation"}, {"text": "or a goal to attack, which is Bar标"},
            {"text": "标"}, {"text": "and that is all there is to say about it"}]
    out = [c["text"] for c in strip_foreign_script(cues)]
    assert out == ["In the sense that it has an operation", "or a goal to attack, which is Bar",
                   "and that is all there is to say about it"]
    # bilingual transcript: the second script is common, nothing is removed
    mixed = [{"text": "Hello 你好 world 世界"}] * 5
    assert [c["text"] for c in strip_foreign_script(mixed)] == ["Hello 你好 world 世界"] * 5
    # Czech diacritics are Latin, never touched
    cz = [{"text": "Příliš žluťoučký kůň úpěl ďábelské ódy"}]
    assert strip_foreign_script(cz) == cz


def test_merge_short_cues_glues_flashing_cues():
    from utils.subtitle_utils import merge_short_cues
    # a 10 ms Whisper segment right before the next line: no free time, so it is glued to its neighbour
    cues = [{"start": 0.50, "end": 0.51, "text": "Yeah"},
            {"start": 0.60, "end": 3.00, "text": "so what I was saying"}]
    out = merge_short_cues(cues, min_duration=0.8, min_gap=0.08, max_duration=7.0, limit_chars=84)
    assert len(out) == 1
    assert out[0]["text"] == "Yeah so what I was saying"
    assert out[0]["start"] == 0.50 and out[0]["end"] == 3.00


def test_merge_short_cues_prefers_free_time_over_merging():
    from utils.subtitle_utils import merge_short_cues
    cues = [{"start": 1.0, "end": 1.1, "text": "Hi"}, {"start": 5.0, "end": 7.0, "text": "there"}]
    out = merge_short_cues(cues, min_duration=0.8, min_gap=0.08, max_duration=7.0, limit_chars=84)
    assert len(out) == 2                                   # a 3.9 s gap follows: just stretch into it
    assert abs(out[0]["end"] - 1.8) < 1e-6


def test_merge_short_cues_refuses_when_text_would_not_fit():
    from utils.subtitle_utils import merge_short_cues
    long = "x" * 84
    cues = [{"start": 0.0, "end": 0.05, "text": "No"}, {"start": 0.05, "end": 4.0, "text": long}]
    out = merge_short_cues(cues, min_duration=0.8, min_gap=0.08, max_duration=7.0, limit_chars=84)
    assert len(out) == 2 and out[0]["text"] == "No"        # the joined text would not fit: no merge
    assert abs(out[0]["end"] - out[0]["start"] - 0.8) < 1e-6   # but it still gets its reading time,
    assert abs(out[1]["end"] - out[1]["start"] - 3.12) < 1e-6  # borrowed from a neighbour that has it


def test_merge_short_cues_borrows_from_a_neighbour_when_nothing_fits():
    from utils.subtitle_utils import merge_short_cues
    # a full line squeezed to 80 ms between two full lines: no free time, and no merge can fit the layout
    cues = [{"start": 2012.258, "end": 2016.380, "text": "why so often when you're arguing with " + "x" * 40},
            {"start": 2016.460, "end": 2016.540, "text": "come up and you're like oh my god " + "x" * 48},
            {"start": 2016.620, "end": 2022.860, "text": "you know and then things explode " + "x" * 44}]
    out = merge_short_cues(cues, min_duration=0.8, min_gap=0.08, max_duration=7.0, limit_chars=84)
    assert len(out) == 3
    assert all(c["end"] - c["start"] >= 0.8 - 1e-6 for c in out)
    assert all(out[i + 1]["start"] - out[i]["end"] >= 0.08 - 1e-6 for i in range(2))
    assert out[0]["start"] == 2012.258 and out[2]["end"] == 2022.860   # the run keeps its outer timing
    assert abs(out[1]["end"] - 2016.540) < 1e-6      # the flashing cue kept its end, it starts earlier


def test_merge_short_cues_keeps_sentences_whole():
    from utils.subtitle_utils import merge_short_cues
    cues = [{"start": 0.0, "end": 1.0, "text": "That was the end."},
            {"start": 1.05, "end": 1.10, "text": "And"},
            {"start": 1.15, "end": 3.0, "text": "then we left"}]
    out = merge_short_cues(cues, min_duration=0.8, min_gap=0.05, max_duration=7.0, limit_chars=84)
    assert [c["text"] for c in out] == ["That was the end.", "And then we left"]


def test_merge_short_cues_carries_words_and_chains():
    from utils.subtitle_utils import merge_short_cues
    cues = [{"start": 0.0, "end": 0.05, "text": "A", "words": [{"start": 0.0, "end": 0.05, "word": "A"}]},
            {"start": 0.06, "end": 0.10, "text": "B", "words": [{"start": 0.06, "end": 0.10, "word": "B"}]},
            {"start": 0.11, "end": 0.90, "text": "C", "words": [{"start": 0.11, "end": 0.90, "word": "C"}]}]
    out = merge_short_cues(cues, min_duration=0.8, min_gap=0.02, max_duration=7.0, limit_chars=84)
    assert len(out) == 1 and out[0]["text"] == "A B C"
    assert [w["word"] for w in out[0]["words"]] == ["A", "B", "C"]


def test_no_cue_shorter_than_the_minimum_after_snapping():
    from utils.subtitle_utils import merge_short_cues
    regions = [(1.00, 1.06), (1.10, 4.00)]
    cues = [{"start": 1.00, "end": 1.06, "text": "Hi"}, {"start": 1.10, "end": 4.00, "text": "there everyone"}]
    snapped = snap_to_speech(cues, regions, max_shift=0.6, end_padding=0.2, min_duration=0.8, min_gap=0.08)
    assert any(c["end"] - c["start"] < 0.8 for c in snapped)          # snapping alone cannot fix this
    out = merge_short_cues(snapped, min_duration=0.8, min_gap=0.08, max_duration=7.0, limit_chars=84)
    assert all(c["end"] - c["start"] >= 0.8 for c in out)
    assert [c["text"] for c in out] == ["Hi there everyone"]


def test_merge_short_cues_capitalises_across_a_full_stop():
    from utils.subtitle_utils import merge_short_cues
    cues = [{"start": 0.50, "end": 0.51, "text": "Yeah."}, {"start": 0.60, "end": 3.0, "text": "so I said"}]
    assert merge_short_cues(cues, 0.8, 0.08, 7.0, 84)[0]["text"] == "Yeah. So I said"
    assert merge_short_cues(cues, 0.8, 0.08, 7.0, 84, capitalize=False)[0]["text"] == "Yeah. so I said"


def _timed(text, t0, words):
    """words: [(word, duration, gap before it)] -> a segment with word timestamps"""
    out, t = [], t0
    for w, dur, gap in words:
        t += gap
        out.append({"start": round(t, 3), "end": round(t + dur, 3), "word": " " + w})
        t += dur
    return {"start": t0, "end": round(t, 3), "text": text, "words": out}


def test_repair_drops_a_full_stop_the_speaker_never_made():
    from utils.subtitle_utils import repair_sentence_breaks
    seg = _timed("When someone is. First about to embark on a minor task,", 8.5,
                 [("When", .25, 0), ("someone", .3, .02), ("is.", .2, .02), ("First", .3, .04), ("about", .25, .02),
                  ("to", .1, .02), ("embark", .35, .02), ("on", .1, .02), ("a", .08, .02), ("minor", .3, .02),
                  ("task,", .4, .02)])
    out = repair_sentence_breaks([seg])[0]
    assert out["text"] == "When someone is first about to embark on a minor task,"
    assert [w["word"] for w in out["words"]][2] == " is"        # the word list still spells the text


def test_repair_keeps_a_real_sentence_end():
    from utils.subtitle_utils import repair_sentence_breaks
    seg = _timed("That's a common trope. It's the flight of the hero.", 35.6,
                 [("That's", .3, 0), ("a", .1, .02), ("common", .3, .02), ("trope.", .4, .02), ("It's", .3, .70),
                  ("the", .15, .02), ("flight", .3, .02), ("of", .1, .02), ("the", .1, .02), ("hero.", .4, .02)])
    assert repair_sentence_breaks([seg])[0]["text"] == seg["text"]      # 700 ms of silence: the speaker stopped


def test_repair_spans_cues_and_keeps_names():
    from utils.subtitle_utils import repair_sentence_breaks
    a = _timed("he encounters the whale Monstro. That", 30.0,
               [("he", .2, 0), ("encounters", .4, .02), ("the", .1, .02), ("whale", .3, .02), ("Monstro.", .5, .02),
                ("That", .3, .03)])
    b = _timed("Monstro lives at the bottom.", 31.68,          # only 100 ms after "Monstro." - no pause
               [("Monstro", .5, 0), ("lives", .3, .02), ("at", .1, .02), ("the", .1, .02), ("bottom.", .4, .02)])
    a["words"] = a["words"][:-1]
    a["text"] = "he encounters the whale Monstro."
    out = repair_sentence_breaks([a, b])
    # no pause after "Monstro." -> the stop goes, but the name keeps its capital (seen mid-sentence elsewhere)
    assert out[0]["text"] == "he encounters the whale Monstro"
    assert out[1]["text"] == "Monstro lives at the bottom."


def test_repair_leaves_abbreviations_ellipses_and_decimals_alone():
    from utils.subtitle_utils import repair_sentence_breaks
    for text in ["I spoke to Dr. Smith about it", "so I was thinking... that we should go", "it cost 3.5 million",
                 "he moved to the U.S. and stayed"]:
        seg = _timed(text, 0.0, [(w, .2, .01) for w in text.split()])
        assert repair_sentence_breaks([seg])[0]["text"] == text, text


def test_repair_without_word_timestamps_only_touches_impossible_endings():
    from utils.subtitle_utils import repair_sentence_breaks
    cues = [{"start": 0, "end": 3, "text": "behind the facade of the. Something else"},
            {"start": 3, "end": 6, "text": "Magical. Agents of transformation."},
            {"start": 6, "end": 9, "text": "an indefinite number of things to attend to. So if you were painting"}]
    out = [c["text"] for c in repair_sentence_breaks(cues)]
    assert out[0] == "behind the facade of the something else"   # "the." cannot end a sentence, ever
    assert out[1] == "Magical. Agents of transformation."        # no evidence: left alone
    # a stranded preposition is a perfectly good sentence end - without a measured pause, hands off
    assert out[2] == "an indefinite number of things to attend to. So if you were painting"


def test_enforce_min_duration_fixes_a_resynced_subtitle_without_touching_text():
    from utils.subtitle_utils import enforce_min_duration
    # the shape a resync leaves behind: imported two-line cues, one of them squeezed to 80 ms by snapping.
    # Merging is not allowed to rewrite someone else's file, so the timing alone has to fix it.
    cues = [{"start": 2012.258, "end": 2016.380, "text": "why so often when you're arguing with\nsomeone you love some trivial thing will"},
            {"start": 2016.460, "end": 2016.540, "text": "come up and you're like oh my god I don't\nknow what the hell's wrong with this car"},
            {"start": 2016.620, "end": 2022.860, "text": "you know and then things explode around\nit's because that trivial thing is an"}]
    out = enforce_min_duration(cues, min_duration=0.8, min_gap=0.08)
    assert [c["text"] for c in out] == [c["text"] for c in cues]      # not one character rewritten
    assert all(c["end"] - c["start"] >= 0.8 - 1e-6 for c in out)
    assert all(out[i + 1]["start"] - out[i]["end"] >= 0.08 - 1e-6 for i in range(2))
    assert out[0]["start"] == 2012.258 and out[2]["end"] == 2022.860  # the run keeps its outer timing


def test_enforce_min_duration_leaves_a_run_it_cannot_fix_alone():
    from utils.subtitle_utils import enforce_min_duration
    # every neighbour is at the floor already: nothing to borrow, and nothing is made worse by trying
    cues = [{"start": 0.0, "end": 0.8, "text": "a"}, {"start": 0.88, "end": 0.98, "text": "b"},
            {"start": 1.06, "end": 1.86, "text": "c"}]
    out = enforce_min_duration(cues, min_duration=0.8, min_gap=0.08)
    assert [round(c["start"], 3) for c in out] == [0.0, 0.88, 1.06]
    assert [round(c["end"], 3) for c in out] == [0.8, 0.98, 1.86]
