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
