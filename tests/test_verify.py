import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.verify_utils import speech_fraction, verify_second_pass, vocabulary_prompt  # noqa: E402


def seg(start, end, text, **extra):
    words = text.split()
    step = (end - start) / max(1, len(words))
    return {"start": start, "end": end, "text": text,
            "words": [{"start": start + i * step, "end": start + (i + 1) * step, "word": w}
                      for i, w in enumerate(words)], **extra}


def test_speech_fraction():
    assert speech_fraction([(0.0, 1.0), (2.0, 3.0)], 0.0, 4.0) == 0.5
    assert speech_fraction(None, 0.0, 4.0) is None


def test_hallucination_over_silence_is_dropped():
    first = [seg(0.0, 3.0, "the interview starts here"), seg(58.0, 62.0, "Thank you for watching this video")]
    second = [seg(0.0, 3.0, "the interview starts here")]        # the second decode heard nothing at 58 s
    regions = [(0.0, 3.2)]                                       # and neither did the VAD
    out, report = verify_second_pass(first, second, regions)
    assert [s["text"] for s in out] == ["the interview starts here"]
    assert report["dropped"] == 1 and report["confirmed"] == 1
    assert report["removed"][0]["text"].startswith("Thank you")


def test_a_cue_the_vad_backs_is_kept_even_if_the_second_pass_missed_it():
    first = [seg(0.0, 3.0, "one two three four"), seg(10.0, 13.0, "five six seven eight")]
    second = [seg(0.0, 3.0, "one two three four")]
    regions = [(0.0, 3.2), (10.0, 13.0)]                         # there really is speech at 10 s
    out, report = verify_second_pass(first, second, regions)
    assert len(out) == 2 and report["dropped"] == 0


def test_agreement_keeps_the_first_pass_verbatim():
    first = [seg(0.0, 4.0, "so what I was saying about the budget")]
    second = [seg(0.0, 4.0, "So, what I was saying about the budget.")]
    out, report = verify_second_pass(first, second, [(0.0, 4.0)])
    assert out[0]["text"] == "so what I was saying about the budget"   # punctuation is not a disagreement
    assert report["confirmed"] == 1 and report["replaced"] == 0


def test_the_better_supported_decode_wins_a_disagreement():
    first = [seg(0.0, 4.0, "the mitochondria is the powerhouse", avg_logprob=-1.4, compression_ratio=2.9)]
    second = [seg(0.0, 4.0, "the immigration is the power source", avg_logprob=-0.2)]
    out, report = verify_second_pass(first, second, [(0.0, 4.0)])
    assert out[0]["text"] == "the immigration is the power source"
    assert report["replaced"] == 1
    assert out[0]["start"] == 0.0 and out[0]["end"] == 4.0        # timing is the first pass's, always


def test_a_worse_second_decode_does_not_win():
    first = [seg(0.0, 4.0, "the mitochondria is the powerhouse", avg_logprob=-0.2)]
    second = [seg(0.0, 4.0, "the immigration is the power source", avg_logprob=-1.5, compression_ratio=2.8)]
    out, report = verify_second_pass(first, second, [(0.0, 4.0)])
    assert out[0]["text"] == "the mitochondria is the powerhouse"
    assert report["replaced"] == 0


def test_speech_the_first_pass_skipped_is_recovered():
    first = [seg(0.0, 3.0, "one two three four")]
    second = [seg(0.0, 3.0, "one two three four"), seg(20.0, 24.0, "and then the second speaker answered")]
    regions = [(0.0, 3.0), (20.0, 24.0)]
    out, report = verify_second_pass(first, second, regions)
    assert [round(s["start"], 1) for s in out] == [0.0, 20.0]
    assert report["added"] == 1


def test_nothing_is_added_without_a_vad_to_back_it():
    first = [seg(0.0, 3.0, "one two three four")]
    second = [seg(0.0, 3.0, "one two three four"), seg(20.0, 24.0, "and then the second speaker answered")]
    out, report = verify_second_pass(first, second, None)
    assert len(out) == 1 and report["added"] == 0


def test_an_empty_second_pass_changes_nothing():
    first = [seg(0.0, 3.0, "one two three four")]
    out, report = verify_second_pass(first, [], [(0.0, 3.0)])
    assert out == first and report["dropped"] == 0 and "note" in report


def test_vocabulary_prompt_collects_repeated_names_only():
    segments = [{"text": "we asked Bismuth about it."}, {"text": "Bismuth said the Hypernode was fine."},
                {"text": "Later that day nothing happened."}, {"text": "the Hypernode came back up."}]
    prompt = vocabulary_prompt(segments)
    assert "Bismuth" in prompt and "Hypernode" in prompt
    assert "Later" not in prompt                                 # a sentence start is not a name
    assert vocabulary_prompt(segments, "medical terms").startswith("medical terms ")


def test_a_replacement_never_borrows_the_neighbour_s_words():
    first = [seg(0.0, 2.0, "alpha bravo charlie", avg_logprob=-1.6, compression_ratio=2.9),
             seg(2.0, 4.0, "delta echo foxtrot", avg_logprob=-0.1)]
    second = [seg(0.0, 2.0, "alpha bravo tango", avg_logprob=-0.1),
              seg(2.0, 4.0, "delta echo foxtrot", avg_logprob=-0.1)]
    out, report = verify_second_pass(first, second, [(0.0, 4.0)])
    assert report["replaced"] == 1
    assert out[0]["text"] == "alpha bravo tango"          # not "... delta" from the cue next door
    assert out[1]["text"] == "delta echo foxtrot"


def decode(segments, covers=None):
    return {"segments": segments, "covers": covers}


def test_three_passes_settle_a_disagreement_by_majority():
    from utils.verify_utils import resolve_passes
    first = [seg(0.0, 4.0, "the mitochondria is the powerhouse", avg_logprob=-0.1)]
    second = [seg(0.0, 4.0, "the immigration is the power source", avg_logprob=-0.9)]
    third = [seg(0.0, 4.0, "the immigration is the power source", avg_logprob=-0.9)]
    out, report, unresolved = resolve_passes([decode(first), decode(second), decode(third)], [(0.0, 4.0)])
    # two of the three say the same thing: they win, however sure the first pass was of itself
    assert out[0]["text"] == "the immigration is the power source"
    assert report["majority"] == 1 and report["unresolved"] == 0 and unresolved == []


def test_a_third_pass_that_agrees_with_the_first_confirms_it():
    from utils.verify_utils import resolve_passes
    first = [seg(0.0, 4.0, "the mitochondria is the powerhouse", avg_logprob=-0.9)]
    second = [seg(0.0, 4.0, "the immigration is the power source", avg_logprob=-0.1)]
    third = [seg(0.0, 4.0, "the mitochondria is the powerhouse", avg_logprob=-0.9)]
    out, report, unresolved = resolve_passes([decode(first), decode(second), decode(third)], [(0.0, 4.0)])
    assert out[0]["text"] == "the mitochondria is the powerhouse"
    assert report["confirmed"] == 1 and not unresolved


def test_three_way_disagreement_stays_unresolved():
    from utils.verify_utils import resolve_passes
    first = [seg(0.0, 4.0, "alpha bravo charlie delta")]
    second = [seg(0.0, 4.0, "whiskey tango foxtrot romeo")]
    third = [seg(0.0, 4.0, "sierra kilo november zulu")]
    out, report, unresolved = resolve_passes([decode(first), decode(second), decode(third)], [(0.0, 4.0)])
    assert len(out) == 1 and report["unresolved"] == 1
    assert unresolved == [(0.0, 4.0)]                        # hand it back: it wants another decode


def test_a_targeted_pass_only_votes_where_it_looked():
    from utils.verify_utils import resolve_passes
    first = [seg(0.0, 4.0, "one two three four"), seg(100.0, 104.0, "five six seven eight")]
    second = [seg(0.0, 4.0, "one two three four"), seg(100.0, 104.0, "five six seven eight")]
    # a third pass aimed at the first window only: it says nothing about the cue at 100 s, so it is not
    # allowed to vote it away
    third = decode([seg(0.0, 4.0, "one two three four")], covers=[(0.0, 30.0)])
    out, report, _ = resolve_passes([decode(first), decode(second), third], [(0.0, 4.0), (100.0, 104.0)])
    assert len(out) == 2 and report["dropped"] == 0


def test_merge_windows_groups_neighbours_into_whole_encoder_windows():
    from utils.verify_utils import merge_windows
    assert merge_windows([(41.0, 44.0), (52.0, 55.0)]) == [(30.0, 60.0)]
    assert merge_windows([(10.0, 12.0), (400.0, 402.0)]) == [(0.0, 30.0), (390.0, 420.0)]
    assert merge_windows([(95.0, 99.0)], duration=97.0) == [(90.0, 97.0)]
    assert merge_windows([]) == []


def test_a_replacement_keeps_words_that_sit_just_outside_the_cue():
    from utils.verify_utils import resolve_passes
    # the passes cut segments in slightly different places: pass 2's last word ends after the cue does.
    # Taking only the exact span used to drop it from the file altogether.
    first = [{"start": 0.0, "end": 2.0, "text": "alpha bravo charlie", "avg_logprob": -1.6,
              "compression_ratio": 2.9,
              "words": [{"start": 0.0, "end": 0.6, "word": "alpha"}, {"start": 0.7, "end": 1.3, "word": " bravo"},
                        {"start": 1.4, "end": 2.0, "word": " charlie"}]},
             seg(3.0, 5.0, "delta echo foxtrot", avg_logprob=-0.1)]
    second = [{"start": 0.0, "end": 2.5, "text": "sierra kilo tango", "avg_logprob": -0.1,
               "words": [{"start": 0.0, "end": 0.6, "word": "sierra"}, {"start": 0.7, "end": 1.3, "word": " kilo"},
                         {"start": 2.1, "end": 2.5, "word": " tango"}]},   # middle at 2.3, past the cue end
              seg(3.0, 5.0, "delta echo foxtrot", avg_logprob=-0.1)]
    out, report, _ = resolve_passes([decode(first), decode(second)], [(0.0, 5.0)])
    assert report["scored"] == 1
    assert out[0]["text"] == "sierra kilo tango"           # nothing dropped at the boundary
    assert out[1]["text"] == "delta echo foxtrot"          # and nothing borrowed from the neighbour


def test_a_long_cue_with_real_speech_in_it_is_not_voted_away():
    from utils.verify_utils import resolve_passes
    # a 6 s cue holding 1.2 s of speech: the fraction is low, the words are real, and one pass missing it
    # must not be enough to delete it
    first = [seg(0.0, 6.0, "yes exactly")]
    second = [{"segments": [], "covers": None}]
    out, report, _ = resolve_passes([decode(first), second[0]], [(2.0, 3.2)])
    assert len(out) == 1 and report["dropped"] == 0


def test_a_cue_over_real_silence_is_still_dropped():
    from utils.verify_utils import resolve_passes
    first = [seg(0.0, 3.0, "the interview starts here"), seg(58.0, 62.0, "Thank you for watching this video")]
    second = [seg(0.0, 3.0, "the interview starts here")]
    out, report, _ = resolve_passes([decode(first), decode(second)], [(0.0, 3.2), (59.0, 59.1)])
    assert [s["text"] for s in out] == ["the interview starts here"]
    assert report["dropped"] == 1


def test_a_windowed_pass_never_supplies_text_for_a_cue_it_only_half_heard():
    from utils.verify_utils import resolve_passes
    # a pass aimed at 0-30 s stops mid-cue at 30 s: it may vote, but its truncated text must not replace
    # the cue, or the words past the cut are lost
    first = [seg(28.0, 32.0, "Ying and Yang and the whole of it", avg_logprob=-1.5, compression_ratio=2.8)]
    second = [seg(28.0, 32.0, "Ying and Yang and the whole of it", avg_logprob=-1.5)]
    third = decode([seg(28.0, 30.0, "Ying and", avg_logprob=-0.05)], covers=[(0.0, 30.0)])
    out, report, _ = resolve_passes([decode(first), decode(second), third], [(28.0, 32.0)])
    assert out[0]["text"] == "Ying and Yang and the whole of it"
    assert report["dropped"] == 0
