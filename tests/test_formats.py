import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SUBTITLE_FORMATS  # noqa: E402
from modules.process_manager import delivery_mode  # noqa: E402
from utils.subtitle_utils import to_ass, to_sub, write_subtitles  # noqa: E402


def segs():
    return [{"start": 1.0, "end": 3.5, "text": "First cue, with a comma"},
            {"start": 4.0, "end": 5.25, "text": "Second | cue"}]


def test_ass_is_a_playable_script():
    out = to_ass(segs(), 42, 2)
    assert "[Script Info]" in out and "[V4+ Styles]" in out and "[Events]" in out
    line = [l for l in out.splitlines() if l.startswith("Dialogue:")][0]
    assert line.startswith("Dialogue: 0,0:00:01.00,0:00:03.50,Default,,0,0,0,,")
    assert line.endswith("First cue, with a comma")          # commas in the text survive, Text is last


def test_ass_breaks_lines_the_ass_way():
    assert r"\N" in to_ass([{"start": 0.0, "end": 4.0, "text": "one two three four five six"}], 10, 2)


def test_microdvd_counts_frames_and_declares_its_rate():
    out = to_sub(segs(), 42, 2, 25.0).splitlines()
    assert out[0] == "{1}{1}25.000"
    assert out[1].startswith("{25}{88}")                     # 1.0 s and 3.5 s at 25 fps
    assert out[2].startswith("{100}{131}")


def test_microdvd_rate_changes_the_frame_numbers():
    at24 = to_sub(segs(), 42, 2, 24.0).splitlines()[1]
    assert at24.startswith("{24}{84}")


def test_microdvd_never_ends_before_it_starts_and_protects_the_pipe():
    out = to_sub([{"start": 2.0, "end": 2.0, "text": "a | b"}], 42, 2, 25.0).splitlines()[1]
    assert out.startswith("{50}{51}")                        # a zero-length cue still gets one frame
    assert "|" not in out.split("}")[-1]                     # the separator cannot appear in the text


def test_a_bad_rate_falls_back_to_25():
    assert to_sub(segs(), 42, 2, 0).splitlines()[0] == "{1}{1}25.000"


def test_write_subtitles_writes_the_new_formats(tmp_path):
    base = str(tmp_path / "film")
    written = write_subtitles(segs(), base, ["ass", "sub"], 42, 2, {}, True, 30.0)
    assert [os.path.basename(p) for p in written] == ["film.ass", "film.sub"]
    with open(base + ".sub", encoding="utf-8") as fh:
        assert fh.readline().strip() == "{1}{1}30.000"


def test_every_offered_format_is_actually_written(tmp_path):
    base = str(tmp_path / "all")
    written = write_subtitles(segs(), base, list(SUBTITLE_FORMATS), 42, 2, {"language": "en"}, True)
    assert len(written) == len(SUBTITLE_FORMATS)             # an unknown format would be skipped silently
    assert all(os.path.getsize(p) > 0 for p in written)


def test_delivery_mode_reads_the_old_checkbox():
    assert delivery_mode({"delivery": "hardcode"}) == "hardcode"
    assert delivery_mode({"mux_subtitles": True}) == "embed"
    assert delivery_mode({"mux_subtitles": False}) == "file"
    assert delivery_mode({}) == "file"
    assert delivery_mode({"delivery": "nonsense", "mux_subtitles": True}) == "embed"
