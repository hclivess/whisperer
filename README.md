# whisperer

Batch **subtitle generator** GUI for video and audio files, powered by OpenAI Whisper models.
Drop files in, pick a model, press *Start* — `.srt` / `.vtt` / `.txt` / `.json` subtitles are written next to
your videos, optionally embedded into a copy of the video as a soft subtitle track.
A sibling of [videer](https://github.com/hclivess/videer) (same queue / progress / presets workflow) built for speech-to-text.

![thumb](thumb.png)

## Features

- **Two engines**
  - [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) — built in, CPU or NVIDIA CUDA,
    models download automatically on first use, built-in VAD silence filter, word timestamps
  - [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — drives an external `whisper-cli` binary
    (CPU / CUDA / Vulkan / Metal builds), GGML models
- All model sizes: `tiny` … `large-v3`, `large-v3-turbo`, `distil-large-v3`, English-only `*.en` variants
- English by default; ~30 languages or auto-detect; *translate to English* task
- Live queue: add files / folders or drag & drop while a run is in progress, reorder by dragging, pause / resume / stop
- Live transcript panel — segments appear as they are decoded
- Progress panel: per-file and overall bars, ETA, speed (× realtime), position, segment count
- Subtitle layout control: max characters per line, max lines per cue, max cue duration — long segments are re-split
  on word timestamps and lines are balanced (no orphan words)
- Output next to the source or into a custom folder, `video.en.srt` naming that media players auto-detect, suffix, overwrite guard
- **Embed** the result as a soft subtitle track into an `.mkv` / `.mp4` copy (FFmpeg stream copy, no re-encode)
- Presets menu (Fast / Balanced / Accurate / Best), save / load / import / export your own, user defaults
- Settings sanity check before starting (English-only model + foreign language, translate with `*.en`, missing binaries …)
- Cross-platform: Windows, Linux, macOS

## Requirements

- Python 3.9+ and `pip install -r requirements.txt` (PySide6, faster-whisper, psutil) — or grab a prebuilt binary from the
  [latest release](https://github.com/hclivess/whisperer/releases/latest) (no Python needed)
- [FFmpeg](https://ffmpeg.org/) in PATH or next to the app — used to extract audio (16 kHz mono WAV), read durations
  and embed subtitles. faster-whisper can decode most files by itself, but FFmpeg is strongly recommended and required for whisper.cpp.
- Optional: NVIDIA GPU. faster-whisper on CUDA needs the CUDA 12 + cuDNN 9 runtime:
  `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` (Linux) — see the faster-whisper README for Windows.
- Optional: `whisper-cli` from whisper.cpp plus GGML models (`ggml-small.en.bin` …) from
  [huggingface.co/ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/main)

## Run

```
pip install -r requirements.txt
python main.py
```

Windows: double-click `run.cmd`. Linux/macOS: `./run.sh`.

## Which model?

| Model | Size | Notes |
|---|---|---|
| `base.en` | 145 MB | very fast, fine for clear speech |
| `small.en` | 480 MB | **default** — good balance on CPU |
| `medium.en` | 1.5 GB | noticeably better on accents / noise |
| `large-v3-turbo` | 1.6 GB | best speed/quality on a GPU |
| `large-v3` | 3 GB | highest accuracy |

Tips: keep **VAD filter** on (skips silence and prevents hallucinated text in quiet parts), use an
**initial prompt** to teach the model names and jargon, and turn on **word timestamps** for tighter cue splitting.

## Output

For `movie.mp4` with language English you get `movie.en.srt` (and any other selected formats). With *Embed into video*
enabled an additional `movie.subbed.mkv` (or `.mp4`) is written containing the original streams plus the subtitle track.

## Building binaries

`python build.py` produces a standalone Nuitka build in `dist/`. Tagged pushes build Windows / Linux / macOS packages on
GitHub Actions and attach them to the release.

## License

MIT
