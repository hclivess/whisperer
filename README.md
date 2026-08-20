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
- Optional: NVIDIA GPU — see [GPU acceleration](#gpu-acceleration-nvidia) below.
- Optional: `whisper-cli` from whisper.cpp plus GGML models (`ggml-small.en.bin` …) from
  [huggingface.co/ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/main)

## Run

```
pip install -r requirements.txt
python main.py
```

Windows: double-click `run.cmd`. Linux/macOS: `./run.sh`.

## GPU acceleration (NVIDIA)

faster-whisper runs on CUDA 12 and needs the **cuBLAS 12** and **cuDNN 9** runtime libraries. whisperer finds them
automatically and loads them itself — no `PATH` / `LD_LIBRARY_PATH` editing:

| You run… | Do this |
|---|---|
| from source | `pip install -r requirements-cuda.txt` (adds `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, ~1 GB) |
| the prebuilt binary (Windows) | drop `cublas64_12.dll`, `cublasLt64_12.dll`, `cudnn64_9.dll` (+ their `cudnn_*64_9.dll` companions) into a `cuda` folder next to `whisperer.exe`, or point **Model → CUDA libraries folder** at wherever they live (e.g. `C:\Program Files\NVIDIA\CUDNN\v9.x\bin`, a CUDA Toolkit `bin` folder, or the `nvidia\cublas\bin` + `nvidia\cudnn\bin` folders of the pip wheels). Ready-made bundle: [Purfview's cuBLAS+cuDNN zip](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs) |
| the prebuilt binary (Linux) | `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` into any Python, or copy `libcublas.so.12`, `libcublasLt.so.12`, `libcudnn*.so.9` into a `cuda` folder next to the binary |

The **Model** tab shows what whisperer sees: GPU name / driver (from `nvidia-smi`), which libraries were found and whether
CTranslate2 can use the device, with a *Re-check GPU* button. Leave *Device* on `auto` (CUDA when usable, CPU otherwise) and
*Compute type* on `default` (`float16` on GPU, `int8` on CPU). Expect roughly 10–30× realtime with `large-v3-turbo` on a
mid-range card versus ~1–3× on a CPU with `small.en`.

whisper.cpp uses the GPU according to how the binary was built (CUDA / Vulkan / Metal); Device = `cpu` passes `-ng`.
macOS has no CUDA — use whisper.cpp's Metal build there, or faster-whisper on the CPU.

## Which model?

| Model | Size | Notes |
|---|---|---|
| `base.en` | 145 MB | very fast, fine for clear speech |
| `small.en` | 480 MB | **default** — good balance on CPU |
| `medium.en` | 1.5 GB | noticeably better on accents / noise |
| `large-v3-turbo` | 1.6 GB | best speed/quality on a GPU |
| `large-v3` | 3 GB | highest accuracy |

Pick the model in the **Model** tab (the dropdown is editable — any faster-whisper Hugging Face repo id such as
`deepdml/faster-whisper-large-v3-turbo-ct2` works too). *Download / check model* fetches it up front; the label next to it
shows whether the model is already cached.

Tips: keep **VAD filter** on (skips silence and prevents hallucinated text in quiet parts), use an
**initial prompt** to teach the model names and jargon, and turn on **word timestamps** for tighter cue splitting.

## Output

For `movie.mp4` with language English you get `movie.en.srt` (and any other selected formats). With *Embed into video*
enabled an additional `movie.subbed.mkv` (or `.mp4`) is written containing the original streams plus the subtitle track.

## Building binaries

`python build.py` (needs `pip install pyinstaller`) produces a standalone PyInstaller build in `dist/`. Setting
`WHISPERER_SELFTEST=<media file>` makes the app transcribe that file headlessly with `tiny.en` and exit — the CI uses it to
verify the frozen build. Tagged pushes build Windows / Linux / macOS packages on
GitHub Actions and attach them to the release.

## License

MIT
