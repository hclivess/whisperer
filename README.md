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
- **Timing that follows the audio** — cues are snapped to speech on/offsets found by a voice-activity detector
  (Silero VAD), held a moment after speech, never overlapping; word-level alignment is used to build the cues
- **Resync existing subtitles** (the SubSync idea, built in): match an `.srt` / `.vtt` against a transcript of the
  audio, fit offset + speed robustly, rewrite the timing — the text stays untouched
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
| the prebuilt binary (any OS) | **Model tab → Download CUDA libraries** — fetches cuBLAS 12, cuDNN 9 and NVRTC from PyPI's official NVIDIA wheels (~1 GB, resumable) into a `cuda` folder next to the app. Done once. |
| the prebuilt binary (Windows, manual) | drop `cublas64_12.dll`, `cublasLt64_12.dll`, `cudnn64_9.dll` (+ their `cudnn_*64_9.dll` companions) into a `cuda` folder next to `whisperer.exe`, or point **Model → CUDA libraries folder** at wherever they live (e.g. `C:\Program Files\NVIDIA\CUDNN\v9.x\bin`, a CUDA Toolkit `bin` folder, or the `nvidia\cublas\bin` + `nvidia\cudnn\bin` folders of the pip wheels). Ready-made bundle: [Purfview's cuBLAS+cuDNN zip](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs) |
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

## Sync

![sync tab](thumb-sync.png)

Whisper's own timestamps are decoder guesses quantised to 20 ms; at segment edges they routinely start early and
linger over silence, and a few containers make things worse by storing an audio stream that starts later than the
video. whisperer 1.3 fixes the timing from the audio itself:

| Problem | What whisperer does |
|---|---|
| Cue appears before the line is spoken / stays on after it | **Snap cues to detected speech** (Sync tab, on by default): every cue start/end is moved onto the nearest speech onset/offset within *Max shift* (600 ms), cues that run over silence are cut, then *Hold after speech* (200 ms), *Min cue duration* and *Min gap* are applied. Word timestamps are used automatically. |
| A cue flashes on screen for a fraction of a second | **Merge cues that stay too short into their neighbour** (Sync tab, on by default): Whisper emits segments as short as 10 ms, and a cue squeezed against the next one cannot reach *Min cue duration* on its own. Such a cue is first stretched into the free time after it and, if there is none, glued to the cue beside it — two short lines shown together read fine, a line held over the next one's speech does not. Merging is skipped when the joined text would not fit the cue layout, would run past *Max cue duration*, or the two cues are over 1.5 s apart. |
| Everything early by a constant amount | Audio streams with a non-zero start time are now extracted in place (`aresample=first_pts=0`), and timestamp gaps are filled with silence so nothing drifts over dropped packets. If you still want a nudge: **Global offset**. |
| Subtitles from somewhere else are off or drift over the film | **Resync an existing subtitle file**: whisperer transcribes the audio, matches the words of your `.srt`/`.vtt` against it, fits `audio_time = speed × sub_time + offset` with a robust (median-slope + inlier least-squares) fit, applies it to every cue and VAD-snaps the result. Speed is only fitted on clips longer than two minutes and only in the 0.9–1.1 range (23.976 ↔ 25 fps conversions). Output: `movie.synced.srt`; the log shows offset, speed, drift per hour and how many words agreed. |

Resync picks `movie.srt` / `movie.vtt` next to the video unless you choose a file. A file that does not match the
audio (wrong language, different cut) is rejected with an explanation rather than guessed at.

## Output

For `movie.mp4` with language English you get `movie.en.srt` (and any other selected formats). With *Embed into video*
enabled an additional `movie.subbed.mkv` (or `.mp4`) is written containing the original streams plus the subtitle track.

## Windows says the app is not safe

The Windows build is not code-signed yet, so SmartScreen shows *"Windows protected your PC — unknown
publisher"* the first time you run it. Nothing is wrong with the file; an unsigned executable from a small
project simply has no reputation with Microsoft.

- **To run it:** *More info* → *Run anyway*. If the whole folder came out of a downloaded `.zip`, Windows also
  tags it with the mark-of-the-web; `Unblock-File .\whisperer\*` in PowerShell clears that.
- **To check you got what we built:** every release ships a `.sha256` next to the archive.
  `Get-FileHash whisperer-1.3.5-windows-x64.zip` must print the same digest.
- **If Defender quarantines it outright** (rather than just warning), that is a false positive on the
  PyInstaller runtime — please report it at <https://www.microsoft.com/wdsi/filesubmission> and open an issue.

The executable carries full publisher/product metadata and is built with `--onedir` (no self-extracting stub)
and without UPX, which is what antivirus heuristics react to. The build is signature-ready: setting the
`WINDOWS_PFX_BASE64` / `WINDOWS_PFX_PASSWORD` repository secrets makes CI sign and timestamp the `.exe`.

## Building binaries

`python build.py` (needs `pip install pyinstaller`) produces a standalone PyInstaller build in `dist/`. Setting
`WHISPERER_SELFTEST=<media file>` makes the app transcribe that file headlessly with `tiny.en` and exit — the CI uses it to
verify the frozen build. Tagged pushes build Windows / Linux / macOS packages on
GitHub Actions and attach them to the release.

## Changes in 1.3.5

- **No more one-frame subtitles**: a cue that cannot reach *Min cue duration* in the free time around it is merged
  with its neighbour instead of being extended over the next line's speech (Sync tab, on by default). Previously
  *Min cue duration* was silently overridden whenever the next cue started immediately, and with snapping off it
  was not applied at all — Whisper's 10 ms segments went straight into the file.

## Changes in 1.3.4

- Queue: files already queued are skipped with a status message (paths compared case-/form-insensitively); folder
  scans are in natural order folder by folder (`ep2` before `ep10`).

## Changes in 1.3.3

- **No orphaned helpers**: every ffmpeg / whisper-cli we start is tracked and killed when the app quits — including
  a crash or Task Manager kill (Windows Job object, Linux parent-death signal).
- **Remove stray foreign-script characters** (Subtitles tab, on by default): Whisper occasionally hallucinates a
  single Chinese / Korean / Cyrillic character inside the text (`Bar标`). Letters of a script that makes up under 5 %
  of the transcript are removed; cues consisting only of such characters are dropped; bilingual files are untouched.

## Changes in 1.3.2

- Subtitle files are written to a temporary name and renamed when complete; the muxed `.subbed.mkv/.mp4` likewise,
  and Stop now interrupts the mux. No partial output is ever left under the final name.

## Changes in 1.3.1

- **Capitalise sentence starts** (Subtitles tab, on by default): Whisper emits a lower-case first word when the audio
  starts cold or right after a VAD cut; the first cue and every cue after a full stop now start upper-case.

## Changes in 1.3

- **Sync tab**: VAD-snapped cue timing (on by default), hold-after-speech, min duration, min gap, global offset.
- **Resync existing subtitles** to the audio (offset + speed, SubSync model) — `movie.synced.srt`.
- **Fixed**: audio streams that start after the video lost their lead during extraction, making every cue early
  by that amount; timestamp gaps in the audio no longer shift later cues.
- Self-test now runs real speech through both paths (generate, then shift by 2 s and resync) on the frozen build.

## License

MIT
