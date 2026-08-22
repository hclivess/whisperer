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
- **Multi-pass verification against hallucinations** (3 passes by default) — the file is decoded more than
  once and only what the decodes agree on is kept; text with no audio under it (*"Thank you for watching"*,
  repetition loops) is dropped on the evidence of the other passes and the VAD, and speech the first pass
  skipped is recovered. Passes 3+ only re-decode the spans nothing agreed on
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
| A sentence is cut in half and a new one starts mid-phrase | **Remove full stops the speaker never made** (Subtitles tab, on by default): Whisper punctuates by language model, not by ear, and regularly ends a sentence inside a phrase — *"When someone is. First about to embark on a minor task"*. People pause between sentences, so a full stop with less than *Shortest pause between sentences* (250 ms) of silence around it is the model's invention: it is removed and the next word goes back to lower case. Names (seen capitalised mid-sentence elsewhere) and *I* keep their capital; abbreviations, initials, decimals and `...` are never touched. Without word timestamps almost nothing is repaired — only `a.` / `an.` / `the.`, which cannot end an English sentence under any reading. |
| A cue flashes on screen for a fraction of a second | **Merge cues that stay too short into their neighbour** (Sync tab, on by default): Whisper emits segments as short as 10 ms, and a cue squeezed against the next one cannot reach *Min cue duration* on its own. Such a cue is first stretched into the free time after it and, if there is none, glued to the cue beside it — two short lines shown together read fine, a line held over the next one's speech does not. The cue layout is a preference here, not a veto: when the joined text does not fit *Max lines*, the cues are merged anyway and the text takes an extra line (lines never go over *Max line length*) — the merged cue spans exactly the two it replaces, so no boundary moves and nothing loses sync. This runs in every mode, **resync included**. Merging is only refused when the result would run past *Max cue duration* or the two cues are over 1.5 s apart; such a cue borrows the missing time from a neighbour that has it to spare, so it still reaches *Min cue duration*. |
| Whole paragraphs nobody said (*"Thank you for watching"*, a line repeated forty times) | **Passes** (Transcription tab, 3 by default): the file is decoded more than once — each pass with no context carried over (a repetition loop cannot feed itself twice), a different beam, then sampling temperature, and the names the first pass settled on handed back as the prompt. A hallucination is text with no audio under it, so the text alone can never prove it: **agreement decides**, because invented text is often fluent and scores well but two decodes rarely invent the *same* words. Text two passes agree on is kept, a cue no other pass heard anything under **and** the VAD finds no speech in is dropped, and speech the first pass skipped but a later pass and the VAD both found is recovered. Only when no two passes agree does Whisper's own confidence (`avg_logprob`, `no_speech_prob`, compression ratio, and how much speech is really in the span) pick the winner — and that cue is reported as unresolved rather than quietly trusted. Passes 3+ re-decode only the unresolved spans, snapped out to Whisper's 30 s encoder window, so they cost a fraction of a full pass. Timing is always the first pass's and no words are ever rewritten. Set *Passes* to 1 for a single decode. |
| A capital in the middle of a phrase (*"a rebuke to A stale order"*), or a sentence that never gets its full stop | **Settle capitals with no full stop in front of them** (Subtitles tab, on by default): Whisper decodes in 30-second windows and starts each one as if it were a fresh utterance. The word timestamps say which of the two things happened — a real silence before the capital means the speaker did stop and the full stop is added; no silence means the capital is the window boundary and it goes back to lower case; anything in between is left alone. Adding is held to the higher standard, because Whisper cuts its windows *at* silences: no full stop is invented after a word a sentence does not plausibly end on (*to*, *the*, *and*) or over punctuation already there. A word the transcript never writes in lower case may be a name and keeps its capital either way. Without word timestamps nothing is measured and nothing is changed. |
| A word decoded as the commoner word it half sounds like (*"mating game"* → *"marriage"*) | **Hotwords** (Transcription tab): terms to weight the decoder towards, comma separated. No amount of re-decoding fixes this one — every pass hears the same audio the same way, so all of them agree on the wrong word and verification confirms it. Naming the words is the only thing that prevents it. Unlike the *Initial prompt* these are not text the model may echo or imitate, and the names the first pass settled on are added to them automatically for the later passes (faster-whisper only). To find the ones that slipped through, switch on **Write a review list** (Subtitles tab, off by default): it writes `<name>.review.txt` next to the subtitles with every span the passes disagreed on — including the ones they agreed on but *worded* differently, which is where a meaning-changing mis-hearing hides — plus the cues the decoder was least sure of. Nothing is rewritten; a mis-hearing every pass agrees on can only be found by reading. |
| A sentence written twice, the second copy running over the words that follow it | **Remove a sentence the model wrote twice** (Subtitles tab, on by default): after a window boundary Whisper sometimes re-emits the sentence it has just written and then carries on, so the copy sits on top of the speech that was really there. Only the repeated run is removed — the cue keeps its real continuation and starts where its own words start. **A speaker repeating himself is not this**: saying a phrase twice takes roughly the same time both times, while a re-emission is squeezed into whatever room is left beside the real words, so the copy is compared with the original by the clock and only removed when it runs at under 60 % of it. Short repetitions are never touched at all. |
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
  `Get-FileHash whisperer-1.3.7-windows-x64.zip` must print the same digest.
- **If Defender quarantines it outright** (rather than just warning), that is a false positive on the
  PyInstaller runtime — please report it at <https://www.microsoft.com/wdsi/filesubmission> and open an issue.

The executable carries full publisher/product metadata and is built with `--onedir` (no self-extracting stub)
and without UPX, which is what antivirus heuristics react to. Code signing through
[SignPath Foundation](https://signpath.org/opensource) (free for open-source projects) is wired into the release
workflow and switches on as soon as the `SIGNPATH_API_TOKEN` secret exists.

## Building binaries

`python build.py` (needs `pip install pyinstaller`) produces a standalone PyInstaller build in `dist/`. Setting
`WHISPERER_SELFTEST=<media file>` makes the app transcribe that file headlessly with `tiny.en` and exit — the CI uses it to
verify the frozen build. Tagged pushes build Windows / Linux / macOS packages on
GitHub Actions and attach them to the release.

## Changes in 1.7.2

- **A sentence the model wrote twice is removed** (Subtitles tab, on by default). After a window boundary
  Whisper sometimes writes the sentence it has just written again and then carries on, so the second copy
  lies over the words that were really spoken there. Only the repeated run goes; the cue keeps its real
  continuation and starts where its own words start.
- The test that separates the model from the speaker is **the clock, not the words**: saying a phrase twice
  takes roughly the same time both times, a re-emission is squeezed into whatever room is left. Nothing is
  removed unless the copy runs at under 60 % of the time the same words took before, is at least six words
  long, and is near-verbatim — so a speaker's own repetitions survive however fast he talks.

## Changes in 1.7.1

- **Fixed: one odd segment switched the capital repair off for the whole file.** A segment whose word list
  does not spell its text — a `[MUSIC]` cue, a backend oddity — cannot be edited through its words, and the
  repair gave up on the *entire file* at the first one it met. In a fifty-minute lecture that is close to
  certain, so in practice the repair often did nothing at all. Such a segment is now simply left alone
  (pauses across it are still not measured, since it has nothing reliable to measure) and every other
  segment is repaired as it should have been.
- **Fixed: a replaced cue could repeat a contraction.** A word was carried through the comparison once per
  token it splits into, and *"he's"* splits into two, so a cue the verifier replaced could come back with
  the word twice. A word is one entry now, whatever it tokenises to.
- **A Title-Cased reading can no longer take over a cue.** Capitals On Every Word is what a pass looks like
  when it imitates a prompt or a sample goes wrong, and it can carry a *good* confidence score while doing
  it. Such a reading is refused even when it wins the vote; and where the first pass is the Title-Cased one
  and another pass says the same words in ordinary prose, the cleaner rendering is taken.
- Repeated words and heavy comma use are **never** treated as defects: people stutter, and a speaker who
  pauses that often has earned the commas. Only the rendering is judged, never the words.
- Later passes now vary by **beam before temperature** — a sampled decode is likelier to come apart — and
  the learned vocabulary goes to the decoder as hotwords rather than as a prompt (1.7.0), because a prompt
  that is a comma-separated list of capitalised names is a style the model will imitate.

## Changes in 1.7.0

- **Hotwords** (Transcription tab): terms to weight decoding towards, without the *Initial prompt*'s side effects.
  This is the answer to a word being decoded as the commoner word it half sounds like — *"mating game"* as
  *"marriage"* — which no number of passes can fix, because every pass hears the same audio the same way and
  they all agree on it. The names the first pass settled on are now handed to the later passes as hotwords
  rather than as a prompt, so the prompt stays exactly what you wrote. faster-whisper only.
- **Optional review list** (Subtitles tab, off by default): writes `<name>.review.txt` next to the subtitles
  listing every span the passes disagreed on and what each of them said, the spans they agreed on but
  *worded* differently — where a meaning-changing mis-hearing hides — and the cues the decoder was least
  sure of. Nothing is rewritten: it is a list of places worth a human eye.

## Changes in 1.6.1

- **Fixed: the third way multi-pass could drop a word.** A pass aimed at a few windows has hard edges — a
  cue lying across one of them was decoded up to the cut and no further. Such a pass could still win the
  cue on score and hand over its truncated text, losing the words past the cut. It may still vote (it heard
  most of the cue), but only a pass that heard a cue's whole stretch may supply its text.

## Changes in 1.6.0

- **Fixed: multi-pass verification could drop words.** Two ways, both now closed. A replaced cue took only
  the words whose middle fell inside its own span, so a word the other pass timed a fraction later — the
  passes cut segments in slightly different places — belonged to nobody and vanished; each cue now owns the
  audio out to the middle of the gap to its neighbour, so every word lands in exactly one cue and none is
  used twice. And the VAD could vote a cue away for having a low speech *fraction*, which is what a long cue
  with a short utterance in it looks like; it may now only vote for silence when there is next to no speech
  under the cue at all, by fraction **and** by seconds. If you ran 1.4.0 or 1.5.0, re-run those files.
- **Capitals with no full stop in front of them are settled** (Subtitles tab, on by default): the pause
  decides whether Whisper forgot to close a sentence or started one at a window boundary — the full stop is
  added, or the capital goes back to lower case. Names, *I*, acronyms and initials keep their capital, and
  nothing is invented after a word a sentence does not end on.

## Changes in 1.5.0

- **Passes: 1–5, three by default** (Transcription tab). The second pass of 1.4.0 became a number. Every pass differs from the
  ones before it — no context carried over, another beam, then sampling temperature — because identical
  settings would only repeat the same computation and prove nothing.
- **Agreement decides, not confidence.** Text that two passes produce is kept whatever the scores say;
  invented text is often fluent and scores well, but two independent decodes hardly ever invent the same
  words. Whisper's confidence numbers now only break a tie where *no* two passes agree, and that cue is
  counted as unresolved in the report instead of being quietly trusted.
- **Passes 3+ only re-decode what is still contested**, snapped out to Whisper's 30 s encoder window so a run
  of neighbouring disagreements costs one decode rather than ten. Past a quarter of the file it decodes the
  whole thing again instead, because that is cheaper. A pass aimed at a few windows only votes inside them.
- Deleting a cue is decided by votes alone (the other passes plus the VAD's vote for silence) — no confidence
  number can remove a line on its own.

## Changes in 1.4.0

- **Second pass, on by default.** whisperer now transcribes each file twice and keeps what both decodes agree
  on. Whisper invents differently every time it is asked — different beam, no context carried over — while
  real speech comes back the same, so the second decode is *evidence* about the first. Hallucinated paragraphs
  are dropped when neither the second pass nor the VAD finds anything under them, disagreements go to the
  decode the audio supports, and speech the first pass skipped entirely is recovered. Cue timing is always the
  first pass's and no words are rewritten; every drop, correction and recovery is reported in the log and in
  the JSON output. Untick *Second pass* on the Transcription tab to go back to one decode.
- The second pass gets the names the first pass settled on as its prompt, so a name Whisper spells three
  different ways in one file comes back consistent.
- Whisper's own quality numbers (`avg_logprob`, `no_speech_prob`, `compression_ratio`, `temperature`) are kept
  on every segment instead of being thrown away, and land in the JSON output.

## Changes in 1.3.9

- **The cue layout no longer outranks readability.** A cue too short to read was left alone whenever the
  merge would not fit *Max lines* × *Max line length* — which is exactly the case that produces the flash: a
  full two-line cue squeezed between two other full-length cues. Such cues are now merged anyway and the text
  takes an extra line, lines never going over *Max line length*. The merged cue spans exactly the two it
  replaces, so every remaining cue boundary is where it was: nothing can lose sync.
- **Merging now runs on resynced subtitles too.** It only ever joins two cues into their union, so an
  imported file keeps all of its text and all of its timing — it just stops flashing. The text itself is left
  exactly as it was written (no capitalisation is applied to a file that is not ours).

## Changes in 1.3.8

- **Min cue duration is now a real floor, in every mode.** It is applied one last time just before the file
  is written, as timing only: a cue that is too short is stretched into the free time around it and, if there
  is none, borrows the rest from a neighbour that stays above the minimum itself. This is what a *resynced*
  subtitle was missing entirely — merging is not allowed to rewrite an imported file, so snapping could leave
  a full two-line cue on screen for 80 ms and nothing downstream ever fixed it.
- **A cue can no longer stay unreadably short.** A full-length line squeezed between two other full-length
  lines had no way out: there was no free time around it, and no merge could fit the two texts inside the cue
  layout, so it was written out as it was — an 80 ms flash of a full sentence. Such a cue now borrows the
  missing time from a neighbour that has it to spare (from the earlier one first, so the next line still
  comes up with its own speech), and a donor never drops below *Min cue duration* itself.

## Changes in 1.3.7

- The no-word-timestamps fallback of the sentence repair no longer touches full stops after a preposition.
  English strands prepositions at the end of sentences all the time (*"things to attend to."*, *"what you're
  looking at."*) and those were being joined into the next sentence. Only `a.` / `an.` / `the.` are repaired
  without a measured pause now.

## Changes in 1.3.6

- **No more invented sentence breaks**: full stops Whisper places in the middle of a phrase are removed when the
  speaker did not actually pause there, and the following word goes back to lower case (Subtitles tab, on by
  default, threshold configurable). Names and *I* keep their capital; abbreviations, initials, decimals and
  ellipses are left alone. This is measured from the word timestamps, so it works in any language.

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
