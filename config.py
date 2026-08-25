"""
whisperer - configuration and constants
"""

APP_NAME = "whisperer"
APP_VERSION = "1.8.2"
WINDOW_MIN_WIDTH = 820
WINDOW_MIN_HEIGHT = 560

# Input files the queue will accept (video + audio)
MEDIA_EXTENSIONS = [
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg",
    ".ts", ".m2ts", ".mts", ".vob", ".3gp", ".ogv",
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".mka",
]

# Transcription engines
ENGINES = {
    "faster_whisper": "faster-whisper (built-in, CTranslate2)",
    "whisper_cpp": "whisper.cpp (external whisper-cli)",
}

# Model sizes understood by both engines
MODEL_SIZES = [
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v2", "large-v3", "large-v3-turbo", "distil-large-v3",
]
DEFAULT_MODEL = "small.en"

# faster-whisper loads any CTranslate2 Whisper model from the Hugging Face Hub, not just the sizes
# above, and a fine-tune of one language usually beats plain Whisper on that language. These are
# offered below the sizes in the Model dropdown; models.json next to the app overrides the list, so
# a model that is not here can be added without touching the code.
MODEL_CATALOG_FILE = "models.json"
MODEL_CATALOG = [
    {"id": "kiendt/PhoWhisper-large-ct2",
     "label": "PhoWhisper large — Vietnamese (VinAI fine-tune)", "language": "vi"},
    {"id": "distil-whisper/distil-large-v3.5-ct2",
     "label": "distil-large-v3.5 — English, faster than large-v3", "language": "en"},
    {"id": "avazir/faster-distil-whisper-large-v3-ru",
     "label": "distil-large-v3 — Russian fine-tune", "language": "ru"},
    {"id": "jimmymeister/whisper-large-v3-turbo-german-ct2",
     "label": "large-v3-turbo — German fine-tune", "language": "de"},
    {"id": "kotoba-tech/kotoba-whisper-v2.0-faster",
     "label": "Kotoba-Whisper v2.0 — Japanese", "language": "ja"},
    {"id": "ivrit-ai/whisper-large-v3-turbo-ct2",
     "label": "ivrit.ai large-v3-turbo — Hebrew", "language": "he"},
    {"id": "techiaith/whisper-large-ft-cy-en-ct2",
     "label": "Whisper large fine-tune — Welsh / English", "language": "cy"},
    {"id": "SoybeanMilk/faster-whisper-Breeze-ASR-25",
     "label": "Breeze-ASR-25 — Mandarin, and Mandarin/English code-switching", "language": "zh"},
]

DEVICES = ["auto", "cpu", "cuda"]
COMPUTE_TYPES = ["default", "int8", "int8_float16", "float16", "float32"]

# Languages: code -> display name (English first, it is the default)
LANGUAGES = {
    "en": "English",
    "auto": "Auto-detect",
    "cs": "Czech", "de": "German", "es": "Spanish", "fr": "French", "it": "Italian",
    "ja": "Japanese", "ko": "Korean", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese",
    "ru": "Russian", "sk": "Slovak", "sv": "Swedish", "tr": "Turkish", "uk": "Ukrainian",
    "zh": "Chinese", "ar": "Arabic", "hi": "Hindi", "hu": "Hungarian", "fi": "Finnish",
    "da": "Danish", "no": "Norwegian", "el": "Greek", "ro": "Romanian", "vi": "Vietnamese",
    "he": "Hebrew", "cy": "Welsh",
}

SYNC_MODES = {
    "generate": "Generate subtitles from the audio",
    "resync": "Resync an existing subtitle file to the audio",
}

TASKS = {"transcribe": "Transcribe (same language)", "translate": "Translate to English"}

# srt is the default; ass carries styling, sub (MicroDVD) counts frames rather than seconds
SUBTITLE_FORMATS = ["srt", "vtt", "ass", "sub", "txt", "json"]

# Containers that can carry a soft subtitle track, with the codec to use
MUX_CONTAINERS = {"mkv": "srt", "mp4": "mov_text"}

# What comes out at the end. The subtitle files are written either way; the video is extra work
# on top, and hardcoding is the only one that survives a player which can display no subtitles.
DELIVERY_MODES = {
    "file": "Subtitle file",
    "embed": "Embed in the video — soft track, no re-encode",
    "hardcode": "Hardcode into the video — burned into the picture, re-encodes",
}

# Hardcoding quality: label, x264 CRF (lower is better and larger), x264 preset (slower is smaller)
HARDCODE_QUALITY = {
    "high": ("High — bigger file, slower", 18, "slow"),
    "balanced": ("Balanced", 20, "medium"),
    "small": ("Smaller file, faster", 24, "fast"),
}

DEFAULT_SETTINGS = {
    # Engine / model
    "engine": "faster_whisper",
    "model": DEFAULT_MODEL,
    "device": "auto",
    "compute_type": "default",
    "model_dir": "",
    "whisper_cli_path": "",
    "cpu_threads": 0,
    "cuda_lib_dir": "",
    # Transcription
    "language": "en",
    "task": "transcribe",
    "vad_filter": True,
    "vad_min_silence_ms": 500,
    "beam_size": 5,
    "word_timestamps": False,
    "condition_on_previous_text": True,
    "passes": 3,                    # decode the file this many times and keep what the decodes agree on
    "initial_prompt": "",
    "hotwords": "",                 # terms to weight decoding towards (faster-whisper)
    "extra_args": "",
    # Sync
    "sync_mode": "generate",        # generate | resync
    "snap_to_speech": True,
    "snap_max_shift_ms": 600,
    "end_padding_ms": 200,
    "min_cue_ms": 800,
    "merge_short_cues": True,
    "min_gap_ms": 80,
    "global_offset_ms": 0,
    "resync_file": "",              # empty = <video>.srt / .vtt next to the source
    "resync_fit_speed": True,
    # Subtitles
    "formats": ["srt"],
    "max_line_chars": 42,
    "max_lines": 2,
    "max_segment_seconds": 7.0,
    "capitalize_sentences": True,
    "repair_sentence_breaks": True,
    "repair_sentence_starts": True,
    "drop_repeated_text": True,     # remove a sentence the model wrote a second time
    "unify_word_case": True,        # a name gets the case the rest of the file gives it
    "sentence_pause_ms": 250,
    "strip_foreign_script": True,
    "review_list": False,           # write <name>.review.txt: where the passes disagreed
    "output_mode": "same",          # same | custom
    "output_dir": "",
    "suffix": "",
    "language_suffix": True,        # file.en.srt
    "overwrite": False,
    "delivery": "file",             # file | embed | hardcode
    "mux_container": "mkv",
    "hardcode_quality": "balanced",
}

# Suggested quality presets (menu)
QUALITY_PRESETS = {
    "fast": {
        "name": "Fast (base.en, greedy)",
        "settings": {"model": "base.en", "beam_size": 1, "vad_filter": True, "language": "en"},
    },
    "balanced": {
        "name": "Balanced (small.en)",
        "settings": {"model": "small.en", "beam_size": 5, "vad_filter": True, "language": "en"},
    },
    "accurate": {
        "name": "Accurate (large-v3-turbo)",
        "settings": {"model": "large-v3-turbo", "beam_size": 5, "vad_filter": True, "language": "en"},
    },
    "best": {
        "name": "Best (large-v3, beam 5)",
        "settings": {"model": "large-v3", "beam_size": 5, "vad_filter": True, "language": "en"},
    },
}
