"""
whisperer - configuration and constants
"""

APP_NAME = "whisperer"
APP_VERSION = "1.7.3"
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
}

SYNC_MODES = {
    "generate": "Generate subtitles from the audio",
    "resync": "Resync an existing subtitle file to the audio",
}

TASKS = {"transcribe": "Transcribe (same language)", "translate": "Translate to English"}

SUBTITLE_FORMATS = ["srt", "vtt", "txt", "json"]

# Containers that can carry a soft subtitle track, with the codec to use
MUX_CONTAINERS = {"mkv": "srt", "mp4": "mov_text"}

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
    "sentence_pause_ms": 250,
    "strip_foreign_script": True,
    "review_list": False,           # write <name>.review.txt: where the passes disagreed
    "output_mode": "same",          # same | custom
    "output_dir": "",
    "suffix": "",
    "language_suffix": True,        # file.en.srt
    "overwrite": False,
    "mux_subtitles": False,
    "mux_container": "mkv",
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
