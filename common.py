"""Shared paths, config, and logging helpers for the PianoSM pipeline.

The original scripts hardcoded Windows paths (F:\\PianoSM) which made the
project impossible to run anywhere else. Everything now lives under
data/ inside the repo, and every stage writes a log file there so the
whole run (calibration choices, every note press/release, tempo
estimation) is inspectable after the fact.
"""

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR = REPO_ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"
LOG_DIR = DATA_DIR / "logs"

DEFAULT_VIDEO_PATH = VIDEO_DIR / "downloaded_video.mp4"
DEFAULT_KEYMAP_PATH = DATA_DIR / "keyboard_map.json"
DEFAULT_TRACKED_JSON = DATA_DIR / "tracked_notes.json"
DEFAULT_DEBUG_IMAGE = DATA_DIR / "played_transition_debug.png"
DEFAULT_KEYMAP_DEBUG_IMAGE = DATA_DIR / "keymap_calibration_debug.png"
DEFAULT_MUSICXML = DATA_DIR / "piano_sheet.musicxml"
DEFAULT_MIDI = DATA_DIR / "piano_sheet.mid"

# 88-key MIDI range, A0..C8.
MIDI_LOW = 21
MIDI_HIGH = 108


def get_logger(name: str, log_file: Path) -> logging.Logger:
    """Console + file logger. The file gets DEBUG-level detail (every
    tracked event); the console stays readable at INFO."""

    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger
