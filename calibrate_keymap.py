#!/usr/bin/env python3
"""Generate keyboard_map.json (88 key x-positions) automatically from a
frame of the tutorial video, instead of requiring someone to hand-measure
88 pixel positions.

Manually built key maps are a major source of the "inaccurate" symptom:
if a single key's x-coordinate is off, the detector samples the wrong
column and either misses that note or attributes a neighbor's bar to it,
and there's no way to tell the two failure modes apart downstream.

This script:
  1. Finds the row where the piano keybed starts (top of the keys),
     instead of assuming a fixed pixel value tuned to one specific video.
  2. Finds the left/right edges of the white-key strip.
  3. Lays out all 88 keys using standard piano key proportions (52 equal
     white keys, black keys positioned between their neighbors using the
     usual visual offsets).
  4. Always writes a debug overlay image so the layout can be checked by
     eye and corrected with --left-edge/--right-edge/--keyboard-top if a
     particular video's UI doesn't match the assumptions.

The black-key offsets are a best-effort approximation of how these are
typically drawn — verify against keymap_calibration_debug.png before
trusting a run.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from music21 import pitch as m21pitch

from common import DEFAULT_KEYMAP_DEBUG_IMAGE, DEFAULT_KEYMAP_PATH, DEFAULT_VIDEO_PATH, LOG_DIR, MIDI_HIGH, MIDI_LOW, get_logger

WHITE_PITCH_CLASSES = {0, 2, 4, 5, 7, 9, 11}

# How far each black key sits between its two flanking white-key centers,
# as a fraction from the left neighbor (0.0) to the right neighbor (1.0).
BLACK_KEY_LEAN = {
    1: 0.42,   # C#
    3: 0.58,   # D#
    6: 0.38,   # F#
    8: 0.50,   # G#
    10: 0.62,  # A#
}

NEAR_WHITE_V = 190
NEAR_WHITE_S = 60


def detect_keyboard_top(frame: np.ndarray, logger) -> int:
    """Find the y-row where the piano keybed begins: the first row (scanning
    downward) where a wide near-white band appears and holds for a
    consistent band below it. Falls back to 2/3 of frame height if nothing
    matches."""

    height, width, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]

    near_white = (v >= NEAR_WHITE_V) & (s <= NEAR_WHITE_S)
    row_frac = near_white.mean(axis=1)

    hold = max(5, int(round(height * 0.02)))
    threshold = 0.30

    for y in range(0, height - hold):
        if row_frac[y] >= threshold and np.all(row_frac[y : y + hold] >= threshold * 0.6):
            logger.info(f"Detected keyboard top at y={y} (white-row fraction {row_frac[y]:.2f})")
            return y

    fallback = int(round(height * 2 / 3))
    logger.warning(f"Could not confidently detect keyboard top; falling back to y={fallback}")
    return fallback


def detect_keybed_edges(frame: np.ndarray, keyboard_top: int, logger) -> Tuple[int, int]:
    """Find the left/right x-edges of the white-key strip by sampling a row
    deep in the key area (near the bottom of frame, where only white keys
    are visible — black keys don't reach that far down)."""

    height, width, _ = frame.shape
    sample_y = keyboard_top + int(round((height - keyboard_top) * 0.85))
    sample_y = min(sample_y, height - 2)

    row = frame[sample_y]
    hsv_row = cv2.cvtColor(row.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)[0]
    bright = (hsv_row[:, 2] >= NEAR_WHITE_V) & (hsv_row[:, 1] <= NEAR_WHITE_S)

    xs = np.nonzero(bright)[0]

    if len(xs) == 0:
        logger.warning(f"No white-key pixels found at y={sample_y}; falling back to full frame width")
        return 0, width - 1

    left_edge, right_edge = int(xs.min()), int(xs.max())
    logger.info(f"Detected keybed span at y={sample_y}: x=[{left_edge}, {right_edge}] (width={right_edge - left_edge})")
    return left_edge, right_edge


def build_keymap(left_edge: int, right_edge: int) -> List[dict]:
    span = right_edge - left_edge
    white_notes = [m for m in range(MIDI_LOW, MIDI_HIGH + 1) if m % 12 in WHITE_PITCH_CLASSES]
    n_white = len(white_notes)

    w = span / n_white
    white_x = {m: left_edge + (i + 0.5) * w for i, m in enumerate(white_notes)}

    keys = []
    for m in range(MIDI_LOW, MIDI_HIGH + 1):
        pc = m % 12
        if pc in WHITE_PITCH_CLASSES:
            x = white_x[m]
            key_type = "white"
        else:
            left_white = white_x[m - 1]
            right_white = white_x[m + 1]
            x = left_white + BLACK_KEY_LEAN[pc] * (right_white - left_white)
            key_type = "black"

        keys.append(
            {
                "midi": m,
                "pitch": m21pitch.Pitch(midi=m).nameWithOctave,
                "type": key_type,
                "x": round(x, 2),
            }
        )

    return keys


def write_debug_overlay(frame: np.ndarray, keys: List[dict], keyboard_top: int, out_path: Path, logger) -> None:
    debug = frame.copy()
    height, width, _ = frame.shape

    cv2.line(debug, (0, keyboard_top), (width - 1, keyboard_top), (0, 0, 255), 2)

    for key in keys:
        cx = int(round(key["x"]))
        color = (255, 255, 255) if key["type"] == "white" else (0, 200, 255)
        cv2.line(debug, (cx, keyboard_top), (cx, height - 1), color, 1)

        if key["midi"] % 12 == 0:  # mark every C
            cv2.putText(
                debug, key["pitch"], (max(0, cx - 10), height - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), debug)
    logger.info(f"Calibration debug overlay saved: {out_path}")
    logger.info("Check that each vertical line lands on its matching physical key before running the detector.")


def calibrate(
    video_path: Path,
    out_path: Path,
    debug_path: Path,
    logger,
    sample_time: float = 1.0,
    keyboard_top_override: Optional[int] = None,
    left_edge_override: Optional[int] = None,
    right_edge_override: Optional[int] = None,
) -> dict:

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise SystemExit(f"ERROR: Could not open {video_path}")

    fps = float(video.get(cv2.CAP_PROP_FPS)) or 30.0
    video.set(cv2.CAP_PROP_POS_FRAMES, int(round(sample_time * fps)))
    ok, frame = video.read()
    video.release()

    if not ok:
        raise SystemExit(f"ERROR: Could not read a frame at t={sample_time:.2f}s from {video_path}")

    height, width, _ = frame.shape

    keyboard_top = keyboard_top_override if keyboard_top_override is not None else detect_keyboard_top(frame, logger)

    if left_edge_override is not None and right_edge_override is not None:
        left_edge, right_edge = left_edge_override, right_edge_override
        logger.info(f"Using manually specified keybed span: x=[{left_edge}, {right_edge}]")
    else:
        left_edge, right_edge = detect_keybed_edges(frame, keyboard_top, logger)

    keys = build_keymap(left_edge, right_edge)

    payload = {
        "reference_width": width,
        "reference_height": height,
        "keyboard_top": keyboard_top,
        "keys": keys,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Saved keymap ({len(keys)} keys): {out_path}")

    write_debug_overlay(frame, keys, keyboard_top, debug_path, logger)

    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(DEFAULT_VIDEO_PATH))
    parser.add_argument("--out", default=str(DEFAULT_KEYMAP_PATH))
    parser.add_argument("--debug-image", default=str(DEFAULT_KEYMAP_DEBUG_IMAGE))
    parser.add_argument("--sample-time", type=float, default=1.0, help="Timestamp to sample the keyboard frame from.")
    parser.add_argument("--keyboard-top", type=int, default=None, help="Override auto-detected keybed top row (pixels).")
    parser.add_argument("--left-edge", type=int, default=None, help="Override auto-detected left edge of the keybed (pixels).")
    parser.add_argument("--right-edge", type=int, default=None, help="Override auto-detected right edge of the keybed (pixels).")
    args = parser.parse_args(argv)

    logger = get_logger("calibrate_keymap", LOG_DIR / "calibrate_keymap.log")

    calibrate(
        Path(args.video),
        Path(args.out),
        Path(args.debug_image),
        logger,
        sample_time=args.sample_time,
        keyboard_top_override=args.keyboard_top,
        left_edge_override=args.left_edge,
        right_edge_override=args.right_edge,
    )


if __name__ == "__main__":
    main()
