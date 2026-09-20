"""Renders a synthetic "falling note" piano tutorial video with known
ground-truth notes, so the detection pipeline can be checked against a
known answer instead of guessing at real footage.

The rendered geometry (keyboard top, left/right edges) is defined as
fractions of the frame size, so the exact same scene can be rendered at
two different resolutions — this is what lets the tests prove the
resolution-scaling fix in keyboard_detector.py actually works, by
generating a keymap from one resolution and detecting against another.
"""

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

KEYBOARD_TOP_FRAC = 0.67
LEFT_EDGE_FRAC = 0.05
RIGHT_EDGE_FRAC = 0.95

LOOKAHEAD_SECONDS = 1.4


def hsv_to_bgr(h: int, s: int, v: int):
    arr = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(arr, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def scene_geometry(width: int, height: int):
    keyboard_top = int(round(height * KEYBOARD_TOP_FRAC))
    left_edge = int(round(width * LEFT_EDGE_FRAC))
    right_edge = int(round(width * RIGHT_EDGE_FRAC))
    return keyboard_top, left_edge, right_edge


def draw_keyboard(frame: np.ndarray, keys: List[dict], keyboard_top: int):
    height, width, _ = frame.shape

    white_keys = [k for k in keys if k["type"] == "white"]
    if len(white_keys) >= 2:
        white_w = (white_keys[-1]["x"] - white_keys[0]["x"]) / (len(white_keys) - 1)
    else:
        white_w = 10.0

    left = int(round(white_keys[0]["x"] - white_w / 2))
    right = int(round(white_keys[-1]["x"] + white_w / 2))

    cv2.rectangle(frame, (left, keyboard_top), (right, height - 1), (235, 235, 235), -1)

    black_h = int(round((height - keyboard_top) * 0.6))
    for k in keys:
        if k["type"] != "black":
            continue
        cx = int(round(k["x"]))
        half = max(1, int(round(white_w * 0.28)))
        cv2.rectangle(frame, (cx - half, keyboard_top), (cx + half, keyboard_top + black_h), (10, 10, 10), -1)

    return white_w


def render_video(
    path: Path,
    width: int,
    height: int,
    fps: float,
    duration_s: float,
    ground_truth: List[dict],
    keys: List[dict],
    hue_map: Optional[Dict[str, int]] = None,
):
    """ground_truth: list of {midi, start, end, hand, color?} dicts.

    `color` selects which entry of `hue_map` a note is rendered in; if
    omitted, falls back to the note's `hand` (so old two-hue tests that
    never set `color` keep working unchanged). `hue_map` defaults to the
    original two-hue {"left": 85, "right": 150} scheme, but callers can
    pass a third (or more) distinctly-hued entry to exercise multi-color
    detection, mirroring a real video that uses more than two colors for
    reasons unrelated to hand (a white/black-key color variant, a
    sustained-voice highlight, etc.).
    """

    if hue_map is None:
        hue_map = {"left": 85, "right": 150}

    keyboard_top, left_edge, right_edge = scene_geometry(width, height)
    key_by_midi = {k["midi"]: k for k in keys}

    pastel = {name: hsv_to_bgr(hue, 130, 195) for name, hue in hue_map.items()}
    bright = {name: hsv_to_bgr(hue, 235, 235) for name, hue in hue_map.items()}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")

    speed = keyboard_top / LOOKAHEAD_SECONDS
    n_frames = int(round(duration_s * fps))

    white_w_ref = None

    for frame_idx in range(n_frames):
        t = frame_idx / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        white_w_ref = draw_keyboard(frame, keys, keyboard_top)

        bar_half = max(2, int(round(white_w_ref * 0.32)))

        for note in ground_truth:
            key = key_by_midi[note["midi"]]
            cx = int(round(key["x"]))

            start, end = note["start"], note["end"]
            color_key = note.get("color", note["hand"])

            full_bottom = keyboard_top - speed * (start - t)
            full_top = keyboard_top - speed * (end - t)

            visible_bottom = min(full_bottom, keyboard_top - 1)
            visible_top = max(full_top, 0)

            if visible_top > visible_bottom:
                continue

            # Frame time and note start/end are both derived from small
            # rational fractions (fps, quarter-note length) that don't
            # always round to the same float, so a note landing exactly on
            # a frame boundary can compare on the wrong side by ~1e-13.
            # Nudge the comparison so the intended frame renders bright.
            eps = 1e-6
            color = bright[color_key] if (start - eps) <= t < (end - eps) else pastel[color_key]

            cv2.rectangle(
                frame,
                (cx - bar_half, int(round(visible_top))),
                (cx + bar_half, int(round(visible_bottom))),
                color,
                -1,
            )

        writer.write(frame)

    writer.release()
    return keyboard_top, left_edge, right_edge
