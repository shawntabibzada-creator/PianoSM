#!/usr/bin/env python3
"""Detect played notes in a falling-note ("Synthesia-style") piano tutorial
video.

The visualizer draws two appearances for a falling note bar:

    UNPLAYED = pastel/light bar color
    PLAYED   = saturated bright color

A note is "played" when the bright color region is attached to the
bottom edge of the falling bar (i.e. the bar is touching the keybed), not
just whenever a bright pixel appears anywhere in frame. This module finds,
for every one of the 88 keys, the falling bar nearest the keyboard and asks
whether its bottom is bright.

Compared to the original prototype this version fixes:

  * Key x-positions (and every pixel threshold) were never rescaled when
    the downloaded video wasn't exactly 1080p, silently misaligning every
    sampling column against the actual key positions on anything else.
  * Color calibration only ever looked at a fixed 2.8s-6.5s window, so a
    video with a longer intro (or notes starting immediately) calibrated
    against silence or missed the bright color range entirely.
  * The "is this bar's column real" check used a fixed pixel-count
    threshold regardless of how wide the sampling column actually was.
  * Calibration hardcoded exactly two colors ("left hand"/"right hand").
    Some tutorials use a third (or more) distinct color for reasons that
    have nothing to do with which hand is playing -- a sustained/pedal
    voice, a highlight color, etc. -- and that whole voice went completely
    undetected, since only the two strongest hue peaks were ever kept.
    Detection now tracks however many distinct colors the video actually
    uses; which staff a note is notated on is decided by its pitch (the
    standard convention when true hand data isn't available), not by
    which of those colors it happened to be drawn in.
  * Nothing was logged beyond a final count, so a wrong run was
    unreviewable after the fact.
"""

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from common import (
    DEFAULT_DEBUG_IMAGE,
    DEFAULT_KEYMAP_PATH,
    DEFAULT_TRACKED_JSON,
    DEFAULT_VIDEO_PATH,
    LOG_DIR,
    get_logger,
)

KEYBOARD_TOP_1080P = 722

# Used only if calibration can't find any distinct hue peak at all.
FALLBACK_HUES = [83.9, 150.4]

# How many distinct note-bar colors to look for. Two is the common case
# (one per hand), but some tutorials add a third or fourth color for a
# sustained/pedal voice, a highlight, etc. -- missing one of those means
# an entire voice goes undetected.
MAX_HUES = 4

# Hue peaks closer together than this (out of 180) are treated as the
# same color rather than two distinct ones.
MIN_HUE_SEPARATION = 25.0


@dataclass
class DetectorConfig:
    """All thresholds, pre-scaled for the video's actual resolution.

    Every constant here was tuned against 1080p footage. Scaling them by
    `scale = height / 1080` is what makes the detector work on a 720p or
    1440p download instead of silently sampling the wrong pixels.
    """

    scale: float = 1.0

    hue_tolerance: float = 15.0
    general_hue_tolerance: float = 25.0

    min_bar_sat: int = 45
    min_bar_value: int = 70

    min_bright_sat: int = 175
    min_bright_value: int = 100

    key_half_width_base: float = 5.0

    min_bar_height_base: float = 8.0
    bottom_band_pixels_base: float = 14.0
    min_bottom_bright_ratio: float = 0.38
    min_bright_rows_base: float = 4.0
    bottom_touch_rows_base: float = 3.0

    # How close a run's own bottom edge must be to the real keyboard for
    # it to ever count as "currently touching the keys". Without this, an
    # isolated colored artifact anywhere in the falling-note area (a
    # watermark sliver, a compression blip) that happens to satisfy the
    # *relative* bottom-touch check within its own few-pixel-tall run can
    # be reported as an active note, no matter how far it actually is
    # from the keyboard. Measured against real footage, a genuinely
    # touching bar's own reported bottom can jitter by 30-45px frame to
    # frame (general_bar_mask's row detection isn't pixel-perfect), so a
    # tight gap here falsely drops "still touching" to "not touching".
    max_keyboard_gap_base: float = 50.0

    # Minimum fraction of a sampling column's pixels that must carry the
    # bar color for a row to count as "part of the bar". Fixed at 2 out
    # of 11 columns in the original code; expressed as a ratio here so it
    # holds regardless of how wide the scaled column actually is.
    min_bar_col_ratio: float = 0.18

    # A fast passage (16th notes, trills) can have a real "active" window
    # only a couple of frames wide, and the tail of *every* note's bar
    # naturally shrinks below min_bar_height as it finishes crossing the
    # keybed. Requiring 2+ consecutive confirmed frames before accepting
    # a press cost genuinely fast notes their entire detectable window.
    # The composite "active" test (hue-specific brightness, ratio,
    # bottom-touch, row count) is already a strong filter on its own, so
    # press confirmation no longer adds a debounce delay; min_note_frames
    # below still discards single-frame flicker from the output.
    press_confirm_frames: int = 1
    release_confirm_frames: int = 3
    min_note_frames: int = 2

    # Notes at or above this MIDI number are notated on the treble staff,
    # below on the bass staff (60 = middle C). Which color a video draws a
    # note in doesn't reliably indicate hand -- see MAX_HUES above -- so
    # staff assignment is decided by pitch instead, same as most
    # auto-transcription tools do when true hand data isn't available.
    hand_split_midi: int = 60

    @property
    def key_half_width(self) -> int:
        return max(1, int(round(self.key_half_width_base * self.scale)))

    @property
    def min_bar_height(self) -> int:
        return max(1, int(round(self.min_bar_height_base * self.scale)))

    @property
    def bottom_band_pixels(self) -> int:
        return max(1, int(round(self.bottom_band_pixels_base * self.scale)))

    @property
    def min_bright_rows(self) -> int:
        return max(1, int(round(self.min_bright_rows_base * self.scale)))

    @property
    def max_keyboard_gap(self) -> int:
        return max(1, int(round(self.max_keyboard_gap_base * self.scale)))

    @property
    def bottom_touch_rows(self) -> int:
        return max(1, int(round(self.bottom_touch_rows_base * self.scale)))


# ============================================================
# HUE HELPERS
# ============================================================


def hue_distance(h: np.ndarray, target: float) -> np.ndarray:
    d = np.abs(h.astype(np.float32) - float(target))
    return np.minimum(d, 180.0 - d)


def clamp_hue_tolerance_for_separation(cfg: DetectorConfig, hues: List[float], logger) -> None:
    """If two of the tracked colors happen to be close together on the
    hue wheel, the default tolerances can make their "is this pixel this
    color" zones overlap, so a pixel near the midpoint gets matched as
    both (or neither cleanly). Shrink the tolerances in place so no two
    zones can touch, whatever the calibrated hues turn out to be.
    """

    if len(hues) < 2:
        return

    min_gap = min(
        min(abs(hues[i] - hues[j]), 180.0 - abs(hues[i] - hues[j]))
        for i in range(len(hues))
        for j in range(i + 1, len(hues))
    )
    safety_margin = 2.0
    max_safe = max(3.0, min_gap / 2.0 - safety_margin)

    if cfg.hue_tolerance > max_safe or cfg.general_hue_tolerance > max_safe:
        logger.warning(
            f"Calibrated hues are only {min_gap:.1f} degrees apart at closest; shrinking hue tolerance "
            f"{cfg.hue_tolerance:.1f}/{cfg.general_hue_tolerance:.1f} -> {max_safe:.1f} "
            "so one color can't be matched as another."
        )
        cfg.hue_tolerance = min(cfg.hue_tolerance, max_safe)
        cfg.general_hue_tolerance = min(cfg.general_hue_tolerance, max_safe)


# ============================================================
# MASKS
# ============================================================


def _any_hue_mask(h: np.ndarray, hues: List[float], tolerance: float) -> np.ndarray:
    mask = np.zeros(h.shape, dtype=bool)
    for hue in hues:
        mask |= hue_distance(h, hue) <= tolerance
    return mask


def general_bar_mask(hsv: np.ndarray, hues: List[float], cfg: DetectorConfig) -> np.ndarray:
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return (s >= cfg.min_bar_sat) & (v >= cfg.min_bar_value) & _any_hue_mask(h, hues, cfg.general_hue_tolerance)


def bright_mask(hsv: np.ndarray, hues: List[float], cfg: DetectorConfig) -> np.ndarray:
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return (s >= cfg.min_bright_sat) & (v >= cfg.min_bright_value) & _any_hue_mask(h, hues, cfg.hue_tolerance)


# ============================================================
# CALIBRATION — adaptive, not a fixed 2.8s-6.5s window
# ============================================================


def _adapt_bright_thresholds(
    video: cv2.VideoCapture,
    scanned_indices: List[int],
    keyboard_top: int,
    hues: List[float],
    cfg: DetectorConfig,
    logger,
    min_samples: int = 200,
) -> None:
    """min_bright_sat/value (175/100) were tuned against one reference
    video's "played" color. If a different video's played color is less
    saturated than that, nothing ever crosses the threshold: real notes
    go undetected for the whole video, and whatever few things ARE
    saturated enough (a watermark, a UI accent) can end up dominating
    calibration instead. Look at the actual saturation/value distribution
    of this video's own bar-colored pixels and split pastel from played
    with Otsu's method, rather than trusting a fixed number.
    """

    sat_samples = []
    val_samples = []

    for fi in scanned_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = video.read()

        if not ok:
            continue

        hsv = cv2.cvtColor(frame[:keyboard_top], cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        bar = (s >= cfg.min_bar_sat) & (v >= cfg.min_bar_value)
        mask = bar & _any_hue_mask(h, hues, cfg.general_hue_tolerance)

        if np.any(mask):
            sat_samples.append(s[mask])
            val_samples.append(v[mask])

    if not val_samples:
        logger.debug("No bar-colored samples for bright-threshold adaptation; keeping defaults")
        return

    all_sat = np.concatenate(sat_samples)
    all_val = np.concatenate(val_samples)

    if len(all_val) < min_samples or int(all_val.max()) == int(all_val.min()):
        logger.debug("Too few/uniform bar-colored samples to adapt bright threshold; keeping defaults")
        return

    val_u8 = all_val.astype(np.uint8).reshape(-1, 1)
    otsu_val, _ = cv2.threshold(val_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    below = int(np.count_nonzero(all_val < otsu_val))
    above = len(all_val) - below
    if below < 0.05 * len(all_val) or above < 0.05 * len(all_val):
        logger.debug(
            f"Value channel of bar pixels doesn't look bimodal (Otsu={otsu_val:.0f}, "
            f"{below}/{above} below/above split); keeping default bright/pastel threshold"
        )
        return

    sat_u8 = all_sat.astype(np.uint8).reshape(-1, 1)
    otsu_sat, _ = cv2.threshold(sat_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    new_min_value = int(otsu_val) + 5
    new_min_sat = int(otsu_sat) + 5

    logger.info(
        f"Adapted bright/pastel split from this video's own colors "
        f"({len(all_val)} samples): sat {cfg.min_bright_sat}->{new_min_sat}, "
        f"value {cfg.min_bright_value}->{new_min_value}"
    )
    cfg.min_bright_sat = new_min_sat
    cfg.min_bright_value = new_min_value


def learn_bright_hues(
    video: cv2.VideoCapture,
    fps: float,
    frame_count: int,
    keyboard_top: int,
    cfg: DetectorConfig,
    logger,
    max_scan_seconds: float = 30.0,
    target_bar_px: int = 4000,
    min_frames_scanned: int = 6,
    max_hues: int = MAX_HUES,
) -> Tuple[List[float], dict]:
    """Scan forward from the start of the video, accumulating a hue
    histogram of bar-colored pixels, until either enough pixels have been
    seen or `max_scan_seconds` is exhausted, then keep up to `max_hues`
    well-separated peaks -- however many distinct colors the video
    actually uses, not a hardcoded two.

    The original version only ever looked at 2.8s-6.5s, which assumed
    every video's first note lands in that window. Some intros are
    longer, some tutorials start on beat one — either way that fixed
    window could calibrate against near-silence.
    """

    hist = np.zeros(180, dtype=np.float64)
    step = max(1, int(round(fps / 6)))
    max_frame = min(frame_count - 1, int(round(max_scan_seconds * fps)))

    scanned_indices: List[int] = []
    total_bar_px = 0

    fi = 0
    while fi <= max_frame:
        video.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = video.read()

        if not ok:
            break

        hsv = cv2.cvtColor(frame[:keyboard_top], cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Use the loose "any bar pixel" mask here, not the strict bright
        # threshold: which saturation/value actually means "played" is
        # itself something this function figures out below, from this
        # same video. Gating the hue histogram on a hardcoded bright
        # threshold first is circular — if this video's played color
        # never reaches that fixed threshold, the histogram comes back
        # empty (or dominated by an unrelated saturated element, like a
        # watermark) before we ever get a chance to learn the real one.
        bar = (s >= cfg.min_bar_sat) & (v >= cfg.min_bar_value)
        n = int(np.count_nonzero(bar))

        if n:
            hist += np.bincount(h[bar].ravel(), minlength=180)
            total_bar_px += n

        scanned_indices.append(fi)
        fi += step

        if total_bar_px >= target_bar_px and len(scanned_indices) >= min_frames_scanned:
            break

    logger.info(
        f"Calibration scanned {len(scanned_indices)} frames "
        f"(up to t={scanned_indices[-1] / fps:.2f}s), "
        f"collected {total_bar_px} bar-colored pixels"
    )

    pad = np.concatenate([hist[-5:], hist, hist[:5]])
    smooth = np.convolve(
        pad, np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], dtype=np.float64), mode="same"
    )[5:-5]

    peaks = sorted(((smooth[hue], hue) for hue in range(180)), reverse=True)

    candidate_hues: List[float] = []
    for strength, hue in peaks:
        if strength <= 0:
            break
        if all(min(abs(hue - old), 180 - abs(hue - old)) >= MIN_HUE_SEPARATION for old in candidate_hues):
            candidate_hues.append(float(hue))
        if len(candidate_hues) >= max_hues:
            break

    if candidate_hues:
        hues = candidate_hues
        used_fallback = False
    else:
        hues = list(FALLBACK_HUES)
        used_fallback = True
        logger.warning("Could not find any distinct hue peaks; using fallback hues")

    clamp_hue_tolerance_for_separation(cfg, hues, logger)
    _adapt_bright_thresholds(video, scanned_indices, keyboard_top, hues, cfg, logger)

    logger.info(f"Detected {len(hues)} distinct note-bar color(s): {', '.join(f'{h:.1f}' for h in hues)}")

    info = {
        "hues": hues,
        "used_fallback": used_fallback,
        "frames_scanned": len(scanned_indices),
        "bar_pixels_seen": total_bar_px,
    }

    return hues, info


# ============================================================
# PER-KEY BAR ANALYSIS
# ============================================================


def vertical_run(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0

    row_has = np.any(mask, axis=1)

    best = 0
    current = 0

    for value in row_has:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def compute_key_half_widths(keys: List[dict], cfg: DetectorConfig) -> Dict[int, int]:
    """A fixed sampling half-width can be wider than the gap to a
    neighboring key — especially for black keys, which sit closer to
    their neighbors than the white-key pitch spacing suggests. When that
    happens, one key's sampling column bleeds into the next key's bar and
    both get reported as played. Cap each key's half-width at a fraction
    of its distance to the nearest adjacent key so columns never overlap.
    """

    ordered = sorted(keys, key=lambda k: k["midi"])
    xs = [k["x"] * cfg.scale for k in ordered]
    n = len(ordered)

    result: Dict[int, int] = {}

    for i, key in enumerate(ordered):
        gap_left = xs[i] - xs[i - 1] if i > 0 else float("inf")
        gap_right = xs[i + 1] - xs[i] if i < n - 1 else float("inf")
        min_gap = min(gap_left, gap_right)

        half = cfg.key_half_width
        if np.isfinite(min_gap):
            half = min(half, max(1, int(np.floor(0.45 * min_gap))))

        result[key["midi"]] = max(1, half)

    return result


def analyze_key(
    general: np.ndarray,
    bright: np.ndarray,
    key: dict,
    cfg: DetectorConfig,
    width: int,
    keyboard_top: int,
    half_width: int,
) -> Optional[dict]:

    # Key x-positions in keyboard_map.json are calibrated against a
    # specific reference resolution; without this scale factor every
    # column samples the wrong pixels on any other resolution.
    cx = int(round(key["x"] * cfg.scale))
    half = half_width

    x1 = max(0, cx - half)
    x2 = min(width, cx + half + 1)

    column = general[:keyboard_top, x1:x2]

    if column.size == 0:
        return None

    column_width = x2 - x1
    min_cols = max(1, int(round(column_width * cfg.min_bar_col_ratio)))

    row_count = np.count_nonzero(column, axis=1)
    row_has = row_count >= min_cols

    runs = []
    run_start = None

    for y, present in enumerate(row_has):
        if present and run_start is None:
            run_start = y
        elif not present and run_start is not None:
            run_end = y - 1
            length = run_end - run_start + 1
            if length >= cfg.min_bar_height:
                runs.append((run_start, run_end))
            run_start = None

    if run_start is not None:
        run_end = keyboard_top - 1
        length = run_end - run_start + 1
        if length >= cfg.min_bar_height:
            runs.append((run_start, run_end))

    if not runs:
        return None

    top, bottom = max(runs, key=lambda r: r[1])

    # NOTE: earlier versions rejected any run taller than 90% of the
    # falling-note area as a likely background/mask artifact. That
    # silently broke detection of any sustained note whose remaining
    # bar length (duration-to-go) exceeds the "lookahead" window the
    # video happens to show above the keyboard — a whole note or a held
    # pedal tone routinely produces a run that tall right when it starts
    # being played. The bottom-touch brightness checks below are what
    # actually distinguishes a real played bar from noise, so no extra
    # height cap is applied here.

    bright_crop = bright[top : bottom + 1, x1:x2]

    band_height = min(cfg.bottom_band_pixels, bottom - top + 1)
    by1 = bottom - band_height + 1

    bright_bottom = bright[by1 : bottom + 1, x1:x2]
    ratio = float(np.mean(bright_bottom)) if bright_bottom.size else 0.0

    bright_ys = np.nonzero(bright[top : bottom + 1, x1:x2])[0]
    reaches_bottom = len(bright_ys) > 0 and (bottom - (top + int(bright_ys.max())) <= cfg.bottom_touch_rows)

    rows = vertical_run(bright_crop)

    # A run can satisfy every "bright touches the bottom of itself" check
    # while sitting nowhere near the actual keyboard — e.g. a small
    # isolated artifact elsewhere in the falling-note area. Require the
    # run's own bottom to be near the real keyboard before it can ever
    # count as a currently-played note.
    touches_keyboard = (keyboard_top - 1 - bottom) <= cfg.max_keyboard_gap

    active = (
        touches_keyboard
        and ratio >= cfg.min_bottom_bright_ratio
        and reaches_bottom
        and rows >= cfg.min_bright_rows
    )

    return {
        "active": active,
        "top": top,
        "bottom": bottom,
        "touches_keyboard": touches_keyboard,
        "ratio": ratio,
        "reaches_bottom": reaches_bottom,
        "rows": rows,
    }


def decide_active(result: Optional[dict], was_active: bool, cfg: DetectorConfig) -> bool:
    """Starting a new note requires the full onset brightness threshold
    (`result["active"]`). Continuing a note already confirmed active only
    requires the bar to still be touching the keyboard, not brightness at
    all: some visualizers pulse the "played" brightness rhythmically
    rather than holding it constant while a note sustains, confirmed
    against real per-frame data (a single continuously-touching bar with
    brightness spiking for ~2 frames every 0.4-0.7s and sitting at
    exactly 0 in between). Requiring brightness on every frame chopped
    one long sustained note into dozens of near-zero-length fragments.
    """

    if result is None:
        return False

    if was_active and result["touches_keyboard"]:
        return True

    return result["active"]


# ============================================================
# STATE MACHINE
# ============================================================


def make_states(keys: List[dict]) -> Dict[int, dict]:
    states = {}
    for key in keys:
        states[key["midi"]] = {
            "key": key,
            "active": False,
            "start": None,
            "last": None,
            "off": 0,
            "candidate_start": None,
            "candidate_count": 0,
        }
    return states


def close_note(state: dict, notes: list, fps: float, cfg: DetectorConfig, logger=None, final_frame=None):
    if not state["active"]:
        return

    start = state["start"]

    if final_frame is None:
        final_frame = state["last"]

    if start is None or final_frame is None:
        state["active"] = False
        return

    count = int(final_frame) - int(start) + 1

    if count >= cfg.min_note_frames:
        start_time = start / fps
        end_time = (final_frame + 1) / fps
        hand = "left" if state["key"]["midi"] < cfg.hand_split_midi else "right"

        note = {
            "pitch": state["key"]["pitch"],
            "midi": int(state["key"]["midi"]),
            "type": state["key"]["type"],
            "hand": hand,
            "start_frame": int(start),
            "end_frame": int(final_frame),
            "start_time": float(start_time),
            "end_time": float(end_time),
            "duration": float(end_time - start_time),
        }
        notes.append(note)

        if logger is not None:
            logger.debug(
                f"RELEASE {note['pitch']:<4} hand={hand:<5} "
                f"start={note['start_time']:.3f}s dur={note['duration']:.3f}s"
            )

    state["active"] = False
    state["start"] = None
    state["last"] = None
    state["off"] = 0
    state["candidate_start"] = None
    state["candidate_count"] = 0


def track_video(
    video: cv2.VideoCapture,
    keys: List[dict],
    hues: List[float],
    cfg: DetectorConfig,
    fps: float,
    frame_count: int,
    width: int,
    keyboard_top: int,
    logger,
    progress: bool = True,
) -> List[dict]:

    states = make_states(keys)
    half_widths = compute_key_half_widths(keys, cfg)
    notes: List[dict] = []

    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_number = 0

    while True:
        ok, frame = video.read()
        if not ok:
            break

        hsv = cv2.cvtColor(frame[:keyboard_top], cv2.COLOR_BGR2HSV)

        general = general_bar_mask(hsv, hues, cfg)
        bright = bright_mask(hsv, hues, cfg)

        for midi, state in states.items():
            result = analyze_key(general, bright, state["key"], cfg, width, keyboard_top, half_widths[midi])
            is_active = decide_active(result, state["active"], cfg)

            if is_active:
                state["off"] = 0

                if state["active"]:
                    state["last"] = frame_number
                    continue

                if state["candidate_start"] is None:
                    state["candidate_start"] = frame_number
                state["candidate_count"] += 1

                if state["candidate_count"] >= cfg.press_confirm_frames:
                    state["active"] = True
                    state["start"] = state["candidate_start"]
                    state["last"] = frame_number
                    state["off"] = 0

                    if logger is not None:
                        logger.debug(
                            f"PRESS   {state['key']['pitch']:<4} "
                            f"frame={state['start']} t={state['start']/fps:.3f}s"
                        )

                    state["candidate_start"] = None
                    state["candidate_count"] = 0

            else:
                state["candidate_start"] = None
                state["candidate_count"] = 0

                if state["active"]:
                    state["off"] += 1
                    if state["off"] >= cfg.release_confirm_frames:
                        close_note(state, notes, fps, cfg, logger)

        frame_number += 1

        if progress and frame_number % 100 == 0:
            percent = (frame_number / max(frame_count, 1)) * 100.0
            print(f"\rProgress: {percent:6.2f}%", end="")

    for state in states.values():
        if state["active"]:
            close_note(state, notes, fps, cfg, logger, final_frame=frame_number - 1)

    if progress:
        print()

    notes.sort(key=lambda n: (n["start_frame"], n["midi"]))
    return notes


# ============================================================
# I/O HELPERS
# ============================================================


def load_keymap(path: Path) -> Tuple[List[dict], int, int]:
    """Returns (keys, reference_height, reference_keyboard_top).

    Supports both the new calibrate_keymap.py output (a dict carrying the
    resolution the x-positions were measured at) and the original flat
    list of keys, which is assumed to have been measured at 1080p — the
    same assumption the original script made. Without recording the
    reference resolution, there is no way to scale key positions
    correctly for a video downloaded at any other resolution.
    """

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "keys" in raw:
        raw_keys = raw["keys"]
        reference_height = int(raw.get("reference_height", 1080))
        reference_keyboard_top = int(raw.get("keyboard_top", KEYBOARD_TOP_1080P))
    else:
        raw_keys = raw
        reference_height = 1080
        reference_keyboard_top = KEYBOARD_TOP_1080P

    keys = sorted(
        [
            {
                "midi": int(k["midi"]),
                "pitch": str(k["pitch"]),
                "type": str(k["type"]).lower(),
                "x": float(k["x"]),
            }
            for k in raw_keys
        ],
        key=lambda k: k["midi"],
    )

    if len(keys) != 88:
        raise SystemExit(f"ERROR: Expected 88 keys, found {len(keys)}")

    return keys, reference_height, reference_keyboard_top


def download_video(url: str, out_path: Path, logger) -> None:
    import yt_dlp

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "outtmpl": str(out_path),
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
    }

    logger.info(f"Downloading {url} -> {out_path}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def write_debug_image(
    video: cv2.VideoCapture,
    keys: List[dict],
    hues: List[float],
    cfg: DetectorConfig,
    fps: float,
    width: int,
    keyboard_top: int,
    debug_path: Path,
    debug_time: float,
    logger,
) -> None:

    video.set(cv2.CAP_PROP_POS_FRAMES, int(round(debug_time * fps)))
    ok, debug_frame = video.read()

    if not ok:
        logger.warning(f"Could not read frame at t={debug_time:.2f}s for debug image")
        return

    hsv = cv2.cvtColor(debug_frame[:keyboard_top], cv2.COLOR_BGR2HSV)
    gm = general_bar_mask(hsv, hues, cfg)
    bm = bright_mask(hsv, hues, cfg)

    debug = debug_frame.copy()
    active = []
    half_widths = compute_key_half_widths(keys, cfg)

    for key in keys:
        half = half_widths[key["midi"]]
        result = analyze_key(gm, bm, key, cfg, width, keyboard_top, half)
        cx = int(round(key["x"] * cfg.scale))

        if result and result["active"]:
            hand = "left" if key["midi"] < cfg.hand_split_midi else "right"
            border = (255, 80, 0) if hand == "left" else (255, 0, 255)
            thickness = 4
            active.append(f"{key['pitch']} {hand} {result['bottom']}")
        else:
            border = (80, 80, 80)
            thickness = 1

        cv2.rectangle(debug, (cx - half, 0), (cx + half, keyboard_top - 1), border, thickness)
        cv2.putText(
            debug,
            key["pitch"],
            (max(0, cx - 14), keyboard_top + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.27,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if result:
            y = int(result["bottom"])
            cv2.line(debug, (cx - half - 2, y), (cx + half + 2, y), (0, 255, 255), 2)

    cv2.line(debug, (0, keyboard_top), (width - 1, keyboard_top), (0, 0, 255), 2)
    cv2.putText(
        debug,
        f"DEBUG t={debug_time:.2f}s",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    debug_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_path), debug)

    logger.info(f"Debug image saved: {debug_path}")
    logger.info(f"Active keys at t={debug_time:.2f}s: {', '.join(active) if active else '(none)'}")


def find_first_active_time(
    video: cv2.VideoCapture,
    keys: List[dict],
    hues: List[float],
    cfg: DetectorConfig,
    fps: float,
    frame_count: int,
    width: int,
    keyboard_top: int,
    max_scan_seconds: float = 30.0,
) -> float:
    """Find the first timestamp with an active (played) key, for the debug
    image. Falls back to 3.5s (the old hardcoded default) if nothing is
    found, rather than assuming every video's first note lands there."""

    step = max(1, int(round(fps / 4)))
    max_frame = min(frame_count - 1, int(round(max_scan_seconds * fps)))
    half_widths = compute_key_half_widths(keys, cfg)

    for fi in range(0, max_frame + 1, step):
        video.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = video.read()
        if not ok:
            continue

        hsv = cv2.cvtColor(frame[:keyboard_top], cv2.COLOR_BGR2HSV)
        gm = general_bar_mask(hsv, hues, cfg)
        bm = bright_mask(hsv, hues, cfg)

        for key in keys:
            result = analyze_key(gm, bm, key, cfg, width, keyboard_top, half_widths[key["midi"]])
            if result and result["active"]:
                return fi / fps

    return 3.5


# ============================================================
# MAIN
# ============================================================


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=None, help="YouTube URL to download. Skipped if --video already exists and this is omitted.")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO_PATH))
    parser.add_argument("--keymap", default=str(DEFAULT_KEYMAP_PATH))
    parser.add_argument("--out", default=str(DEFAULT_TRACKED_JSON))
    parser.add_argument("--debug-image", default=str(DEFAULT_DEBUG_IMAGE))
    parser.add_argument("--hues", type=str, default=None, help="Comma-separated hue values (0-179) to use instead of automatic calibration, e.g. '149,175,90'.")
    parser.add_argument("--hand-split-midi", type=int, default=None, help="MIDI note number at/above which notes are notated on the treble staff (default 60 = middle C).")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)

    video_path = Path(args.video)
    keymap_path = Path(args.keymap)
    out_path = Path(args.out)
    debug_path = Path(args.debug_image)

    logger = get_logger("keyboard_detector", LOG_DIR / "keyboard_detector.log")

    if args.url:
        download_video(args.url, video_path, logger)
    elif not video_path.exists():
        raise SystemExit(
            f"ERROR: {video_path} does not exist and no --url was given to download it."
        )

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise SystemExit(f"ERROR: Could not open {video_path}")

    fps = float(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        raise SystemExit("ERROR: Invalid FPS.")

    keys, reference_height, reference_keyboard_top = load_keymap(keymap_path)

    # Key x-positions (and every pixel threshold below) were measured at
    # `reference_height`. If the actual downloaded video is a different
    # resolution, everything must be rescaled or the sampling columns
    # land on the wrong pixels entirely.
    scale = height / float(reference_height)
    keyboard_top = int(round(reference_keyboard_top * scale))

    logger.info(f"Resolution: {width}x{height}  FPS: {fps:.2f}  Frames: {frame_count}  scale={scale:.4f}")
    logger.info(f"Keyboard top: y={keyboard_top} (reference {reference_keyboard_top} @ {reference_height}p)")

    cfg = DetectorConfig(scale=scale)
    if args.hand_split_midi is not None:
        cfg.hand_split_midi = args.hand_split_midi

    if args.hues:
        hues = [float(x) for x in args.hues.split(",")]
        cal_info = {"hues": hues, "used_fallback": False, "manual_override": True}
        logger.info(f"Using manually specified hues: {', '.join(f'{h:.1f}' for h in hues)}")
        clamp_hue_tolerance_for_separation(cfg, hues, logger)
    else:
        hues, cal_info = learn_bright_hues(video, fps, frame_count, keyboard_top, cfg, logger)
        logger.info(f"Calibrated hues: {', '.join(f'{h:.1f}' for h in hues)} (fallback={cal_info['used_fallback']})")

    debug_time = find_first_active_time(video, keys, hues, cfg, fps, frame_count, width, keyboard_top)
    write_debug_image(video, keys, hues, cfg, fps, width, keyboard_top, debug_path, debug_time, logger)

    logger.info("Tracking full video...")
    t0 = time.time()
    notes = track_video(
        video, keys, hues, cfg, fps, frame_count, width, keyboard_top, logger,
        progress=not args.no_progress,
    )
    elapsed = time.time() - t0

    video.release()

    logger.info(f"Tracked {len(notes)} notes in {elapsed:.1f}s")

    events_csv = out_path.parent / "note_events.csv"
    with open(events_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start_time", "end_time", "duration", "pitch", "midi", "hand"])
        for n in notes:
            writer.writerow([f"{n['start_time']:.4f}", f"{n['end_time']:.4f}", f"{n['duration']:.4f}", n["pitch"], n["midi"], n["hand"]])
    logger.info(f"Per-note event log: {events_csv}")

    output = {
        "meta": {
            "fps": fps,
            "width": width,
            "height": height,
            "scale": scale,
            "keyboard_top": keyboard_top,
            "hues": hues,
            "hand_split_midi": cfg.hand_split_midi,
            "calibration": cal_info,
            "note_count": len(notes),
        },
        "notes": notes,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved: {out_path}")

    print()
    print(f"Notes detected: {len(notes)}")
    for note in notes[:30]:
        print(f"{note['start_time']:7.3f}s  {note['pitch']:<4}  {note['hand']:<5}  {note['duration']:.3f}s")


if __name__ == "__main__":
    main()
