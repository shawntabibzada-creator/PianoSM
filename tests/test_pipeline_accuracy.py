"""End-to-end accuracy tests against a synthetic video with known ground
truth. Real YouTube footage isn't available in this environment, so this
is what stands in for it: a rendered scene where the "right answer" is
known exactly, used to prove the detector and sheet generator recover it
-- and specifically to prove the resolution-scaling bug fix works, by
calibrating a keymap on one resolution and detecting on a different one
(exactly how these scripts get used in practice: one keyboard_map.json
reused across videos downloaded at whatever resolution YouTube serves).
"""

import json
import logging
import sys

import pytest

import calibrate_keymap
import keyboard_detector
import json_to_sheet
from tests.synthetic_video import render_video, scene_geometry

FPS = 30.0
QUARTER_SECONDS = 0.4  # 150 BPM; chosen so 8th/16th notes land on exact frames at 30fps.

GROUND_TRUTH_SPEC = [
    # (start_in_quarters, duration_in_quarters, midi, hand)
    (0.0, 1.0, 60, "right"),   # C4  -- simultaneous with E4 below: a chord
    (0.0, 1.0, 64, "right"),   # E4
    (1.0, 0.5, 62, "right"),   # D4  eighth note
    (1.5, 0.5, 65, "right"),   # F4  eighth note
    (2.0, 4.0, 48, "left"),    # C3  whole note in the left hand, crosses the
                               #     measure boundary at quarter=4 (tie test)
    (2.0, 1.0, 67, "right"),   # G4  simultaneous with the sustained left note
    (3.0, 1.0, 69, "right"),   # A4
    (6.0, 0.25, 60, "right"),  # a run of 16th notes
    (6.25, 0.25, 62, "right"),
    (6.5, 0.25, 64, "right"),
    (6.75, 0.25, 65, "right"),
    (7.0, 1.0, 67, "right"),
]


def make_logger():
    logger = logging.getLogger("test_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers = [logging.StreamHandler(sys.stdout)]
    logger.propagate = False
    return logger


def build_ground_truth():
    notes = []
    for start_q, dur_q, midi, hand in GROUND_TRUTH_SPEC:
        start = start_q * QUARTER_SECONDS
        end = (start_q + dur_q) * QUARTER_SECONDS
        notes.append({"midi": midi, "start": start, "end": end, "hand": hand})
    return notes


@pytest.fixture(scope="module")
def logger():
    return make_logger()


@pytest.fixture(scope="module")
def ground_truth():
    return build_ground_truth()


@pytest.fixture(scope="module")
def calibrated_keymap(tmp_path_factory, ground_truth, logger):
    """Render a low-res (640x360) video and calibrate a keymap from it --
    standing in for "the one keyboard_map.json someone generated once"."""

    tmp = tmp_path_factory.mktemp("calibration")
    video_path = tmp / "calibration_video.mp4"

    width, height = 640, 360
    keyboard_top, left_edge, right_edge = scene_geometry(width, height)
    keys = calibrate_keymap.build_keymap(left_edge, right_edge)

    duration_s = max(n["end"] for n in ground_truth) + 1.0
    render_video(video_path, width, height, FPS, duration_s, ground_truth, keys)

    keymap_path = tmp / "keyboard_map.json"
    debug_path = tmp / "keymap_debug.png"

    payload = calibrate_keymap.calibrate(video_path, keymap_path, debug_path, logger)

    return {
        "path": keymap_path,
        "payload": payload,
        "true_keyboard_top": keyboard_top,
        "true_left_edge": left_edge,
        "true_right_edge": right_edge,
        "width": width,
        "height": height,
    }


def test_calibration_finds_keyboard_geometry(calibrated_keymap):
    payload = calibrated_keymap["payload"]

    assert abs(payload["keyboard_top"] - calibrated_keymap["true_keyboard_top"]) <= 4
    assert payload["reference_height"] == calibrated_keymap["height"]
    assert len(payload["keys"]) == 88

    keys_by_midi = {k["midi"]: k for k in payload["keys"]}
    # Spot-check a white and a black key land close to where they were drawn.
    true_keys = {k["midi"]: k for k in calibrate_keymap.build_keymap(
        calibrated_keymap["true_left_edge"], calibrated_keymap["true_right_edge"]
    )}
    for midi in (21, 60, 61, 108):
        assert abs(keys_by_midi[midi]["x"] - true_keys[midi]["x"]) <= 3.0


@pytest.fixture(scope="module")
def detection_video(tmp_path_factory, ground_truth, calibrated_keymap):
    """Render the SAME scene at exactly 2x the calibration resolution --
    this is the case that silently broke before the scale fix."""

    tmp = tmp_path_factory.mktemp("detection")
    video_path = tmp / "detection_video.mp4"

    width, height = 1280, 720
    _, left_edge, right_edge = scene_geometry(width, height)
    keys = calibrate_keymap.build_keymap(left_edge, right_edge)

    duration_s = max(n["end"] for n in ground_truth) + 1.0
    render_video(video_path, width, height, FPS, duration_s, ground_truth, keys)

    return {"path": video_path, "width": width, "height": height}


def _match_notes(detected, ground_truth, tol=3.0 / FPS):
    """Greedy match by (midi, hand); returns (matched_pairs, unmatched_gt, unmatched_detected)."""

    remaining = list(detected)
    matched = []
    unmatched_gt = []

    for gt in ground_truth:
        best = None
        for d in remaining:
            if d["midi"] != gt["midi"] or d["hand"] != gt["hand"]:
                continue
            if abs(d["start_time"] - gt["start"]) <= tol and abs(d["end_time"] - gt["end"]) <= tol:
                best = d
                break
        if best is not None:
            matched.append((gt, best))
            remaining.remove(best)
        else:
            unmatched_gt.append(gt)

    return matched, unmatched_gt, remaining


def test_detection_with_correct_scale_recovers_ground_truth(detection_video, calibrated_keymap, ground_truth, logger):
    keys, reference_height, reference_keyboard_top = keyboard_detector.load_keymap(calibrated_keymap["path"])

    video = keyboard_detector.cv2.VideoCapture(str(detection_video["path"]))
    assert video.isOpened()

    fps = float(video.get(keyboard_detector.cv2.CAP_PROP_FPS))
    frame_count = int(video.get(keyboard_detector.cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(keyboard_detector.cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(keyboard_detector.cv2.CAP_PROP_FRAME_HEIGHT))

    scale = height / float(reference_height)
    keyboard_top = int(round(reference_keyboard_top * scale))

    assert scale == pytest.approx(2.0)

    cfg = keyboard_detector.DetectorConfig(scale=scale)
    left_hue, right_hue, cal_info = keyboard_detector.learn_bright_hues(video, fps, frame_count, keyboard_top, cfg, logger)

    detected = keyboard_detector.track_video(
        video, keys, left_hue, right_hue, cfg, fps, frame_count, width, keyboard_top, logger, progress=False,
    )
    video.release()

    matched, unmatched_gt, unmatched_detected = _match_notes(detected, ground_truth)

    logger.info(f"Matched {len(matched)}/{len(ground_truth)} ground-truth notes")
    if unmatched_gt:
        logger.error(f"Missed notes: {unmatched_gt}")
    if unmatched_detected:
        logger.error(f"Spurious detections: {unmatched_detected}")

    assert not unmatched_gt, f"Missed ground-truth notes: {unmatched_gt}"
    assert not unmatched_detected, f"Spurious detected notes: {unmatched_detected}"


def test_detection_without_scale_fix_fails(detection_video, calibrated_keymap, ground_truth, logger):
    """Regression guard: sampling with scale=1.0 (the old bug — never
    rescaling key x-positions to the video's actual resolution) should
    detect essentially nothing real at 2x resolution, demonstrating why
    the fix in load_keymap/main() matters."""

    keys, reference_height, reference_keyboard_top = keyboard_detector.load_keymap(calibrated_keymap["path"])

    video = keyboard_detector.cv2.VideoCapture(str(detection_video["path"]))
    assert video.isOpened()

    fps = float(video.get(keyboard_detector.cv2.CAP_PROP_FPS))
    frame_count = int(video.get(keyboard_detector.cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(keyboard_detector.cv2.CAP_PROP_FRAME_WIDTH))

    # Deliberately use the *unscaled* keyboard_top and scale=1.0, matching
    # the pre-fix behavior of assuming the reference resolution always.
    cfg = keyboard_detector.DetectorConfig(scale=1.0)
    broken_keyboard_top = reference_keyboard_top

    left_hue, right_hue, _ = keyboard_detector.learn_bright_hues(video, fps, frame_count, broken_keyboard_top, cfg, logger)

    detected = keyboard_detector.track_video(
        video, keys, left_hue, right_hue, cfg, fps, frame_count, width, broken_keyboard_top, logger, progress=False,
    )
    video.release()

    matched, unmatched_gt, _ = _match_notes(detected, ground_truth)
    logger.info(f"Unscaled run matched only {len(matched)}/{len(ground_truth)} — confirms the scale fix is load-bearing")

    assert len(matched) < len(ground_truth) / 2


@pytest.fixture(scope="module")
def detected_notes_json(tmp_path_factory, detection_video, calibrated_keymap, logger):
    keys, reference_height, reference_keyboard_top = keyboard_detector.load_keymap(calibrated_keymap["path"])

    video = keyboard_detector.cv2.VideoCapture(str(detection_video["path"]))
    fps = float(video.get(keyboard_detector.cv2.CAP_PROP_FPS))
    frame_count = int(video.get(keyboard_detector.cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(keyboard_detector.cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(keyboard_detector.cv2.CAP_PROP_FRAME_HEIGHT))

    scale = height / float(reference_height)
    keyboard_top = int(round(reference_keyboard_top * scale))
    cfg = keyboard_detector.DetectorConfig(scale=scale)

    left_hue, right_hue, cal_info = keyboard_detector.learn_bright_hues(video, fps, frame_count, keyboard_top, cfg, logger)
    notes = keyboard_detector.track_video(
        video, keys, left_hue, right_hue, cfg, fps, frame_count, width, keyboard_top, logger, progress=False,
    )
    video.release()

    tmp = tmp_path_factory.mktemp("tracked")
    out_path = tmp / "tracked_notes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"fps": fps, "width": width, "height": height}, "notes": notes}, f)

    return out_path


def test_sheet_generation_recovers_tempo_and_ties(detected_notes_json, tmp_path, logger):
    out_musicxml = tmp_path / "out.musicxml"
    out_midi = tmp_path / "out.mid"

    score = json_to_sheet.convert(detected_notes_json, out_musicxml, out_midi, logger)

    assert out_musicxml.exists()
    assert out_midi.exists()

    quarter_seconds, grid, info = json_to_sheet.estimate_tempo_and_grid(
        json_to_sheet.load_notes(detected_notes_json, logger)[0], logger
    )
    true_bpm = 60.0 / QUARTER_SECONDS
    assert info["bpm"] == pytest.approx(true_bpm, rel=0.05)

    # The left-hand whole note starting at quarter 2 with duration 4 crosses
    # the measure boundary at quarter 4 -- it must show up as a tie, not be
    # silently truncated.
    left_part = next(p for p in score.parts if p.id == "LH")
    tied_notes = [n for m in left_part.getElementsByClass("Measure") for n in m.notes if n.tie is not None]
    assert tied_notes, "Expected the sustained left-hand note to be split into tied fragments across the barline"
