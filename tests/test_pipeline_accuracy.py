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

import numpy as np
import pytest

import calibrate_keymap
import keyboard_detector
import json_to_sheet
from tests.synthetic_video import render_video, scene_geometry

FPS = 30.0
QUARTER_SECONDS = 0.4  # 150 BPM; chosen so 8th/16th notes land on exact frames at 30fps.

# A real tutorial video was found to use a third, distinctly different
# note-bar color (30 = orange, far from the primary two) alongside the
# usual two -- confirmed by the person testing this against actual
# YouTube footage: colors there encode hand *and* white/black key type,
# so more than two colors is a real, common case, not an edge case.
# Calibration hardcoded to exactly two hues silently dropped that whole
# voice. HUE_MAP below is what synthetic_video.py renders notes in;
# "third" is used by one note to exercise that fix directly.
HUE_MAP = {"left": 85, "right": 150, "third": 30}

GROUND_TRUTH_SPEC = [
    # (start_in_quarters, duration_in_quarters, midi, hand, color)
    (0.0, 1.0, 60, "right", None),   # C4  -- simultaneous with E4 below: a chord
    (0.0, 1.0, 64, "right", None),   # E4
    (1.0, 0.5, 62, "right", None),   # D4  eighth note
    (1.5, 0.5, 65, "right", None),   # F4  eighth note
    (2.0, 4.0, 48, "left", None),    # C3  whole note in the left hand, crosses the
                                      #     measure boundary at quarter=4 (tie test)
    (2.0, 1.0, 67, "right", None),   # G4  simultaneous with the sustained left note
    (3.0, 1.0, 69, "right", None),   # A4
    (6.0, 0.25, 60, "right", None),  # a run of 16th notes
    (6.25, 0.25, 62, "right", None),
    (6.5, 0.25, 64, "right", None),
    (6.75, 0.25, 65, "right", None),
    (7.0, 1.0, 67, "right", None),
    (0.0, 1.0, 72, "right", "third"),  # C5 -- rendered in the third color;
                                        # hand is still decided by pitch (>=60),
                                        # not by which color it was drawn in.
]


def make_logger():
    logger = logging.getLogger("test_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers = [logging.StreamHandler(sys.stdout)]
    logger.propagate = False
    return logger


def build_ground_truth():
    notes = []
    for start_q, dur_q, midi, hand, color in GROUND_TRUTH_SPEC:
        start = start_q * QUARTER_SECONDS
        end = (start_q + dur_q) * QUARTER_SECONDS
        note = {"midi": midi, "start": start, "end": end, "hand": hand}
        if color is not None:
            note["color"] = color
        notes.append(note)
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
    render_video(video_path, width, height, FPS, duration_s, ground_truth, keys, hue_map=HUE_MAP)

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
    render_video(video_path, width, height, FPS, duration_s, ground_truth, keys, hue_map=HUE_MAP)

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
    hues, cal_info = keyboard_detector.learn_bright_hues(video, fps, frame_count, keyboard_top, cfg, logger)

    assert len(hues) >= 3, f"Expected calibration to find all 3 distinct colors, got {hues}"

    detected = keyboard_detector.track_video(
        video, keys, hues, cfg, fps, frame_count, width, keyboard_top, logger, progress=False,
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

    hues, _ = keyboard_detector.learn_bright_hues(video, fps, frame_count, broken_keyboard_top, cfg, logger)

    detected = keyboard_detector.track_video(
        video, keys, hues, cfg, fps, frame_count, width, broken_keyboard_top, logger, progress=False,
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

    hues, cal_info = keyboard_detector.learn_bright_hues(video, fps, frame_count, keyboard_top, cfg, logger)
    notes = keyboard_detector.track_video(
        video, keys, hues, cfg, fps, frame_count, width, keyboard_top, logger, progress=False,
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


def test_ottava_bracket_transposes_display_pitch_only(logger):
    """Unit-level check on apply_ottava_brackets: a note above the high
    threshold gets wrapped in an 8va spanner and its displayed pitch
    shifts down an octave; a note below the low threshold gets an 8vb
    spanner and shifts up an octave. Notes inside the normal range are
    left untouched."""

    from music21 import note as m21note
    from music21 import spanner as m21spanner
    from music21 import stream as m21stream

    part = m21stream.Part()
    part.id = "RH"
    m = m21stream.Measure(number=1)

    high_note = m21note.Note(midi=96)   # C7 -- above OTTAVA_HIGH_MIDI (84)
    normal_note = m21note.Note(midi=60)  # C4 -- untouched
    low_note = m21note.Note(midi=28)    # E1 -- below OTTAVA_LOW_MIDI (36)

    m.insert(0, high_note)
    m.insert(1, normal_note)
    m.insert(2, low_note)
    part.append(m)

    bracket_count = json_to_sheet.apply_ottava_brackets(part, logger)

    assert bracket_count == 2  # one high run, one low run (normal_note breaks them apart)
    assert high_note.pitch.midi == 96 - 12
    assert normal_note.pitch.midi == 60
    assert low_note.pitch.midi == 28 + 12

    ottavas = list(part.getElementsByClass(m21spanner.Ottava))
    assert len(ottavas) == 2
    types = sorted(o.type for o in ottavas)
    assert types == ["8va", "8vb"]


def test_convert_keeps_true_pitch_in_midi_but_shifts_musicxml(tmp_path, logger):
    """End-to-end: a note far above the staff should play back at its real
    detected pitch in the MIDI file, while the MusicXML shows it inside an
    8va bracket with the notated pitch shifted down an octave."""

    from music21 import converter as m21converter
    from music21 import spanner as m21spanner

    notes = [
        {"pitch": "C4", "midi": 60, "type": "white", "hand": "right", "start_time": 0.0, "end_time": 0.4},
        {"pitch": "D4", "midi": 62, "type": "white", "hand": "right", "start_time": 0.4, "end_time": 0.8},
        {"pitch": "C7", "midi": 96, "type": "white", "hand": "right", "start_time": 0.8, "end_time": 1.2},
        {"pitch": "E4", "midi": 64, "type": "white", "hand": "right", "start_time": 1.2, "end_time": 1.6},
    ]

    tracked_path = tmp_path / "tracked_notes.json"
    with open(tracked_path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"fps": 30.0}, "notes": notes}, f)

    out_musicxml = tmp_path / "ottava.musicxml"
    out_midi = tmp_path / "ottava.mid"

    json_to_sheet.convert(tracked_path, out_musicxml, out_midi, logger)

    # Part ids ("RH"/"LH") aren't preserved through a MusicXML round-trip
    # (music21 reassigns its own on reload), so just inspect the whole score.
    reloaded = m21converter.parse(str(out_musicxml))

    ottavas = list(reloaded.recurse().getElementsByClass(m21spanner.Ottava))
    assert ottavas, "Expected an ottava bracket around the C7 note in the MusicXML"

    xml_notes = list(reloaded.recurse().notes)
    shifted = [n for n in xml_notes if n.pitch.midi == 96 - 12]
    assert shifted, "Expected the C7 note to be notated an octave lower under the 8va bracket"

    midi_score = m21converter.parse(str(out_midi))
    midi_pitches = {n.pitch.midi for n in midi_score.recurse().notes}
    assert 96 in midi_pitches, "MIDI playback must keep the true detected pitch, not the notated one"


def test_isolated_artifact_far_from_keyboard_is_not_active():
    """Regression test for a false positive seen on real footage: an
    isolated colored blob anywhere in the falling-note area (a watermark
    sliver, a compression artifact) could satisfy every "bright touches
    the bottom of its own run" check while sitting nowhere near the
    actual keyboard, and get reported as a currently-played note.
    analyze_key must also check that the run's bottom is near the real
    keyboard before calling it active."""

    width = 200
    keyboard_top = 200
    cfg = keyboard_detector.DetectorConfig(scale=1.0)
    key = {"midi": 60, "pitch": "C4", "type": "white", "x": 100.0}
    half_width = cfg.key_half_width

    general = np.zeros((keyboard_top, width), dtype=bool)
    bright = np.zeros((keyboard_top, width), dtype=bool)

    x1, x2 = 100 - half_width, 100 + half_width + 1

    # An isolated bright blob near the TOP of the frame, far from the
    # keyboard at row 199 -- tall enough and bright enough to pass every
    # per-run check on its own.
    general[0:15, x1:x2] = True
    bright[0:15, x1:x2] = True

    result = keyboard_detector.analyze_key(general, bright, key, cfg, width, keyboard_top, half_width)

    assert result is not None
    assert result["active"] is False, "An artifact far from the keyboard must not be reported as a played note"

    # Sanity check the fix doesn't also break real detections: the same
    # kind of bar, but actually touching the keyboard at the bottom row.
    general2 = np.zeros((keyboard_top, width), dtype=bool)
    bright2 = np.zeros((keyboard_top, width), dtype=bool)

    general2[keyboard_top - 15 : keyboard_top, x1:x2] = True
    bright2[keyboard_top - 15 : keyboard_top, x1:x2] = True

    result2 = keyboard_detector.analyze_key(general2, bright2, key, cfg, width, keyboard_top, half_width)

    assert result2 is not None
    assert result2["active"] is True


def test_sustain_uses_touching_not_brightness():
    """Regression test for a real failure mode found on actual footage: a
    single continuously-touching bar (one long sustained note, confirmed
    frame-by-frame) had its brightness spike to ~1.0 for ~2 frames every
    0.4-0.7s and sit at exactly 0.0 the rest of the time -- this
    visualizer pulses the "played" brightness rhythmically rather than
    holding it constant while a note sustains. Requiring brightness on
    every frame chopped one long note into dozens of near-zero-length
    fragments. Continuing an already-active note must only require the
    bar to still be touching the keyboard, regardless of brightness;
    starting a brand new note still requires the full onset brightness."""

    cfg = keyboard_detector.DetectorConfig(scale=1.0)

    bright_onset = {
        "active": True,
        "touches_keyboard": True,
        "ratio": 0.9, "reaches_bottom": True, "rows": 10,
    }
    # Still touching the keyboard, but completely dark -- the "in between
    # pulses" state observed on real footage.
    dark_but_touching = {
        "active": False,
        "touches_keyboard": True,
        "ratio": 0.0, "reaches_bottom": False, "rows": 0,
    }
    no_longer_touching = {
        "active": False,
        "touches_keyboard": False,
        "ratio": 0.0, "reaches_bottom": False, "rows": 0,
    }

    assert keyboard_detector.decide_active(bright_onset, False, cfg) is True

    # Continuing an already-active note through a completely dark (but
    # still touching) frame must survive.
    assert keyboard_detector.decide_active(dark_but_touching, True, cfg) is True

    # The same dark frame must NOT be enough to start a brand new note
    # from scratch.
    assert keyboard_detector.decide_active(dark_but_touching, False, cfg) is False

    # Once the bar genuinely stops touching the keyboard, the note ends
    # even if it was previously active.
    assert keyboard_detector.decide_active(no_longer_touching, True, cfg) is False
