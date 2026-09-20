#!/usr/bin/env python3
"""Dump per-frame brightness diagnostics for ONE key over a time window.

Not part of the normal pipeline -- a debugging tool to see exactly how a
key's color evolves frame-by-frame during a real press, instead of
guessing from aggregate calibration numbers. Run it over a time range
where you know (by ear/eye) a note is being held, and compare the
printed ratio against what analyze_key requires to call it "active".

Usage:
    python diagnose_key.py --video path.mp4 --keymap keyboard_map.json --midi 63 --start 3.5 --end 6.5
"""

import argparse
from pathlib import Path

import cv2

import keyboard_detector as kd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--keymap", required=True)
    parser.add_argument("--midi", type=int, required=True, help="MIDI note number to inspect (e.g. 63 for Eb4).")
    parser.add_argument("--start", type=float, required=True, help="Start time in seconds.")
    parser.add_argument("--end", type=float, required=True, help="End time in seconds.")
    parser.add_argument("--hues", type=str, default=None, help="Comma-separated hue values to use instead of automatic calibration.")
    args = parser.parse_args()

    import logging
    logger = logging.getLogger("diagnose")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    keys, reference_height, reference_keyboard_top = kd.load_keymap(Path(args.keymap))
    key = next((k for k in keys if k["midi"] == args.midi), None)
    if key is None:
        raise SystemExit(f"No key with midi={args.midi} in keymap")

    video = cv2.VideoCapture(args.video)
    if not video.isOpened():
        raise SystemExit(f"Could not open {args.video}")

    fps = float(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = height / float(reference_height)
    keyboard_top = int(round(reference_keyboard_top * scale))
    cfg = kd.DetectorConfig(scale=scale)

    if args.hues:
        hues = [float(x) for x in args.hues.split(",")]
        print(f"Using manual hues: {', '.join(f'{h:.1f}' for h in hues)}")
    else:
        hues, cal_info = kd.learn_bright_hues(video, fps, frame_count, keyboard_top, cfg, logger)
        print(f"Calibrated hues: {', '.join(f'{h:.1f}' for h in hues)}")

    print(f"min_bright_sat={cfg.min_bright_sat} min_bright_value={cfg.min_bright_value}")
    print(f"onset ratio threshold={cfg.min_bottom_bright_ratio:.3f}")
    print(f"key: midi={key['midi']} pitch={key['pitch']} x={key['x']}")
    print()

    half_widths = kd.compute_key_half_widths(keys, cfg)
    half = half_widths[args.midi]

    start_frame = int(round(args.start * fps))
    end_frame = int(round(args.end * fps))

    print(f"{'frame':>6} {'t':>7} {'top':>5} {'bot':>5} {'touchKB':>8} {'active':>7} {'ratio':>8} {'reach':>7} {'rows':>6}")

    for fi in range(start_frame, end_frame + 1):
        video.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = video.read()
        if not ok:
            print(f"{fi:>6}  (could not read frame)")
            continue

        hsv = cv2.cvtColor(frame[:keyboard_top], cv2.COLOR_BGR2HSV)
        general = kd.general_bar_mask(hsv, hues, cfg)
        bright = kd.bright_mask(hsv, hues, cfg)

        result = kd.analyze_key(general, bright, key, cfg, width, keyboard_top, half)

        if result is None:
            print(f"{fi:>6} {fi/fps:7.3f}  (no bar found in this column)")
            continue

        print(
            f"{fi:>6} {fi/fps:7.3f} {result['top']:>5} {result['bottom']:>5} "
            f"{str(result['touches_keyboard']):>8} {str(result['active']):>7} "
            f"{result['ratio']:>8.3f} {str(result['reaches_bottom']):>7} {result['rows']:>6}"
        )

    video.release()


if __name__ == "__main__":
    main()
