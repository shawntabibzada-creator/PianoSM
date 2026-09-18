# PianoSM

Turns a "falling note" piano tutorial video (Synthesia-style) into sheet
music by watching each key's falling color bar, tracking when it switches
from its pastel (falling) color to its saturated "played" color at the
keybed, and reconstructing notes, timing, and hands from that.

## Pipeline

```
1. calibrate_keymap.py  video.mp4  -> keyboard_map.json   (88 key x-positions)
2. keyboard_detector.py video.mp4  -> tracked_notes.json  (press/release events)
3. json_to_sheet.py     tracked_notes.json -> piano_sheet.musicxml / .mid
```

```bash
pip install -r requirements.txt

python calibrate_keymap.py --video data/videos/downloaded_video.mp4
python keyboard_detector.py --url "https://youtube.com/watch?v=..." \
    --keymap data/keyboard_map.json
python json_to_sheet.py --input data/tracked_notes.json
```

All defaults point at `data/`. Every stage writes a log to `data/logs/`
and `keyboard_detector.py` also writes `data/note_events.csv` — a
plain per-note press/release table — so a run is fully inspectable
after the fact instead of only reporting a final count.

### 1. calibrate_keymap.py

Reuse a `keyboard_map.json` across multiple videos rather than
regenerating it per-video if they share the same visualizer/layout, but
regenerate it whenever the layout or aspect ratio changes.

Auto-detects:
- the row where the keybed starts (`keyboard_top`)
- the left/right pixel edges of the white-key strip
- all 88 key x-positions, laid out with standard piano proportions
  (52 equal white keys; black keys positioned with approximate offsets)

The black-key placement is a best-effort approximation, not measured from
the actual video. **Always check `keymap_calibration_debug.png`** — every
key gets a vertical line at its detected x-position; if a line doesn't
land on its matching physical key, override with `--left-edge`,
`--right-edge`, or `--keyboard-top`, or hand-edit `keyboard_map.json`'s
`x` values for the keys that are off.

### 2. keyboard_detector.py

Reads `keyboard_map.json`, which records the resolution it was measured
at (`reference_height`). If the actual video is a different resolution
(a very common case — the same keymap gets reused across videos
downloaded at whatever resolution YouTube happens to serve), every key
position and pixel threshold is rescaled accordingly before sampling.
Getting this wrong was the single biggest source of misdetection in the
original prototype: it silently assumed every video was exactly 1080p.

Hue calibration (which color is "left hand" vs "right hand") scans
forward from the start of the video accumulating a histogram of bright
pixels until it has enough samples, rather than assuming the first note
always lands in a fixed 2.8s-6.5s window.

Output `tracked_notes.json` has the shape:

```json
{
  "meta": {"fps": 30.0, "width": 1920, "height": 1080, "scale": 1.0, ...},
  "notes": [{"pitch": "C4", "midi": 60, "hand": "right", "start_time": 1.2, "end_time": 1.6, ...}]
}
```

### 3. json_to_sheet.py

Tempo and the quantization grid (does this piece need 8th-note or
16th-note resolution?) are inferred from the spacing *between note
onsets*, not from how long notes were held — held duration reflects
articulation (staccato/legato), not the notated rhythm, and using it
gave the original prototype fragile, easily-wrong tempo estimates.

Notes that would cross a measure barline are split into tied fragments
instead of being truncated with the remainder dropped, and both hands
always share the same measure count so the grand staff stays aligned.

## Testing

There's no real YouTube footage available in this environment, so
`tests/` renders a synthetic video with **known ground-truth notes**
(`tests/synthetic_video.py`) and runs the real pipeline against it:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Notably, one test calibrates a keymap from a low-resolution rendering of
a scene and then runs detection against the *same scene rendered at 2x
the resolution* — this is what actually exercises (and would catch a
regression in) the resolution-scaling fix described above. Another test
runs detection with the scaling deliberately disabled to confirm it's
load-bearing (fails without it).

## Known limitations

- Black-key x-positions from `calibrate_keymap.py` are geometric
  approximations, not measured — verify against the debug overlay.
- Same-hand notes are rendered as a single monophonic line per staff; a
  genuinely polyphonic single-hand passage (e.g. a sustained note under
  a moving line) gets its sustain clipped at the next onset rather than
  drawn as a second voice.
- Tempo estimation assumes a single fixed tempo for the whole piece; it
  does not detect tempo changes, swing, or triplets beyond a basic
  triplet-eighth grid candidate.
