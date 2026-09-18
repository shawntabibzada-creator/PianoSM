#!/usr/bin/env python3
"""Convert tracked_notes.json (output of keyboard_detector.py) into
MusicXML + MIDI sheet music.

Fixes over the original prototype:

  * Tempo/grid was inferred from *held-key duration*, which reflects how
    staccato or legato someone plays, not the notated rhythm. It is now
    inferred from the *spacing between note onsets* (inter-onset
    intervals), which is what actually encodes tempo regardless of
    articulation.
  * The quantization grid was a hardcoded eighth-note (0.5). It is now
    chosen from the data — a piece with sixteenth-note passages no longer
    gets every fast note rounded to the nearest eighth.
  * FPS was hardcoded to 30 for de-noising durations; it is now read from
    the detector's own metadata (falls back to 30 with a warning only if
    that metadata is missing).
  * A note whose duration crossed a measure boundary was silently
    truncated at the barline with the remainder just dropped. Notes are
    now split into tied fragments across the measures they span.
  * The two hands could end up with different numbers of measures
    (whichever hand's last note ended first), breaking grand-staff
    alignment. Both parts now share one measure count.
  * Same-hand notes with detected durations overlapping the next note's
    onset (common when the detector reports a delayed release, e.g. from
    a lingering bright frame) produced overlapping notation. They are now
    clipped to end at the next onset in that hand.
  * Notes far above the treble staff or below the bass staff were printed
    with a wall of ledger lines instead of an 8va/8vb bracket, which is
    both hard to read and not how engraved sheet music is normally done.
"""

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from music21 import chord, clef, duration, meter, metadata, note, spanner, stream, tempo, tie

from common import DEFAULT_MIDI, DEFAULT_MUSICXML, DEFAULT_TRACKED_JSON, LOG_DIR, get_logger

TIME_SIGNATURE = "4/4"
MEASURE_LENGTH_QUARTERS = 4.0

FALLBACK_FPS = 30.0

# Candidate quarter-note lengths to test, in seconds (46-240 BPM).
QUARTER_CANDIDATES = np.arange(0.25, 1.30, 0.005)
GRID_CANDIDATES_COARSE_TO_FINE = [1.0, 0.5, 1.0 / 3.0, 0.25]
ONSET_MATCH_TOLERANCE = 0.07

# Above/below these, engraved sheet music normally uses an 8va/8vb
# bracket instead of a wall of ledger lines. C6 and C2 are conservative
# defaults (some engravers go a couple of semitones further before
# switching) — tune per taste.
OTTAVA_HIGH_MIDI = 84  # C6
OTTAVA_LOW_MIDI = 36  # C2


# ============================================================
# LOAD DATA
# ============================================================


def load_notes(path: Path, logger) -> Tuple[List[dict], float]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "notes" in raw:
        raw_notes = raw["notes"]
        fps = float(raw.get("meta", {}).get("fps", FALLBACK_FPS))
    else:
        raw_notes = raw
        fps = FALLBACK_FPS
        logger.warning(
            f"{path} has no embedded fps metadata; assuming {FALLBACK_FPS}. "
            "Re-run keyboard_detector.py to get accurate metadata."
        )

    logger.info(f"Loaded {len(raw_notes)} detected notes (fps={fps})")

    notes = []
    for n in raw_notes:
        try:
            midi = int(n["midi"])
            start = float(n["start_time"])
            end = float(n["end_time"])
            hand = str(n.get("hand", "right")).lower()

            if hand not in ("left", "right"):
                hand = "right"

            if end <= start:
                continue

            notes.append({"midi": midi, "start": start, "end": end, "duration": end - start, "hand": hand})
        except (KeyError, TypeError, ValueError):
            continue

    if not notes:
        raise ValueError("No usable notes found.")

    notes.sort(key=lambda x: (x["start"], x["midi"]))
    return notes, fps


# ============================================================
# TEMPO / GRID ESTIMATION FROM ONSET SPACING
# ============================================================


def _coverage(gaps: np.ndarray, unit: float, tol: float = ONSET_MATCH_TOLERANCE) -> float:
    ratios = gaps / unit
    nearest = np.maximum(np.round(ratios), 1)
    err = np.abs(ratios - nearest) / nearest
    return float(np.mean(err < tol))


def _mean_relative_error(gaps: np.ndarray, unit: float) -> float:
    ratios = gaps / unit
    nearest = np.maximum(np.round(ratios), 1)
    return float(np.mean(np.abs(ratios - nearest) / nearest))


def estimate_tempo_and_grid(notes: List[dict], logger) -> Tuple[float, float, dict]:
    """Estimate quarter-note length (seconds) and quantization grid
    (fraction of a quarter note) from the spacing between note onsets,
    not from how long notes were held."""

    onsets = sorted({round(n["start"], 4) for n in notes})
    gaps = np.array([b - a for a, b in zip(onsets, onsets[1:]) if b - a > 0.03])

    if len(gaps) < 3:
        logger.warning("Too few distinct onsets to estimate tempo reliably; defaulting to 120 BPM / eighth-note grid")
        return 0.5, 0.5, {"method": "fallback", "reason": "insufficient_onsets"}

    scored = [(_coverage(gaps, q / 4.0), q) for q in QUARTER_CANDIDATES]
    best_coverage = max(cov for cov, _ in scored)

    # Many nearby quarter-note lengths can all "cover" the same set of
    # gaps within tolerance (a wide plateau, not a sharp peak), so break
    # ties by which candidate actually fits the data most tightly rather
    # than always preferring the coarsest — that biased the estimate
    # toward a slower tempo than the one actually played.
    top = [q for cov, q in scored if cov >= best_coverage - 1e-9]
    quarter_seconds = min(top, key=lambda q: _mean_relative_error(gaps, q / 4.0))
    bpm = 60.0 / quarter_seconds

    logger.info(f"Tempo estimation: {len(onsets)} onsets, {len(gaps)} usable gaps, {len(top)} candidates tied at coverage {best_coverage:.3f}")
    ranked = sorted(top, key=lambda q: _mean_relative_error(gaps, q / 4.0))[:5]
    logger.debug("Top 5 quarter-note candidates by fit within the best-coverage tier:")
    for q in ranked:
        logger.debug(f"  {q:.4f}s ({60.0/q:6.2f} BPM) -> mean error {_mean_relative_error(gaps, q/4.0):.4f}")

    chosen_grid = GRID_CANDIDATES_COARSE_TO_FINE[-1]
    for g in GRID_CANDIDATES_COARSE_TO_FINE:
        cov = _coverage(gaps, quarter_seconds * g, tol=ONSET_MATCH_TOLERANCE)
        logger.debug(f"Grid candidate {g:.4f} quarters -> coverage {cov:.3f}")
        if cov >= 0.85:
            chosen_grid = g
            break

    logger.info(f"Estimated tempo: {quarter_seconds:.4f}s/quarter ({bpm:.1f} BPM), coverage={best_coverage:.3f}")
    logger.info(f"Quantization grid: {chosen_grid:.4f} of a quarter note")

    info = {
        "method": "onset_interval",
        "quarter_seconds": quarter_seconds,
        "bpm": bpm,
        "coverage": best_coverage,
        "grid": chosen_grid,
        "onset_count": len(onsets),
    }
    return quarter_seconds, chosen_grid, info


def quantize(value: float, step: float) -> float:
    return round(value / step) * step


# ============================================================
# QUANTIZE NOTES
# ============================================================


def quantize_notes(notes: List[dict], quarter_seconds: float, grid: float, logger) -> List[dict]:
    first_start = min(n["start"] for n in notes)
    logger.info(f"First detected note at t={first_start:.3f}s (used as t=0 for notation)")

    large_shifts = 0

    for n in notes:
        relative_start = n["start"] - first_start
        relative_duration = n["duration"]

        start_quarters = relative_start / quarter_seconds
        duration_quarters = relative_duration / quarter_seconds

        q_start = quantize(start_quarters, grid)
        q_duration = quantize(duration_quarters, grid)

        if q_duration < grid:
            q_duration = grid

        shift = abs(q_start - start_quarters)
        if shift > grid * 0.75:
            large_shifts += 1
            logger.debug(
                f"Large quantization shift: midi={n['midi']} hand={n['hand']} "
                f"raw_start={start_quarters:.3f}q -> {q_start:.3f}q (shift={shift:.3f}q)"
            )

        n["q_start"] = round(q_start, 6)
        n["q_duration"] = round(q_duration, 6)

    if large_shifts:
        logger.warning(
            f"{large_shifts} note(s) had a quantization shift over 0.75 grid units — "
            "likely detector timing noise, worth checking note_events.csv."
        )

    return notes


def dedupe_notes(notes: List[dict], logger) -> List[dict]:
    deduped = {}
    for n in notes:
        key = (n["hand"], n["midi"], n["q_start"], n["q_duration"])
        deduped[key] = n

    result = list(deduped.values())
    if len(result) != len(notes):
        logger.info(f"Removed {len(notes) - len(result)} duplicate note(s)")
    return result


def group_by_hand(notes: List[dict]) -> Dict[str, Dict[float, List[dict]]]:
    grouped: Dict[str, Dict[float, List[dict]]] = {"left": defaultdict(list), "right": defaultdict(list)}
    for n in notes:
        grouped[n["hand"]][n["q_start"]].append(n)
    return grouped


def clip_overlaps(hand_data: Dict[float, List[dict]], grid: float, hand_name: str, logger) -> None:
    """A note's detected duration should never run past the next onset in
    the same hand — that would mean two notes sound at once in a single
    monophonic-per-hand staff line, which comes out as broken notation.
    This is usually the detector reporting a released key a frame or two
    late (trailing bright pixels), not an intentional overlap."""

    starts_sorted = sorted(hand_data.keys())

    for i, s in enumerate(starts_sorted):
        if i + 1 >= len(starts_sorted):
            continue

        next_start = starts_sorted[i + 1]
        group = hand_data[s]

        for n in group:
            if s + n["q_duration"] > next_start + 1e-9:
                capped = max(grid, next_start - s)
                if capped < n["q_duration"] - 1e-9:
                    logger.debug(
                        f"{hand_name}: clipped overlap midi={n['midi']} at start={s:.3f}q "
                        f"({n['q_duration']:.3f}q -> {capped:.3f}q)"
                    )
                n["q_duration"] = capped


# ============================================================
# MEASURE / TIE CONSTRUCTION
# ============================================================


def split_across_measures(start: float, duration_q: float, measure_length: float) -> List[Tuple[int, float, float]]:
    """Split a (start, duration) span — in quarter-length units from t=0 —
    into per-measure (measure_index, local_start, local_duration) pieces,
    so a note that crosses a barline becomes tied fragments instead of
    being truncated with the remainder dropped."""

    pieces = []
    remaining = duration_q
    cur_start = start

    while remaining > 1e-9:
        m_idx = int(cur_start // measure_length)
        m_start = m_idx * measure_length
        m_end = m_start + measure_length
        avail = m_end - cur_start

        piece_dur = min(remaining, avail)
        local_start = cur_start - m_start

        pieces.append((m_idx, local_start, piece_dur))

        remaining -= piece_dur
        cur_start += piece_dur

    return pieces


def build_part(hand_data: Dict[float, List[dict]], clef_obj, num_measures: int, measure_length: float, bpm: float) -> stream.Part:
    part = stream.Part()

    measure_events: Dict[int, list] = defaultdict(list)

    for start in sorted(hand_data.keys()):
        group = hand_data[start]
        midi_values = sorted(set(n["midi"] for n in group))
        max_duration = max(n["q_duration"] for n in group)

        pieces = split_across_measures(start, max_duration, measure_length)

        for i, (m_idx, local_start, piece_dur) in enumerate(pieces):
            if len(pieces) == 1:
                tie_type = None
            elif i == 0:
                tie_type = "start"
            elif i == len(pieces) - 1:
                tie_type = "stop"
            else:
                tie_type = "continue"

            measure_events[m_idx].append((local_start, piece_dur, midi_values, tie_type))

    for measure_number in range(num_measures):
        m = stream.Measure(number=measure_number + 1)

        if measure_number == 0:
            m.insert(0, clef_obj)
            m.insert(0, meter.TimeSignature(TIME_SIGNATURE))
            m.insert(0, tempo.MetronomeMark(number=bpm))

        events = sorted(measure_events.get(measure_number, []), key=lambda e: e[0])
        current_local = 0.0

        for local_start, piece_dur, midi_values, tie_type in events:
            gap = local_start - current_local
            if gap > 0.001:
                r = note.Rest()
                r.duration = duration.Duration(gap)
                m.insert(current_local, r)

            if len(midi_values) == 1:
                obj = note.Note(midi=midi_values[0])
            else:
                obj = chord.Chord(midi_values)

            obj.duration = duration.Duration(piece_dur)
            if tie_type:
                obj.tie = tie.Tie(tie_type)

            m.insert(local_start, obj)
            current_local = max(current_local, local_start + piece_dur)

        remaining = measure_length - current_local
        if remaining > 0.001:
            r = note.Rest()
            r.duration = duration.Duration(remaining)
            m.insert(current_local, r)

        part.append(m)

    return part


# ============================================================
# OTTAVA (8va/8vb) BRACKETS
# ============================================================


def _ottava_zone(el, high_threshold: int, low_threshold: int) -> Optional[str]:
    if isinstance(el, chord.Chord):
        pitches = el.pitches
    elif isinstance(el, note.Note):
        pitches = [el.pitch]
    else:
        return None

    if not pitches:
        return None

    if max(p.midi for p in pitches) >= high_threshold:
        return "high"
    if min(p.midi for p in pitches) <= low_threshold:
        return "low"
    return None


def apply_ottava_brackets(
    part: stream.Part,
    logger,
    high_threshold: int = OTTAVA_HIGH_MIDI,
    low_threshold: int = OTTAVA_LOW_MIDI,
) -> int:
    """Wrap runs of consecutive notes above/below the given thresholds in
    8va/8vb spanners, transposing the notated pitch by an octave so the
    bracket and the noteheads agree. Applied to a copy of the score used
    only for the MusicXML export — the MIDI export must keep the true
    detected pitches, not the notated ones.
    """

    elements = sorted(part.recurse().notes, key=lambda n: n.getOffsetInHierarchy(part))

    run: List = []
    run_zone: Optional[str] = None
    bracket_count = 0

    def flush():
        nonlocal run, run_zone, bracket_count
        if run_zone and run:
            ott_type = "8va" if run_zone == "high" else "8vb"
            ott = spanner.Ottava(*run, type=ott_type)
            part.insert(0, ott)

            # music21's own performTransposition() goes from written pitch
            # to sounding pitch (up an octave for '8va') — the opposite of
            # what we need: our notes already hold the true sounding pitch
            # and we need the *written* (notated) pitch, which is a
            # semitone-perfect octave shift the other way.
            shift = -12 if ott_type == "8va" else 12
            for el in run:
                el.transpose(shift, inPlace=True)
            ott.transposing = False

            bracket_count += 1
            start_q = float(run[0].getOffsetInHierarchy(part))
            logger.debug(f"{part.id}: {ott_type} bracket over {len(run)} note(s) starting at {start_q:.2f}q")
        run = []
        run_zone = None

    for el in elements:
        zone = _ottava_zone(el, high_threshold, low_threshold)
        if zone is not None and zone == run_zone:
            run.append(el)
        else:
            flush()
            if zone is not None:
                run = [el]
                run_zone = zone

    flush()
    return bracket_count


# ============================================================
# MAIN
# ============================================================


def convert(input_json: Path, output_musicxml: Path, output_midi: Path, logger) -> stream.Score:
    notes, fps = load_notes(input_json, logger)

    quarter_seconds, grid, tempo_info = estimate_tempo_and_grid(notes, logger)
    bpm = tempo_info.get("bpm", 60.0 / quarter_seconds)

    notes = quantize_notes(notes, quarter_seconds, grid, logger)
    notes = dedupe_notes(notes, logger)

    grouped = group_by_hand(notes)

    clip_overlaps(grouped["left"], grid, "LH", logger)
    clip_overlaps(grouped["right"], grid, "RH", logger)

    max_end = 0.0
    for hand_data in grouped.values():
        for start, group in hand_data.items():
            for n in group:
                max_end = max(max_end, start + n["q_duration"])

    num_measures = max(1, int(np.ceil(max_end / MEASURE_LENGTH_QUARTERS))) if max_end > 0 else 1
    logger.info(f"Score spans {num_measures} measures at {MEASURE_LENGTH_QUARTERS} quarters/measure")

    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = "Converted Piano Performance"

    right = build_part(grouped["right"], clef.TrebleClef(), num_measures, MEASURE_LENGTH_QUARTERS, bpm)
    left = build_part(grouped["left"], clef.BassClef(), num_measures, MEASURE_LENGTH_QUARTERS, bpm)
    right.id = "RH"
    left.id = "LH"

    score.insert(0, right)
    score.insert(0, left)

    logger.info("Generating measures and notation...")
    for part in score.parts:
        try:
            part.makeNotation(inPlace=True)
        except Exception as e:
            logger.warning(f"Notation warning: {e}")

    # MIDI must play back the true detected pitches, so it's written from
    # this score before any ottava transposition touches it.
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    score.write("midi", fp=str(output_midi))
    logger.info(f"Saved MIDI: {output_midi}")

    xml_score = copy.deepcopy(score)
    total_brackets = 0
    for part in xml_score.parts:
        total_brackets += apply_ottava_brackets(part, logger)
    if total_brackets:
        logger.info(f"Added {total_brackets} 8va/8vb bracket(s) for notes beyond the staff")

    output_musicxml.parent.mkdir(parents=True, exist_ok=True)
    xml_score.write("musicxml", fp=str(output_musicxml))
    logger.info(f"Saved MusicXML: {output_musicxml}")

    return score


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_TRACKED_JSON))
    parser.add_argument("--musicxml", default=str(DEFAULT_MUSICXML))
    parser.add_argument("--midi", default=str(DEFAULT_MIDI))
    args = parser.parse_args(argv)

    logger = get_logger("json_to_sheet", LOG_DIR / "json_to_sheet.log")

    convert(Path(args.input), Path(args.musicxml), Path(args.midi), logger)

    logger.info("DONE.")


if __name__ == "__main__":
    main()
