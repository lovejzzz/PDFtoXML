"""Musical post-processing to correct common prediction errors.

Rules:
1. Key signature filtering — flag notes outside the key
2. Measure duration enforcement — adjust to match time signature
3. Remove duplicate consecutive identical notes (stuttering)
4. Merge very short rests that shouldn't be there
5. Fix common octave errors (notes too far from neighbors)
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

from src.types import (
    CANONICAL_DIVISIONS_PER_QUARTER,
    NoteEvent,
    RestEvent,
    MeasureEvents,
    ScoreMeta,
    ScoreData,
    duration_name_to_divisions,
)

# Key signature → set of natural pitch classes (no accidentals)
KEY_PITCHES = {
    0: {'C', 'D', 'E', 'F', 'G', 'A', 'B'},           # C major
    1: {'C', 'D', 'E', 'F#', 'G', 'A', 'B'},           # G major
    -1: {'C', 'D', 'E', 'F', 'G', 'A', 'Bb'},          # F major
    2: {'C#', 'D', 'E', 'F#', 'G', 'A', 'B'},          # D major
    -2: {'C', 'D', 'Eb', 'F', 'G', 'A', 'Bb'},         # Bb major
    3: {'C#', 'D', 'E', 'F#', 'G#', 'A', 'B'},         # A major
    -3: {'C', 'D', 'Eb', 'F', 'G', 'Ab', 'Bb'},        # Eb major
    4: {'C#', 'D#', 'E', 'F#', 'G#', 'A', 'B'},        # E major
    -4: {'C', 'Db', 'Eb', 'F', 'G', 'Ab', 'Bb'},       # Ab major
    -5: {'C', 'Db', 'Eb', 'F', 'Gb', 'Ab', 'Bb'},      # Db major
    5: {'C#', 'D#', 'E', 'F#', 'G#', 'A#', 'B'},       # B major
}

# Common pitch classes for jazz (chromatic, but some more common)
CHROMATIC_PITCHES = {'C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F',
                     'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B'}


def _extract_pitch_class(pitch: str) -> str:
    """Extract pitch class from pitch string like 'Eb4' → 'Eb'."""
    for i, c in enumerate(pitch):
        if c.isdigit():
            return pitch[:i]
    return pitch


def _extract_octave(pitch: str) -> int:
    """Extract octave from pitch string like 'Eb4' → 4."""
    for i, c in enumerate(pitch):
        if c.isdigit():
            return int(pitch[i:])
    return 4


def _set_octave(pitch: str, octave: int) -> str:
    """Set octave of a pitch string."""
    pc = _extract_pitch_class(pitch)
    return f"{pc}{octave}"


def remove_stuttering(score: ScoreData) -> ScoreData:
    """Remove consecutive duplicate notes (same pitch + duration)."""
    for measure in score.measures:
        cleaned = []
        prev = None
        for ev in measure.events:
            if isinstance(ev, NoteEvent) and prev is not None:
                if (isinstance(prev, NoteEvent) and
                    ev.pitch == prev.pitch and
                    ev.duration_name == prev.duration_name):
                    continue  # skip duplicate
            cleaned.append(ev)
            prev = ev
        measure.events = cleaned
    return score


def fix_octave_jumps(score: ScoreData, max_interval: int = 12) -> ScoreData:
    """Fix notes that jump too far from their neighbors (likely octave errors)."""
    NOTE_TO_MIDI = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

    def pitch_to_midi(pitch: str) -> int:
        pc = _extract_pitch_class(pitch)
        octave = _extract_octave(pitch)
        base = pc[0]
        midi = NOTE_TO_MIDI.get(base, 0) + octave * 12
        if '#' in pc: midi += 1
        elif 'b' in pc: midi -= 1
        return midi

    for measure in score.measures:
        notes = [(i, ev) for i, ev in enumerate(measure.events)
                 if isinstance(ev, NoteEvent)]

        for idx in range(1, len(notes) - 1):
            prev_midi = pitch_to_midi(notes[idx-1][1].pitch)
            curr_midi = pitch_to_midi(notes[idx][1].pitch)
            next_midi = pitch_to_midi(notes[idx+1][1].pitch)

            # If current note is far from both neighbors but neighbors are close
            if (abs(curr_midi - prev_midi) > max_interval and
                abs(curr_midi - next_midi) > max_interval and
                abs(prev_midi - next_midi) <= max_interval):

                # Move to closest octave near the average of neighbors
                target = (prev_midi + next_midi) // 2
                pc = _extract_pitch_class(notes[idx][1].pitch)
                base_midi = NOTE_TO_MIDI.get(pc[0], 0)
                if '#' in pc: base_midi += 1
                elif 'b' in pc: base_midi -= 1

                # Find the octave that puts this note closest to target
                best_oct = 4
                best_dist = 999
                for oct in range(2, 7):
                    midi = base_midi + oct * 12
                    dist = abs(midi - target)
                    if dist < best_dist:
                        best_dist = dist
                        best_oct = oct

                new_pitch = _set_octave(notes[idx][1].pitch, best_oct)
                measure.events[notes[idx][0]].pitch = new_pitch

    return score


def enforce_measure_duration(score: ScoreData) -> ScoreData:
    """Trim or pad measures to match the time signature."""
    for measure in score.measures:
        expected = (measure.time_beats * CANONICAL_DIVISIONS_PER_QUARTER *
                    4 // measure.time_beat_type)
        actual = sum(ev.duration_divisions for ev in measure.events)

        if actual > expected and measure.events:
            # Trim from the end
            trimmed = []
            running = 0
            for ev in measure.events:
                if running + ev.duration_divisions <= expected:
                    trimmed.append(ev)
                    running += ev.duration_divisions
                else:
                    # Shorten this event to fit
                    remaining = expected - running
                    if remaining > 0:
                        ev.duration_divisions = remaining
                        trimmed.append(ev)
                    break
            measure.events = trimmed

    return score


def remove_tiny_rests(score: ScoreData, min_dur: int = 6) -> ScoreData:
    """Remove rests shorter than min_dur divisions (likely artifacts)."""
    for measure in score.measures:
        measure.events = [
            ev for ev in measure.events
            if not (isinstance(ev, RestEvent) and ev.duration_divisions < min_dur)
        ]
    return score


def postprocess(score: ScoreData) -> ScoreData:
    """Apply all post-processing rules."""
    score = remove_stuttering(score)
    score = fix_octave_jumps(score)
    score = enforce_measure_duration(score)
    score = remove_tiny_rests(score)

    # Recompute offsets
    for measure in score.measures:
        offset = 0
        for ev in measure.events:
            ev.offset_divisions = offset
            offset += ev.duration_divisions

    return score


def postprocess_predictions(pred_dir: str, output_dir: str):
    """Apply post-processing to all prediction files."""
    os.makedirs(output_dir, exist_ok=True)

    for f in sorted(Path(pred_dir).glob("*.json")):
        with open(f) as fh:
            score = ScoreData.from_dict(json.load(fh))

        score = postprocess(score)

        out_path = os.path.join(output_dir, f.name)
        with open(out_path, "w") as fh:
            fh.write(score.to_json())

    print(f"Post-processed {len(list(Path(pred_dir).glob('*.json')))} files → {output_dir}")


if __name__ == "__main__":
    from src.eval import evaluate_all

    pred_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/predictions"
    output_dir = pred_dir + "_postprocessed"

    postprocess_predictions(pred_dir, output_dir)

    evaluate_all(
        pred_dir=output_dir,
        gold_dir=os.path.join(os.path.dirname(__file__), "..", "..", "data", "events"),
        commit="postproc",
        description=f"Post-processed predictions from {pred_dir}",
    )
