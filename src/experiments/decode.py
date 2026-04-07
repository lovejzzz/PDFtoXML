"""Decoding and post-processing for event-sequence predictions.

Converts model output (token index lists) back to ScoreData objects
for evaluation with src/eval.py.
"""

import json
import os
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

# Reverse mapping: token string → event
DURATION_NAMES = ["whole", "half", "quarter", "eighth", "16th", "32nd", "64th"]


def _parse_note_token(token: str) -> NoteEvent | None:
    """Parse a NOTE_* token into a NoteEvent."""
    # Format: NOTE_{pitch}_{duration}[_DOT]*
    parts = token.split("_")
    if len(parts) < 3 or parts[0] != "NOTE":
        return None

    pitch = parts[1]

    # Find duration name and dots
    dur_parts = parts[2:]
    dots = dur_parts.count("DOT")
    dur_name_parts = [p for p in dur_parts if p != "DOT"]
    dur_name = "_".join(dur_name_parts).lower() if dur_name_parts else ""

    # Map back to standard names
    dur_name_map = {
        "whole": "whole", "half": "half", "quarter": "quarter",
        "eighth": "eighth", "16th": "16th", "32nd": "32nd", "64th": "64th",
    }
    dur_name = dur_name_map.get(dur_name, dur_name)

    dur_divs = duration_name_to_divisions(dur_name, dots)

    return NoteEvent(
        pitch=pitch,
        duration_name=dur_name,
        duration_divisions=dur_divs,
        dots=dots,
    )


def _parse_rest_token(token: str) -> RestEvent | None:
    """Parse a REST_* token into a RestEvent."""
    parts = token.split("_")
    if len(parts) < 2 or parts[0] != "REST":
        return None

    dur_parts = parts[1:]
    dots = dur_parts.count("DOT")
    dur_name_parts = [p for p in dur_parts if p != "DOT"]
    dur_name = "_".join(dur_name_parts).lower() if dur_name_parts else ""

    dur_name_map = {
        "whole": "whole", "half": "half", "quarter": "quarter",
        "eighth": "eighth", "16th": "16th", "32nd": "32nd", "64th": "64th",
    }
    dur_name = dur_name_map.get(dur_name, dur_name)

    dur_divs = duration_name_to_divisions(dur_name, dots)

    return RestEvent(
        duration_name=dur_name,
        duration_divisions=dur_divs,
        dots=dots,
    )


def tokens_to_score(tokens: list[str], title: str = "") -> ScoreData:
    """Convert a flat token sequence back to ScoreData.

    Reconstructs measure structure, offsets, and metadata from tokens.
    """
    meta = ScoreMeta(title=title)
    measures = []
    current_events = []
    current_offset = 0
    measure_num = 0

    # Parse header tokens
    for tok in tokens:
        if tok.startswith("CLEF_"):
            meta.clef = tok.split("_", 1)[1]
        elif tok.startswith("KEY_"):
            try:
                meta.key_fifths = int(tok.split("_", 1)[1])
            except ValueError:
                pass
        elif tok.startswith("TIME_"):
            parts = tok.split("_")
            if len(parts) == 3:
                try:
                    meta.time_beats = int(parts[1])
                    meta.time_beat_type = int(parts[2])
                except ValueError:
                    pass
        elif tok == "MEASURE_START":
            if current_events:
                measures.append(MeasureEvents(
                    measure_number=measure_num,
                    events=current_events,
                    time_beats=meta.time_beats,
                    time_beat_type=meta.time_beat_type,
                ))
            measure_num += 1
            current_events = []
            current_offset = 0
        elif tok == "BARLINE":
            continue
        elif tok.startswith("NOTE_"):
            ev = _parse_note_token(tok)
            if ev:
                ev.offset_divisions = current_offset
                current_offset += ev.duration_divisions
                current_events.append(ev)
        elif tok.startswith("REST_"):
            ev = _parse_rest_token(tok)
            if ev:
                ev.offset_divisions = current_offset
                current_offset += ev.duration_divisions
                current_events.append(ev)

    # Flush last measure
    if current_events:
        measures.append(MeasureEvents(
            measure_number=measure_num,
            events=current_events,
            time_beats=meta.time_beats,
            time_beat_type=meta.time_beat_type,
        ))

    return ScoreData(meta=meta, measures=measures)


def save_predictions(
    predictions: dict[str, list[str]],
    output_dir: str,
):
    """Save predicted token sequences as ScoreData JSON files.

    predictions: {file_id: [token_strings]}
    """
    os.makedirs(output_dir, exist_ok=True)
    for file_id, tokens in predictions.items():
        score = tokens_to_score(tokens, title=file_id)
        out_path = os.path.join(output_dir, f"{file_id}.json")
        with open(out_path, "w") as f:
            f.write(score.to_json())
