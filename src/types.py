"""Canonical event schema for music-pdf2xml.

All durations and offsets are integers in canonical divisions.
CANONICAL_DIVISIONS_PER_QUARTER = 48 handles triplets, 16ths, dotted values
without floating-point issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Union
import json

CANONICAL_DIVISIONS_PER_QUARTER = 48

# Duration name -> fraction of a quarter note
DURATION_FRACTIONS: dict[str, float] = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "16th": 0.25,
    "32nd": 0.125,
    "64th": 0.0625,
}


def duration_name_to_divisions(name: str, dots: int = 0) -> int:
    """Convert a duration name (+ dots) to canonical divisions."""
    base = DURATION_FRACTIONS.get(name)
    if base is None:
        return 0
    total = base
    dot_val = base
    for _ in range(dots):
        dot_val /= 2
        total += dot_val
    return int(round(total * CANONICAL_DIVISIONS_PER_QUARTER))


@dataclass
class NoteEvent:
    type: str = "note"
    pitch: str = ""           # e.g. "C5", "Eb4", "F#3"
    duration_name: str = ""   # e.g. "eighth", "quarter"
    duration_divisions: int = 0
    offset_divisions: int = 0
    voice: int = 1
    dots: int = 0
    tie_start: bool = False
    tie_stop: bool = False
    accidental: str = ""      # "sharp", "flat", "natural", ""
    tuplet_actual: int = 0    # e.g. 3 for triplet
    tuplet_normal: int = 0    # e.g. 2 for triplet

    def to_token(self) -> str:
        acc = ""
        if "b" in self.pitch[1:] or "#" in self.pitch[1:]:
            acc = ""  # already in pitch string
        dur = self.duration_name.upper()
        if self.dots:
            dur += "_DOT" * self.dots
        return f"NOTE_{self.pitch}_{dur}"


@dataclass
class RestEvent:
    type: str = "rest"
    duration_name: str = ""
    duration_divisions: int = 0
    offset_divisions: int = 0
    voice: int = 1
    dots: int = 0

    def to_token(self) -> str:
        dur = self.duration_name.upper()
        if self.dots:
            dur += "_DOT" * self.dots
        return f"REST_{dur}"


@dataclass
class MeasureEvents:
    measure_number: int = 0
    events: list[Union[NoteEvent, RestEvent]] = field(default_factory=list)
    time_beats: int = 4
    time_beat_type: int = 4

    def to_tokens(self) -> list[str]:
        tokens = ["MEASURE_START"]
        for ev in self.events:
            tokens.append(ev.to_token())
        tokens.append("BARLINE")
        return tokens


@dataclass
class ScoreMeta:
    title: str = ""
    key_fifths: int = 0
    clef: str = "G"
    divisions: int = 1        # original MusicXML divisions
    canonical_divisions: int = CANONICAL_DIVISIONS_PER_QUARTER
    time_beats: int = 4
    time_beat_type: int = 4


@dataclass
class ScoreData:
    meta: ScoreMeta = field(default_factory=ScoreMeta)
    measures: list[MeasureEvents] = field(default_factory=list)

    def to_tokens(self) -> list[str]:
        tokens = [
            f"CLEF_{self.meta.clef}",
            f"KEY_{self.meta.key_fifths}",
            f"TIME_{self.meta.time_beats}_{self.meta.time_beat_type}",
        ]
        for m in self.measures:
            tokens.extend(m.to_tokens())
        return tokens

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "ScoreData":
        meta = ScoreMeta(**d["meta"])
        measures = []
        for md in d["measures"]:
            events = []
            for ed in md["events"]:
                if ed["type"] == "note":
                    events.append(NoteEvent(**{k: v for k, v in ed.items()}))
                else:
                    events.append(RestEvent(**{k: v for k, v in ed.items()}))
            measures.append(MeasureEvents(
                measure_number=md["measure_number"],
                events=events,
                time_beats=md["time_beats"],
                time_beat_type=md["time_beat_type"],
            ))
        return cls(meta=meta, measures=measures)

    @classmethod
    def from_json(cls, s: str) -> "ScoreData":
        return cls.from_dict(json.loads(s))
