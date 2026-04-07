"""Canonical events → MusicXML 3.0 writer.

Deterministic conversion from ScoreData to valid MusicXML.
Handles Tier 1 features; Tier 2 best-effort.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

from src.types import (
    CANONICAL_DIVISIONS_PER_QUARTER,
    NoteEvent,
    RestEvent,
    ScoreData,
)

EVENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "events")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "reconstructed_xml")

# We use canonical divisions directly as MusicXML divisions
MUSICXML_DIVISIONS = CANONICAL_DIVISIONS_PER_QUARTER

# Key fifths → major key name (for display)
KEY_NAMES = {
    -7: "Cb", -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F",
    0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#",
}

# Pitch string parsing
ACCIDENTAL_MAP = {"#": 1, "b": -1, "##": 2, "bb": -2}

DURATION_DIV_TO_TYPE = {}


def _build_duration_map():
    """Build reverse map from canonical divisions to MusicXML type name."""
    from src.types import DURATION_FRACTIONS
    for name, frac in DURATION_FRACTIONS.items():
        divs = int(round(frac * CANONICAL_DIVISIONS_PER_QUARTER))
        DURATION_DIV_TO_TYPE[divs] = name
        # Dotted
        dotted = int(round(frac * 1.5 * CANONICAL_DIVISIONS_PER_QUARTER))
        DURATION_DIV_TO_TYPE[dotted] = (name, 1)
        # Double dotted
        dd = int(round(frac * 1.75 * CANONICAL_DIVISIONS_PER_QUARTER))
        DURATION_DIV_TO_TYPE[dd] = (name, 2)


_build_duration_map()


def _parse_pitch_string(pitch_str: str) -> tuple[str, int, int]:
    """Parse 'Eb4' → ('E', 4, -1)  i.e. (step, octave, alter)."""
    step = pitch_str[0]
    rest = pitch_str[1:]

    alter = 0
    if rest.startswith("bb"):
        alter = -2
        rest = rest[2:]
    elif rest.startswith("##"):
        alter = 2
        rest = rest[2:]
    elif rest.startswith("b"):
        alter = -1
        rest = rest[1:]
    elif rest.startswith("#"):
        alter = 1
        rest = rest[1:]

    octave = int(rest) if rest else 4
    return step, octave, alter


def _get_type_and_dots(duration_divisions: int, duration_name: str, dots: int) -> tuple[str, int]:
    """Get MusicXML type name and dot count from event data."""
    # Prefer the stored name if available
    if duration_name:
        return duration_name, dots

    # Try reverse lookup
    entry = DURATION_DIV_TO_TYPE.get(duration_divisions)
    if entry is None:
        return "quarter", 0
    if isinstance(entry, tuple):
        return entry[0], entry[1]
    return entry, 0


def score_to_musicxml(score: ScoreData) -> str:
    """Convert ScoreData to MusicXML string."""
    root = ET.Element("score-partwise", version="3.0")

    # Work title
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = score.meta.title

    # Identification
    ident = ET.SubElement(root, "identification")
    encoding = ET.SubElement(ident, "encoding")
    ET.SubElement(encoding, "software").text = "music-pdf2xml"

    # Part list
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Music"

    # Part
    part = ET.SubElement(root, "part", id="P1")

    first_measure = True
    prev_time_beats = None
    prev_time_beat_type = None

    for measure_data in score.measures:
        measure = ET.SubElement(part, "measure", number=str(measure_data.measure_number))

        # Attributes (first measure always; later only on changes)
        need_attrs = first_measure
        if not first_measure:
            if (measure_data.time_beats != prev_time_beats or
                    measure_data.time_beat_type != prev_time_beat_type):
                need_attrs = True

        if need_attrs:
            attrs = ET.SubElement(measure, "attributes")
            if first_measure:
                ET.SubElement(attrs, "divisions").text = str(MUSICXML_DIVISIONS)
                key = ET.SubElement(attrs, "key")
                ET.SubElement(key, "fifths").text = str(score.meta.key_fifths)
                clef = ET.SubElement(attrs, "clef")
                ET.SubElement(clef, "sign").text = score.meta.clef
                ET.SubElement(clef, "line").text = "2"

            time_el = ET.SubElement(attrs, "time")
            ET.SubElement(time_el, "beats").text = str(measure_data.time_beats)
            ET.SubElement(time_el, "beat-type").text = str(measure_data.time_beat_type)

        prev_time_beats = measure_data.time_beats
        prev_time_beat_type = measure_data.time_beat_type
        first_measure = False

        # Notes
        for ev in measure_data.events:
            note = ET.SubElement(measure, "note")

            if isinstance(ev, RestEvent):
                ET.SubElement(note, "rest")
                ET.SubElement(note, "duration").text = str(ev.duration_divisions)
                ET.SubElement(note, "voice").text = str(ev.voice)
                type_name, dot_count = _get_type_and_dots(
                    ev.duration_divisions, ev.duration_name, ev.dots)
                if type_name:
                    ET.SubElement(note, "type").text = type_name
                for _ in range(dot_count):
                    ET.SubElement(note, "dot")

            elif isinstance(ev, NoteEvent):
                # Pitch
                pitch = ET.SubElement(note, "pitch")
                step, octave, alter = _parse_pitch_string(ev.pitch)
                ET.SubElement(pitch, "step").text = step
                if alter != 0:
                    ET.SubElement(pitch, "alter").text = str(alter)
                ET.SubElement(pitch, "octave").text = str(octave)

                ET.SubElement(note, "duration").text = str(ev.duration_divisions)

                # Ties
                if ev.tie_start:
                    ET.SubElement(note, "tie", type="start")
                if ev.tie_stop:
                    ET.SubElement(note, "tie", type="stop")

                ET.SubElement(note, "voice").text = str(ev.voice)

                type_name, dot_count = _get_type_and_dots(
                    ev.duration_divisions, ev.duration_name, ev.dots)
                if type_name:
                    ET.SubElement(note, "type").text = type_name
                for _ in range(dot_count):
                    ET.SubElement(note, "dot")

                # Accidental
                if ev.accidental:
                    ET.SubElement(note, "accidental").text = ev.accidental

                # Tuplet time modification
                if ev.tuplet_actual > 0 and ev.tuplet_normal > 0:
                    time_mod = ET.SubElement(note, "time-modification")
                    ET.SubElement(time_mod, "actual-notes").text = str(ev.tuplet_actual)
                    ET.SubElement(time_mod, "normal-notes").text = str(ev.tuplet_normal)

                # Tie notations
                if ev.tie_start or ev.tie_stop:
                    notations = ET.SubElement(note, "notations")
                    if ev.tie_stop:
                        ET.SubElement(notations, "tied", type="stop")
                    if ev.tie_start:
                        ET.SubElement(notations, "tied", type="start")

    # Pretty print
    rough = ET.tostring(root, encoding="unicode", xml_declaration=False)
    header = '<?xml version="1.0" encoding="UTF-8"?>\n'
    header += '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.0 Partwise//EN" '
    header += '"http://www.musicxml.org/dtds/partwise.dtd">\n'

    try:
        dom = minidom.parseString(rough)
        pretty = dom.toprettyxml(indent="  ", encoding=None)
        # Remove the xml declaration that minidom adds (we have our own)
        lines = pretty.split("\n")
        if lines[0].startswith("<?xml"):
            lines = lines[1:]
        return header + "\n".join(lines)
    except Exception:
        return header + rough


def write_score(score: ScoreData, output_path: str):
    """Write ScoreData to a MusicXML file."""
    xml_str = score_to_musicxml(score)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)


def roundtrip_test(events_dir: str = EVENTS_DIR, output_dir: str = OUTPUT_DIR):
    """Round-trip test: events → XML → re-parse → compare Tier 1 fields."""
    from src.prepare_data import parse_musicxml

    os.makedirs(output_dir, exist_ok=True)
    event_files = sorted(Path(events_dir).glob("*.json"))

    if not event_files:
        print(f"No event files in {events_dir}")
        return

    passed = 0
    failed = 0
    tier2_mismatches = 0

    for ef in event_files:
        file_id = ef.stem

        # Load original events
        with open(ef) as f:
            original = ScoreData.from_dict(json.load(f))

        # Write to XML
        xml_path = os.path.join(output_dir, f"{file_id}.xml")
        write_score(original, xml_path)

        # Re-parse
        try:
            reparsed, _ = parse_musicxml(xml_path)
        except Exception as e:
            print(f"  FAIL {file_id}: re-parse error: {e}")
            failed += 1
            continue

        # Compare Tier 1 fields per measure
        ok = True
        for orig_m, rep_m in zip(original.measures, reparsed.measures):
            if orig_m.measure_number != rep_m.measure_number:
                print(f"  FAIL {file_id}: measure number mismatch "
                      f"{orig_m.measure_number} vs {rep_m.measure_number}")
                ok = False
                break

            if len(orig_m.events) != len(rep_m.events):
                print(f"  FAIL {file_id}: m{orig_m.measure_number} event count "
                      f"{len(orig_m.events)} vs {len(rep_m.events)}")
                ok = False
                break

            for i, (oe, re_) in enumerate(zip(orig_m.events, rep_m.events)):
                if oe.type != re_.type:
                    print(f"  FAIL {file_id}: m{orig_m.measure_number} e{i} "
                          f"type {oe.type} vs {re_.type}")
                    ok = False
                    break

                if oe.duration_divisions != re_.duration_divisions:
                    print(f"  FAIL {file_id}: m{orig_m.measure_number} e{i} "
                          f"dur {oe.duration_divisions} vs {re_.duration_divisions}")
                    ok = False
                    break

                if oe.offset_divisions != re_.offset_divisions:
                    print(f"  FAIL {file_id}: m{orig_m.measure_number} e{i} "
                          f"offset {oe.offset_divisions} vs {re_.offset_divisions}")
                    ok = False
                    break

                if isinstance(oe, NoteEvent) and isinstance(re_, NoteEvent):
                    if oe.pitch != re_.pitch:
                        print(f"  FAIL {file_id}: m{orig_m.measure_number} e{i} "
                              f"pitch {oe.pitch} vs {re_.pitch}")
                        ok = False
                        break
            if not ok:
                break

        if ok:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    print(f"\nRound-trip test: {passed}/{total} passed, {failed} failed")


if __name__ == "__main__":
    if "--roundtrip" in sys.argv:
        roundtrip_test()
    else:
        print("Usage: python -m src.xml_writer --roundtrip")
        print("  Runs round-trip test on all event files")
