"""MusicXML → canonical event sequences.

Parses MusicXML files into ScoreData objects using the canonical 48-divisions-per-quarter grid.
Handles Tier 1 (notes, rests, measures, time/key sig, clef) and Tier 2 best-effort
(ties, accidentals, tuplets, chord symbols).
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from src.types import (
    CANONICAL_DIVISIONS_PER_QUARTER,
    DURATION_FRACTIONS,
    NoteEvent,
    RestEvent,
    MeasureEvents,
    ScoreMeta,
    ScoreData,
)

XML_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "xml")
EVENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "events")
DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "parse_debug")

# Map alter values to accidental symbols in pitch string
ALTER_MAP = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}


def _title_to_id(title: str) -> str:
    """Convert a title to a stable filename-stem ID."""
    return title.lower().replace(" ", "_").replace("'", "").replace(".", "")


def _compute_canonical_duration(
    duration_xml: int,
    divisions: int,
    type_name: str,
    dots: int,
    tuplet_actual: int,
    tuplet_normal: int,
) -> int:
    """Compute duration in canonical divisions.

    Primary method: scale from MusicXML duration using divisions.
    Fallback: compute from type name + dots + tuplet ratio.
    """
    if divisions > 0 and duration_xml > 0:
        # MusicXML duration is in divisions; a quarter = divisions
        canonical = (duration_xml * CANONICAL_DIVISIONS_PER_QUARTER) / divisions
        result = int(round(canonical))
        if result > 0:
            return result

    # Fallback from type name
    base_frac = DURATION_FRACTIONS.get(type_name, 0)
    if base_frac == 0:
        return 0
    total = base_frac
    dot_val = base_frac
    for _ in range(dots):
        dot_val /= 2
        total += dot_val
    # Apply tuplet ratio
    if tuplet_actual > 0 and tuplet_normal > 0:
        total = total * tuplet_normal / tuplet_actual
    return int(round(total * CANONICAL_DIVISIONS_PER_QUARTER))


def parse_musicxml(xml_path: str) -> tuple[ScoreData, dict]:
    """Parse a MusicXML file into a ScoreData object.

    Returns (ScoreData, debug_info) where debug_info has parse statistics and warnings.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract title
    title = ""
    work_title = root.find(".//work-title")
    if work_title is not None and work_title.text:
        title = work_title.text.strip()

    # Parse first part only (these are single-staff lead sheets)
    part = root.find(".//part")
    if part is None:
        raise ValueError(f"No <part> found in {xml_path}")

    divisions = 1
    key_fifths = 0
    clef_sign = "G"
    time_beats = 4
    time_beat_type = 4
    measures_out = []

    # Debug counters
    stats = {
        "total_notes": 0,
        "total_rests": 0,
        "total_measures": 0,
        "ties": 0,
        "accidentals": 0,
        "tuplets": 0,
        "chord_symbols": 0,
        "dots": 0,
        "warnings": [],
        "unique_pitches": set(),
        "duration_names": {},
    }

    for measure_el in part.findall("measure"):
        measure_num = int(measure_el.get("number", "0"))
        stats["total_measures"] += 1

        # Check for attribute changes
        attrs = measure_el.find("attributes")
        if attrs is not None:
            div_el = attrs.find("divisions")
            if div_el is not None and div_el.text:
                divisions = int(div_el.text)

            key_el = attrs.find("key/fifths")
            if key_el is not None and key_el.text:
                key_fifths = int(key_el.text)

            time_el = attrs.find("time")
            if time_el is not None:
                beats_el = time_el.find("beats")
                bt_el = time_el.find("beat-type")
                if beats_el is not None and beats_el.text:
                    time_beats = int(beats_el.text)
                if bt_el is not None and bt_el.text:
                    time_beat_type = int(bt_el.text)

            clef_el = attrs.find("clef/sign")
            if clef_el is not None and clef_el.text:
                clef_sign = clef_el.text

        # Count chord symbols
        for _ in measure_el.findall("harmony"):
            stats["chord_symbols"] += 1

        # Parse notes and rests
        events = []
        canonical_offset_acc = 0  # accumulated canonical offset (sum of durations)

        for note_el in measure_el.findall("note"):
            # Check for chord (simultaneous note) — skip offset advance
            is_chord = note_el.find("chord") is not None

            # Duration
            dur_el = note_el.find("duration")
            duration_xml = int(dur_el.text) if dur_el is not None and dur_el.text else 0

            # Type name
            type_el = note_el.find("type")
            type_name = type_el.text if type_el is not None and type_el.text else ""

            # Dots
            dot_count = len(note_el.findall("dot"))
            if dot_count:
                stats["dots"] += dot_count

            # Tuplet info
            time_mod = note_el.find("time-modification")
            tuplet_actual = 0
            tuplet_normal = 0
            if time_mod is not None:
                act_el = time_mod.find("actual-notes")
                norm_el = time_mod.find("normal-notes")
                if act_el is not None and act_el.text:
                    tuplet_actual = int(act_el.text)
                if norm_el is not None and norm_el.text:
                    tuplet_normal = int(norm_el.text)
                stats["tuplets"] += 1

            # Compute canonical duration
            canonical_dur = _compute_canonical_duration(
                duration_xml, divisions, type_name, dot_count,
                tuplet_actual, tuplet_normal,
            )

            # Canonical offset: accumulate canonical durations for consistency
            # This ensures round-trip stability (no rounding drift from raw offsets)
            if is_chord:
                canonical_offset = events[-1].offset_divisions if events else 0
            else:
                canonical_offset = canonical_offset_acc

            is_rest = note_el.find("rest") is not None

            if is_rest:
                stats["total_rests"] += 1
                ev = RestEvent(
                    duration_name=type_name,
                    duration_divisions=canonical_dur,
                    offset_divisions=canonical_offset,
                    voice=1,
                    dots=dot_count,
                )
                events.append(ev)
            else:
                # Pitch
                pitch_el = note_el.find("pitch")
                if pitch_el is None:
                    # Forward/backup or other non-pitch element
                    if not is_chord:
                        current_offset_xml += duration_xml
                    continue

                step = pitch_el.findtext("step", "")
                octave = pitch_el.findtext("octave", "")
                alter_el = pitch_el.find("alter")
                alter = int(float(alter_el.text)) if alter_el is not None and alter_el.text else 0
                acc_symbol = ALTER_MAP.get(alter, "")
                pitch_str = f"{step}{acc_symbol}{octave}"

                stats["total_notes"] += 1
                stats["unique_pitches"].add(pitch_str)
                stats["duration_names"][type_name] = stats["duration_names"].get(type_name, 0) + 1

                # Ties
                tie_start = False
                tie_stop = False
                for tie_el in note_el.findall("tie"):
                    t = tie_el.get("type", "")
                    if t == "start":
                        tie_start = True
                        stats["ties"] += 1
                    elif t == "stop":
                        tie_stop = True

                # Accidental
                acc_el = note_el.find("accidental")
                accidental_str = acc_el.text if acc_el is not None and acc_el.text else ""
                if accidental_str:
                    stats["accidentals"] += 1

                ev = NoteEvent(
                    pitch=pitch_str,
                    duration_name=type_name,
                    duration_divisions=canonical_dur,
                    offset_divisions=canonical_offset,
                    voice=1,
                    dots=dot_count,
                    tie_start=tie_start,
                    tie_stop=tie_stop,
                    accidental=accidental_str,
                    tuplet_actual=tuplet_actual,
                    tuplet_normal=tuplet_normal,
                )
                events.append(ev)

            # Advance offset (only for non-chord notes)
            if not is_chord:
                canonical_offset_acc += canonical_dur

        # Handle forward/backup elements for offset tracking
        # (Already handled above by tracking current_offset_xml per note)

        measures_out.append(MeasureEvents(
            measure_number=measure_num,
            events=events,
            time_beats=time_beats,
            time_beat_type=time_beat_type,
        ))

    meta = ScoreMeta(
        title=title,
        key_fifths=key_fifths,
        clef=clef_sign,
        divisions=divisions,
        time_beats=time_beats,
        time_beat_type=time_beat_type,
    )

    score = ScoreData(meta=meta, measures=measures_out)

    # Convert set to list for JSON
    stats["unique_pitches"] = sorted(stats["unique_pitches"])

    return score, stats


def prepare_all(xml_dir: str = XML_DIR, events_dir: str = EVENTS_DIR, debug_dir: str = DEBUG_DIR):
    """Parse all MusicXML files and save canonical event sequences."""
    os.makedirs(events_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    xml_files = sorted(Path(xml_dir).glob("*.xml"))
    if not xml_files:
        print(f"No XML files found in {xml_dir}")
        return

    all_stats = {}
    total_notes = 0
    total_rests = 0
    total_measures = 0

    for xml_path in xml_files:
        stem = xml_path.stem
        file_id = _title_to_id(stem)

        try:
            score, stats = parse_musicxml(str(xml_path))
        except Exception as e:
            print(f"  ERROR parsing {stem}: {e}")
            continue

        # Save event JSON
        out_path = os.path.join(events_dir, f"{file_id}.json")
        with open(out_path, "w") as f:
            f.write(score.to_json())

        # Save token sequence
        tokens = score.to_tokens()
        token_path = os.path.join(events_dir, f"{file_id}.tokens")
        with open(token_path, "w") as f:
            f.write(" ".join(tokens))

        all_stats[file_id] = stats
        total_notes += stats["total_notes"]
        total_rests += stats["total_rests"]
        total_measures += stats["total_measures"]

        print(f"  {stem}: {stats['total_measures']} measures, "
              f"{stats['total_notes']} notes, {stats['total_rests']} rests, "
              f"{stats['tuplets']} tuplets, {stats['ties']} ties")

    # Save per-piece debug
    for file_id, stats in all_stats.items():
        debug_path = os.path.join(debug_dir, f"{file_id}.json")
        with open(debug_path, "w") as f:
            json.dump(stats, f, indent=2)

    # Summary
    summary = {
        "total_files": len(all_stats),
        "total_notes": total_notes,
        "total_rests": total_rests,
        "total_measures": total_measures,
        "tier2_coverage": {
            "ties": sum(s["ties"] for s in all_stats.values()),
            "accidentals": sum(s["accidentals"] for s in all_stats.values()),
            "tuplets": sum(s["tuplets"] for s in all_stats.values()),
            "chord_symbols": sum(s["chord_symbols"] for s in all_stats.values()),
            "dots": sum(s["dots"] for s in all_stats.values()),
        },
    }
    summary_path = os.path.join(debug_dir, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal: {len(all_stats)} files, {total_measures} measures, "
          f"{total_notes} notes, {total_rests} rests")
    print(f"Tier 2: {summary['tier2_coverage']}")
    print(f"Events saved to {events_dir}")
    print(f"Debug saved to {debug_dir}")


if __name__ == "__main__":
    prepare_all()
