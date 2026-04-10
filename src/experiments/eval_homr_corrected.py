"""Evaluate homr's predictions after octave-shift correction.

homr produces pitches that are octave-shifted from the Omnibook gold (probably
because it's reading a different clef or transposition). Try shifting up by
1, 2, or 3 octaves to see what works best.
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
EVENTS_DIR = os.path.join(PROJECT_ROOT, "data", "events")
PAGES_DIR = os.path.join(PROJECT_ROOT, "data", "pages")
HOMR_DIR = os.path.join(PROJECT_ROOT, "outputs", "homr_predictions")
PRED_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions_homr_shifted")
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "data_manifest")


def shift_octave(pitch_str: str, shift: int) -> str:
    """Shift a pitch like 'C4' by N octaves."""
    if not pitch_str or not pitch_str[-1].isdigit():
        return pitch_str
    # Find where octave starts
    for i, c in enumerate(pitch_str):
        if c.isdigit():
            base = pitch_str[:i]
            octave = int(pitch_str[i:])
            new_oct = max(0, min(8, octave + shift))
            return f"{base}{new_oct}"
    return pitch_str


def shift_score_octaves(score_dict: dict, shift: int) -> dict:
    """Shift all pitches in a score by N octaves."""
    new_score = json.loads(json.dumps(score_dict))  # deep copy
    for m in new_score["measures"]:
        for e in m["events"]:
            if e["type"] == "note":
                e["pitch"] = shift_octave(e["pitch"], shift)
    return new_score


def main():
    from src.prepare_data import parse_musicxml
    from src.types import ScoreData
    from src.eval import evaluate_pair

    with open(os.path.join(MANIFEST_DIR, "manual_page_map.json")) as f:
        page_map = json.load(f)

    # Try different shifts
    for shift in [0, 1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"OCTAVE SHIFT = {shift}")
        print(f"{'='*60}")

        os.makedirs(PRED_DIR, exist_ok=True)

        results = []
        for file_id, entry in sorted(page_map.items()):
            # Get homr output for first page
            page_idx = entry["page_indices"][0]
            xml_path = os.path.join(PAGES_DIR, f"page_{page_idx + 1:03d}.musicxml")

            if not os.path.exists(xml_path):
                continue

            try:
                pred_score, _ = parse_musicxml(xml_path)
            except Exception:
                continue

            # Shift octaves
            score_dict = pred_score.to_dict()
            shifted = shift_score_octaves(score_dict, shift)
            shifted_score = ScoreData.from_dict(shifted)

            gold_path = os.path.join(EVENTS_DIR, f"{file_id}.json")
            if not os.path.exists(gold_path):
                continue
            with open(gold_path) as f:
                gold = ScoreData.from_dict(json.load(f))

            metrics = evaluate_pair(gold, shifted_score)
            metrics["file_id"] = file_id
            results.append(metrics)

        if results:
            n = len(results)
            agg_score = sum(m["score"] for m in results) / n
            agg_pitch = sum(m["pitch_acc_global"] for m in results) / n
            agg_rhythm = sum(m["rhythm_acc_global"] for m in results) / n
            agg_meas = sum(m["measure_validity"] for m in results) / n
            print(f"  N={n}")
            print(f"  Score:    {agg_score:.4f}")
            print(f"  Pitch:    {agg_pitch:.4f}")
            print(f"  Rhythm:   {agg_rhythm:.4f}")
            print(f"  Measure:  {agg_meas:.4f}")


if __name__ == "__main__":
    main()
