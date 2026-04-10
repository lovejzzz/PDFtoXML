"""Fix homr's clef detection: shift bass-clef pitches to treble range.

homr reads staff 1 correctly (treble G2) but staves 2+ as bass clef (F4),
putting pitches 2 octaves too low. This script:
1. Parses homr MusicXML
2. Shifts all notes below C4 up by 2 octaves
3. Evaluates against gold
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
EVENTS_DIR = os.path.join(PROJECT_ROOT, "data", "events")
PAGES_DIR = os.path.join(PROJECT_ROOT, "data", "pages")
PRED_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions_homr_fixed")
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "data_manifest")

NOTE_ORDER = ['C', 'D', 'E', 'F', 'G', 'A', 'B']


def fix_pitch(pitch_str: str) -> str:
    """Shift pitches in bass range up to treble range."""
    if not pitch_str or not pitch_str[-1].isdigit():
        return pitch_str

    for i, c in enumerate(pitch_str):
        if c.isdigit() or (c == '-' and i > 0):
            base = pitch_str[:i]
            octave = int(pitch_str[i:])
            # If in bass range (octave 1-3), shift up 2
            if octave <= 3:
                octave += 2
            return f"{base}{octave}"
    return pitch_str


def fix_score(score_dict: dict) -> dict:
    """Fix all pitches in a score."""
    new_score = json.loads(json.dumps(score_dict))
    for m in new_score["measures"]:
        for e in m["events"]:
            if e["type"] == "note":
                e["pitch"] = fix_pitch(e["pitch"])
    return new_score


def main():
    from src.prepare_data import parse_musicxml
    from src.types import ScoreData
    from src.eval import evaluate_pair, evaluate_all
    from src.experiments.decode import save_predictions

    os.makedirs(PRED_DIR, exist_ok=True)

    with open(os.path.join(MANIFEST_DIR, "manual_page_map.json")) as f:
        page_map = json.load(f)
    with open(os.path.join(MANIFEST_DIR, "splits.json")) as f:
        splits = json.load(f)

    all_metrics = []
    dev_metrics = []

    for file_id, entry in sorted(page_map.items()):
        page_idx = entry["page_indices"][0]
        xml_path = os.path.join(PAGES_DIR, f"page_{page_idx + 1:03d}.musicxml")
        if not os.path.exists(xml_path):
            continue

        try:
            pred_score, _ = parse_musicxml(xml_path)
        except Exception:
            continue

        # Fix pitches
        fixed_dict = fix_score(pred_score.to_dict())
        fixed_score = ScoreData.from_dict(fixed_dict)

        # Save
        pred_path = os.path.join(PRED_DIR, f"{file_id}.json")
        with open(pred_path, "w") as f:
            f.write(fixed_score.to_json())

        # Eval
        gold_path = os.path.join(EVENTS_DIR, f"{file_id}.json")
        if not os.path.exists(gold_path):
            continue
        with open(gold_path) as f:
            gold = ScoreData.from_dict(json.load(f))

        metrics = evaluate_pair(gold, fixed_score)
        metrics["file_id"] = file_id
        all_metrics.append(metrics)
        if splits.get(file_id) == "dev":
            dev_metrics.append(metrics)

        print(f"  {file_id} [{splits.get(file_id, '?')}]: "
              f"score={metrics['score']:.4f} pitch={metrics['pitch_acc_global']:.4f}")

    # Aggregate
    for label, mlist in [("ALL", all_metrics), ("DEV", dev_metrics)]:
        if not mlist:
            continue
        n = len(mlist)
        agg = {k: sum(m[k] for m in mlist) / n
               for k in ["event_f1", "pitch_acc_global", "rhythm_acc_global",
                          "measure_validity", "xml_parse_rate"]}
        agg["score"] = (0.30 * agg["event_f1"] + 0.25 * agg["pitch_acc_global"] +
                        0.20 * agg["rhythm_acc_global"] + 0.15 * agg["measure_validity"] +
                        0.10 * agg["xml_parse_rate"])
        print(f"\n{'='*60}")
        print(f"HOMR CLEF-FIXED — {label} ({n} tunes)")
        print(f"  Score:    {agg['score']:.4f}")
        print(f"  Event F1: {agg['event_f1']:.4f}")
        print(f"  Pitch:    {agg['pitch_acc_global']:.4f}")
        print(f"  Rhythm:   {agg['rhythm_acc_global']:.4f}")
        print(f"  Measure:  {agg['measure_validity']:.4f}")


if __name__ == "__main__":
    main()
