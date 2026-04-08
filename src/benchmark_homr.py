"""Benchmark homr OMR tool against our ground truth.

Runs homr on all labeled Omnibook pages, parses the MusicXML output
with our prepare_data parser, and evaluates with eval.py.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES_DIR = os.path.join(PROJECT_ROOT, "data", "pages")
EVENTS_DIR = os.path.join(PROJECT_ROOT, "data", "events")
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "data_manifest")
HOMR_OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "homr_predictions")


def run_homr_on_all():
    """Run homr on all labeled pages and collect MusicXML outputs."""
    os.makedirs(HOMR_OUT_DIR, exist_ok=True)

    with open(os.path.join(MANIFEST_DIR, "manual_page_map.json")) as f:
        page_map = json.load(f)

    with open(os.path.join(MANIFEST_DIR, "splits.json")) as f:
        splits = json.load(f)

    results = {}

    for file_id, entry in sorted(page_map.items()):
        # Process all pages for this tune
        all_xmls = []
        for page_idx in entry["page_indices"]:
            page_path = os.path.join(PAGES_DIR, f"page_{page_idx + 1:03d}.png")
            if not os.path.exists(page_path):
                continue

            # Check if homr already produced output
            xml_out = page_path.replace(".png", ".musicxml")
            if not os.path.exists(xml_out):
                print(f"  Running homr on page_{page_idx + 1:03d}...")
                try:
                    result = subprocess.run(
                        ["homr", page_path],
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                except subprocess.TimeoutExpired:
                    print(f"    TIMEOUT on page_{page_idx + 1:03d}")
                    continue
                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue

            if os.path.exists(xml_out):
                all_xmls.append(xml_out)

        if all_xmls:
            results[file_id] = {
                "xml_paths": all_xmls,
                "split": splits.get(file_id, "train"),
                "title": entry.get("title", file_id),
            }
            print(f"  {file_id}: {len(all_xmls)} page(s) processed")
        else:
            print(f"  {file_id}: FAILED - no output")

    print(f"\nProcessed: {len(results)}/{len(page_map)} tunes")
    return results


def convert_and_evaluate(results: dict):
    """Convert homr MusicXML output to our canonical format and evaluate."""
    from src.prepare_data import parse_musicxml
    from src.eval import evaluate_pair
    from src.types import ScoreData

    os.makedirs(HOMR_OUT_DIR, exist_ok=True)

    all_metrics = []
    dev_metrics = []

    for file_id, info in sorted(results.items()):
        # Parse homr output (use first page's XML for now)
        xml_path = info["xml_paths"][0]
        try:
            pred_score, _ = parse_musicxml(xml_path)
        except Exception as e:
            print(f"  {file_id}: parse error: {e}")
            continue

        # Save as canonical JSON for eval
        pred_json_path = os.path.join(HOMR_OUT_DIR, f"{file_id}.json")
        with open(pred_json_path, "w") as f:
            f.write(pred_score.to_json())

        # Load gold standard
        gold_json_path = os.path.join(EVENTS_DIR, f"{file_id}.json")
        if not os.path.exists(gold_json_path):
            continue

        with open(gold_json_path) as f:
            gold_score = ScoreData.from_dict(json.load(f))

        # Evaluate
        metrics = evaluate_pair(gold_score, pred_score)
        metrics["file_id"] = file_id
        metrics["split"] = info["split"]
        all_metrics.append(metrics)

        if info["split"] == "dev":
            dev_metrics.append(metrics)

        print(f"  {file_id} [{info['split']}]: "
              f"score={metrics['score']:.4f} "
              f"pitch={metrics['pitch_acc_global']:.4f} "
              f"rhythm={metrics['rhythm_acc_global']:.4f}")

    # Aggregate
    if all_metrics:
        n = len(all_metrics)
        agg = {
            "event_f1": sum(m["event_f1"] for m in all_metrics) / n,
            "pitch_acc_global": sum(m["pitch_acc_global"] for m in all_metrics) / n,
            "rhythm_acc_global": sum(m["rhythm_acc_global"] for m in all_metrics) / n,
            "measure_validity": sum(m["measure_validity"] for m in all_metrics) / n,
            "xml_parse_rate": sum(m["xml_parse_rate"] for m in all_metrics) / n,
        }
        agg["score"] = (0.30 * agg["event_f1"] +
                        0.25 * agg["pitch_acc_global"] +
                        0.20 * agg["rhythm_acc_global"] +
                        0.15 * agg["measure_validity"] +
                        0.10 * agg["xml_parse_rate"])

        print(f"\n{'='*60}")
        print(f"HOMR BENCHMARK — ALL ({n} tunes)")
        print(f"{'='*60}")
        print(f"  Combined Score: {agg['score']:.4f}")
        print(f"  Event F1:       {agg['event_f1']:.4f}")
        print(f"  Pitch Acc:      {agg['pitch_acc_global']:.4f}")
        print(f"  Rhythm Acc:     {agg['rhythm_acc_global']:.4f}")
        print(f"  Measure Valid:  {agg['measure_validity']:.4f}")
        print(f"  XML Parse:      {agg['xml_parse_rate']:.4f}")

    if dev_metrics:
        n = len(dev_metrics)
        dev_agg = {
            "event_f1": sum(m["event_f1"] for m in dev_metrics) / n,
            "pitch_acc_global": sum(m["pitch_acc_global"] for m in dev_metrics) / n,
            "rhythm_acc_global": sum(m["rhythm_acc_global"] for m in dev_metrics) / n,
            "measure_validity": sum(m["measure_validity"] for m in dev_metrics) / n,
            "xml_parse_rate": sum(m["xml_parse_rate"] for m in dev_metrics) / n,
        }
        dev_agg["score"] = (0.30 * dev_agg["event_f1"] +
                            0.25 * dev_agg["pitch_acc_global"] +
                            0.20 * dev_agg["rhythm_acc_global"] +
                            0.15 * dev_agg["measure_validity"] +
                            0.10 * dev_agg["xml_parse_rate"])

        print(f"\n{'='*60}")
        print(f"HOMR BENCHMARK — DEV SET ({n} tunes)")
        print(f"{'='*60}")
        print(f"  Combined Score: {dev_agg['score']:.4f}")
        print(f"  Event F1:       {dev_agg['event_f1']:.4f}")
        print(f"  Pitch Acc:      {dev_agg['pitch_acc_global']:.4f}")
        print(f"  Rhythm Acc:     {dev_agg['rhythm_acc_global']:.4f}")
        print(f"  Measure Valid:  {dev_agg['measure_validity']:.4f}")
        print(f"  XML Parse:      {dev_agg['xml_parse_rate']:.4f}")


if __name__ == "__main__":
    print("Running homr on all labeled Omnibook pages...")
    results = run_homr_on_all()
    print("\nEvaluating homr output...")
    convert_and_evaluate(results)
