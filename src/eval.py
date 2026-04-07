"""Evaluation harness for music-pdf2xml.

Compares predicted canonical event sequences against gold standard.
Computes: event_f1, pitch_acc, rhythm_acc, measure_validity, xml_parse_rate.
Logs error categories and appends results to results.tsv.
"""

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

from src.types import (
    CANONICAL_DIVISIONS_PER_QUARTER,
    NoteEvent,
    RestEvent,
    ScoreData,
)

EVENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "events")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results.tsv")


def _event_key(ev, measure_num: int) -> tuple:
    """Tier 1 exact match key for an event."""
    if isinstance(ev, NoteEvent):
        return ("note", measure_num, ev.offset_divisions, ev.pitch, ev.duration_divisions)
    else:
        return ("rest", measure_num, ev.offset_divisions, ev.duration_divisions)


def _align_events(gold_events: list, pred_events: list) -> list[tuple]:
    """Align gold and pred event lists using greedy matching.

    Returns list of (gold_idx | None, pred_idx | None) pairs.
    """
    used_pred = set()
    alignments = []

    for gi, ge in enumerate(gold_events):
        best_pi = None
        for pi, pe in enumerate(pred_events):
            if pi in used_pred:
                continue
            if ge.type != pe.type:
                continue
            # Exact match on all Tier 1 fields
            if isinstance(ge, NoteEvent) and isinstance(pe, NoteEvent):
                if (ge.pitch == pe.pitch and
                        ge.duration_divisions == pe.duration_divisions and
                        ge.offset_divisions == pe.offset_divisions):
                    best_pi = pi
                    break
            elif isinstance(ge, RestEvent) and isinstance(pe, RestEvent):
                if (ge.duration_divisions == pe.duration_divisions and
                        ge.offset_divisions == pe.offset_divisions):
                    best_pi = pi
                    break

        if best_pi is not None:
            used_pred.add(best_pi)
            alignments.append((gi, best_pi))
        else:
            alignments.append((gi, None))

    # Extra predictions
    for pi in range(len(pred_events)):
        if pi not in used_pred:
            alignments.append((None, pi))

    return alignments


def evaluate_pair(gold: ScoreData, pred: ScoreData) -> dict:
    """Evaluate a single gold/pred pair.

    Returns dict with all metrics and error categories.
    """
    # Flatten events with measure context
    gold_events = []
    pred_events = []

    for m in gold.measures:
        for ev in m.events:
            gold_events.append((m.measure_number, ev))

    for m in pred.measures:
        for ev in m.events:
            pred_events.append((m.measure_number, ev))

    # Build keys
    gold_keys = [_event_key(ev, mn) for mn, ev in gold_events]
    pred_keys = [_event_key(ev, mn) for mn, ev in pred_events]

    gold_key_set = Counter(gold_keys)
    pred_key_set = Counter(pred_keys)

    # Event F1 (using multiset intersection)
    tp = sum((gold_key_set & pred_key_set).values())
    fp = sum(pred_key_set.values()) - tp
    fn = sum(gold_key_set.values()) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    event_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Note-specific metrics
    gold_notes = [(mn, ev) for mn, ev in gold_events if isinstance(ev, NoteEvent)]
    pred_notes = [(mn, ev) for mn, ev in pred_events if isinstance(ev, NoteEvent)]

    # Align notes for matched metrics
    alignments = _align_events(
        [ev for _, ev in gold_notes],
        [ev for _, ev in pred_notes],
    )

    matched_count = sum(1 for gi, pi in alignments if gi is not None and pi is not None)
    pitch_correct_matched = 0
    rhythm_correct_matched = 0
    pitch_correct_global = 0
    rhythm_correct_global = 0

    error_categories = Counter()

    for gi, pi in alignments:
        if gi is None:
            error_categories["extra_note"] += 1
            continue
        if pi is None:
            error_categories["missing_note"] += 1
            continue

        ge = gold_notes[gi][1]
        pe = pred_notes[pi][1]

        # Pitch check
        if ge.pitch == pe.pitch:
            pitch_correct_matched += 1
            pitch_correct_global += 1
        else:
            # Classify pitch error
            g_step = ge.pitch[0]
            p_step = pe.pitch[0]
            g_oct = ge.pitch[-1] if ge.pitch[-1].isdigit() else ""
            p_oct = pe.pitch[-1] if pe.pitch[-1].isdigit() else ""

            if g_step == p_step and g_oct != p_oct:
                error_categories["octave_error"] += 1
            elif g_step != p_step and g_oct == p_oct:
                # Check if it's an accidental issue
                if g_step == p_step:
                    error_categories["accidental_omission"] += 1
                else:
                    error_categories["pitch_substitution"] += 1
            else:
                error_categories["pitch_substitution"] += 1

        # Rhythm check
        if ge.duration_divisions == pe.duration_divisions:
            rhythm_correct_matched += 1
            rhythm_correct_global += 1
        else:
            error_categories["duration_error"] += 1

    total_gold_notes = len(gold_notes)

    pitch_acc_matched = pitch_correct_matched / matched_count if matched_count > 0 else 0
    pitch_acc_global = pitch_correct_global / total_gold_notes if total_gold_notes > 0 else 0
    rhythm_acc_matched = rhythm_correct_matched / matched_count if matched_count > 0 else 0
    rhythm_acc_global = rhythm_correct_global / total_gold_notes if total_gold_notes > 0 else 0

    # Measure validity
    valid_measures = 0
    total_checked_measures = 0
    for m in pred.measures:
        expected_dur = (m.time_beats * CANONICAL_DIVISIONS_PER_QUARTER *
                        4 // m.time_beat_type)
        actual_dur = sum(ev.duration_divisions for ev in m.events)

        # Skip anacrustic/pickup measures (first measure with less than expected)
        if m.measure_number == 1 and actual_dur < expected_dur:
            continue
        # Skip empty measures
        if not m.events:
            continue

        total_checked_measures += 1
        if actual_dur == expected_dur:
            valid_measures += 1
        else:
            error_categories["barline_shift"] += 1

    measure_validity = valid_measures / total_checked_measures if total_checked_measures > 0 else 0

    # XML parse rate (always 1.0 when evaluating from ScoreData)
    xml_parse_rate = 1.0

    # Combined score
    score = (0.30 * event_f1 +
             0.25 * pitch_acc_global +
             0.20 * rhythm_acc_global +
             0.15 * measure_validity +
             0.10 * xml_parse_rate)

    return {
        "event_f1": event_f1,
        "pitch_acc_matched": pitch_acc_matched,
        "pitch_acc_global": pitch_acc_global,
        "rhythm_acc_matched": rhythm_acc_matched,
        "rhythm_acc_global": rhythm_acc_global,
        "measure_validity": measure_validity,
        "xml_parse_rate": xml_parse_rate,
        "score": score,
        "error_categories": dict(error_categories),
        "total_gold_events": len(gold_events),
        "total_pred_events": len(pred_events),
        "total_gold_notes": total_gold_notes,
        "matched_notes": matched_count,
    }


def evaluate_all(
    pred_dir: str,
    gold_dir: str = EVENTS_DIR,
    results_path: str = RESULTS_PATH,
    commit: str = "manual",
    description: str = "",
) -> dict:
    """Evaluate all predictions against gold standard.

    Returns aggregate metrics dict.
    """
    gold_files = {f.stem: f for f in Path(gold_dir).glob("*.json")}
    pred_files = {f.stem: f for f in Path(pred_dir).glob("*.json")}

    common = sorted(set(gold_files) & set(pred_files))
    if not common:
        print("No matching files found between pred and gold directories")
        return {}

    all_metrics = []
    for file_id in common:
        with open(gold_files[file_id]) as f:
            gold = ScoreData.from_dict(json.load(f))
        with open(pred_files[file_id]) as f:
            pred = ScoreData.from_dict(json.load(f))

        metrics = evaluate_pair(gold, pred)
        metrics["file_id"] = file_id
        all_metrics.append(metrics)

        print(f"  {file_id}: score={metrics['score']:.4f} "
              f"f1={metrics['event_f1']:.4f} "
              f"pitch={metrics['pitch_acc_global']:.4f} "
              f"rhythm={metrics['rhythm_acc_global']:.4f}")

    # Aggregate
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

    print(f"\nAggregate ({n} files):")
    print(f"  score={agg['score']:.4f}")
    print(f"  event_f1={agg['event_f1']:.4f}")
    print(f"  pitch_acc={agg['pitch_acc_global']:.4f}")
    print(f"  rhythm_acc={agg['rhythm_acc_global']:.4f}")
    print(f"  measure_validity={agg['measure_validity']:.4f}")
    print(f"  xml_parse_rate={agg['xml_parse_rate']:.4f}")

    # Aggregate errors
    total_errors = Counter()
    for m in all_metrics:
        for k, v in m["error_categories"].items():
            total_errors[k] += v
    if total_errors:
        print(f"\n  Error categories: {dict(total_errors)}")

    # Log to results.tsv
    status = "keep" if agg["score"] > 0 else "crash"
    _log_result(results_path, commit, agg, status, description)

    return agg


def _log_result(results_path: str, commit: str, agg: dict, status: str, description: str):
    """Append a result row to results.tsv."""
    header = "commit\tscore\tevent_f1\tpitch_acc_global\trhythm_acc_global\tmeasure_validity\txml_parse_rate\tstatus\tdescription"

    write_header = not os.path.exists(results_path) or os.path.getsize(results_path) == 0

    with open(results_path, "a") as f:
        if write_header:
            f.write(header + "\n")
        row = (f"{commit}\t{agg['score']:.4f}\t{agg['event_f1']:.4f}\t"
               f"{agg['pitch_acc_global']:.4f}\t{agg['rhythm_acc_global']:.4f}\t"
               f"{agg['measure_validity']:.4f}\t{agg['xml_parse_rate']:.4f}\t"
               f"{status}\t{description}")
        f.write(row + "\n")

    print(f"\n  Result logged to {results_path}")


def roundtrip_eval(gold_dir: str = EVENTS_DIR):
    """Run evaluation comparing gold events to themselves (should score ~1.0)."""
    print("Round-trip eval (gold vs gold — should be ~1.0):")
    return evaluate_all(gold_dir, gold_dir, commit="roundtrip", description="self-eval sanity check")


if __name__ == "__main__":
    if "--roundtrip-test" in sys.argv:
        roundtrip_eval()
    elif "--pred" in sys.argv and "--gold" in sys.argv:
        pred_idx = sys.argv.index("--pred") + 1
        gold_idx = sys.argv.index("--gold") + 1
        pred_dir = sys.argv[pred_idx]
        gold_dir = sys.argv[gold_idx]
        evaluate_all(pred_dir, gold_dir)
    else:
        print("Usage:")
        print("  python -m src.eval --roundtrip-test")
        print("  python -m src.eval --pred outputs/pred/ --gold data/events/")
