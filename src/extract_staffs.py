"""Extract individual staff crops from Omnibook pages using homr's segmentation.

For each page:
1. Run homr's UNet segmentation to detect staff positions
2. Crop each staff with padding
3. Split the ground-truth token sequence across staves by measure count
4. Save crops + per-staff token files for training

Output: data/staff_crops/{file_id}/staff_NN.png + staff_NN.tokens
"""

import json
import math
import os
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES_DIR = os.path.join(PROJECT_ROOT, "data", "pages")
EVENTS_DIR = os.path.join(PROJECT_ROOT, "data", "events")
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "data_manifest")
CROPS_DIR = os.path.join(PROJECT_ROOT, "data", "staff_crops")

# Padding around each staff crop (pixels)
CROP_PAD = 20


def _detect_staffs(image_path: str):
    """Use homr to detect staff positions and bar line counts in an image."""
    from homr.main import detect_staffs_in_image, ProcessingConfig
    config = ProcessingConfig(False, False, False, False, None)
    multi_staffs, image, debug, title = detect_staffs_in_image(image_path, config)

    staffs = []
    for ms in multi_staffs:
        staff = ms.staffs[0]
        y1 = max(0, int(staff.min_y) - CROP_PAD)
        y2 = min(image.shape[0], int(staff.max_y) + CROP_PAD)
        x1 = max(0, int(staff.min_x) - CROP_PAD)
        x2 = min(image.shape[1], int(staff.max_x) + CROP_PAD)
        # Get bar lines on this staff (estimate measure count)
        try:
            bar_lines = staff.get_bar_lines()
            n_bars = len(bar_lines) if bar_lines is not None else 0
        except Exception:
            n_bars = 0
        # Number of measures = bar lines (approximately)
        # Each staff has at least 1 measure
        n_measures = max(1, n_bars)
        staffs.append({
            "y1": y1, "y2": y2, "x1": x1, "x2": x2,
            "crop": image[y1:y2, x1:x2],
            "n_measures": n_measures,
        })

    return staffs, image


def _split_tokens_by_staff(
    tokens: list[str],
    n_staffs: int,
    measures_per_staff_list: list[int] | None = None,
) -> list[list[str]]:
    """Split a full token sequence across N staves using per-staff measure counts.

    If measures_per_staff_list is provided (from homr bar line detection),
    use it to allocate measures more accurately. Otherwise, split evenly.
    """
    # Find measure boundaries
    measure_starts = []
    for i, tok in enumerate(tokens):
        if tok == "MEASURE_START":
            measure_starts.append(i)

    n_measures = len(measure_starts)
    if n_measures == 0 or n_staffs <= 0:
        return [tokens]

    # Extract header tokens (before first measure)
    header = tokens[:measure_starts[0]] if measure_starts[0] > 0 else []

    # Compute per-staff measure allocation
    if measures_per_staff_list and len(measures_per_staff_list) == n_staffs:
        # Scale to match total measures
        total_detected = sum(measures_per_staff_list)
        if total_detected > 0:
            scale = n_measures / total_detected
            scaled = [max(1, int(round(m * scale))) for m in measures_per_staff_list]
            # Adjust for rounding errors
            diff = n_measures - sum(scaled)
            if diff != 0 and len(scaled) > 0:
                # Distribute correction across staves
                idx = 0
                while diff > 0:
                    scaled[idx % len(scaled)] += 1
                    diff -= 1
                    idx += 1
                while diff < 0 and any(s > 1 for s in scaled):
                    if scaled[idx % len(scaled)] > 1:
                        scaled[idx % len(scaled)] -= 1
                        diff += 1
                    idx += 1
            measures_per_staff = scaled
        else:
            measures_per_staff = [max(1, n_measures // n_staffs)] * n_staffs
    else:
        # Even split
        per = max(1, math.ceil(n_measures / n_staffs))
        measures_per_staff = [per] * n_staffs

    # Build per-staff token lists
    staff_tokens = []
    cur_measure = 0
    for s in range(n_staffs):
        n_for_this = measures_per_staff[s]
        start_measure = cur_measure
        end_measure = min(cur_measure + n_for_this, n_measures)
        cur_measure = end_measure

        if start_measure >= n_measures:
            # No more measures - give an empty staff (just header)
            staff_tokens.append(list(header))
            continue

        tok_start = measure_starts[start_measure]
        if end_measure < n_measures:
            tok_end = measure_starts[end_measure]
        else:
            tok_end = len(tokens)

        staff_toks = list(header) + tokens[tok_start:tok_end]
        staff_tokens.append(staff_toks)

    return staff_tokens


def extract_all_staffs():
    """Extract staff crops for all labeled Omnibook pages."""
    os.makedirs(CROPS_DIR, exist_ok=True)

    with open(os.path.join(MANIFEST_DIR, "manual_page_map.json")) as f:
        page_map = json.load(f)

    with open(os.path.join(MANIFEST_DIR, "splits.json")) as f:
        splits = json.load(f)

    manifest = []
    total_crops = 0
    errors = 0

    for file_id, entry in sorted(page_map.items()):
        # Load full token sequence
        token_path = os.path.join(EVENTS_DIR, f"{file_id}.tokens")
        if not os.path.exists(token_path):
            continue

        with open(token_path) as f:
            full_tokens = f.read().strip().split()

        tune_dir = os.path.join(CROPS_DIR, file_id)
        os.makedirs(tune_dir, exist_ok=True)

        # Process all pages for this tune
        all_page_staffs = []
        for page_idx in entry["page_indices"]:
            page_path = os.path.join(PAGES_DIR, f"page_{page_idx + 1:03d}.png")
            if not os.path.exists(page_path):
                continue

            try:
                staffs, _ = _detect_staffs(page_path)
                all_page_staffs.extend(staffs)
            except Exception as e:
                print(f"  ERROR {file_id} page {page_idx}: {e}")
                errors += 1
                continue

        if not all_page_staffs:
            print(f"  {file_id}: no staffs detected")
            continue

        # Split tokens across all staves using per-staff measure counts
        n_staffs = len(all_page_staffs)
        measures_per_staff = [s.get("n_measures", 1) for s in all_page_staffs]
        staff_token_lists = _split_tokens_by_staff(
            full_tokens, n_staffs, measures_per_staff
        )

        # Save crops and tokens
        for i, staff_info in enumerate(all_page_staffs):
            crop_path = os.path.join(tune_dir, f"staff_{i:02d}.png")
            cv2.imwrite(crop_path, staff_info["crop"])

            if i < len(staff_token_lists):
                staff_tokens = staff_token_lists[i]
                tok_path = os.path.join(tune_dir, f"staff_{i:02d}.tokens")
                with open(tok_path, "w") as f:
                    f.write(" ".join(staff_tokens))

                manifest.append({
                    "file_id": file_id,
                    "staff_index": i,
                    "crop_path": crop_path,
                    "token_path": tok_path,
                    "n_tokens": len(staff_tokens),
                    "crop_height": staff_info["y2"] - staff_info["y1"],
                    "crop_width": staff_info["x2"] - staff_info["x1"],
                    "split": splits.get(file_id, "train"),
                })

            total_crops += 1

        print(f"  {file_id}: {n_staffs} staves, "
              f"{len(staff_token_lists)} token groups")

    # Save manifest
    manifest_path = os.path.join(CROPS_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    splits_count = {"train": 0, "dev": 0, "test": 0}
    for m in manifest:
        splits_count[m["split"]] = splits_count.get(m["split"], 0) + 1

    print(f"\nExtraction complete:")
    print(f"  Total staff crops: {total_crops}")
    print(f"  With tokens: {len(manifest)}")
    print(f"  Splits: {splits_count}")
    print(f"  Errors: {errors}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    extract_all_staffs()
