"""Semi-automatic alignment of MusicXML files to PDF pages.

Strategy:
1. Try PyMuPDF text extraction on each page
2. Fall back to OCR (Tesseract) if available
3. Fuzzy match titles against page text
4. Load manual corrections from manual_page_map.json
5. Generate dataset.csv manifest and splits.json
"""

import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import fitz  # PyMuPDF
from rapidfuzz import fuzz, process

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "charlie-parker-omnibook.pdf")
EVENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "events")
XML_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "xml")
MANIFEST_DIR = os.path.join(os.path.dirname(__file__), "..", "data_manifest")
DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "alignment_debug")
MANUAL_MAP_PATH = os.path.join(MANIFEST_DIR, "manual_page_map.json")

SPLIT_SEED = 42
SPLIT_RATIOS = {"train": 0.70, "dev": 0.15, "test": 0.15}


def _title_to_id(title: str) -> str:
    return title.lower().replace(" ", "_").replace("'", "").replace(".", "")


def _extract_page_texts(pdf_path: str) -> list[str]:
    """Extract text from each PDF page using PyMuPDF."""
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    doc.close()
    return texts


def _try_ocr_page(pdf_path: str, page_index: int, dpi: int = 150) -> str:
    """Try OCR on a single page using Tesseract (if available)."""
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        img_bytes = pix.tobytes("png")
        doc.close()

        result = subprocess.run(
            ["tesseract", "stdin", "stdout", "--psm", "6"],
            input=img_bytes,
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.decode("utf-8", errors="replace")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def _ocr_all_pages(pdf_path: str, page_texts: list[str]) -> dict[int, str]:
    """Run OCR once on all pages that lack text. Returns {page_idx: ocr_text}."""
    # Check if Tesseract is available
    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  Tesseract not available — skipping OCR")
        return {}

    print("  Running OCR on pages without text (one pass)...")
    doc = fitz.open(pdf_path)
    results = {}
    for i, text in enumerate(page_texts):
        if text.strip():
            continue
        try:
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72))
            img_bytes = pix.tobytes("png")
            result = subprocess.run(
                ["tesseract", "stdin", "stdout", "--psm", "6"],
                input=img_bytes,
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                results[i] = result.stdout.decode("utf-8", errors="replace")
        except (subprocess.TimeoutExpired, Exception):
            continue
        if (i + 1) % 20 == 0:
            print(f"    OCR'd {i + 1}/{len(page_texts)} pages...")
    doc.close()
    print(f"  OCR complete: {len(results)} pages processed")
    return results


def _collect_xml_titles(xml_dir: str) -> dict[str, str]:
    """Collect {file_id: title} from event JSONs or XML filenames."""
    titles = {}
    events_dir = os.path.join(os.path.dirname(xml_dir), "events")

    for f in sorted(Path(xml_dir).glob("*.xml")):
        file_id = _title_to_id(f.stem)
        # Try to get title from event JSON first
        event_path = os.path.join(events_dir, f"{file_id}.json")
        if os.path.exists(event_path):
            with open(event_path) as jf:
                data = json.load(jf)
                title = data.get("meta", {}).get("title", f.stem.replace("_", " "))
        else:
            title = f.stem.replace("_", " ")
        titles[file_id] = title
    return titles


def _count_measures(file_id: str) -> int:
    """Count measures from event JSON."""
    event_path = os.path.join(EVENTS_DIR, f"{file_id}.json")
    if os.path.exists(event_path):
        with open(event_path) as f:
            data = json.load(f)
            return len(data.get("measures", []))
    return 0


def align_data(
    pdf_path: str = PDF_PATH,
    xml_dir: str = XML_DIR,
    manifest_dir: str = MANIFEST_DIR,
    debug_dir: str = DEBUG_DIR,
):
    os.makedirs(manifest_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    print("Extracting text from PDF pages...")
    page_texts = _extract_page_texts(pdf_path)
    print(f"  {len(page_texts)} pages extracted")

    # Check how many pages have useful text
    pages_with_text = sum(1 for t in page_texts if len(t.strip()) > 10)
    print(f"  {pages_with_text} pages have extractable text")

    # Collect XML titles
    titles = _collect_xml_titles(xml_dir)
    print(f"  {len(titles)} XML files to match")

    # Load manual corrections
    manual_map = {}
    if os.path.exists(MANUAL_MAP_PATH):
        with open(MANUAL_MAP_PATH) as f:
            manual_map = json.load(f)
        print(f"  Loaded {len(manual_map)} manual corrections")

    # Match titles to pages
    matches = []
    unmatched = []
    ocr_texts = None  # lazy-loaded OCR cache

    for file_id, title in titles.items():
        # Check manual map first
        if file_id in manual_map:
            entry = manual_map[file_id]
            matches.append({
                "id": file_id,
                "title": entry.get("title", title),
                "page_indices": entry["page_indices"],
                "match_source": "manual",
                "confidence": 1.0,
                "status": "manual_verified",
            })
            continue

        # Try text matching on each page
        best_score = 0
        best_pages = []

        for page_idx, text in enumerate(page_texts):
            if not text.strip():
                continue
            # Try fuzzy matching the title against page text
            score = fuzz.partial_ratio(title.lower(), text.lower())
            if score > best_score:
                best_score = score
                best_pages = [page_idx]
            elif score == best_score and score > 70:
                best_pages.append(page_idx)

        if best_score >= 75 and best_pages:
            # Keep only the first match (most likely the title page)
            matches.append({
                "id": file_id,
                "title": title,
                "page_indices": [best_pages[0]],
                "match_source": "text",
                "confidence": best_score / 100.0,
                "status": "auto_matched",
            })
        else:
            # No text match — try OCR cache if available
            if ocr_texts is None:
                # One-time OCR pass over pages without text
                ocr_texts = _ocr_all_pages(pdf_path, page_texts)

            ocr_best_score = 0
            ocr_best_page = -1
            for page_idx, ocr_text in ocr_texts.items():
                if ocr_text:
                    score = fuzz.partial_ratio(title.lower(), ocr_text.lower())
                    if score > ocr_best_score:
                        ocr_best_score = score
                        ocr_best_page = page_idx

            if ocr_best_score >= 70 and ocr_best_page >= 0:
                matches.append({
                    "id": file_id,
                    "title": title,
                    "page_indices": [ocr_best_page],
                    "match_source": "ocr",
                    "confidence": ocr_best_score / 100.0,
                    "status": "auto_matched",
                })
            else:
                unmatched.append({
                    "id": file_id,
                    "title": title,
                    "best_text_score": best_score,
                })

    # Generate splits (by tune, not by page)
    all_ids = sorted([m["id"] for m in matches] + [u["id"] for u in unmatched])
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(all_ids)

    n = len(all_ids)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_dev = int(n * SPLIT_RATIOS["dev"])

    splits = {}
    for i, fid in enumerate(all_ids):
        if i < n_train:
            splits[fid] = "train"
        elif i < n_train + n_dev:
            splits[fid] = "dev"
        else:
            splits[fid] = "test"

    splits_path = os.path.join(manifest_dir, "splits.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    split_counts = {"train": 0, "dev": 0, "test": 0}
    for s in splits.values():
        split_counts[s] += 1
    print(f"  Splits: {split_counts}")

    # Write manifest CSV
    csv_path = os.path.join(manifest_dir, "dataset.csv")
    fieldnames = [
        "id", "title", "xml_path", "event_json_path", "page_indices",
        "system_indices", "crop_bbox", "status", "match_source",
        "title_match_confidence", "num_measures", "split",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for m in sorted(matches, key=lambda x: x["id"]):
            writer.writerow({
                "id": m["id"],
                "title": m["title"],
                "xml_path": f"data/xml/{_find_xml_filename(xml_dir, m['id'])}",
                "event_json_path": f"data/events/{m['id']}.json",
                "page_indices": json.dumps(m["page_indices"]),
                "system_indices": "",
                "crop_bbox": "",
                "status": m["status"],
                "match_source": m["match_source"],
                "title_match_confidence": f"{m['confidence']:.2f}",
                "num_measures": _count_measures(m["id"]),
                "split": splits.get(m["id"], "train"),
            })

        for u in sorted(unmatched, key=lambda x: x["id"]):
            writer.writerow({
                "id": u["id"],
                "title": u["title"],
                "xml_path": f"data/xml/{_find_xml_filename(xml_dir, u['id'])}",
                "event_json_path": f"data/events/{u['id']}.json",
                "page_indices": "",
                "system_indices": "",
                "crop_bbox": "",
                "status": "unmatched",
                "match_source": "",
                "title_match_confidence": "0.00",
                "num_measures": _count_measures(u["id"]),
                "split": splits.get(u["id"], "train"),
            })

    # Debug output
    report = {
        "total_xml": len(titles),
        "matched": len(matches),
        "unmatched": len(unmatched),
        "match_sources": {},
        "unmatched_list": unmatched,
    }
    for m in matches:
        src = m["match_source"]
        report["match_sources"][src] = report["match_sources"].get(src, 0) + 1

    report_path = os.path.join(debug_dir, "alignment_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nAlignment complete:")
    print(f"  Matched: {len(matches)}/{len(titles)}")
    print(f"  Unmatched: {len(unmatched)}")
    print(f"  Sources: {report['match_sources']}")
    print(f"  Manifest: {csv_path}")
    print(f"  Splits: {splits_path}")

    if unmatched:
        print(f"\n  Unmatched tunes (need manual_page_map.json entries):")
        for u in unmatched:
            print(f"    - {u['title']} (id: {u['id']}, best score: {u['best_text_score']})")


def _find_xml_filename(xml_dir: str, file_id: str) -> str:
    """Find the actual XML filename for a file_id."""
    for f in Path(xml_dir).glob("*.xml"):
        if _title_to_id(f.stem) == file_id:
            return f.name
    return f"{file_id}.xml"


if __name__ == "__main__":
    align_data()
