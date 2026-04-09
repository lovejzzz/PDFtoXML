"""Test-time augmentation: predict each staff multiple times with different
augmentations and pick the most consistent prediction.
"""

import json
import os
import sys
import random
from collections import Counter

import numpy as np
import torch
from PIL import Image, ImageEnhance

from src.experiments.dataset import Vocabulary
from src.experiments.staff_dataset import StaffCropDataset
from src.experiments.model import OMRModel
from src.experiments.decode import save_predictions
from src.eval import evaluate_all

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
PRED_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")
EVENTS_DIR = os.path.join(PROJECT_ROOT, "data", "events")
VOCAB_PATH = os.path.join(PROJECT_ROOT, "data", "vocab_combined.json")


def _augment_variants(img: Image.Image, n: int = 3) -> list[Image.Image]:
    """Generate N augmented variants of an image."""
    variants = [img]  # original always included

    for i in range(n):
        v = img.copy()
        # Random brightness
        v = ImageEnhance.Brightness(v).enhance(random.uniform(0.85, 1.15))
        # Random contrast
        v = ImageEnhance.Contrast(v).enhance(random.uniform(0.85, 1.15))
        variants.append(v)

    return variants


def _preprocess(img: Image.Image, h: int, w: int) -> torch.Tensor:
    img = img.convert("L")
    iw, ih = img.size
    scale = h / ih
    new_w = min(int(iw * scale), w)
    new_h = h
    img = img.resize((new_w, new_h), Image.LANCZOS)
    padded = Image.new("L", (w, h), 255)
    padded.paste(img, (0, 0))
    arr = np.array(padded, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def main(checkpoint: str = "primus_finetuned_v2.pt", n_augments: int = 3):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}, augments={n_augments}")

    vocab = Vocabulary.load(VOCAB_PATH)

    ckpt = torch.load(
        os.path.join(CHECKPOINTS_DIR, checkpoint),
        map_location=device, weights_only=True,
    )
    model = OMRModel(
        vocab_size=vocab.size,
        d_model=256, nhead=4, num_decoder_layers=4, dim_feedforward=512,
        max_seq_len=1400, pad_idx=vocab.pad_idx,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dev = StaffCropDataset(
        "dev", vocab,
        img_height=128, img_width=1024,
        max_seq_len=600, augment=False,
    )
    print(f"Dev staff crops: {len(dev)}")

    tune_predictions: dict[str, list[list[str]]] = {}

    for i, sample in enumerate(dev.samples):
        # Load image
        img = Image.open(sample["crop_path"])
        variants = _augment_variants(img, n_augments)

        # Get prediction for each variant
        all_preds = []
        for v in variants:
            tensor = _preprocess(v, 128, 1024).to(device)
            ids = model.generate(
                tensor,
                sos_idx=vocab.sos_idx,
                eos_idx=vocab.eos_idx,
                max_len=600,
            )[0]
            all_preds.append(vocab.decode(ids))

        # Pick the median-length prediction (avoid both over and under generation)
        sorted_preds = sorted(all_preds, key=lambda p: len(p))
        best_pred = sorted_preds[len(sorted_preds) // 2]

        file_id = sample.get("file_id", "")
        if file_id not in tune_predictions:
            tune_predictions[file_id] = []
        tune_predictions[file_id].append(best_pred)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(dev)}")

    assembled = {}
    header_prefixes = ("CLEF_", "KEY_", "TIME_")
    for file_id, staff_lists in tune_predictions.items():
        combined = []
        for si, staff_tokens in enumerate(staff_lists):
            if si == 0:
                combined.extend(staff_tokens)
            else:
                for tok in staff_tokens:
                    if not tok.startswith(header_prefixes):
                        combined.append(tok)
        assembled[file_id] = combined

    save_predictions(assembled, PRED_DIR)
    print(f"Assembled {len(assembled)} tunes")

    import subprocess
    commit_hash = "tta"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            commit_hash = result.stdout.strip()
    except Exception:
        pass

    evaluate_all(
        pred_dir=PRED_DIR,
        gold_dir=EVENTS_DIR,
        commit=commit_hash,
        description=f"TTA n={n_augments} ckpt={checkpoint}",
    )


if __name__ == "__main__":
    ckpt = "primus_finetuned_v2.pt"
    n = 3
    for arg in sys.argv[1:]:
        if arg.startswith("n="):
            n = int(arg.split("=")[1])
        elif not arg.startswith("--"):
            ckpt = arg
    main(ckpt, n)
