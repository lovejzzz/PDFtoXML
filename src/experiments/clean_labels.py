"""Self-cleaning: use the model to identify and filter noisy training labels.

The per-staff token splits are approximate. Some staves have correctly aligned
labels, others don't. Use the trained model to predict each training staff,
compare pred vs label, and keep only staves where they're similar.
Then retrain on the cleaned subset.
"""

import json
import os
import sys
from difflib import SequenceMatcher

import torch

from src.experiments.dataset import Vocabulary
from src.experiments.staff_dataset import StaffCropDataset
from src.experiments.model import OMRModel

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
VOCAB_PATH = os.path.join(PROJECT_ROOT, "data", "vocab_combined.json")
CROPS_DIR = os.path.join(PROJECT_ROOT, "data", "staff_crops")


def clean_labels(
    checkpoint: str = "primus_finetuned_v9.pt",
    similarity_threshold: float = 0.3,
):
    """Score each training staff by pred-vs-label similarity, filter noisy ones."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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

    train_ds = StaffCropDataset(
        "train", vocab,
        img_height=128, img_width=1024,
        max_seq_len=600, augment=False,
    )
    print(f"Training staves: {len(train_ds)}")

    # Score each staff
    scores = []
    for i, sample in enumerate(train_ds.samples):
        # Get label tokens
        with open(sample["token_path"]) as f:
            label_tokens = f.read().strip().split()

        # Get prediction
        img_tensor = train_ds[i][0].unsqueeze(0).to(device)
        pred_ids = model.generate(
            img_tensor,
            sos_idx=vocab.sos_idx,
            eos_idx=vocab.eos_idx,
            max_len=600,
        )[0]
        pred_tokens = vocab.decode(pred_ids)

        # Compare sequences
        sm = SequenceMatcher(None, pred_tokens, label_tokens)
        similarity = sm.ratio()

        scores.append({
            "index": i,
            "file_id": sample["file_id"],
            "similarity": similarity,
            "pred_len": len(pred_tokens),
            "label_len": len(label_tokens),
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(train_ds)}")

    # Sort by similarity
    scores.sort(key=lambda x: x["similarity"], reverse=True)

    # Report
    sims = [s["similarity"] for s in scores]
    print(f"\nSimilarity stats:")
    print(f"  Mean: {sum(sims)/len(sims):.4f}")
    print(f"  Min: {min(sims):.4f}, Max: {max(sims):.4f}")
    print(f"  >0.5: {sum(1 for s in sims if s > 0.5)}")
    print(f"  >0.3: {sum(1 for s in sims if s > 0.3)}")
    print(f"  >0.2: {sum(1 for s in sims if s > 0.2)}")
    print(f"  >0.1: {sum(1 for s in sims if s > 0.1)}")

    # Save cleaned manifest (only keep staves above threshold)
    kept = [s for s in scores if s["similarity"] >= similarity_threshold]
    print(f"\nKeeping {len(kept)}/{len(scores)} staves (threshold={similarity_threshold})")

    # Update the manifest
    manifest_path = os.path.join(CROPS_DIR, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Create cleaned manifest
    kept_indices = {s["index"] for s in kept}
    train_samples = [m for m in manifest if m["split"] == "train"]
    cleaned_manifest = []
    for i, m in enumerate(train_samples):
        if i in kept_indices:
            cleaned_manifest.append(m)

    # Add all dev/test samples unchanged
    for m in manifest:
        if m["split"] != "train":
            cleaned_manifest.append(m)

    cleaned_path = os.path.join(CROPS_DIR, "manifest_cleaned.json")
    with open(cleaned_path, "w") as f:
        json.dump(cleaned_manifest, f, indent=2)

    train_kept = sum(1 for m in cleaned_manifest if m["split"] == "train")
    print(f"Cleaned manifest: {train_kept} train, "
          f"{sum(1 for m in cleaned_manifest if m['split'] == 'dev')} dev")
    print(f"Saved to: {cleaned_path}")

    return cleaned_path


if __name__ == "__main__":
    threshold = 0.3
    ckpt = "primus_finetuned_v9.pt"
    for arg in sys.argv[1:]:
        if arg.startswith("threshold="):
            threshold = float(arg.split("=")[1])
        elif not arg.startswith("--"):
            ckpt = arg
    clean_labels(ckpt, threshold)
