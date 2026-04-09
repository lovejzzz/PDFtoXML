"""Evaluate the PrIMuS-pretrained model directly on Omnibook without fine-tuning.

Hypothesis: Fine-tuning on noisy per-staff labels (where token splits don't
align with actual staff content) may be hurting accuracy. The pretrained
model on PrIMuS is already very strong (val loss 0.34).
"""

import json
import os
import sys

import torch

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


def main(checkpoint_name: str = "primus_pretrained.pt"):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    vocab = Vocabulary.load(VOCAB_PATH)
    print(f"Vocab size: {vocab.size}")

    ckpt_path = os.path.join(CHECKPOINTS_DIR, checkpoint_name)
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    model = OMRModel(
        vocab_size=vocab.size,
        d_model=256, nhead=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        max_seq_len=1400,
        pad_idx=vocab.pad_idx,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load dev staff crops
    dev = StaffCropDataset(
        "dev", vocab,
        img_height=128, img_width=1024,
        max_seq_len=600, augment=False,
    )
    print(f"Dev staff crops: {len(dev)}")

    # Predict per staff
    tune_predictions: dict[str, list[list[str]]] = {}
    for i, sample in enumerate(dev.samples):
        img_tensor = dev[i][0].unsqueeze(0).to(device)
        token_ids = model.generate(
            img_tensor,
            sos_idx=vocab.sos_idx,
            eos_idx=vocab.eos_idx,
            max_len=600,
        )[0]
        token_strs = vocab.decode(token_ids)

        file_id = sample.get("file_id", "")
        if file_id not in tune_predictions:
            tune_predictions[file_id] = []
        tune_predictions[file_id].append(token_strs)

    # Assemble per tune
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
    commit_hash = "pretrained_only"
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
        description=f"PrIMuS pretrained ONLY (no FT) {checkpoint_name}",
    )


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "primus_pretrained.pt"
    main(ckpt)
