"""Evaluate model using beam search instead of greedy decoding."""

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


def main(checkpoint_name: str = "primus_finetuned_v2.pt", beam_size: int = 5):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}, beam_size={beam_size}")

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

    dev = StaffCropDataset(
        "dev", vocab,
        img_height=128, img_width=1024,
        max_seq_len=600, augment=False,
    )
    print(f"Dev staff crops: {len(dev)}")

    tune_predictions: dict[str, list[list[str]]] = {}
    for i, sample in enumerate(dev.samples):
        img_tensor = dev[i][0].unsqueeze(0).to(device)
        token_ids = model.generate_beam(
            img_tensor,
            sos_idx=vocab.sos_idx,
            eos_idx=vocab.eos_idx,
            beam_size=beam_size,
            max_len=600,
        )
        token_strs = vocab.decode(token_ids)

        file_id = sample.get("file_id", "")
        if file_id not in tune_predictions:
            tune_predictions[file_id] = []
        tune_predictions[file_id].append(token_strs)

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
    commit_hash = "beam_search"
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
        description=f"Beam search beam={beam_size} ckpt={checkpoint_name}",
    )


if __name__ == "__main__":
    ckpt = "primus_finetuned_v2.pt"
    beam = 5
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith("beam="):
            beam = int(arg.split("=")[1])
        elif not arg.startswith("--"):
            ckpt = arg
    main(ckpt, beam)
