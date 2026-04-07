"""Semi-supervised learning via pseudo-labeling.

Pipeline:
1. Run inference on unlabeled Omnibook pages using the best model
2. Score each prediction by average token confidence (softmax probability)
3. Filter out low-confidence predictions
4. Save pseudo-labels as token files for training
5. Support iterative self-training (predict → filter → retrain)

Also supports using ALL labeled pages (including multi-page tunes)
as additional training data with the model predicting per-page.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.experiments.dataset import (
    Vocabulary,
    PAGES_DIR,
    EVENTS_DIR,
    MANIFEST_DIR,
)
from src.experiments.model import OMRModel

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
PSEUDO_DIR = os.path.join(PROJECT_ROOT, "data", "pseudo_labels")
VOCAB_PATH = os.path.join(PROJECT_ROOT, "data", "vocab.json")


def _load_model(checkpoint_path: str, vocab: Vocabulary, device: torch.device) -> OMRModel:
    """Load model from checkpoint, inferring config from saved state."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Infer model config from checkpoint
    state_dict = checkpoint["model_state_dict"]
    config = checkpoint.get("config", {})

    # Infer d_model from embedding weight shape
    if "token_embed.weight" in state_dict:
        d_model = state_dict["token_embed.weight"].shape[1]
    else:
        d_model = config.get("d_model", 256)

    # Infer num_layers by counting decoder layer keys
    layer_indices = set()
    for key in state_dict:
        if key.startswith("decoder.layers."):
            idx = int(key.split(".")[2])
            layer_indices.add(idx)
    num_layers = len(layer_indices) if layer_indices else config.get("num_layers", 4)

    # Infer nhead and dim_ff
    nhead = config.get("nhead", 4)
    dim_ff = config.get("dim_ff", d_model * 2)

    model = OMRModel(
        vocab_size=vocab.size,
        d_model=d_model,
        nhead=nhead,
        num_decoder_layers=num_layers,
        dim_feedforward=dim_ff,
        max_seq_len=config.get("max_seq_len", 1400),
        pad_idx=vocab.pad_idx,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _preprocess_page(page_path: str, img_height: int = 256, img_width: int = 192) -> torch.Tensor:
    """Load and preprocess a page image."""
    img = Image.open(page_path).convert("L")
    # Resize preserving aspect ratio
    w, h = img.size
    scale = min(img_width / w, img_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    # Pad
    padded = Image.new("L", (img_width, img_height), 255)
    padded.paste(img, (0, 0))
    arr = np.array(padded, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


@torch.no_grad()
def predict_with_confidence(
    model: OMRModel,
    image: torch.Tensor,
    vocab: Vocabulary,
    max_len: int = 1400,
) -> tuple[list[str], float, list[float]]:
    """Generate prediction with per-token confidence scores.

    Returns: (token_strings, avg_confidence, per_token_confidences)
    """
    device = next(model.parameters()).device
    image = image.to(device)

    # Encode
    enc_out = model.encoder(image)
    enc_out = model.enc_pos(enc_out)

    # Greedy decode with confidence tracking
    generated = torch.full((1, 1), vocab.sos_idx, dtype=torch.long, device=device)
    confidences = []

    for _ in range(max_len - 1):
        tgt_emb = model.token_embed(generated) * math.sqrt(model.d_model)
        tgt_emb = model.dec_pos(tgt_emb)

        T = generated.size(1)
        causal_mask = model._make_causal_mask(T, device)

        dec_out = model.decoder(
            tgt=tgt_emb,
            memory=enc_out,
            tgt_mask=causal_mask,
        )

        logits = dec_out[:, -1, :]  # (1, vocab)
        probs = torch.softmax(logits, dim=-1)
        confidence, next_token = probs.max(dim=-1)

        confidences.append(confidence.item())
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        if next_token.item() == vocab.eos_idx:
            break

    # Convert to tokens, strip SOS/EOS
    token_ids = generated[0].tolist()[1:]  # remove SOS
    if vocab.eos_idx in token_ids:
        eos_pos = token_ids.index(vocab.eos_idx)
        token_ids = token_ids[:eos_pos]
        confidences = confidences[:eos_pos]

    token_strs = vocab.decode(token_ids)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return token_strs, avg_conf, confidences


def generate_pseudo_labels(
    checkpoint_path: str | None = None,
    confidence_threshold: float = 0.3,
    img_height: int = 256,
    img_width: int = 192,
):
    """Generate pseudo-labels for all unlabeled Omnibook pages.

    Also generates predictions for labeled pages (for analysis/comparison).
    """
    os.makedirs(PSEUDO_DIR, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load vocab and model
    vocab = Vocabulary.load(VOCAB_PATH)
    ckpt_path = checkpoint_path or os.path.join(CHECKPOINTS_DIR, "best.pt")
    print(f"Loading model from {ckpt_path}")
    model = _load_model(ckpt_path, vocab, device)

    # Find unlabeled pages
    with open(os.path.join(MANIFEST_DIR, "manual_page_map.json")) as f:
        page_map = json.load(f)

    used_pages = set()
    for entry in page_map.values():
        for p in entry["page_indices"]:
            used_pages.add(p)

    total_pages = len(list(Path(PAGES_DIR).glob("*.png")))
    unlabeled_pages = sorted(set(range(total_pages)) - used_pages)

    print(f"Total pages: {total_pages}, Labeled: {len(used_pages)}, "
          f"Unlabeled: {len(unlabeled_pages)}")

    # Generate predictions for unlabeled pages
    results = []
    kept = 0

    for page_idx in unlabeled_pages:
        page_path = os.path.join(PAGES_DIR, f"page_{page_idx + 1:03d}.png")
        if not os.path.exists(page_path):
            continue

        image = _preprocess_page(page_path, img_height, img_width)
        tokens, avg_conf, per_token_conf = predict_with_confidence(
            model, image, vocab,
        )

        result = {
            "page_index": page_idx,
            "page_path": page_path,
            "num_tokens": len(tokens),
            "avg_confidence": avg_conf,
            "min_confidence": min(per_token_conf) if per_token_conf else 0,
            "kept": avg_conf >= confidence_threshold,
        }
        results.append(result)

        status = "KEEP" if result["kept"] else "SKIP"
        print(f"  page_{page_idx + 1:03d}: {len(tokens)} tokens, "
              f"conf={avg_conf:.3f} [{status}]")

        if result["kept"]:
            # Save pseudo-label
            pseudo_id = f"pseudo_page_{page_idx + 1:03d}"
            token_path = os.path.join(PSEUDO_DIR, f"{pseudo_id}.tokens")
            with open(token_path, "w") as f:
                f.write(" ".join(tokens))

            # Save metadata
            meta_path = os.path.join(PSEUDO_DIR, f"{pseudo_id}.meta.json")
            with open(meta_path, "w") as f:
                json.dump({
                    "page_index": page_idx,
                    "avg_confidence": avg_conf,
                    "num_tokens": len(tokens),
                    "per_token_confidence": per_token_conf,
                }, f, indent=2)

            kept += 1

    # Save summary
    summary = {
        "total_unlabeled": len(unlabeled_pages),
        "kept": kept,
        "skipped": len(unlabeled_pages) - kept,
        "confidence_threshold": confidence_threshold,
        "checkpoint": ckpt_path,
        "results": results,
    }
    summary_path = os.path.join(PSEUDO_DIR, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPseudo-labeling complete:")
    print(f"  Kept: {kept}/{len(unlabeled_pages)} "
          f"(threshold={confidence_threshold})")
    print(f"  Saved to: {PSEUDO_DIR}")

    return summary


def iterative_self_train(
    num_rounds: int = 3,
    initial_threshold: float = 0.4,
    threshold_decay: float = 0.05,
    epochs_per_round: int = 40,
):
    """Iterative self-training loop.

    Each round:
    1. Generate pseudo-labels with current best model
    2. Retrain with labeled + pseudo-labeled data
    3. Evaluate on dev set
    4. Lower confidence threshold slightly for next round
    """
    print("=" * 60)
    print("ITERATIVE SELF-TRAINING")
    print("=" * 60)

    for round_num in range(1, num_rounds + 1):
        threshold = max(0.2, initial_threshold - (round_num - 1) * threshold_decay)

        print(f"\n--- Round {round_num}/{num_rounds} (threshold={threshold:.2f}) ---")

        # Step 1: Generate pseudo-labels
        print("\nStep 1: Generating pseudo-labels...")
        summary = generate_pseudo_labels(confidence_threshold=threshold)

        if summary["kept"] == 0:
            print("  No pseudo-labels above threshold, stopping.")
            break

        # Step 2: Retrain with pseudo-labels
        print(f"\nStep 2: Retraining with {summary['kept']} pseudo-labels...")

        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "src.experiments.train",
                f"epochs={epochs_per_round}",
                "batch_size=8",
                "lr=0.0002",
                "img_height=256",
                "img_width=192",
                "use_synthetic=1",
                "eval_every=5",
                "patience=15",
                "use_pseudo=1",
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=3600,
        )

        print(result.stdout[-500:] if result.stdout else "No output")
        if result.returncode != 0:
            print(f"  Training failed: {result.stderr[-300:]}")
            break

        print(f"\n  Round {round_num} complete.")

    print("\n" + "=" * 60)
    print("SELF-TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    if "--generate" in sys.argv:
        threshold = 0.3
        for arg in sys.argv:
            if arg.startswith("--threshold="):
                threshold = float(arg.split("=")[1])
        generate_pseudo_labels(confidence_threshold=threshold)
    elif "--self-train" in sys.argv:
        rounds = 3
        for arg in sys.argv:
            if arg.startswith("--rounds="):
                rounds = int(arg.split("=")[1])
        iterative_self_train(num_rounds=rounds)
    else:
        print("Usage:")
        print("  python -m src.experiments.pseudo_label --generate [--threshold=0.3]")
        print("  python -m src.experiments.pseudo_label --self-train [--rounds=3]")
