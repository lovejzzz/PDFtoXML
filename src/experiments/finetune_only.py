"""Fine-tune only: use existing PrIMuS-pretrained checkpoint, fine-tune on staff crops.

Skips the slow pre-training stage. Use this when iterating on:
- Better staff alignment
- Different fine-tuning strategies
- More epochs
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.experiments.dataset import Vocabulary, collate_fn
from src.experiments.staff_dataset import StaffCropDataset
from src.experiments.model import OMRModel
from src.experiments.decode import save_predictions
from src.eval import evaluate_all

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
PRED_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")
EVENTS_DIR = os.path.join(PROJECT_ROOT, "data", "events")
VOCAB_PATH = os.path.join(PROJECT_ROOT, "data", "vocab_combined.json")


def main(
    epochs: int = 80,
    batch_size: int = 16,
    lr: float = 5e-5,
    patience: int = 25,
    eval_every: int = 5,
    img_height: int = 128,
    img_width: int = 1024,
    max_seq_len: int = 600,
    pretrained_ckpt: str = "primus_pretrained.pt",
    output_ckpt: str = "primus_finetuned_v2.pt",
    desc: str = "FT-only",
    label_smoothing: float = 0.0,
):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    vocab = Vocabulary.load(VOCAB_PATH)
    print(f"Vocab size: {vocab.size}")

    # Load pretrained model
    print(f"Loading pretrained: {pretrained_ckpt}")
    ckpt = torch.load(
        os.path.join(CHECKPOINTS_DIR, pretrained_ckpt),
        map_location=device, weights_only=True,
    )

    model = OMRModel(
        vocab_size=vocab.size,
        d_model=256, nhead=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        max_seq_len=max(max_seq_len, 1400),
        pad_idx=vocab.pad_idx,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Datasets
    train_ds = StaffCropDataset(
        "train", vocab,
        img_height=img_height, img_width=img_width,
        max_seq_len=max_seq_len, augment=True,
    )
    dev_ds = StaffCropDataset(
        "dev", vocab,
        img_height=img_height, img_width=img_width,
        max_seq_len=max_seq_len, augment=False,
    )
    print(f"Train: {len(train_ds)}, Dev: {len(dev_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx,
        label_smoothing=label_smoothing,
    )

    best_dev = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tl, tt = 0, 0
        for imgs, tokens, lengths in train_loader:
            imgs = imgs.to(device)
            tokens = tokens.to(device)
            tgt_in, tgt_out = tokens[:, :-1], tokens[:, 1:]

            logits = model(imgs, tgt_in)
            loss = criterion(logits.reshape(-1, vocab.size), tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tl += loss.item() * tgt_out.numel()
            tt += (tgt_out != vocab.pad_idx).sum().item()

        scheduler.step()
        avg_train = tl / max(tt, 1)

        if epoch % eval_every == 0 or epoch == 1:
            model.eval()
            dl, dt = 0, 0
            with torch.no_grad():
                for imgs, tokens, lengths in dev_loader:
                    imgs = imgs.to(device)
                    tokens = tokens.to(device)
                    tgt_in, tgt_out = tokens[:, :-1], tokens[:, 1:]
                    logits = model(imgs, tgt_in)
                    loss = criterion(logits.reshape(-1, vocab.size), tgt_out.reshape(-1))
                    dl += loss.item() * tgt_out.numel()
                    dt += (tgt_out != vocab.pad_idx).sum().item()

            avg_dev = dl / max(dt, 1)
            print(f"  Epoch {epoch}/{epochs}  train={avg_train:.4f}  dev={avg_dev:.4f}")

            if avg_dev < best_dev:
                best_dev = avg_dev
                best_epoch = epoch
                no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab.size,
                    "dev_loss": avg_dev,
                }, os.path.join(CHECKPOINTS_DIR, output_ckpt))
            else:
                no_improve += 1

            if no_improve >= patience // eval_every:
                print(f"  Early stop at epoch {epoch} (best: {best_epoch})")
                break

    print(f"\nBest dev loss: {best_dev:.4f} at epoch {best_epoch}")

    # Evaluate
    print("\nEvaluating...")
    ckpt = torch.load(
        os.path.join(CHECKPOINTS_DIR, output_ckpt),
        map_location=device, weights_only=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tune_predictions: dict[str, list[list[str]]] = {}
    for i, sample in enumerate(dev_ds.samples):
        img_tensor = dev_ds[i][0].unsqueeze(0).to(device)
        token_ids = model.generate(
            img_tensor,
            sos_idx=vocab.sos_idx,
            eos_idx=vocab.eos_idx,
            max_len=max_seq_len,
        )[0]
        token_strs = vocab.decode(token_ids)

        file_id = sample.get("file_id", "")
        if file_id not in tune_predictions:
            tune_predictions[file_id] = []
        tune_predictions[file_id].append(token_strs)

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
    commit_hash = "ft_only"
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
        description=desc,
    )


if __name__ == "__main__":
    kwargs = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, val = arg.split("=", 1)
            key = key.lstrip("-")
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
            kwargs[key] = val
    main(**kwargs)
