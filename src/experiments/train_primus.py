"""Pre-train on PrIMuS/CameraPrIMuS, then fine-tune on Omnibook.

Strategy:
1. Build combined vocabulary (PrIMuS + Omnibook tokens)
2. Pre-train CNN-Transformer on 87K PrIMuS system images
3. Fine-tune on Omnibook synthetic renders + real scans
4. Evaluate on dev set

PrIMuS images are single-staff incipits (128xW), matching the guide's
recommended "cropped system → event sequence" baseline.
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from src.experiments.dataset import (
    Vocabulary,
    OMRDataset,
    build_vocabulary,
    collate_fn,
    EVENTS_DIR,
)
from src.experiments.primus_loader import (
    PrIMuSDataset,
    build_primus_vocabulary,
    convert_semantic_to_tokens,
)
from src.experiments.model import OMRModel
from src.experiments.decode import save_predictions
from src.eval import evaluate_all

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
PRED_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")
VOCAB_PATH = os.path.join(PROJECT_ROOT, "data", "vocab_combined.json")


def build_combined_vocabulary() -> Vocabulary:
    """Build vocabulary from both Omnibook and PrIMuS tokens."""
    vocab = Vocabulary()

    # Omnibook tokens
    from pathlib import Path
    for f in sorted(Path(EVENTS_DIR).glob("*.tokens")):
        with open(f) as fh:
            for tok in fh.read().strip().split():
                vocab._add(tok)

    # PrIMuS tokens (sample first 10K for speed)
    primus_tokens = build_primus_vocabulary(max_samples=10000)
    for tok in sorted(primus_tokens):
        vocab._add(tok)

    return vocab


def train_primus(
    # Pre-training params
    pretrain_epochs: int = 15,
    pretrain_lr: float = 3e-4,
    pretrain_batch: int = 32,
    pretrain_samples: int = 87000,
    # Fine-tune params
    finetune_epochs: int = 60,
    finetune_lr: float = 1e-4,
    finetune_batch: int = 8,
    # Model params
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 4,
    dim_ff: int = 512,
    dropout: float = 0.1,
    # PrIMuS image size (single staff)
    primus_height: int = 128,
    primus_width: int = 1024,
    primus_seq_len: int = 600,
    # Omnibook image size (full page)
    omni_height: int = 256,
    omni_width: int = 192,
    omni_seq_len: int = 1400,
    # Device
    device_name: str = "auto",
):
    # Device
    if device_name == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    # Build combined vocabulary
    print("Building combined vocabulary...")
    vocab = build_combined_vocabulary()
    vocab.save(VOCAB_PATH)
    print(f"  Combined vocab size: {vocab.size}")

    # ============================================================
    # STAGE 1: Pre-train on PrIMuS
    # ============================================================
    print(f"\n{'='*60}")
    print("STAGE 1: Pre-training on PrIMuS ({} samples)".format(pretrain_samples))
    print(f"{'='*60}")

    primus_train = PrIMuSDataset(
        vocab, img_height=primus_height, img_width=primus_width,
        max_seq_len=primus_seq_len, max_samples=pretrain_samples,
        use_distorted=True, augment=True, split="train",
    )
    primus_val = PrIMuSDataset(
        vocab, img_height=primus_height, img_width=primus_width,
        max_seq_len=primus_seq_len, max_samples=pretrain_samples,
        use_distorted=True, augment=False, split="val",
    )
    print(f"  PrIMuS train: {len(primus_train)}, val: {len(primus_val)}")

    primus_train_loader = DataLoader(
        primus_train, batch_size=pretrain_batch, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    primus_val_loader = DataLoader(
        primus_val, batch_size=pretrain_batch, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Build model with PrIMuS image dimensions
    # Note: we'll need to handle the size mismatch when fine-tuning on Omnibook
    model = OMRModel(
        vocab_size=vocab.size,
        d_model=d_model, nhead=nhead,
        num_decoder_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=dropout,
        max_seq_len=max(primus_seq_len, omni_seq_len),
        pad_idx=vocab.pad_idx,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=pretrain_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        train_loss = 0
        train_tokens = 0

        for batch_idx, (imgs, tokens, lengths) in enumerate(primus_train_loader):
            imgs = imgs.to(device)
            tokens = tokens.to(device)

            tgt_input = tokens[:, :-1]
            tgt_output = tokens[:, 1:]

            logits = model(imgs, tgt_input)
            loss = criterion(logits.reshape(-1, vocab.size), tgt_output.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * tgt_output.numel()
            train_tokens += (tgt_output != vocab.pad_idx).sum().item()

            if (batch_idx + 1) % 100 == 0:
                avg = train_loss / max(train_tokens, 1)
                print(f"    Batch {batch_idx+1}/{len(primus_train_loader)} loss={avg:.4f}")

        scheduler.step()
        avg_train = train_loss / max(train_tokens, 1)

        # Validate
        model.eval()
        val_loss = 0
        val_tokens = 0
        with torch.no_grad():
            for imgs, tokens, lengths in primus_val_loader:
                imgs = imgs.to(device)
                tokens = tokens.to(device)
                tgt_input = tokens[:, :-1]
                tgt_output = tokens[:, 1:]
                logits = model(imgs, tgt_input)
                loss = criterion(logits.reshape(-1, vocab.size), tgt_output.reshape(-1))
                val_loss += loss.item() * tgt_output.numel()
                val_tokens += (tgt_output != vocab.pad_idx).sum().item()

        avg_val = val_loss / max(val_tokens, 1)
        print(f"  Epoch {epoch}/{pretrain_epochs}  train={avg_train:.4f}  val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab.size,
                "val_loss": avg_val,
                "stage": "pretrain",
            }, os.path.join(CHECKPOINTS_DIR, "primus_pretrained.pt"))

    print(f"\n  Pre-training done. Best val loss: {best_val_loss:.4f}")

    # ============================================================
    # STAGE 2: Fine-tune on Omnibook
    # ============================================================
    print(f"\n{'='*60}")
    print("STAGE 2: Fine-tuning on Omnibook")
    print(f"{'='*60}")

    # Load best pretrained checkpoint
    ckpt = torch.load(
        os.path.join(CHECKPOINTS_DIR, "primus_pretrained.pt"),
        map_location=device, weights_only=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    # Use staff crops if available, otherwise full pages
    from src.experiments.staff_dataset import StaffCropDataset
    staff_manifest = os.path.join(PROJECT_ROOT, "data", "staff_crops", "manifest.json")
    if os.path.exists(staff_manifest):
        print("  Using staff-level crops (aligned with PrIMuS format)")
        omni_train = StaffCropDataset(
            "train", vocab,
            img_height=primus_height, img_width=primus_width,
            max_seq_len=primus_seq_len, augment=True,
        )
        omni_dev = StaffCropDataset(
            "dev", vocab,
            img_height=primus_height, img_width=primus_width,
            max_seq_len=primus_seq_len, augment=False,
        )
    else:
        print("  No staff crops found, using full pages")
        omni_train = OMRDataset(
            "train", vocab,
            img_height=omni_height, img_width=omni_width,
            max_seq_len=omni_seq_len, augment=True,
            use_synthetic=True,
        )
        omni_dev = OMRDataset(
            "dev", vocab,
            img_height=omni_height, img_width=omni_width,
            max_seq_len=omni_seq_len, augment=False,
        )
    print(f"  Omnibook train: {len(omni_train)}, dev: {len(omni_dev)}")

    omni_train_loader = DataLoader(
        omni_train, batch_size=finetune_batch, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    omni_dev_loader = DataLoader(
        omni_dev, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    ft_optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=0.01)
    ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ft_optimizer, T_max=finetune_epochs)

    best_dev_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, finetune_epochs + 1):
        model.train()
        train_loss = 0
        train_tokens = 0

        for imgs, tokens, lengths in omni_train_loader:
            imgs = imgs.to(device)
            tokens = tokens.to(device)

            tgt_input = tokens[:, :-1]
            tgt_output = tokens[:, 1:]

            logits = model(imgs, tgt_input)
            loss = criterion(logits.reshape(-1, vocab.size), tgt_output.reshape(-1))

            ft_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ft_optimizer.step()

            train_loss += loss.item() * tgt_output.numel()
            train_tokens += (tgt_output != vocab.pad_idx).sum().item()

        ft_scheduler.step()
        avg_train = train_loss / max(train_tokens, 1)

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            dev_loss = 0
            dev_tokens = 0
            with torch.no_grad():
                for imgs, tokens, lengths in omni_dev_loader:
                    imgs = imgs.to(device)
                    tokens = tokens.to(device)
                    tgt_input = tokens[:, :-1]
                    tgt_output = tokens[:, 1:]
                    logits = model(imgs, tgt_input)
                    loss = criterion(logits.reshape(-1, vocab.size), tgt_output.reshape(-1))
                    dev_loss += loss.item() * tgt_output.numel()
                    dev_tokens += (tgt_output != vocab.pad_idx).sum().item()

            avg_dev = dev_loss / max(dev_tokens, 1)
            print(f"  FT Epoch {epoch}/{finetune_epochs}  train={avg_train:.4f}  dev={avg_dev:.4f}")

            if avg_dev < best_dev_loss:
                best_dev_loss = avg_dev
                best_epoch = epoch
                no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab.size,
                    "dev_loss": avg_dev,
                    "stage": "finetune",
                }, os.path.join(CHECKPOINTS_DIR, "primus_finetuned.pt"))
            else:
                no_improve += 1

            if no_improve >= 4:
                print(f"  Early stop at epoch {epoch} (best: {best_epoch})")
                break

    print(f"\n  Fine-tuning done. Best dev loss: {best_dev_loss:.4f} at epoch {best_epoch}")

    # ============================================================
    # STAGE 3: Evaluate
    # ============================================================
    print(f"\n{'='*60}")
    print("STAGE 3: Evaluation")
    print(f"{'='*60}")

    ckpt = torch.load(
        os.path.join(CHECKPOINTS_DIR, "primus_finetuned.pt"),
        map_location=device, weights_only=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    predictions = {}
    for i, sample in enumerate(omni_dev.samples):
        img_tensor = omni_dev[i][0].unsqueeze(0).to(device)
        token_ids = model.generate(
            img_tensor,
            sos_idx=vocab.sos_idx,
            eos_idx=vocab.eos_idx,
            max_len=omni_seq_len,
        )[0]
        token_strs = vocab.decode(token_ids)
        predictions[sample["file_id"]] = token_strs

    save_predictions(predictions, PRED_DIR)

    import subprocess
    commit_hash = "primus_pretrained"
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
        description=f"PrIMuS pretrain {pretrain_samples}→FT d={d_model} L={num_layers} ep={best_epoch}",
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
    train_primus(**kwargs)
