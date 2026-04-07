"""Training script for system-image → event-sequence model.

This is the primary editable research surface. See program.md for rules.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.experiments.dataset import (
    OMRDataset,
    Vocabulary,
    build_vocabulary,
    collate_fn,
    EVENTS_DIR,
)
from src.experiments.model import OMRModel
from src.experiments.decode import tokens_to_score, save_predictions
from src.eval import evaluate_all

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
PRED_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")
VOCAB_PATH = os.path.join(PROJECT_ROOT, "data", "vocab.json")


def train(
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 3e-4,
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 4,
    dim_ff: int = 512,
    dropout: float = 0.1,
    img_height: int = 512,
    img_width: int = 384,
    max_seq_len: int = 1400,
    eval_every: int = 10,
    patience: int = 20,
    device_name: str = "auto",
    use_synthetic: int = 0,
):
    """Train the OMR model."""
    # Device
    if device_name == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    print(f"Using device: {device}")

    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocabulary()
    vocab.save(VOCAB_PATH)
    print(f"  Vocab size: {vocab.size}")

    # Datasets
    print("Loading datasets...")
    train_ds = OMRDataset(
        "train", vocab,
        img_height=img_height, img_width=img_width,
        max_seq_len=max_seq_len, augment=True,
        use_synthetic=bool(use_synthetic),
    )
    dev_ds = OMRDataset(
        "dev", vocab,
        img_height=img_height, img_width=img_width,
        max_seq_len=max_seq_len, augment=False,
    )
    print(f"  Train: {len(train_ds)} samples, Dev: {len(dev_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Model
    model = OMRModel(
        vocab_size=vocab.size,
        d_model=d_model,
        nhead=nhead,
        num_decoder_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=dropout,
        max_seq_len=max_seq_len,
        pad_idx=vocab.pad_idx,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    best_dev_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_tokens = 0

        for imgs, tokens, lengths in train_loader:
            imgs = imgs.to(device)
            tokens = tokens.to(device)

            # Teacher forcing: input is tokens[:-1], target is tokens[1:]
            tgt_input = tokens[:, :-1]
            tgt_output = tokens[:, 1:]

            logits = model(imgs, tgt_input)  # (B, T-1, vocab)

            loss = criterion(
                logits.reshape(-1, vocab.size),
                tgt_output.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * tgt_output.numel()
            train_tokens += (tgt_output != vocab.pad_idx).sum().item()

        scheduler.step()
        avg_train_loss = train_loss / max(train_tokens, 1)

        # Evaluate on dev set periodically
        if epoch % eval_every == 0 or epoch == 1:
            model.eval()
            dev_loss = 0
            dev_tokens = 0

            with torch.no_grad():
                for imgs, tokens, lengths in dev_loader:
                    imgs = imgs.to(device)
                    tokens = tokens.to(device)

                    tgt_input = tokens[:, :-1]
                    tgt_output = tokens[:, 1:]

                    logits = model(imgs, tgt_input)
                    loss = criterion(
                        logits.reshape(-1, vocab.size),
                        tgt_output.reshape(-1),
                    )

                    dev_loss += loss.item() * tgt_output.numel()
                    dev_tokens += (tgt_output != vocab.pad_idx).sum().item()

            avg_dev_loss = dev_loss / max(dev_tokens, 1)

            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={avg_train_loss:.4f}  "
                  f"dev_loss={avg_dev_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

            if avg_dev_loss < best_dev_loss:
                best_dev_loss = avg_dev_loss
                best_epoch = epoch
                no_improve = 0
                # Save best checkpoint
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dev_loss": avg_dev_loss,
                    "vocab_size": vocab.size,
                    "config": {
                        "d_model": d_model, "nhead": nhead,
                        "num_layers": num_layers, "dim_ff": dim_ff,
                        "dropout": dropout, "img_height": img_height,
                        "img_width": img_width, "max_seq_len": max_seq_len,
                    },
                }, os.path.join(CHECKPOINTS_DIR, "best.pt"))
            else:
                no_improve += 1

            if no_improve >= patience // eval_every:
                print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
                break
        else:
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}/{epochs}  train_loss={avg_train_loss:.4f}")

    print(f"\nBest dev loss: {best_dev_loss:.4f} at epoch {best_epoch}")

    # Stage 2: Fine-tune on real data only (if trained with synthetic)
    if use_synthetic:
        print(f"\n--- Stage 2: Fine-tuning on real data only ---")
        real_ds = OMRDataset(
            "train", vocab,
            img_height=img_height, img_width=img_width,
            max_seq_len=max_seq_len, augment=True,
            use_synthetic=False,
        )
        real_loader = DataLoader(
            real_ds, batch_size=max(1, batch_size // 2), shuffle=True,
            collate_fn=collate_fn, num_workers=0,
        )
        print(f"  Real train samples: {len(real_ds)}")

        # Load best pretrained checkpoint
        checkpoint = torch.load(
            os.path.join(CHECKPOINTS_DIR, "best.pt"),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        ft_optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=0.01)
        ft_epochs = min(30, epochs // 2)
        ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ft_optimizer, T_max=ft_epochs)

        ft_best_dev_loss = best_dev_loss
        ft_best_epoch = 0
        ft_no_improve = 0

        for epoch in range(1, ft_epochs + 1):
            model.train()
            ft_train_loss = 0
            ft_train_tokens = 0

            for imgs, tokens, lengths in real_loader:
                imgs = imgs.to(device)
                tokens = tokens.to(device)
                tgt_input = tokens[:, :-1]
                tgt_output = tokens[:, 1:]

                logits = model(imgs, tgt_input)
                loss = criterion(
                    logits.reshape(-1, vocab.size),
                    tgt_output.reshape(-1),
                )

                ft_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                ft_optimizer.step()

                ft_train_loss += loss.item() * tgt_output.numel()
                ft_train_tokens += (tgt_output != vocab.pad_idx).sum().item()

            ft_scheduler.step()

            if epoch % 5 == 0 or epoch == 1:
                model.eval()
                dev_loss = 0
                dev_tokens = 0
                with torch.no_grad():
                    for imgs, tokens, lengths in dev_loader:
                        imgs = imgs.to(device)
                        tokens = tokens.to(device)
                        tgt_input = tokens[:, :-1]
                        tgt_output = tokens[:, 1:]
                        logits = model(imgs, tgt_input)
                        loss = criterion(
                            logits.reshape(-1, vocab.size),
                            tgt_output.reshape(-1),
                        )
                        dev_loss += loss.item() * tgt_output.numel()
                        dev_tokens += (tgt_output != vocab.pad_idx).sum().item()

                avg_ft_dev = dev_loss / max(dev_tokens, 1)
                avg_ft_train = ft_train_loss / max(ft_train_tokens, 1)
                print(f"  FT Epoch {epoch:3d}/{ft_epochs}  "
                      f"train_loss={avg_ft_train:.4f}  "
                      f"dev_loss={avg_ft_dev:.4f}")

                if avg_ft_dev < ft_best_dev_loss:
                    ft_best_dev_loss = avg_ft_dev
                    ft_best_epoch = epoch
                    ft_no_improve = 0
                    torch.save({
                        "epoch": best_epoch + epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": ft_optimizer.state_dict(),
                        "dev_loss": avg_ft_dev,
                        "vocab_size": vocab.size,
                        "config": checkpoint.get("config", {}),
                    }, os.path.join(CHECKPOINTS_DIR, "best.pt"))
                else:
                    ft_no_improve += 1

                if ft_no_improve >= 4:
                    print(f"  FT early stop at epoch {epoch} (best: {ft_best_epoch})")
                    break

        best_dev_loss = ft_best_dev_loss
        best_epoch = best_epoch + ft_best_epoch
        print(f"  FT best dev loss: {ft_best_dev_loss:.4f} at FT epoch {ft_best_epoch}")

    # Generate predictions on dev set with best model
    print("\nGenerating dev predictions with best model...")
    checkpoint = torch.load(
        os.path.join(CHECKPOINTS_DIR, "best.pt"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = {}
    for i, sample in enumerate(dev_ds.samples):
        img_tensor = dev_ds[i][0].unsqueeze(0).to(device)
        token_ids = model.generate(
            img_tensor,
            sos_idx=vocab.sos_idx,
            eos_idx=vocab.eos_idx,
            max_len=max_seq_len,
        )[0]
        token_strs = vocab.decode(token_ids)
        predictions[sample["file_id"]] = token_strs

    # Save predictions
    save_predictions(predictions, PRED_DIR)
    print(f"  Predictions saved to {PRED_DIR}")

    # Evaluate
    print("\nEvaluating on dev set...")
    commit_hash = "baseline_v1"
    try:
        import subprocess
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
        description=f"CNN-Transformer baseline d={d_model} L={num_layers} ep={best_epoch}",
    )


if __name__ == "__main__":
    # Parse simple CLI overrides
    kwargs = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, val = arg.split("=", 1)
            key = key.lstrip("-")
            # Try int, then float, then string
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
            kwargs[key] = val

    train(**kwargs)
