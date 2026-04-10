"""Dataset for staff-level OMR training.

Loads individual staff crops from the Omnibook (extracted by extract_staffs.py)
with their per-staff token sequences. Compatible with PrIMuS pre-trained models
since both use single-staff images.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CROPS_DIR = os.path.join(PROJECT_ROOT, "data", "staff_crops")


class StaffCropDataset(Dataset):
    """Dataset of individual staff crops from the Omnibook.

    Each sample: (image_tensor, token_indices, token_length)
    Images are single-staff crops (same format as PrIMuS).
    """

    def __init__(
        self,
        split: str,
        vocab,
        img_height: int = 128,
        img_width: int = 1024,
        max_seq_len: int = 600,
        augment: bool = False,
    ):
        self.vocab = vocab
        self.img_height = img_height
        self.img_width = img_width
        self.max_seq_len = max_seq_len
        self.augment = augment

        # Use cleaned manifest if available, otherwise original
        cleaned_path = os.path.join(CROPS_DIR, "manifest_cleaned.json")
        manifest_path = cleaned_path if os.path.exists(cleaned_path) else os.path.join(CROPS_DIR, "manifest.json")
        if not os.path.exists(manifest_path):
            self.samples = []
            return

        with open(manifest_path) as f:
            manifest = json.load(f)

        self.samples = [
            m for m in manifest
            if m["split"] == split
            and os.path.exists(m["crop_path"])
            and os.path.exists(m["token_path"])
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        img = Image.open(sample["crop_path"]).convert("L")

        if self.augment:
            from src.experiments.scan_augment import scan_augment
            if random.random() < 0.3:
                img = scan_augment(img)

        # Resize to fixed height, pad width
        img = self._resize_pad(img)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        # Load tokens
        with open(sample["token_path"]) as f:
            tokens = f.read().strip().split()

        token_ids = (
            [self.vocab.sos_idx]
            + self.vocab.encode(tokens)
            + [self.vocab.eos_idx]
        )

        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len - 1] + [self.vocab.eos_idx]

        seq_len = len(token_ids)
        token_ids = token_ids + [self.vocab.pad_idx] * (self.max_seq_len - seq_len)
        token_tensor = torch.tensor(token_ids, dtype=torch.long)

        return img_tensor, token_tensor, seq_len

    def _resize_pad(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.img_height / h
        new_w = min(int(w * scale), self.img_width)
        new_h = self.img_height
        img = img.resize((new_w, new_h), Image.LANCZOS)
        padded = Image.new("L", (self.img_width, self.img_height), 255)
        padded.paste(img, (0, 0))
        return padded
