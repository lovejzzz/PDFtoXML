"""Dataset and vocabulary for system-image → event-sequence training.

Loads page images and canonical token sequences, with augmentation.
Multi-page tunes use the first page only (with full target sequence).
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset

EVENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "events")
PAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "pages")
MANIFEST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data_manifest")

# Special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class Vocabulary:
    """Token vocabulary with encode/decode."""

    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self.size = 0

        for tok in SPECIAL_TOKENS:
            self._add(tok)

    def _add(self, token: str) -> int:
        if token not in self.token2idx:
            idx = self.size
            self.token2idx[token] = idx
            self.idx2token[idx] = token
            self.size += 1
        return self.token2idx[token]

    @property
    def pad_idx(self) -> int:
        return self.token2idx[PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.token2idx[SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[EOS_TOKEN]

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.token2idx[UNK_TOKEN]
        return [self.token2idx.get(t, unk) for t in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        return [self.idx2token.get(i, UNK_TOKEN) for i in indices]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"token2idx": self.token2idx}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        v = cls.__new__(cls)
        with open(path) as f:
            data = json.load(f)
        v.token2idx = data["token2idx"]
        v.idx2token = {int(i): t for t, i in v.token2idx.items()}
        v.size = len(v.token2idx)
        return v


def build_vocabulary(events_dir: str = EVENTS_DIR) -> Vocabulary:
    """Build vocabulary from all token files."""
    vocab = Vocabulary()
    for f in sorted(Path(events_dir).glob("*.tokens")):
        with open(f) as fh:
            for tok in fh.read().strip().split():
                vocab._add(tok)
    return vocab


SYNTH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic")
PSEUDO_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "pseudo_labels")


class OMRDataset(Dataset):
    """Dataset for OMR training.

    Each sample: (image_tensor, token_indices, token_length)
    Supports real page images, synthetic rendered images, and pseudo-labels.
    """

    def __init__(
        self,
        split: str,
        vocab: Vocabulary,
        img_height: int = 512,
        img_width: int = 384,
        max_seq_len: int = 1400,
        augment: bool = False,
        use_synthetic: bool = False,
        use_pseudo: bool = False,
    ):
        self.vocab = vocab
        self.img_height = img_height
        self.img_width = img_width
        self.max_seq_len = max_seq_len
        self.augment = augment

        # Load splits
        splits_path = os.path.join(MANIFEST_DIR, "splits.json")
        with open(splits_path) as f:
            all_splits = json.load(f)

        # Load manifest for page mapping
        page_map_path = os.path.join(MANIFEST_DIR, "manual_page_map.json")
        with open(page_map_path) as f:
            page_map = json.load(f)

        # Filter to this split
        self.samples = []
        split_ids = {fid for fid, s in all_splits.items() if s == split}

        # Real page images — first page of each tune
        for file_id in sorted(split_ids):
            token_path = os.path.join(EVENTS_DIR, f"{file_id}.tokens")
            if not os.path.exists(token_path):
                continue

            if file_id not in page_map:
                continue

            first_page_idx = page_map[file_id]["page_indices"][0]
            page_path = os.path.join(PAGES_DIR, f"page_{first_page_idx + 1:03d}.png")
            if not os.path.exists(page_path):
                continue

            with open(token_path) as f:
                tokens = f.read().strip().split()

            self.samples.append({
                "file_id": file_id,
                "page_path": page_path,
                "tokens": tokens,
                "provenance": "real",
            })

        # Synthetic images (train split only)
        if use_synthetic and split == "train":
            synth_manifest_path = os.path.join(SYNTH_DIR, "manifest.json")
            if os.path.exists(synth_manifest_path):
                with open(synth_manifest_path) as f:
                    synth_manifest = json.load(f)

                for entry in synth_manifest:
                    source_id = entry["source_id"]
                    if source_id not in split_ids:
                        continue
                    if not os.path.exists(entry["image_path"]):
                        continue
                    if not os.path.exists(entry["token_path"]):
                        continue

                    with open(entry["token_path"]) as f:
                        tokens = f.read().strip().split()

                    self.samples.append({
                        "file_id": entry["id"],
                        "page_path": entry["image_path"],
                        "tokens": tokens,
                        "provenance": "synthetic",
                    })

        # Pseudo-labeled pages (train split only)
        if use_pseudo and split == "train":
            pseudo_summary = os.path.join(PSEUDO_DIR, "_summary.json")
            if os.path.exists(pseudo_summary):
                for token_file in sorted(Path(PSEUDO_DIR).glob("*.tokens")):
                    pseudo_id = token_file.stem
                    # Extract page index from pseudo_id (format: pseudo_page_NNN)
                    try:
                        page_num = int(pseudo_id.split("_")[-1])
                        page_path = os.path.join(PAGES_DIR, f"page_{page_num:03d}.png")
                    except (ValueError, IndexError):
                        continue

                    if not os.path.exists(page_path):
                        continue

                    with open(token_file) as f:
                        tokens = f.read().strip().split()

                    if tokens:
                        self.samples.append({
                            "file_id": pseudo_id,
                            "page_path": page_path,
                            "tokens": tokens,
                            "provenance": "pseudo",
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and preprocess image
        img = Image.open(sample["page_path"]).convert("L")  # grayscale

        if self.augment:
            # Use scan-like augmentation for synthetic images to bridge domain gap
            if sample.get("provenance") == "synthetic" and random.random() < 0.7:
                from src.experiments.scan_augment import scan_augment
                img = scan_augment(img)
            else:
                img = self._augment(img)

        # Resize preserving aspect ratio, pad to fixed size
        img = self._resize_pad(img)

        # To tensor, normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)

        # Encode tokens with SOS/EOS
        token_ids = (
            [self.vocab.sos_idx]
            + self.vocab.encode(sample["tokens"])
            + [self.vocab.eos_idx]
        )

        # Truncate if needed
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len - 1] + [self.vocab.eos_idx]

        seq_len = len(token_ids)

        # Pad
        token_ids = token_ids + [self.vocab.pad_idx] * (self.max_seq_len - seq_len)
        token_tensor = torch.tensor(token_ids, dtype=torch.long)

        return img_tensor, token_tensor, seq_len

    def _resize_pad(self, img: Image.Image) -> Image.Image:
        """Resize preserving aspect ratio, then pad to target size."""
        w, h = img.size
        scale = min(self.img_width / w, self.img_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Pad to target size (white background)
        padded = Image.new("L", (self.img_width, self.img_height), 255)
        padded.paste(img, (0, 0))
        return padded

    def _augment(self, img: Image.Image) -> Image.Image:
        """Apply random augmentations."""
        # Random rotation (small)
        if random.random() < 0.3:
            angle = random.uniform(-2, 2)
            img = img.rotate(angle, fillcolor=255, expand=False)

        # Random brightness
        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.2)
            img = ImageEnhance.Brightness(img).enhance(factor)

        # Random contrast
        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.2)
            img = ImageEnhance.Contrast(img).enhance(factor)

        # Random blur
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # Random noise
        if random.random() < 0.2:
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, 5, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        return img


def collate_fn(batch):
    """Custom collate that handles variable-length sequences."""
    imgs, tokens, lengths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    tokens = torch.stack(tokens, dim=0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return imgs, tokens, lengths
