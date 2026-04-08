"""PrIMuS/CameraPrIMuS dataset loader.

Converts the PrIMuS semantic encoding to our canonical token format
and provides a PyTorch dataset for pre-training.

PrIMuS semantic format: clef-G2 keySignature-EbM timeSignature-3/4 note-Bb5_quarter ...
Our format:             CLEF_G KEY_-3 TIME_3_4 NOTE_Bb5_QUARTER ...
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PRIMUS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "external", "primus", "Corpus"
)

# Key signature mapping (PrIMuS name → fifths)
KEY_MAP = {
    "CM": 0, "GM": 1, "DM": 2, "AM": 3, "EM": 4, "BM": 5, "F#M": 6, "C#M": 7,
    "FM": -1, "BbM": -2, "EbM": -3, "AbM": -4, "DbM": -5, "GbM": -6, "CbM": -7,
    "Am": 0, "Em": 1, "Bm": 2, "F#m": 3, "C#m": 4, "G#m": 5, "D#m": 6, "A#m": 7,
    "Dm": -1, "Gm": -2, "Cm": -3, "Fm": -4, "Bbm": -5, "Ebm": -6, "Abm": -7,
}

# Duration mapping
DURATION_MAP = {
    "whole": "WHOLE", "half": "HALF", "quarter": "QUARTER",
    "eighth": "EIGHTH", "sixteenth": "16TH", "thirty_second": "32ND",
    "sixty_fourth": "64TH",
    "whole.": "WHOLE_DOT", "half.": "HALF_DOT", "quarter.": "QUARTER_DOT",
    "eighth.": "EIGHTH_DOT", "sixteenth.": "16TH_DOT",
}


def convert_semantic_to_tokens(semantic: str) -> list[str]:
    """Convert PrIMuS semantic encoding to our canonical token format."""
    tokens = []
    parts = semantic.strip().split("\t")

    in_measure = False

    for part in parts:
        if part.startswith("clef-"):
            clef = part.split("-")[1]
            sign = clef[0]  # G, F, C
            tokens.append(f"CLEF_{sign}")

        elif part.startswith("keySignature-"):
            key_name = part.split("-")[1]
            fifths = KEY_MAP.get(key_name, 0)
            tokens.append(f"KEY_{fifths}")

        elif part.startswith("timeSignature-"):
            ts = part.split("-")[1]
            if "/" in ts:
                beats, beat_type = ts.split("/")
                tokens.append(f"TIME_{beats}_{beat_type}")

        elif part.startswith("note-"):
            if not in_measure:
                tokens.append("MEASURE_START")
                in_measure = True

            note_info = part.split("-")[1]
            # Format: Bb5_quarter or C#4_eighth.
            if "_" in note_info:
                pitch, duration = note_info.rsplit("_", 1)
                dur_upper = DURATION_MAP.get(duration, duration.upper())
                tokens.append(f"NOTE_{pitch}_{dur_upper}")

        elif part.startswith("rest-"):
            if not in_measure:
                tokens.append("MEASURE_START")
                in_measure = True

            rest_info = part.split("-")[1]
            dur_upper = DURATION_MAP.get(rest_info, rest_info.upper())
            tokens.append(f"REST_{dur_upper}")

        elif part == "barline":
            tokens.append("BARLINE")
            in_measure = False

        elif part.startswith("multirest"):
            pass  # skip multi-measure rests

        elif part.startswith("tie"):
            pass  # skip ties for now (Tier 2)

        elif part.startswith("gracenote"):
            pass  # skip grace notes

    # Close last measure if open
    if in_measure:
        tokens.append("BARLINE")

    return tokens


def build_primus_vocabulary(primus_dir: str = PRIMUS_DIR, max_samples: int = 5000) -> set[str]:
    """Scan PrIMuS files to collect all unique tokens."""
    all_tokens = set()
    corpus = Path(primus_dir)
    dirs = sorted(corpus.iterdir())[:max_samples]

    for d in dirs:
        if not d.is_dir():
            continue
        sem_files = list(d.glob("*.semantic"))
        if not sem_files:
            continue
        with open(sem_files[0]) as f:
            semantic = f.read()
        tokens = convert_semantic_to_tokens(semantic)
        all_tokens.update(tokens)

    return all_tokens


class PrIMuSDataset(Dataset):
    """PyTorch dataset for PrIMuS/CameraPrIMuS.

    Each sample: (image_tensor, token_indices, token_length)
    Uses distorted (camera) images for training.
    """

    def __init__(
        self,
        vocab,
        primus_dir: str = PRIMUS_DIR,
        img_height: int = 128,
        img_width: int = 1024,
        max_seq_len: int = 600,
        max_samples: int = 0,
        use_distorted: bool = True,
        augment: bool = False,
        split: str = "train",
        split_ratio: float = 0.95,
    ):
        self.vocab = vocab
        self.img_height = img_height
        self.img_width = img_width
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.use_distorted = use_distorted

        corpus = Path(primus_dir)
        all_dirs = sorted([d for d in corpus.iterdir() if d.is_dir()])

        if max_samples > 0:
            all_dirs = all_dirs[:max_samples]

        # Split
        n_train = int(len(all_dirs) * split_ratio)
        if split == "train":
            self.dirs = all_dirs[:n_train]
        else:
            self.dirs = all_dirs[n_train:]

        # Pre-filter valid samples
        self.samples = []
        for d in self.dirs:
            sem_files = list(d.glob("*.semantic"))
            img_ext = "_distorted.jpg" if use_distorted else ".png"
            img_files = list(d.glob(f"*{img_ext}"))

            if sem_files and img_files:
                self.samples.append({
                    "dir": d,
                    "semantic_path": str(sem_files[0]),
                    "image_path": str(img_files[0]),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        img = Image.open(sample["image_path"]).convert("L")

        if self.augment:
            from src.experiments.scan_augment import scan_augment
            if random.random() < 0.5:
                img = scan_augment(img)

        # Resize to fixed height, variable width (pad to max)
        img = self._resize_pad(img)

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)

        # Load and convert tokens
        with open(sample["semantic_path"]) as f:
            semantic = f.read()
        tokens = convert_semantic_to_tokens(semantic)

        token_ids = (
            [self.vocab.sos_idx]
            + self.vocab.encode(tokens)
            + [self.vocab.eos_idx]
        )

        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len - 1] + [self.vocab.eos_idx]

        seq_len = len(token_ids)
        token_ids = token_ids + [self.vocab.pad_idx] * (self.max_seq_len - seq_len)
        token_tensor = torch.tensor(token_ids, dtype=torch.long)

        return img_tensor, token_tensor, seq_len

    def _resize_pad(self, img: Image.Image) -> Image.Image:
        """Resize to fixed height, pad width."""
        w, h = img.size
        scale = self.img_height / h
        new_w = min(int(w * scale), self.img_width)
        new_h = self.img_height
        img = img.resize((new_w, new_h), Image.LANCZOS)

        padded = Image.new("L", (self.img_width, self.img_height), 255)
        padded.paste(img, (0, 0))
        return padded


if __name__ == "__main__":
    # Quick test
    print("Testing PrIMuS loader...")
    sample_dir = Path(PRIMUS_DIR)
    if sample_dir.exists():
        first = sorted(sample_dir.iterdir())[0]
        sem = list(first.glob("*.semantic"))[0]
        with open(sem) as f:
            semantic = f.read()
        tokens = convert_semantic_to_tokens(semantic)
        print(f"Semantic: {semantic[:100]}...")
        print(f"Tokens: {' '.join(tokens[:20])}...")
        print(f"Token count: {len(tokens)}")

        # Check vocab coverage
        vocab_tokens = build_primus_vocabulary(max_samples=1000)
        print(f"\nUnique tokens in first 1000 samples: {len(vocab_tokens)}")
    else:
        print(f"PrIMuS not found at {PRIMUS_DIR}")
