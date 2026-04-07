"""Synthetic data augmentation: render MusicXML → varied notation images.

Uses MuseScore CLI to render XML files to PNG, then applies visual
augmentations to simulate varied engraving styles and scan conditions.
Each variant is tracked by provenance (synthetic vs real).
"""

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

MUSESCORE_BIN = "/Applications/MuseScore 4.app/Contents/MacOS/mscore"
XML_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "xml")
SYNTH_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
EVENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "events")

# Augmentation configs: each produces a visually distinct variant
AUGMENTATION_CONFIGS = [
    {"name": "clean_150", "dpi": 150, "aug": None},
    {"name": "clean_200", "dpi": 200, "aug": None},
    {"name": "clean_300", "dpi": 300, "aug": None},
    {"name": "noisy_200", "dpi": 200, "aug": "noise"},
    {"name": "noisy_300", "dpi": 300, "aug": "noise"},
    {"name": "blur_200", "dpi": 200, "aug": "blur"},
    {"name": "blur_300", "dpi": 300, "aug": "blur"},
    {"name": "contrast_low", "dpi": 250, "aug": "low_contrast"},
    {"name": "contrast_high", "dpi": 250, "aug": "high_contrast"},
    {"name": "skew_cw", "dpi": 250, "aug": "skew_cw"},
    {"name": "skew_ccw", "dpi": 250, "aug": "skew_ccw"},
    {"name": "thick_lines", "dpi": 350, "aug": "erode"},
    {"name": "thin_lines", "dpi": 200, "aug": "dilate"},
    {"name": "scanner_dark", "dpi": 250, "aug": "scanner_dark"},
    {"name": "scanner_light", "dpi": 250, "aug": "scanner_light"},
]


def _render_xml_to_png(xml_path: str, output_path: str, dpi: int = 300) -> bool:
    """Render a MusicXML file to PNG using MuseScore CLI."""
    try:
        result = subprocess.run(
            [MUSESCORE_BIN, "-o", output_path, "-r", str(dpi), xml_path],
            capture_output=True,
            timeout=60,
        )
        # MuseScore may produce multi-page output as file-1.png, file-2.png, etc.
        # Check for the main output or numbered variants
        base = output_path.rsplit(".", 1)[0]
        if os.path.exists(output_path):
            return True
        # Check for -1.png variant
        if os.path.exists(f"{base}-1.png"):
            os.rename(f"{base}-1.png", output_path)
            # Remove additional pages
            for i in range(2, 20):
                extra = f"{base}-{i}.png"
                if os.path.exists(extra):
                    os.remove(extra)
            return True
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  Render failed for {xml_path}: {e}")
        return False


def _apply_augmentation(img: Image.Image, aug_type: str) -> Image.Image:
    """Apply a specific augmentation to an image."""
    if aug_type is None:
        return img

    if aug_type == "noise":
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, 15, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    elif aug_type == "blur":
        radius = random.uniform(0.8, 2.0)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    elif aug_type == "low_contrast":
        return ImageEnhance.Contrast(img).enhance(0.5)

    elif aug_type == "high_contrast":
        return ImageEnhance.Contrast(img).enhance(2.0)

    elif aug_type == "skew_cw":
        return img.rotate(-2, fillcolor=255, expand=False)

    elif aug_type == "skew_ccw":
        return img.rotate(2, fillcolor=255, expand=False)

    elif aug_type == "erode":
        # Make lines thicker (morphological erosion = darker)
        return img.filter(ImageFilter.MinFilter(3))

    elif aug_type == "dilate":
        # Make lines thinner (morphological dilation = lighter)
        return img.filter(ImageFilter.MaxFilter(3))

    elif aug_type == "scanner_dark":
        # Simulate dark scanner: reduce brightness, add slight yellow tint
        img = ImageEnhance.Brightness(img).enhance(0.7)
        return ImageEnhance.Contrast(img).enhance(0.8)

    elif aug_type == "scanner_light":
        # Simulate light/faded scan
        img = ImageEnhance.Brightness(img).enhance(1.3)
        return ImageEnhance.Contrast(img).enhance(0.7)

    return img


def render_all(
    xml_dir: str = XML_DIR,
    synth_dir: str = SYNTH_DIR,
    configs: list[dict] | None = None,
):
    """Render all XML files with all augmentation configs."""
    if configs is None:
        configs = AUGMENTATION_CONFIGS

    os.makedirs(synth_dir, exist_ok=True)

    xml_files = sorted(Path(xml_dir).glob("*.xml"))
    if not xml_files:
        print(f"No XML files in {xml_dir}")
        return

    print(f"Rendering {len(xml_files)} XML files x {len(configs)} configs = "
          f"{len(xml_files) * len(configs)} synthetic images")

    manifest = []
    rendered = 0
    failed = 0

    for config in configs:
        config_name = config["name"]
        dpi = config["dpi"]
        aug = config["aug"]

        config_dir = os.path.join(synth_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)

        for xml_path in xml_files:
            file_id = xml_path.stem.lower().replace(" ", "_").replace("'", "").replace(".", "")
            output_path = os.path.join(config_dir, f"{file_id}.png")

            # Render base image (skip if augmentation-only variant)
            base_path = os.path.join(synth_dir, "clean_300", f"{file_id}.png")

            if aug is None or not os.path.exists(base_path):
                # Need to render from XML
                if not _render_xml_to_png(str(xml_path), output_path, dpi):
                    failed += 1
                    continue
            else:
                # Apply augmentation to base render
                if not os.path.exists(base_path):
                    if not _render_xml_to_png(str(xml_path), output_path, dpi):
                        failed += 1
                        continue
                else:
                    # Re-render at target DPI if different, or copy and augment
                    temp_path = output_path + ".tmp.png"
                    if dpi != 300:
                        if not _render_xml_to_png(str(xml_path), temp_path, dpi):
                            # Fallback: use base and resize
                            img = Image.open(base_path)
                            scale = dpi / 300
                            new_size = (int(img.width * scale), int(img.height * scale))
                            img = img.resize(new_size, Image.LANCZOS)
                            img.save(temp_path)
                    else:
                        shutil.copy2(base_path, temp_path)

                    if os.path.exists(temp_path):
                        img = Image.open(temp_path).convert("L")
                        img = _apply_augmentation(img, aug)
                        img.save(output_path)
                        os.remove(temp_path)
                    else:
                        failed += 1
                        continue

            # Check if output exists
            if os.path.exists(output_path):
                # Get corresponding token file
                token_path = os.path.join(EVENTS_DIR, f"{file_id}.tokens")
                event_path = os.path.join(EVENTS_DIR, f"{file_id}.json")

                manifest.append({
                    "id": f"{file_id}_{config_name}",
                    "source_id": file_id,
                    "image_path": output_path,
                    "token_path": token_path,
                    "event_path": event_path,
                    "config": config_name,
                    "dpi": dpi,
                    "augmentation": aug or "none",
                    "provenance": "synthetic",
                })
                rendered += 1

        print(f"  {config_name}: rendered {sum(1 for m in manifest if m['config'] == config_name)} images")

    # Save manifest
    manifest_path = os.path.join(synth_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nTotal: {rendered} rendered, {failed} failed")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    render_all()
