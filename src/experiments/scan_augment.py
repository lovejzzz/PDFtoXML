"""Domain-adaptive augmentation: make synthetic renders look like scans.

The key insight: our model can read clean synthetic renders but not scanned pages.
This module applies scan-like degradations to synthetic images during training
to bridge the domain gap.

Augmentations designed to simulate:
- Scanner noise and artifacts
- Page curvature / warping
- Ink spread and bleed
- Background texture and yellowing
- Uneven lighting
- Dust and speckle noise
"""

import random

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps


def scan_augment(img: Image.Image) -> Image.Image:
    """Apply random scan-like augmentations to bridge synthetic → real gap."""
    # Convert to grayscale if needed
    if img.mode != "L":
        img = img.convert("L")

    # Apply 2-4 random augmentations
    augmentations = [
        _add_scanner_noise,
        _adjust_ink_weight,
        _add_background_texture,
        _uneven_lighting,
        _slight_warp,
        _add_speckle,
        _jpeg_artifact,
        _threshold_binarize_soft,
        _page_edge_shadow,
    ]

    # Always apply at least background + noise
    img = _add_background_texture(img)
    img = _add_scanner_noise(img)

    # Then 1-3 more random augmentations
    num_extra = random.randint(1, 3)
    chosen = random.sample(augmentations, min(num_extra, len(augmentations)))
    for aug_fn in chosen:
        img = aug_fn(img)

    return img


def _add_scanner_noise(img: Image.Image) -> Image.Image:
    """Add Gaussian noise typical of flatbed scanners."""
    arr = np.array(img, dtype=np.float32)
    sigma = random.uniform(3, 12)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _adjust_ink_weight(img: Image.Image) -> Image.Image:
    """Simulate thicker or thinner ink by adjusting contrast around midtone."""
    factor = random.uniform(0.6, 1.8)
    return ImageEnhance.Contrast(img).enhance(factor)


def _add_background_texture(img: Image.Image) -> Image.Image:
    """Add paper-like background texture (slight yellowing, grain)."""
    arr = np.array(img, dtype=np.float32)

    # Slightly darken the white areas (paper isn't pure white)
    bg_level = random.uniform(225, 250)
    mask = arr > 200  # white areas
    arr[mask] = np.clip(arr[mask] * (bg_level / 255), 0, 255)

    # Add very fine grain
    grain = np.random.normal(0, 2, arr.shape)
    arr = np.clip(arr + grain, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _uneven_lighting(img: Image.Image) -> Image.Image:
    """Simulate uneven scanner lighting (brighter center, darker edges)."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape

    # Create a gradient mask
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    xx, yy = np.meshgrid(x, y)

    # Random center offset
    cx = random.uniform(-0.3, 0.3)
    cy = random.uniform(-0.3, 0.3)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Vignette effect
    intensity = random.uniform(0.05, 0.15)
    vignette = 1.0 - intensity * dist
    arr = np.clip(arr * vignette, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _slight_warp(img: Image.Image) -> Image.Image:
    """Simulate slight page curvature / scanner distortion."""
    # Use affine transform for a subtle perspective effect
    w, h = img.size
    shift = random.uniform(2, 8)

    # Random perspective-like transform via mesh
    coeffs = [
        1 + random.uniform(-0.005, 0.005),  # a
        random.uniform(-0.01, 0.01),  # b
        random.uniform(-shift, shift),  # c
        random.uniform(-0.01, 0.01),  # d
        1 + random.uniform(-0.005, 0.005),  # e
        random.uniform(-shift, shift),  # f
        0, 0,  # perspective
    ]
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, fillcolor=240)


def _add_speckle(img: Image.Image) -> Image.Image:
    """Add random speckle (dust/dirt) noise."""
    arr = np.array(img, dtype=np.uint8)
    num_specks = random.randint(5, 50)

    h, w = arr.shape
    for _ in range(num_specks):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        size = random.randint(1, 3)
        val = random.randint(0, 100)  # dark specks
        y1, y2 = max(0, y - size), min(h, y + size)
        x1, x2 = max(0, x - size), min(w, x + size)
        arr[y1:y2, x1:x2] = val

    return Image.fromarray(arr)


def _jpeg_artifact(img: Image.Image) -> Image.Image:
    """Simulate JPEG compression artifacts."""
    import io
    quality = random.randint(15, 50)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("L")


def _threshold_binarize_soft(img: Image.Image) -> Image.Image:
    """Soft binarization to simulate scanner auto-threshold."""
    arr = np.array(img, dtype=np.float32)
    threshold = random.uniform(140, 180)
    sharpness = random.uniform(0.05, 0.15)

    # Sigmoid-based soft threshold
    result = 255.0 / (1.0 + np.exp(-sharpness * (arr - threshold)))
    return Image.fromarray(result.astype(np.uint8))


def _page_edge_shadow(img: Image.Image) -> Image.Image:
    """Add dark shadow on one or two edges (book binding effect)."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape

    # Pick an edge
    edge = random.choice(["left", "right", "top"])
    shadow_width = random.randint(max(1, w // 20), max(2, w // 8))
    shadow_width = min(shadow_width, w - 1, h - 1)  # clamp to image size
    darkness = random.uniform(0.5, 0.8)

    if edge == "left":
        gradient = np.linspace(darkness, 1.0, shadow_width)
        arr[:, :shadow_width] *= gradient[np.newaxis, :]
    elif edge == "right":
        gradient = np.linspace(1.0, darkness, shadow_width)
        arr[:, -shadow_width:] *= gradient[np.newaxis, :]
    elif edge == "top":
        gradient = np.linspace(darkness, 1.0, shadow_width)
        arr[:shadow_width, :] *= gradient[:, np.newaxis]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
