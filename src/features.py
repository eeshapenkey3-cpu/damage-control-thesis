"""Spectral indices and feature stacking for Sentinel-2 change analysis.

Expected input: each band is a 2-D (H, W) array of surface reflectance on the
same 10 m grid (B11 and B12 are native 20 m, so resample them to 10 m first,
e.g. at export time in Google Earth Engine). If your bands are still raw
digital numbers (0-10000), divide by 10000 before calling these functions.

Output: an (H, W, 8) float32 tensor with channels in FEATURE_NAMES order.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

EPS = 1e-10

RAW_BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")
FEATURE_NAMES = RAW_BANDS + ("NDVI", "NDBI")


def normalized_difference(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> np.ndarray:
    """(a - b) / (a + b + eps), computed element-wise in float32.

    The epsilon only matters where a + b == 0 (for example masked or no-data
    pixels stored as 0): there the result is 0 instead of NaN or inf. NaN
    inputs still propagate as NaN, so cloud masks stored as NaN are preserved.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return (a - b) / (a + b + np.float32(eps))


def ndvi(nir_b8: np.ndarray, red_b4: np.ndarray, eps: float = EPS) -> np.ndarray:
    """NDVI = (B8 - B4) / (B8 + B4)."""
    return normalized_difference(nir_b8, red_b4, eps)


def ndbi(swir_b11: np.ndarray, nir_b8: np.ndarray, eps: float = EPS) -> np.ndarray:
    """NDBI = (B11 - B8) / (B11 + B8)."""
    return normalized_difference(swir_b11, nir_b8, eps)


def stack_features(bands: Mapping[str, np.ndarray], eps: float = EPS) -> np.ndarray:
    """Stack B2, B3, B4, B8, B11, B12 plus NDVI and NDBI into an (H, W, 8) array.

    `bands` maps band names ("B2", "B3", ...) to (H, W) arrays. Channel order
    is given by FEATURE_NAMES.
    """
    missing = [name for name in RAW_BANDS if name not in bands]
    if missing:
        raise KeyError(f"Missing bands: {missing}")

    raw = {name: np.asarray(bands[name], dtype=np.float32) for name in RAW_BANDS}

    shape = raw["B2"].shape
    if len(shape) != 2:
        raise ValueError(f"Bands must be 2-D (H, W); got shape {shape} for B2")
    for name, arr in raw.items():
        if arr.shape != shape:
            raise ValueError(f"Band {name} has shape {arr.shape}, expected {shape}")

    channels = [raw[name] for name in RAW_BANDS]
    channels.append(ndvi(raw["B8"], raw["B4"], eps))
    channels.append(ndbi(raw["B11"], raw["B8"], eps))

    return np.stack(channels, axis=-1)


if __name__ == "__main__":
    # Quick sanity check on random data, including a zero (no-data) pixel.
    rng = np.random.default_rng(0)
    h, w = 4, 5
    demo = {name: rng.uniform(0.01, 0.5, size=(h, w)).astype(np.float32) for name in RAW_BANDS}
    for name in RAW_BANDS:
        demo[name][0, 0] = 0.0

    tensor = stack_features(demo)
    print(tensor.shape, tensor.dtype)
    print("NDVI/NDBI at the zero pixel:", tensor[0, 0, 6], tensor[0, 0, 7])