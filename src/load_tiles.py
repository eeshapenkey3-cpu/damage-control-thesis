"""Open paired pre/post Sentinel-2 GeoTIFF tiles with Rasterio.

Expected layout (what gee_export.py writes):
    data/raw/pre/pre_r00_c00.tif, ...
    data/raw/post/post_r00_c00.tif, ...

Each tile is int16 with 6 bands in the order B2, B3, B4, B8, B11, B12 (raw 0-10000 scale).
Loading a tile:
  * divides by 10000 to get reflectance (float32);
  * turns pixels that are 0 in every band (masked clouds / no data) into NaN, so they
    never look like real "index = 0" pixels downstream.

Usage:
    python load_tiles.py --out data/raw          # print a per-tile summary
or, in your own code:
    from load_tiles import iter_pairs
    for pair in iter_pairs("data/raw"):
        d_ndvi, d_ndbi = pair.deltas()
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from features import RAW_BANDS, stack_features

SCALE_FACTOR = 10_000.0
TILE_NAME = re.compile(r"_(r\d{2}_c\d{2})\.tif$")


@dataclass
class Tile:
    name: str
    bands: np.ndarray  # (6, H, W) float32 reflectance, NaN where masked
    masked: np.ndarray  # (H, W) bool, True where the pixel had no data in any band
    crs: object
    transform: object

    @property
    def shape(self) -> tuple[int, int]:
        return self.bands.shape[1:]

    def features(self) -> np.ndarray:
        """(H, W, 8): the 6 bands plus NDVI and NDBI. NaN stays NaN."""
        return stack_features(dict(zip(RAW_BANDS, self.bands)))


@dataclass
class TilePair:
    name: str
    pre: Tile
    post: Tile

    @property
    def valid(self) -> np.ndarray:
        """(H, W) bool: pixels with data in both the pre and post composite."""
        return ~(self.pre.masked | self.post.masked)

    def deltas(self) -> tuple[np.ndarray, np.ndarray]:
        """(dNDVI, dNDBI), each (H, W) float32, defined as post minus pre."""
        pre_f, post_f = self.pre.features(), self.post.features()
        return post_f[..., 6] - pre_f[..., 6], post_f[..., 7] - pre_f[..., 7]


def to_reflectance(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert (6, H, W) integer DNs to float32 reflectance, masking all-zero pixels."""
    masked = (raw == 0).all(axis=0)
    refl = raw.astype(np.float32) / SCALE_FACTOR
    refl[:, masked] = np.nan
    return refl, masked


def read_tile(path: str | Path) -> Tile:
    """Open one GeoTIFF with Rasterio and return reflectance plus georeferencing."""
    import rasterio

    path = Path(path)
    with rasterio.open(path) as ds:
        if ds.count != len(RAW_BANDS):
            raise ValueError(f"{path.name}: {ds.count} bands, expected {len(RAW_BANDS)}")
        raw = ds.read()  # (bands, H, W)
        crs, transform = ds.crs, ds.transform

    refl, masked = to_reflectance(raw)
    match = TILE_NAME.search(path.name)
    return Tile(match.group(1) if match else path.stem, refl, masked, crs, transform)


def check_aligned(pre: Tile, post: Tile) -> None:
    """Pre and post must share the exact same pixel grid, or differences are meaningless."""
    if pre.shape != post.shape:
        raise ValueError(f"{pre.name}: shape mismatch {pre.shape} vs {post.shape}")
    if pre.crs != post.crs:
        raise ValueError(f"{pre.name}: CRS mismatch {pre.crs} vs {post.crs}")
    if pre.transform != post.transform:
        raise ValueError(f"{pre.name}: pixel grid mismatch {pre.transform} vs {post.transform}")


def load_pair(pre_path: str | Path, post_path: str | Path, reader=read_tile) -> TilePair:
    pre, post = reader(pre_path), reader(post_path)
    check_aligned(pre, post)
    return TilePair(pre.name, pre, post)


def find_pairs(out_dir: str | Path, pre_prefix: str = "pre", post_prefix: str = "post"):
    """Match tiles by their rXX_cXX name. Returns (pairs, only_pre, only_post)."""
    out_dir = Path(out_dir)

    def index(prefix: str) -> dict[str, Path]:
        found = {}
        for p in (out_dir / prefix).glob(f"{prefix}_r*_c*.tif"):
            m = TILE_NAME.search(p.name)
            if m:
                found[m.group(1)] = p
        return found

    pre, post = index(pre_prefix), index(post_prefix)
    pairs = [(name, pre[name], post[name]) for name in sorted(pre.keys() & post.keys())]
    return pairs, sorted(pre.keys() - post.keys()), sorted(post.keys() - pre.keys())


def iter_pairs(out_dir: str | Path, reader=read_tile) -> Iterator[TilePair]:
    """Load one pair at a time so memory stays small."""
    pairs, only_pre, only_post = find_pairs(out_dir)
    if only_pre or only_post:
        print(f"warning: unmatched tiles, pre only {only_pre}, post only {only_post}")
    for _, pre_path, post_path in pairs:
        yield load_pair(pre_path, post_path, reader)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/raw")
    args = p.parse_args()

    print(f"{'tile':8} {'size':>11} {'valid %':>8} {'dNDVI mean':>11} {'dNDVI std':>10} {'dNDBI mean':>11} {'dNDBI std':>10}")
    count = 0
    for pair in iter_pairs(args.out):
        valid = pair.valid
        d_ndvi, d_ndbi = pair.deltas()
        h, w = pair.pre.shape
        if valid.any():
            stats = [np.nanmean(d[valid]) for d in (d_ndvi, d_ndbi)] + [np.nanstd(d[valid]) for d in (d_ndvi, d_ndbi)]
            line = f"{stats[0]:11.4f} {stats[2]:10.4f} {stats[1]:11.4f} {stats[3]:10.4f}"
        else:
            line = f"{'no valid pixels':>45}"
        print(f"{pair.name:8} {f'{h}x{w}':>11} {100 * valid.mean():8.1f} {line}")
        count += 1
    print(f"\n{count} tile pair(s) opened.")


if __name__ == "__main__":
    main()