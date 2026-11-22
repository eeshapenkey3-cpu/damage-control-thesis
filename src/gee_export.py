"""Test export and tiled download of Sentinel-2 composites from Google Earth Engine.

Usage (after `earthengine authenticate`):
    python gee_export.py grid                      # print the tile grid, no network
    python gee_export.py test   --start 2021-06-01 --end 2021-09-01
    python gee_export.py export --start 2021-06-01 --end 2021-09-01 --prefix pre

Set your Earth Engine project with --project or the EE_PROJECT env variable.

Design notes
- Everything is done in UTM 37N (EPSG:32637) on an exact 10 m grid, so tiles line
  up pixel-for-pixel with each other and across the pre/post composites.
- getDownloadURL has a request size cap (about 48 MB) and a per-side pixel cap.
  Tiles are therefore small (default 1024 x 1024) and checked before any request.
- Bands are exported as int16 (raw 0-10000 reflectance scale). Divide by 10000
  after loading, before computing indices. Masked (cloudy) pixels normally
  arrive as 0, which is why the index functions protect against a zero sum.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Study area as lon/lat (xmin, ymin, xmax, ymax). Starting guess: check and adjust
# it against the city outline on the GEE map before the real export.
BBOX_LONLAT = (37.38, 47.00, 37.75, 47.20)
TEST_CENTER_LONLAT = (37.5434, 47.0971)  # roughly central Mariupol

CRS = "EPSG:32637"  # UTM zone 37N covers 36E-42E
SCALE = 10
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
BYTES_PER_PIXEL = 2  # int16
TILE_PX = 1024
GEE_MAX_BYTES = 50_331_648  # getDownloadURL request cap (about 48 MB)
SAFETY_FRACTION = 0.5  # keep the uncompressed estimate under half the cap


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    scale: int = SCALE

    @property
    def width(self) -> int:
        return int(round((self.xmax - self.xmin) / self.scale))

    @property
    def height(self) -> int:
        return int(round((self.ymax - self.ymin) / self.scale))

    @property
    def name(self) -> str:
        return f"r{self.row:02d}_c{self.col:02d}"


# --------------------------------------------------------------------------- #
# Tiling (pure Python, no network)
# --------------------------------------------------------------------------- #

def bbox_to_utm(bbox_lonlat=BBOX_LONLAT, crs: str = CRS):
    """Project a lon/lat bounding box to UTM metres (bounding box of its corners)."""
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xmin, ymin, xmax, ymax = bbox_lonlat
    xs, ys = zip(*(tr.transform(x, y) for x in (xmin, xmax) for y in (ymin, ymax)))
    return min(xs), min(ys), max(xs), max(ys)


def check_tile_size(tile_px: int, n_bands: int = len(BANDS), bytes_per_px: int = BYTES_PER_PIXEL) -> int:
    """Raise if a full tile could approach the request cap. Returns estimated bytes."""
    est = tile_px * tile_px * n_bands * bytes_per_px
    if est > GEE_MAX_BYTES * SAFETY_FRACTION:
        max_px = int(math.sqrt(GEE_MAX_BYTES * SAFETY_FRACTION / (n_bands * bytes_per_px)))
        raise ValueError(
            f"Tile {tile_px}x{tile_px} x {n_bands} bands is about {est / 1e6:.1f} MB; "
            f"use tile_px <= {max_px} to stay well under the {GEE_MAX_BYTES / 1e6:.0f} MB cap."
        )
    return est


def make_tiles(xmin, ymin, xmax, ymax, tile_px: int = TILE_PX, scale: int = SCALE) -> list[Tile]:
    """Cut a UTM extent into a row-major grid (row 0 = north). Edge tiles may be smaller."""
    check_tile_size(tile_px)
    # Snap outwards to the pixel grid so every tile edge is a multiple of `scale`.
    xmin = math.floor(xmin / scale) * scale
    ymin = math.floor(ymin / scale) * scale
    xmax = math.ceil(xmax / scale) * scale
    ymax = math.ceil(ymax / scale) * scale

    tile_m = tile_px * scale
    n_cols = math.ceil((xmax - xmin) / tile_m)
    n_rows = math.ceil((ymax - ymin) / tile_m)

    tiles = []
    for r in range(n_rows):
        top = ymax - r * tile_m
        bottom = max(top - tile_m, ymin)
        for c in range(n_cols):
            left = xmin + c * tile_m
            right = min(left + tile_m, xmax)
            tiles.append(Tile(r, c, left, bottom, right, top, scale))
    return tiles


# --------------------------------------------------------------------------- #
# Earth Engine pieces
# --------------------------------------------------------------------------- #

def init_ee(project: str | None):
    import ee

    ee.Initialize(project=project or os.environ.get("EE_PROJECT"))
    return ee


def build_composite(ee, start: str, end: str, bbox_lonlat=BBOX_LONLAT):
    """Cloud-masked median composite as int16. Replace with your own composite if needed."""
    region = ee.Geometry.Rectangle(list(bbox_lonlat))

    def mask_clouds(img):
        scl = img.select("SCL")
        bad = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))  # shadow, cloud, cirrus
        return img.updateMask(bad.Not())

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .map(mask_clouds)
    )
    return col.median().select(BANDS).toInt16()


def tile_region(ee, tile: Tile):
    return ee.Geometry.Rectangle(
        [tile.xmin, tile.ymin, tile.xmax, tile.ymax], proj=CRS, geodesic=False
    )


def validate_tif(path: Path, tile: Tile) -> tuple[bool, str]:
    """Check a downloaded file is a real GeoTIFF with the expected shape and grid."""
    try:
        import rasterio

        with rasterio.open(path) as ds:
            if ds.count != len(BANDS):
                return False, f"{ds.count} bands, expected {len(BANDS)}"
            if (ds.height, ds.width) != (tile.height, tile.width):
                return False, f"size {ds.height}x{ds.width}, expected {tile.height}x{tile.width}"
            if ds.crs is None or ds.crs.to_epsg() != int(CRS.split(":")[1]):
                return False, f"unexpected CRS {ds.crs}"
            t = ds.transform
            if abs(t.c - tile.xmin) > 1e-6 or abs(t.f - tile.ymax) > 1e-6 or abs(t.a - tile.scale) > 1e-6:
                return False, f"grid misaligned: origin ({t.c}, {t.f}), pixel {t.a}"
            data = ds.read()
            if not data.any():
                return False, "all pixels are zero (empty or fully masked)"
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 - report any read failure as invalid
        return False, f"could not read as GeoTIFF: {exc}"


def download_tile(ee, image, tile: Tile, out_path: Path, retries: int = 3) -> tuple[bool, str]:
    """Download one tile, verify the bytes, and keep it only if validation passes."""
    import requests

    if out_path.exists():
        ok, msg = validate_tif(out_path, tile)
        if ok:
            return True, "already downloaded"
        out_path.unlink()

    params = {
        "region": tile_region(ee, tile),
        "crs": CRS,
        "scale": tile.scale,
        "format": "GEO_TIFF",
    }
    last = "unknown error"
    for attempt in range(1, retries + 1):
        tmp = out_path.with_suffix(".part")
        try:
            url = image.getDownloadURL(params)
            with requests.get(url, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                expected = resp.headers.get("Content-Length")
                written = 0
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        written += len(chunk)
            if expected is not None and int(expected) != written:
                raise IOError(f"truncated download: {written} of {expected} bytes")
            with open(tmp, "rb") as fh:
                magic = fh.read(4)
            if magic[:2] not in (b"II", b"MM"):
                head = tmp.read_bytes()[:200]
                raise IOError(f"not a GeoTIFF (starts with {head!r})")
            tmp.replace(out_path)
            ok, msg = validate_tif(out_path, tile)
            if not ok:
                out_path.unlink(missing_ok=True)
                return False, msg  # a bad shape will not fix itself on retry
            return True, f"{written / 1e6:.1f} MB"
        except ee.EEException as exc:
            return False, f"Earth Engine refused the request: {exc}"  # not transient
        except (requests.RequestException, IOError) as exc:
            last = str(exc)
            tmp.unlink(missing_ok=True)
            time.sleep(2 ** attempt)
    return False, f"failed after {retries} attempts: {last}"


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_grid(args) -> None:
    xmin, ymin, xmax, ymax = bbox_to_utm()
    tiles = make_tiles(xmin, ymin, xmax, ymax, args.tile_px)
    n_rows = max(t.row for t in tiles) + 1
    n_cols = max(t.col for t in tiles) + 1
    est = check_tile_size(args.tile_px)
    print(f"UTM extent: x {xmin:.0f}-{xmax:.0f}, y {ymin:.0f}-{ymax:.0f}")
    print(f"{len(tiles)} tiles ({n_rows} rows x {n_cols} cols), up to {args.tile_px}x{args.tile_px} px")
    print(f"Estimated uncompressed size of a full tile: {est / 1e6:.1f} MB (cap {GEE_MAX_BYTES / 1e6:.0f} MB)")
    for t in tiles:
        print(f"  {t.name}: {t.width}x{t.height} px")


def cmd_test(args) -> None:
    from pyproj import Transformer

    ee = init_ee(args.project)
    image = build_composite(ee, args.start, args.end)

    cx, cy = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(*TEST_CENTER_LONLAT)
    half = args.test_px // 2 * SCALE
    x0 = round((cx - half) / SCALE) * SCALE
    y0 = round((cy - half) / SCALE) * SCALE
    tile = Tile(0, 0, x0, y0, x0 + args.test_px * SCALE, y0 + args.test_px * SCALE)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "test_window.tif"
    if path.exists():
        path.unlink()
    ok, msg = download_tile(ee, image, tile, path)
    print(("PASS" if ok else "FAIL"), msg)
    if ok:
        import rasterio

        with rasterio.open(path) as ds:
            data = ds.read()
            print(f"shape {data.shape}, dtype {data.dtype}, {path.stat().st_size / (tile.width * tile.height):.1f} bytes/pixel on disk")
            for name, band in zip(BANDS, data):
                print(f"  {name}: min {band.min()}, max {band.max()}, zero pixels {100 * (band == 0).mean():.1f}%")
    sys.exit(0 if ok else 1)


def cmd_export(args) -> None:
    ee = init_ee(args.project)
    image = build_composite(ee, args.start, args.end)
    xmin, ymin, xmax, ymax = bbox_to_utm()
    tiles = make_tiles(xmin, ymin, xmax, ymax, args.tile_px)

    out = Path(args.out) / args.prefix
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, tile in enumerate(tiles, 1):
        ok, msg = download_tile(ee, image, tile, out / f"{args.prefix}_{tile.name}.tif")
        print(f"[{i}/{len(tiles)}] {tile.name}: {'ok' if ok else 'FAILED'} ({msg})")
        if not ok:
            failed.append(tile.name)
    print(f"Done. {len(tiles) - len(failed)} ok, {len(failed)} failed: {failed}")
    sys.exit(1 if failed else 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["grid", "test", "export"])
    p.add_argument("--project", help="Earth Engine cloud project (or set EE_PROJECT)")
    p.add_argument("--start", help="composite start date, YYYY-MM-DD")
    p.add_argument("--end", help="composite end date, YYYY-MM-DD")
    p.add_argument("--prefix", default="composite", help="output file prefix, e.g. pre or post")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--tile-px", type=int, default=TILE_PX)
    p.add_argument("--test-px", type=int, default=256, help="side of the small test window in pixels")
    args = p.parse_args()

    if args.command in ("test", "export") and not (args.start and args.end):
        p.error("--start and --end are required for test and export")
    {"grid": cmd_grid, "test": cmd_test, "export": cmd_export}[args.command](args)


if __name__ == "__main__":
    main()