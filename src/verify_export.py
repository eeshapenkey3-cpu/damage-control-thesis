"""Verify exported Sentinel-2 GeoTIFF tiles: complete, uncorrupted, grid-aligned.

Usage (run after gee_export.py export, from the same folder):
    python verify_export.py --out data/raw --prefixes pre post

For every expected tile it checks that the file exists, opens as a GeoTIFF, has 6 bands,
the expected size, CRS and pixel grid, and that every pixel can be read (this catches
truncated files). It also reports leftover .part files, unexpected files, tiles that are
mostly masked, and writes manifest.csv with a SHA-256 per file. On a later run it warns
if any file changed since the manifest was written.

Exit code is 0 only if every expected tile of every prefix is present and valid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import gee_export as ge

MASKED_WARN_PCT = 30.0  # warn if more than this share of a tile is zero in every band
INT16_MAX = 32767


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def tile_stats(path: Path) -> dict:
    """Share of fully masked pixels (zero in all bands) and the value range."""
    import rasterio

    with rasterio.open(path) as ds:
        data = ds.read()
    masked_pct = 100.0 * float((data == 0).all(axis=0).mean())
    return {"masked_pct": masked_pct, "min": int(data.min()), "max": int(data.max())}


def read_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {f"{r['prefix']}/{r['file']}": r["sha256"] for r in csv.DictReader(fh)}


def verify_prefix(directory: Path, prefix: str, tiles, validate=ge.validate_tif, stats=tile_stats) -> dict:
    """Check one set of tiles (e.g. all 'pre' tiles) against the expected grid."""
    expected = {f"{prefix}_{t.name}.tif": t for t in tiles}
    result = {"rows": [], "missing": [], "invalid": [], "partial": [], "unexpected": [], "warnings": []}

    if not directory.exists():
        result["missing"] = sorted(expected)
        return result

    result["partial"] = sorted(p.name for p in directory.glob("*.part"))
    result["unexpected"] = sorted(p.name for p in directory.glob("*.tif") if p.name not in expected)

    for name, tile in sorted(expected.items()):
        path = directory / name
        if not path.exists():
            result["missing"].append(name)
            continue
        ok, msg = validate(path, tile)
        if not ok:
            result["invalid"].append((name, msg))
            continue
        s = stats(path)
        if s["masked_pct"] > MASKED_WARN_PCT:
            result["warnings"].append(f"{name}: {s['masked_pct']:.0f}% of pixels fully masked (cloud mask or no data)")
        if s["max"] >= INT16_MAX:
            result["warnings"].append(f"{name}: values reach the int16 limit, check for saturation")
        result["rows"].append(
            {
                "prefix": prefix,
                "file": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
                "masked_pct": round(s["masked_pct"], 2),
            }
        )
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/raw")
    p.add_argument("--prefixes", nargs="+", default=["pre", "post"])
    p.add_argument("--tile-px", type=int, default=ge.TILE_PX)
    args = p.parse_args()

    tiles = ge.make_tiles(*ge.bbox_to_utm(), args.tile_px)
    out = Path(args.out)
    manifest_path = out / "manifest.csv"
    old_manifest = read_manifest(manifest_path)

    all_rows, problems = [], 0
    valid_sets = {}
    for prefix in args.prefixes:
        res = verify_prefix(out / prefix, prefix, tiles)
        all_rows += res["rows"]
        valid_sets[prefix] = {r["file"].removeprefix(f"{prefix}_") for r in res["rows"]}

        print(f"\n== {prefix}: {len(res['rows'])}/{len(tiles)} tiles valid ==")
        for name in res["missing"]:
            print(f"  MISSING   {name}")
        for name, msg in res["invalid"]:
            print(f"  INVALID   {name}: {msg}")
        for name in res["partial"]:
            print(f"  PARTIAL   {name} (unfinished download, delete it and re-run export)")
        for name in res["unexpected"]:
            print(f"  UNEXPECTED {name} (not part of the expected grid)")
        for w in res["warnings"]:
            print(f"  warning   {w}")
        problems += len(res["missing"]) + len(res["invalid"]) + len(res["partial"])

    for row in all_rows:
        key = f"{row['prefix']}/{row['file']}"
        if key in old_manifest and old_manifest[key] != row["sha256"]:
            print(f"  warning   {key} changed since the last manifest was written")

    if len(args.prefixes) > 1:
        common = set.intersection(*valid_sets.values())
        print(f"\nTiles valid in every set (usable for change analysis): {len(common)}/{len(tiles)}")

    if all_rows:
        out.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["prefix", "file", "bytes", "sha256", "masked_pct"])
            w.writeheader()
            w.writerows(all_rows)
        print(f"Manifest written to {manifest_path}")

    print("\nRESULT:", "all expected tiles present and valid" if problems == 0 else f"{problems} problem(s) found")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())