#!/usr/bin/env python3
"""
End-to-end utility to pull RRFS composite reflectivity (REFC) directly from
NOAA's public S3 bucket, extract only the REFC GRIB message, reproject it to
Web Mercator, and write Leaflet-ready PNG overlays plus metadata.

The script purposefully avoids Git LFS by byte-range downloading just the REFC
message (~7 MB) instead of the full multi‑GB GRIB2 file.

Typical usage (from repo root):
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python scripts/build_rrfs_tiles.py --num-forecasts 6
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Tuple
import time

import numpy as np
import requests
import xarray as xr
from PIL import Image
from pyproj import Transformer
from scipy.spatial import cKDTree

# S3 paths and local locations
S3_BUCKET = "https://noaa-rrfs-pds.s3.amazonaws.com"
S3_PREFIX_ROOT = "rrfs_a"
OUTPUT_ROOT = Path("docs")
TILES_DIR = OUTPUT_ROOT / "tiles"
DATA_DIR = Path("data")

# Official RRFS rotated grid size (used only for sanity checks)
RRFS_NATIVE_SHAPE = (2961, 4881)  # (y, x)

# NWS radar color table (dBZ thresholds, RGBA)
REFC_COLORS: List[Tuple[float, Tuple[int, int, int, int]]] = [
    (-999, (0, 0, 0, 0)),        # transparent background
    (0, (0, 0, 0, 0)),
    (5, (4, 233, 231, 255)),
    (10, (1, 159, 244, 255)),
    (15, (3, 0, 244, 255)),
    (20, (2, 253, 2, 255)),
    (25, (1, 197, 1, 255)),
    (30, (0, 142, 0, 255)),
    (35, (253, 248, 2, 255)),
    (40, (229, 188, 0, 255)),
    (45, (253, 149, 0, 255)),
    (50, (253, 0, 0, 255)),
    (55, (212, 0, 0, 255)),
    (60, (188, 0, 0, 255)),
    (65, (248, 0, 253, 255)),
    (70, (152, 84, 198, 255)),
    (75, (253, 253, 253, 255)),
]


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _s3_list(prefix: str) -> str:
    """List objects under a prefix using S3's anonymous XML listing."""
    resp = fetch_with_retry(
        f"{S3_BUCKET}/",
        params={"delimiter": "/", "prefix": prefix},
    )
    return resp.text


def _parse_common_prefixes(xml_text: str) -> List[str]:
    """Extract CommonPrefixes->Prefix entries from an S3 XML listing."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    prefixes = []
    for cp in root.findall("s3:CommonPrefixes", ns):
        pref = cp.find("s3:Prefix", ns)
        if pref is not None and pref.text:
            prefixes.append(pref.text)
    return prefixes


def _parse_contents(xml_text: str) -> List[str]:
    """Extract Contents->Key entries from an S3 XML listing."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = []
    for ct in root.findall("s3:Contents", ns):
        key_el = ct.find("s3:Key", ns)
        if key_el is not None and key_el.text:
            keys.append(key_el.text)
    return keys


def _ensure_dirs():
    OUTPUT_ROOT.mkdir(exist_ok=True)
    TILES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)


def fetch_with_retry(
    url: str,
    *,
    headers: dict | None = None,
    params: dict | None = None,
    timeout: int = 30,
    max_attempts: int = 4,
) -> requests.Response:
    """Simple retry wrapper to smooth over transient DNS / connection hiccups."""
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # pragma: no cover - network dependent
            if attempt == max_attempts:
                raise
            sleep = 1.5 * attempt
            print(f"Retry {attempt}/{max_attempts} for {url} ({exc}); sleeping {sleep:.1f}s")
            time.sleep(sleep)
    raise RuntimeError(f"Unreachable code for {url}")


# -----------------------------------------------------------------------------
# Cycle discovery and downloading
# -----------------------------------------------------------------------------

def discover_latest_cycle(max_hours_back: int = 12) -> Tuple[str, str]:
    """
    Find the most recent cycle that has natlev files available.

    Returns:
        (cycle_date 'YYYYMMDD', cycle_hour 'HH')
    """
    now = datetime.now(timezone.utc)
    for offset in range(max_hours_back):
        t = now - timedelta(hours=offset)
        day = t.strftime("%Y%m%d")
        hh = f"{t.hour:02d}"
        prefix = f"{S3_PREFIX_ROOT}/rrfs.{day}/{hh}/"
        xml_text = _s3_list(prefix)
        keys = _parse_contents(xml_text)
        natlev = [
            k
            for k in keys
            if "natlev.3km" in k
            and k.endswith(".na.grib2")
            and not k.endswith(".idx")
        ]
        if natlev:
            return day, hh
    raise RuntimeError("No recent RRFS natlev cycles found in the last window")


def list_natlev_files(day: str, hour: str) -> List[str]:
    """Return natlev 3km NA GRIB2 keys for a given cycle."""
    prefix = f"{S3_PREFIX_ROOT}/rrfs.{day}/{hour}/"
    xml_text = _s3_list(prefix)
    keys = _parse_contents(xml_text)
    return sorted(
        k
        for k in keys
        if "natlev.3km" in k
        and k.endswith(".na.grib2")
        and not k.endswith(".idx")
    )


def parse_refc_byte_range(idx_url: str) -> Tuple[int, int]:
    """
    Parse a .idx file and return (start_byte, end_byte) for the REFC message.
    """
    resp = fetch_with_retry(idx_url, timeout=40)
    lines = resp.text.strip().splitlines()
    for i, line in enumerate(lines):
        if "REFC:" in line:
            parts = line.split(":")
            start = int(parts[1])
            if i + 1 < len(lines):
                end = int(lines[i + 1].split(":")[1]) - 1
            else:
                end = None  # last message, download to EOF
            return start, end
    raise RuntimeError("REFC field not found in idx file")


def download_refc_only(key: str, dest: Path) -> None:
    """
    Download only the REFC message from a natlev GRIB2 file using HTTP range.
    """
    grib_url = f"{S3_BUCKET}/{key}"
    idx_url = f"{grib_url}.idx"
    start, end = parse_refc_byte_range(idx_url)

    headers = {"Range": f"bytes={start}-{'' if end is None else end}"}
    resp = fetch_with_retry(grib_url, headers=headers, timeout=300)
    dest.write_bytes(resp.content)


# -----------------------------------------------------------------------------
# Reprojection helpers
# -----------------------------------------------------------------------------

@dataclass
class ReprojectionGrid:
    x_grid: np.ndarray
    y_grid: np.ndarray
    nearest_idx: np.ndarray
    sample_indices: np.ndarray
    shape: Tuple[int, int]
    bounds: dict
    transformer: Transformer

    def reproject(self, data: np.ndarray) -> np.ndarray:
        """Map a native-grid data array to the prepared Web Mercator grid."""
        flat = data.ravel()[self.sample_indices]
        mapped = flat[self.nearest_idx].reshape(self.shape)
        return mapped


def build_reprojection_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    resolution_m: int,
    sample_stride: int,
    extent: dict | None = None,
) -> ReprojectionGrid:
    """
    Build a reusable nearest-neighbor mapping from native grid to Web Mercator.

    We precompute the KD-tree once, then reuse the neighbor indices for every
    forecast hour (massive speedup).
    """
    sample_stride = max(1, int(sample_stride))
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    merc_max = 85.05112878  # Leaflet / EPSG:3857 latitude limit
    clipped_lats = np.clip(lats, -merc_max, merc_max)

    # Determine target extent (either provided or full grid)
    if extent:
        x_min, y_min = transformer.transform(extent["lon_min"], extent["lat_min"])
        x_max, y_max = transformer.transform(extent["lon_max"], extent["lat_max"])
    else:
        xs_full, ys_full = transformer.transform(lons, clipped_lats)
        x_min, x_max = xs_full.min(), xs_full.max()
        y_min, y_max = ys_full.min(), ys_full.max()

    nx = int(math.floor((x_max - x_min) / resolution_m)) + 1
    ny = int(math.floor((y_max - y_min) / resolution_m)) + 1
    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)
    Xi, Yi = np.meshgrid(x_grid, y_grid)

    xs_full, ys_full = transformer.transform(lons, clipped_lats)
    flat_indices = np.arange(xs_full.size, dtype=np.int64)[::sample_stride]
    xs_sample = xs_full.ravel()[flat_indices]
    ys_sample = ys_full.ravel()[flat_indices]

    tree = cKDTree(np.column_stack((xs_sample, ys_sample)))
    _, nearest_idx = tree.query(
        np.column_stack((Xi.ravel(), Yi.ravel())), k=1, workers=-1
    )

    lon_min, lat_min = transformer.transform(x_min, y_min, direction="INVERSE")
    lon_max, lat_max = transformer.transform(x_max, y_max, direction="INVERSE")

    bounds = {
        "lat_min": float(min(lat_min, lat_max)),
        "lat_max": float(max(lat_min, lat_max)),
        "lon_min": float(min(lon_min, lon_max)),
        "lon_max": float(max(lon_min, lon_max)),
        "projection": "EPSG:3857",
        "note": f"Web Mercator grid {x_grid.size}x{y_grid.size} (w×h) at {resolution_m} m ({'custom extent' if extent else 'full native extent'})",
    }

    return ReprojectionGrid(
        x_grid=x_grid,
        y_grid=y_grid,
        nearest_idx=nearest_idx,
        sample_indices=flat_indices,
        shape=Xi.shape,
        bounds=bounds,
        transformer=transformer,
    )


# -----------------------------------------------------------------------------
# Color mapping
# -----------------------------------------------------------------------------

def refc_to_rgba(data: np.ndarray) -> np.ndarray:
    """Map dBZ values to RGBA using the NWS palette."""
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    for i in range(len(REFC_COLORS) - 1):
        low_val, low_color = REFC_COLORS[i]
        high_val, _ = REFC_COLORS[i + 1]
        mask = (data >= low_val) & (data < high_val)
        rgba[mask] = low_color
    # Max bucket
    rgba[data >= REFC_COLORS[-1][0]] = REFC_COLORS[-1][1]
    # Transparent where NaN
    rgba[np.isnan(data)] = (0, 0, 0, 0)
    return rgba


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def process_grib(grib_path: Path, reproj: ReprojectionGrid) -> Tuple[np.ndarray, dict]:
    """
    Read a REFC-only GRIB2, reproject to Web Mercator grid, return (image, stats).
    """
    ds = xr.open_dataset(grib_path, engine="cfgrib")
    var_name = list(ds.data_vars)[0]
    data = ds[var_name].values.astype(np.float32)

    # First call may need lat/lon for grid setup
    if reproj is None:
        lats = ds["latitude"].values
        lons = ds["longitude"].values
        raise ValueError("Reprojection grid must be built before calling process_grib")

    grid_data = reproj.reproject(data)
    stats = {
        "min": float(np.nanmin(grid_data)),
        "max": float(np.nanmax(grid_data)),
        "mean": float(np.nanmean(grid_data)),
    }
    return grid_data, stats


def crop_and_save_tile(
    grid_data: np.ndarray, reproj: ReprojectionGrid, dest: Path
) -> dict:
    """
    Convert grid to RGBA, crop to non-transparent pixels, save PNG, and return bounds.
    """
    rgba = refc_to_rgba(grid_data)
    alpha = rgba[..., 3]
    if not np.any(alpha):
        raise ValueError("Tile is fully transparent")

    rows = np.where(alpha.max(axis=1) > 0)[0]
    cols = np.where(alpha.max(axis=0) > 0)[0]
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])

    cropped = rgba[r0 : r1 + 1, c0 : c1 + 1]

    x_min, x_max = reproj.x_grid[c0], reproj.x_grid[c1]
    y_min, y_max = reproj.y_grid[r0], reproj.y_grid[r1]
    lon_min, lat_min = reproj.transformer.transform(x_min, y_min, direction="INVERSE")
    lon_max, lat_max = reproj.transformer.transform(x_max, y_max, direction="INVERSE")

    cropped = np.flipud(cropped)  # align top row to north
    img = Image.fromarray(cropped, mode="RGBA")
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="PNG", optimize=True)

    return {
        "lat_min": float(min(lat_min, lat_max)),
        "lat_max": float(max(lat_min, lat_max)),
        "lon_min": float(min(lon_min, lon_max)),
        "lon_max": float(max(lon_min, lon_max)),
    }


def select_forecast_hours(keys: List[str], limit: int) -> List[int]:
    """Extract forecast hour integers from file names and select first N ascending."""
    hours = []
    for k in keys:
        try:
            part = k.split(".f")[1].split(".")[0]
            hours.append(int(part))
        except Exception:
            continue
    hours = sorted(set(hours))
    return hours[:limit]


def main():
    parser = argparse.ArgumentParser(description="Build Leaflet-ready RRFS REFC tiles")
    parser.add_argument(
        "--cycle",
        help="Cycle in YYYYMMDDHH (defaults to latest available)",
    )
    parser.add_argument(
        "--num-forecasts",
        type=int,
        default=6,
        help="How many forecast hours to process (starting at f000)",
    )
    parser.add_argument(
        "--resolution-m",
        type=int,
        default=8000,
        help="Output grid resolution in meters (Web Mercator)",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=25,
        help="Stride applied to native grid when building KD-tree (larger=faster)",
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        default=-170.0,
        help="Western bound for output extent (degrees). Ignored with --auto-extent.",
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=-50.0,
        help="Eastern bound for output extent (degrees). Ignored with --auto-extent.",
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        default=5.0,
        help="Southern bound for output extent (degrees). Ignored with --auto-extent.",
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=85.0,
        help="Northern bound for output extent (degrees). Ignored with --auto-extent.",
    )
    parser.add_argument(
        "--auto-extent",
        action="store_true",
        help="Use full native grid extent instead of a North America crop.",
    )
    args = parser.parse_args()

    _ensure_dirs()

    if args.cycle:
        day, hour = args.cycle[:8], args.cycle[8:10]
    else:
        day, hour = discover_latest_cycle()
    print(f"Using cycle {day} {hour}z")

    natlev_keys = list_natlev_files(day, hour)
    if not natlev_keys:
        raise SystemExit("No natlev files found for selected cycle")

    hours_to_do = select_forecast_hours(natlev_keys, args.num_forecasts)
    print(f"Processing forecast hours: {hours_to_do}")

    reproj_grid: ReprojectionGrid | None = None
    forecast_meta = []
    overall_bounds = {
        "lat_min": 90.0,
        "lat_max": -90.0,
        "lon_min": 180.0,
        "lon_max": -180.0,
        "projection": "EPSG:3857",
        "note": "Union of per-frame bounds (cropped to data coverage)",
    }

    # Build reprojection grid using the first file's lat/lon
    first_key = natlev_keys[0]
    first_path = DATA_DIR / f"{Path(first_key).name}.refc.grib2"
    if not first_path.exists():
        print(f"Downloading REFC slice for {first_key}")
        download_refc_only(first_key, first_path)
    ds_first = xr.open_dataset(first_path, engine="cfgrib")
    lats = ds_first["latitude"].values
    lons = ds_first["longitude"].values
    print(f"Native grid shape: {lats.shape}")

    extent = (
        None
        if args.auto_extent
        else {
            "lat_min": args.lat_min,
            "lat_max": args.lat_max,
            "lon_min": args.lon_min,
            "lon_max": args.lon_max,
        }
    )

    reproj_grid = build_reprojection_grid(
        lats=lats,
        lons=lons,
        resolution_m=args.resolution_m,
        sample_stride=args.sample_stride,
        extent=extent,
    )
    ds_first.close()

    # Now loop through requested forecast hours
    for fh in hours_to_do:
        key = f"{S3_PREFIX_ROOT}/rrfs.{day}/{hour}/rrfs.t{hour}z.natlev.3km.f{fh:03d}.na.grib2"
        out_grib = DATA_DIR / f"rrfs.t{hour}z.natlev.f{fh:03d}.refc.grib2"
        if not out_grib.exists():
            print(f"Downloading REFC f{fh:03d}")
            download_refc_only(key, out_grib)
        else:
            print(f"Using cached {out_grib.name}")

        grid, stats = process_grib(out_grib, reproj_grid)
        tile_path = TILES_DIR / f"refc_f{fh:03d}.png"
        tile_bounds = crop_and_save_tile(grid, reproj_grid, tile_path)

        overall_bounds["lat_min"] = min(overall_bounds["lat_min"], tile_bounds["lat_min"])
        overall_bounds["lat_max"] = max(overall_bounds["lat_max"], tile_bounds["lat_max"])
        overall_bounds["lon_min"] = min(overall_bounds["lon_min"], tile_bounds["lon_min"])
        overall_bounds["lon_max"] = max(overall_bounds["lon_max"], tile_bounds["lon_max"])

        cycle_dt = datetime.strptime(day + hour, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        valid = cycle_dt + timedelta(hours=fh)

        forecast_meta.append(
            {
                "forecast_hour": fh,
                "tile": f"tiles/{tile_path.name}",
                "valid_time": valid.isoformat(),
                "stats": stats,
                "bounds": tile_bounds,
            }
        )

    # Build metadata
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": {
            "cycle_date": day,
            "cycle_hour": hour,
            "bucket": S3_BUCKET,
            "domain": "RRFS natlev 3km NA",
        },
        "bounds": overall_bounds,
        "forecasts": forecast_meta,
    }

    OUTPUT_ROOT.mkdir(exist_ok=True)
    (OUTPUT_ROOT / "data.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUTPUT_ROOT/'data.json'}")
    print(f"Tiles directory: {TILES_DIR}")


if __name__ == "__main__":
    main()
