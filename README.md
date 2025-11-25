# RRFS REFC Leaflet Demo

Downloads the latest RRFS composite reflectivity (REFC) straight from NOAA's
public S3 bucket, extracts only the REFC GRIB message via byte‑range, reprojects
to Web Mercator, and builds a Leaflet overlay you can host on GitHub Pages.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# pull the latest cycle, first 6 hours, 8 km Web Mercator grid cropped to NA
python scripts/build_rrfs_tiles.py --num-forecasts 6 --resolution-m 8000 --sample-stride 25
```

Outputs land in `docs/`:
- `docs/data.json` metadata (bounds, timestamps, stats)
- `docs/tiles/refc_fXXX.png` web mercator overlays
- `docs/index.html` Leaflet viewer that reads `data.json`

Commit/push `docs/` to publish with GitHub Pages (project/site mode).

## Why this works
- Grabs only the REFC message (~7 MB) from each natlev GRIB2 using HTTP `Range`
  requests — no Git LFS required.
- Reprojects the native rotated lat/lon grid onto a Web Mercator mesh, so the
  PNG aligns pixel‑for‑pixel with Leaflet’s base map. Default extent is a
  North America crop (`lon -170..-50`, `lat 5..85`) so the overlay fits the
  Leaflet viewport without wrapping.
- Precomputes a KD‑tree once and reuses the neighbor mapping for every frame,
  keeping runtime manageable.

## Tuning
- `--resolution-m`: Web Mercator grid spacing (meters). Smaller = sharper, slower
  (6–8 km is a good balance).
- `--sample-stride`: subsampling factor for building the KD‑tree. Lower = more
  accurate but slower. 20–30 works well.
- `--num-forecasts`: how many forecast hours to fetch starting at f000.
- `--lon-min/--lon-max/--lat-min/--lat-max`: set a custom crop (defaults to NA).
- `--auto-extent`: use the full native grid instead of the crop.

## Notes
- Requires `cfgrib`/`eccodes` to decode GRIB2; the provided `requirements.txt`
  installs wheels that bundle the needed libs.
- Network calls go straight to `https://noaa-rrfs-pds.s3.amazonaws.com`.
- The RRFS grid is huge (2961x4881). Expect ~30–60 seconds per frame with the
  default settings on a laptop.
