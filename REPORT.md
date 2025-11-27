# RRFS REFC Leaflet Pipeline – Integration Notes

Date: 2025-11-26  
Repo: `andrewnakas/rrfs_refc`  
Maintains: Hourly auto-built RRFS composite reflectivity (REFC) overlays for use in Tree60 Weather or other web UIs.

## What the pipeline produces
- **18 forecast frames** (F000–F017) from the latest *complete* RRFS natlev 3 km North America cycle.
- **Tiles:** `docs/tiles/refc_fXXX.png` (Web Mercator, transparent background, NWS reflectivity palette).
- **Metadata:** `docs/data.json` (frame list, bounds, stats, cycle time).
- **Viewer:** `docs/index.html` (Leaflet overlay with timeline + dateline wrapping).
- Deployment target: GitHub Pages (`main` branch, `docs/` artifact).

## Alignment & projection
- Source grid: RRFS native rotated lat/lon (4881 × 2961, pole −35° lat / 247° lon).
- Reprojection: Curvilinear → Web Mercator (EPSG:3857) using `pyproj` with spherical radius 6 371 229 m.
- Dateline handling: Viewer renders the overlay at lon, lon ± 360 so Asia/Russia appears on the correct side when panning.
- Bounds in `data.json` reflect the true extent of each frame (cropped to data envelope, plus full-grid union).

## Automation (GitHub Actions)
- Workflow: `.github/workflows/pages.yml`
- Schedule: `20 * * * *` (HH:20 UTC) — waits for NOAA to finish uploading the cycle.
- Steps: clean old artifacts → install deps → build 18 frames → upload `docs/` → deploy to Pages.
- Storage: old tiles/data are removed each run; no Git LFS required.

## Local regeneration (optional)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Latest complete cycle, 18 frames, 8 km grid, full extent
python scripts/build_rrfs_tiles.py --num-forecasts 18 --resolution-m 8000 --sample-stride 25 --auto-extent
```
Key flags:
- `--cycle YYYYMMDDHH` to pin a cycle.
- `--resolution-m` smaller → sharper, larger PNGs (default 8000 m).
- `--sample-stride` lower → more accurate NN mapping, slower (default 25).
- `--auto-extent` uses the full native domain; omit to set a custom crop via `--lon/lat-*`.

## Using the assets in Tree60 Weather
1) **Consume metadata**  
   Fetch `https://andrewnakas.github.io/rrfs_refc/data.json` (Pages URL).  
   Use `forecasts[*].tile` for image URLs and `forecasts[*].bounds` for the overlay corners.
2) **Leaflet (recommended)**  
   ```js
   const b = frame.bounds;
   const url = `https://andrewnakas.github.io/rrfs_refc/${frame.tile}`;
   L.imageOverlay(url, [[b.lat_min, b.lon_min], [b.lat_max, b.lon_max]], { opacity: 0.72 }).addTo(map);
   // optional dateline wrap:
   L.imageOverlay(url, [[b.lat_min, b.lon_min-360], [b.lat_max, b.lon_max-360]], { opacity: 0.72 }).addTo(map);
   ```
3) **Caching**  
   Append a cache-buster (e.g., `?v=${frame.valid_time}`) to ensure you always load the newest tiles.
4) **Attribution**  
   Data: NOAA RRFS (public). Color table: NWS reflectivity palette.

## Operational behavior
- Chooses the newest cycle that has ≥ requested frames (default 18). Avoids “single-frame” artifacts when NOAA is mid-upload.
- Cleans previous outputs before each build to keep the repo light.
- Uses nearest-neighbor to preserve discrete dBZ levels (no smoothing).

## Known limitations / next options
- Web Mercator distortion at high latitudes is inherent; aspect is preserved per-frame.  
- If sub-km alignment is needed, drop `--resolution-m` to 6000 or 4000 (expect larger tiles and longer build times).
- If you want CONUS-only faster tiles, add a crop (`--lon-min -130 --lon-max -60 --lat-min 20 --lat-max 55`).

## Quick embed snippet (Tree60)
```html
<iframe src="https://andrewnakas.github.io/rrfs_refc/" style="width:100%;height:600px;border:0;"></iframe>
```

## Contacts / handoff
- Repo owner: @andrewnakas
- This report authored by the build maintainer; pipeline is self-contained in the repo.
