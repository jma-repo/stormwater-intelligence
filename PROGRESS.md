# Progress and handoff

Updated 2026-11-13 (session 1). The PoC is complete, reproducible and published.

## Where things stand

- `stormwater-intel fetch` then `stormwater-intel run` rebuilds `data/` and `outputs/` from public APIs. 40 offline unit tests (`pytest`). `stormwater-intel maps` re-renders the map and static figure from `outputs/` without a model run.
- Headline numbers are in `README.md`; every number is in `outputs/metrics.json`.
- **Nothing has to be pulled by hand.** Optional upgrades are listed in `docs/MANUAL_DATA.md`.

## Environment notes

- Python 3.12 venv kept **outside** any cloud-synced folder (this repo may live in OneDrive; a venv or large data there churns the sync):
  `py -3.12 -m venv $env:USERPROFILE\.venvs\stormwater-intel` then `pip install -e ".[dev]"`.
- The machine's default `python` is 3.9 (too old for this package); call the venv's `python.exe`.
- PowerShell is the primary shell. `data/` (about 30 MB) is git-ignored, `outputs/` is committed.
- Ad-hoc scripts that read downloaded data: run with `python -I` and keep the data in `data/raw/` only.
- Package versions at build time: pandas 3.1, xgboost 3.4, geopandas 1.2, scikit-learn 1.9, rasterio 1.5.

## Decisions and why

| Decision | Why |
|----------|-----|
| City rain gauges (5-minute, 44-45 online per season) instead of one airport gauge | convective storms are local; the nearest gauge matches the ECCC airport record at r = 0.93 |
| Target = catch-basin + drainage-concern requests | catch basin runs 5.2x higher in storm windows; sewer maintenance 1.0x (not rain-driven) |
| Events = fixed 3-day tiled windows (>= 2 gauges with >= 10 mm) | 311 has dates only; response peaks on days 0-1; chain-merging made season-long mega events |
| Ranking universe = residential + industrial communities (266) | 99.4 % of target complaints |
| Community level only | open 311 publishes community centrepoints (99.6 % of target requests), not addresses |
| Walk-forward backtest, headline = train 2021-24 / test 2025 | matches the original plan; the other years show how stable it is |
| Rain features use realised rainfall | stands in for a forecast; documented as the main optimism in the README |

## Next steps, in priority order

1. **Storm sewer inventory** (catch basins, mains, ponds) to turn complaints into complaints per asset. Not open data: ask City Water Services or file a FOIP request.
2. Swap realised rainfall for a forecast (ECCC HRDPS / radar) so the backtest is operational; re-score honestly.
3. 2 m LiDAR DEM (`eink-tu9p`, file download) for depressions and flow accumulation; impervious surface 2023 (`7yr5-d2ns`) per community.
4. Radar precipitation instead of inverse-distance interpolation.
5. Dispatcher view: per-asset or per-dwelling intensity, and an alert threshold (probability of >= N complaints).
6. Choose and add a licence file (none yet).

## Known gaps

- Model design was iterated on the same 2019-2026 folds that are reported, so results are PoC-grade rather than a locked-box test.
- 2025 has only 12 storms; confidence intervals are wide (see README).
- The interactive map (`outputs/risk_map.html`) was checked structurally (4 layers, tooltips, no basemap-tile dependency) but not viewed in a browser during the build; the static `outputs/figures/risk_maps.png` was inspected.
- The per-dwelling intensity view uses 2019 census dwellings (stale for new suburbs); a newer dwelling count would improve it.
