# Data you would have to pull by hand

**Nothing here blocks the PoC.** Everything the PoC uses is downloaded by `stormwater-intel fetch`
from public APIs (Open Calgary / Socrata, Environment and Climate Change Canada, the Copernicus DEM
on AWS open data); no keys, no logins.

The items below would make the *next* iteration better, ordered by expected value for the model.

| # | Data | Why it matters | Where / how | Status |
|---|------|----------------|-------------|--------|
| 1 | **Storm sewer inventory**: catch basins (point, install year), storm mains (diameter, material, install year), outfalls, storm ponds, culverts | Complaint counts scale with the number and age of catch basins in a community. This is the single biggest missing input: it turns "complaints" into "complaints per asset" and gives the model physical exposure instead of proxies (population, community age, location). | **Not in Open Calgary** (catalog searched for storm, drain, catch basin, pond, pipe, sewer). Ask City Water Services / the Open Data team, or file a FOIP request. Save under `data/manual/storm_network/`. | Not obtainable automatically |
| 2 | **2 m LiDAR DEM** | Micro-topography: depressions and road sags that pond water, flow accumulation, height above nearest creek. The 30 m public surface model used today only describes where a community sits, not where water collects. | Open Calgary dataset `eink-tu9p` ("Digital Elevation Model (DEM) - ASCII 2m"). It is a **file download**, not an API table, and large. Save under `data/manual/dem_2m/`. | Code for it not written yet |
| 3 | **Impervious surface 2023** | Runoff volume per community. | Open Calgary `7yr5-d2ns` (1.0 M polygons). The API works but is heavy; easier: dataset page, Export, GeoJSON or Shapefile, then save under `data/manual/impervious/`. | Aggregation code not written yet |
| 4 | **Address-level 311 locations** | Open 311 only publishes the *community centrepoint* (`location_type = Community Centrepoint` for 99.6 % of the catch-basin and drainage requests). Grid or street-level hotspots, as in the original week-2 plan, are impossible with open data. | Needs a data-sharing agreement or FOIP request. | Not obtainable automatically |
| 5 | **Gridded / radar rainfall** (optional) | The City gauge network (44-45 gauges online per season, median nearest-neighbour spacing 3.4 km) is interpolated by inverse-distance weighting. Radar-based precipitation analyses resolve storm cells better, and local rainfall is where the model gets its edge over a history-only ranking. | MSC Datamart / GeoMet (public). Large GRIB2 files, so worth doing as its own task. | Not needed for the PoC |

## What was checked and is **not** needed

* **Rain**: the City's own 5-minute gauge network (`d9kv-swk3` historical, `c7sr-67sr` current year, `7rke-qemw` locations) is used instead of the single Calgary airport gauge. Daily totals at the City gauge nearest to the airport station correlate with the Environment Canada record at r = 0.93 (2,185 season days), so ECCC is only a cross-check.
* **Sewer backups**: "WATS - Sewer Maintenance" requests are not rain-driven in 311 (1.01x more per day inside storm windows; catch-basin requests: 5.2x), so they are excluded from the target. See `outputs/category_rain_sensitivity.csv`.
* **Terrain**: a 30 m Copernicus DEM is streamed automatically (`stormwater-intel fetch --only dem`).
* **Boundaries and demographics**: `surr-xmvs` (community polygons) and `rkfr-buzb` (2019 census by community) come from the API.
