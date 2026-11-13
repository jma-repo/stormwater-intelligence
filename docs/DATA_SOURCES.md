# Data sources, quirks and the evidence behind the modelling choices

All inputs are public and fetched by `stormwater-intel fetch` into `data/raw/` (git-ignored).
City datasets: *Contains information licensed under the Open Government Licence - City of Calgary.*

## Inputs

| Input | Source / id | Used as |
|-------|-------------|---------|
| 311 service requests | Open Calgary `iahh-g8bj` (7.56 M rows, 2012 - present); 150,788 drainage-related rows pulled | target (catch basin + drainage requests) and history features (sewer, manhole, bylaw, storm pond, erosion) |
| Rain gauges, 5-minute | Open Calgary `d9kv-swk3` (historical, QA/QC'd yearly) + `c7sr-67sr` (current year, raw) | storm events and community rainfall |
| Gauge locations | Open Calgary `7rke-qemw` (45 gauges) | inverse-distance interpolation |
| Community boundaries | Open Calgary `surr-xmvs` (313 polygons: 223 residential, 43 industrial, 43 residual sub-areas, 4 major parks) | ranking universe = residential + industrial = **266 communities** (99.4 % of the target complaints) |
| Census by community 2019 | Open Calgary `rkfr-buzb` | population, dwellings, dwelling mix |
| Calgary Intl A daily climate | ECCC `api.weather.gc.ca`, climate id 3031092 | cross-check of the City gauges only |
| Copernicus GLO-30 DEM (30 m) | AWS open data, COG tiles N50/N51 x W114/W115 | community terrain features |

## Quirks that matter

1. **311 dates have no time of day** (`requested_date` is a date), so everything runs at daily resolution and a storm's response is a multi-day window.
2. **311 geography is the community centrepoint** (`location_type = Community Centrepoint`) for 99.6 % of the catch-basin and drainage requests (60,308 of 60,549). Community level is the finest resolution open data allows.
3. **Category renames in 2016-17**: `311APP - Catch Basin Concerns` became `WATS - Catch Basin Concerns` (with a transitional `(DO NOT USE)` label). All spellings are mapped to one group in `config.SERVICE_GROUPS`.
4. **Gauge logs are sparse**: a row exists only when the bucket tipped, so a dead gauge looks like a dry one. A gauge-season counts as online only with >= 20 records and a seasonal total >= 25 % of that year's median gauge.
5. **Implausible tips**: 89 of 571,240 five-minute records (2012 onwards) exceed 10 mm (>120 mm/h; the largest is 43.2 mm on 2016-07-15). They are clipped at 10 mm.
6. **Timestamps are local time** (the hour-of-day profile peaks between 15:00 and 19:00).
7. **Gauges record May 1 - Sep 30 only.** There are no spring snow-melt storms and antecedent rainfall for early-May events is partial.
8. **City gauges agree with Environment Canada**: daily totals at the City gauge nearest the airport station (Huntington Hills, 3.9 km away) vs the ECCC record, r = 0.93 over 2,185 season days; days >= 10 mm: 133 (City) vs 136 (ECCC). Reproduced in `outputs/metrics.json` (`eccc_crosscheck`).

## Evidence for the modelling choices

**What is a "drainage complaint"?** Citywide requests per day inside storm windows vs other rain-season days
(`outputs/category_rain_sensitivity.csv`):

| 311 group | per day in storm windows | per day otherwise | ratio |
|-----------|-------------------------:|------------------:|------:|
| catch basin concerns | 24.9 | 4.8 | **5.2x** |
| manhole concerns | 4.1 | 2.3 | 1.8x |
| drainage bylaw | 2.7 | 1.8 | 1.5x |
| erosion and sediment control | 2.7 | 2.2 | 1.3x |
| drainage concerns | 2.6 | 2.2 | 1.2x |
| storm pond concerns | 0.9 | 0.8 | 1.0x |
| sewer maintenance | 11.9 | 11.8 | **1.0x** |

Catch-basin requests are the rain-driven signal. Sewer maintenance (the largest water-related group) does not respond
to storms in 311, so sewer backups are not part of the target. The primary target is catch basin + drainage concerns;
`outputs/metrics.json` reports the result for two other definitions.

**How long does the response last?** Catch-basin requests relative to a quiet baseline, by day relative to the first
storm day (`outputs/lag_profile_catch_basin.csv`, events 2018 onwards): day 0 **7.3x**, day 1 **6.6x**, day 2 3.8x,
day 3 2.7x (days -3 and -2 are already ~3x because storms cluster). Response window = storm day + 2 days.

**Does severity matter?** Yes: complaints per event rise with peak gauge rainfall (`outputs/figures/dose_response.png`),
which is why the ranking is scored per storm and pooled with complaint weights.

**Do complaints concentrate in the same communities?** Yes, which is why history is a strong baseline: the ranking of
communities by the previous four years of post-storm complaints has a Spearman correlation of 0.66-0.78 with the
following year's (`outputs/spatial_persistence.csv`, test years 2019-2026).
