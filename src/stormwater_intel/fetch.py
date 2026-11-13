"""Download raw inputs into data/raw. Idempotent: existing files are skipped unless force=True."""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from . import config as C
from .socrata import fetch_all, get_text, new_session, resource_url

log = logging.getLogger(__name__)

CENSUS_COLS = [
    "comm_code", "name", "class", "sector", "srg", "comm_structure", "res_cnt", "dwell_cnt",
    "sing_famly", "duplex", "multi_plex", "apartment", "town_house", "manuf_home", "conv_struc",
]


def _out(name: str) -> Path:
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    return C.RAW_DIR / name


def _skip(path: Path, force: bool) -> bool:
    if path.exists() and not force:
        log.info("exists, skipping: %s", path.name)
        return True
    return False


def fetch_311(force: bool = False) -> Path:
    out = _out("311_drainage.csv.gz")
    if _skip(out, force):
        return out
    names = sorted(C.SERVICE_TO_GROUP)
    quoted = ", ".join("'" + n.replace("'", "''") + "'" for n in names)
    df = fetch_all(C.DATASET_311, select=",".join(C.COLS_311), where=f"service_name in ({quoted})")
    df.to_csv(out, index=False, compression="gzip")
    log.info("311 drainage-related rows: %d", len(df))
    return out


def fetch_rain(force: bool = False) -> list[Path]:
    paths = []
    for tag, dataset in (("historical", C.DATASET_RAIN_HISTORICAL), ("current", C.DATASET_RAIN_CURRENT)):
        out = _out(f"rain_5min_{tag}.csv.gz")
        paths.append(out)
        if _skip(out, force):
            continue
        df = fetch_all(dataset, select="channel,name,year,timestamp,rainfall")
        df.to_csv(out, index=False, compression="gzip")
        log.info("rain %s rows: %d", tag, len(df))
    return paths


def fetch_gauges(force: bool = False) -> Path:
    out = _out("rain_gauges.csv")
    if _skip(out, force):
        return out
    fetch_all(C.DATASET_RAIN_GAUGES, order=None).to_csv(out, index=False)
    return out


def fetch_communities(force: bool = False) -> Path:
    out = _out("communities.geojson")
    if _skip(out, force):
        return out
    s = new_session()
    text = get_text(s, resource_url(C.DATASET_COMMUNITIES, "geojson"), {"$limit": 5000})
    json.loads(text)  # validate
    out.write_text(text, encoding="utf-8")
    return out


def fetch_census(force: bool = False) -> Path:
    out = _out("census_2019.csv")
    if _skip(out, force):
        return out
    fetch_all(C.DATASET_CENSUS_2019, select=",".join(CENSUS_COLS), order=None).to_csv(out, index=False)
    return out


def fetch_eccc_daily(force: bool = False, first_year: int = 2012) -> Path:
    """Daily climate at Calgary Intl A (cross-check for the City gauge network)."""
    out = _out("eccc_yyc_daily.csv.gz")
    if _skip(out, force):
        return out
    s = new_session()
    rows: list[dict] = []
    for year in range(first_year, date.today().year + 1):
        text = get_text(s, f"{C.ECCC_API}/collections/climate-daily/items",
                        {"f": "json", "limit": 1000, "CLIMATE_IDENTIFIER": C.ECCC_CLIMATE_ID, "LOCAL_YEAR": year})
        rows.extend(f["properties"] for f in json.loads(text).get("features", []))
    pd.DataFrame(rows).to_csv(out, index=False, compression="gzip")
    log.info("ECCC daily rows: %d", len(rows))
    return out


def fetch_dem(force: bool = False, pad_deg: float = 0.06) -> Path:
    """30 m Copernicus DEM mosaic covering the community polygons (+ margin for neighbourhood terrain)."""
    out = _out("dem_cop30_calgary.tif")
    if _skip(out, force):
        return out
    import geopandas as gpd

    from .terrain import download_dem
    west, south, east, north = gpd.read_file(fetch_communities()).total_bounds
    return download_dem((west - pad_deg, south - pad_deg, east + pad_deg, north + pad_deg), out)


STEPS = {
    "311": fetch_311,
    "rain": fetch_rain,
    "gauges": fetch_gauges,
    "communities": fetch_communities,
    "census": fetch_census,
    "eccc": fetch_eccc_daily,
    "dem": fetch_dem,
}


def fetch_all_inputs(only: list[str] | None = None, force: bool = False) -> None:
    for name in only or list(STEPS):
        log.info("== fetch %s", name)
        STEPS[name](force=force)
