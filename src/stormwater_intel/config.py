"""Paths, dataset ids and modelling constants (single source of truth)."""
from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve().parents[2]
    return here if (here / "pyproject.toml").exists() else Path.cwd()


REPO_ROOT = _repo_root()
DATA_DIR = Path(os.environ.get("STORMWATER_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
OUTPUT_DIR = Path(os.environ.get("STORMWATER_OUTPUT_DIR", REPO_ROOT / "outputs"))

# ---- Open Calgary (Socrata) dataset ids ---------------------------------
SOCRATA_HOST = "https://data.calgary.ca"
DATASET_311 = "iahh-g8bj"              # 311 Service Requests, 2012-present
DATASET_RAIN_HISTORICAL = "d9kv-swk3"  # City rain gauges, 5-min, QA/QC'd annually
DATASET_RAIN_CURRENT = "c7sr-67sr"     # current-year raw rain gauge data
DATASET_RAIN_GAUGES = "7rke-qemw"      # gauge locations
DATASET_COMMUNITIES = "surr-xmvs"      # community district boundaries
DATASET_CENSUS_2019 = "rkfr-buzb"      # census by community (2019)

# ---- Environment and Climate Change Canada --------------------------------
ECCC_API = "https://api.weather.gc.ca"
ECCC_CLIMATE_ID = "3031092"            # CALGARY INTL A (station 50430)
YYC_LAT, YYC_LON = 51.1225, -114.0133  # station position, used to pick the nearest City gauge

# ---- 311 service categories ------------------------------------------------
# Names are exactly as they appear in `service_name`. Older names were renamed in
# 2016-2017 (311APP -> WATS), so each group lists all historical spellings.
SERVICE_GROUPS: dict[str, list[str]] = {
    "catch_basin": [
        "WATS - Catch Basin Concerns",
        "311APP - Catch Basin Concerns",
        "WATS - Catch Basin Concerns (DO NOT USE)",
    ],
    "drainage": [
        "WATS - Drainage Concerns",
        "Bylaw - Water and Drainage Concerns",
    ],
    "drainage_bylaw": ["WATS - Drainage Bylaw"],
    "storm_pond": ["WATS - Storm Pond Concerns"],
    "sewer": ["WATS - Sewer Maintenance"],
    "manhole": ["WATS - Manhole Concerns", "z - 311APP - Manhole Concerns"],
    "erosion": [
        "WATS - Erosion and Sediment Control",
        "WATR - Erosion and Sediment Control",
        "z - WATR - Erosion and Sediment Control",
    ],
}
SERVICE_TO_GROUP = {name: grp for grp, names in SERVICE_GROUPS.items() for name in names}

# ---- modelling choices (evidence in docs/DATA_SOURCES.md) -------------------
# catch_basin complaints jump ~5.6x inside storm windows, sewer maintenance only ~1.1x,
# so "drainage complaint" = catch basin + drainage concerns.
TARGET_GROUPS = ["catch_basin", "drainage"]
UNIVERSE_CLASSES = ("Residential", "Industrial")   # 266 communities
FIRST_YEAR = 2012                                  # 311 categories are consistent from 2012
MAX_TIP_MM = 10.0                                  # clip absurd 5-min gauge records (>120 mm/h)
TRIGGER_MM = 10.0                                  # storm day: >= MIN_GAUGES gauges with >= TRIGGER_MM
MIN_GAUGES = 2
WINDOW_DAYS = 3                                    # response window = storm day + 2 days (lag: day0 9x, day1 7x, day2 3x)
TOP_FRAC = 0.10                                    # headline: top 10% riskiest communities

COLS_311 = [
    "service_request_id", "requested_date", "closed_date", "status_description", "source",
    "service_name", "agency_responsible", "comm_code", "comm_name", "location_type",
    "latitude", "longitude",
]
