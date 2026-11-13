"""Load raw downloads into analysis-ready tables."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from . import config as C

ALBERTA_3TM = 3776  # NAD83(CSRS) / Alberta 3TM ref merid 114 W (metres)
CENSUS_NUM = ["res_cnt", "dwell_cnt", "sing_famly", "duplex", "multi_plex", "apartment", "town_house",
              "manuf_home", "conv_struc"]


def _code(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def load_complaints() -> pd.DataFrame:
    """311 drainage-related requests -> [date, comm_code, group], one row per request."""
    df = pd.read_csv(C.RAW_DIR / "311_drainage.csv.gz", dtype=str, keep_default_na=False)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["requested_date"]).dt.normalize(),
        "comm_code": _code(df["comm_code"]),
        "group": df["service_name"].map(C.SERVICE_TO_GROUP),
    })
    return out[(out["comm_code"] != "") & out["group"].notna()].reset_index(drop=True)


def load_communities() -> gpd.GeoDataFrame:
    """Community polygons + centroid (lat/lon) + area (km2)."""
    gdf = gpd.read_file(C.RAW_DIR / "communities.geojson")
    gdf["comm_code"] = _code(gdf["comm_code"])
    gdf = gdf.drop_duplicates("comm_code").reset_index(drop=True)
    proj = gdf.to_crs(ALBERTA_3TM)
    gdf["area_km2"] = proj.area / 1e6
    centroid = proj.centroid.to_crs(4326)
    gdf["lon"], gdf["lat"] = centroid.x.to_numpy(), centroid.y.to_numpy()
    return gdf


def load_census() -> pd.DataFrame:
    """2019 census counts and dwelling-type shares by community."""
    df = pd.read_csv(C.RAW_DIR / "census_2019.csv", dtype=str, keep_default_na=False)
    df["comm_code"] = _code(df["comm_code"])
    for col in CENSUS_NUM:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    dwell = df["dwell_cnt"].where(df["dwell_cnt"] > 0)
    df["share_single"] = df["sing_famly"] / dwell
    df["share_multi"] = (df["duplex"] + df["multi_plex"] + df["town_house"]) / dwell
    df["share_apartment"] = df["apartment"] / dwell
    return df.drop_duplicates("comm_code").set_index("comm_code")


def load_gauges() -> pd.DataFrame:
    """Rain gauge locations indexed by channel."""
    df = pd.read_csv(C.RAW_DIR / "rain_gauges.csv", dtype=str, keep_default_na=False)
    out = pd.DataFrame({
        "channel": pd.to_numeric(df["channel"], errors="coerce"),
        "name": df["name"],
        "elevation_m": pd.to_numeric(df["elevation"], errors="coerce"),
        "lon": pd.to_numeric(df["longtitude"], errors="coerce"),
        "lat": pd.to_numeric(df["latittude"], errors="coerce"),
    }).dropna(subset=["channel", "lon", "lat"])
    out["channel"] = out["channel"].astype(int)
    return out.drop_duplicates("channel").set_index("channel")


def daily_counts(complaints: pd.DataFrame, comm_codes: list[str], groups: list[str],
                 start=None, end=None) -> pd.DataFrame:
    """Dense (date x community) complaint counts for the given 311 groups."""
    sel = complaints[complaints["group"].isin(groups) & complaints["comm_code"].isin(comm_codes)]
    idx = pd.date_range(start or complaints["date"].min(), end or complaints["date"].max(), freq="D")
    tab = sel.groupby(["date", "comm_code"]).size().unstack(fill_value=0)
    return tab.reindex(index=idx, columns=comm_codes, fill_value=0).astype(float)


def community_statics(comm: gpd.GeoDataFrame, census: pd.DataFrame) -> pd.DataFrame:
    """Static per-community features (index = comm_code)."""
    st = comm.set_index("comm_code")[["name", "class", "sector", "srg", "comm_structure", "area_km2", "lat", "lon"]].copy()
    st = st.join(census[["res_cnt", "dwell_cnt", "share_single", "share_multi", "share_apartment"]])
    for col in ("class", "sector", "srg", "comm_structure"):
        st[f"{col}_code"] = st[col].astype("category").cat.codes.replace(-1, np.nan)
    st["res_density"] = st["res_cnt"] / st["area_km2"]
    return st
