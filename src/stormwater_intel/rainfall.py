"""Rain gauge processing.

sparse 5-min tips -> per gauge-day stats -> dense (date x gauge) matrix
-> inverse-distance-weighted rainfall at community centroids -> storm events.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0088
LAT0, LON0 = 51.05, -114.07  # Calgary centre; equirectangular error is <0.1% over the city
SEASON = ((5, 1), (9, 30))  # City gauges only record May 1 - Sep 30


def to_xy_km(lat, lon) -> np.ndarray:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = np.radians(lon - LON0) * np.cos(np.radians(LAT0)) * EARTH_R_KM
    y = np.radians(lat - LAT0) * EARTH_R_KM
    return np.column_stack([x, y])


def distance_matrix_km(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    return np.sqrt(((a_xy[:, None, :] - b_xy[None, :, :]) ** 2).sum(-1))


def load_tips(paths: list[Path], utc_offset_hours: float = 0.0, min_year: int | None = None,
              max_tip_mm: float | None = None) -> pd.DataFrame:
    """Read raw gauge files -> [channel, ts, mm] with positive tips only.

    Earlier paths win when the same (channel, ts) appears twice (list the QA/QC'd
    historical file before the raw current-year file). Timestamps are already local
    time (afternoon convective peak in the hour-of-day profile), so no shift by default.
    `max_tip_mm` clips physically implausible 5-minute records (a handful of >20 mm tips).
    """
    frames = []
    for priority, p in enumerate(paths):
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        frames.append(pd.DataFrame({
            "channel": pd.to_numeric(df["channel"], errors="coerce"),
            "ts": pd.to_datetime(df["timestamp"], errors="coerce"),
            "mm": pd.to_numeric(df["rainfall"], errors="coerce"),
            "priority": priority,
        }))
    tips = pd.concat(frames, ignore_index=True).dropna()
    tips = tips[tips["mm"] > 0].copy()
    tips["channel"] = tips["channel"].astype(int)
    if utc_offset_hours:
        tips["ts"] = tips["ts"] + pd.Timedelta(hours=utc_offset_hours)
    tips = tips.sort_values(["channel", "ts", "priority"]).drop_duplicates(["channel", "ts"], keep="first")
    if min_year is not None:
        tips = tips[tips["ts"].dt.year >= min_year]
    if max_tip_mm is not None:
        tips["mm"] = tips["mm"].clip(upper=max_tip_mm)
    return tips.drop(columns="priority").reset_index(drop=True)


def gauge_daily(tips: pd.DataFrame) -> pd.DataFrame:
    """Per gauge-day: total mm and the max 5/15/60-minute accumulations."""
    out = []
    for channel, g in tips.groupby("channel", sort=True):
        s = g.set_index("ts")["mm"].sort_index()
        d = pd.DataFrame({"mm": s, "r15": s.rolling("15min").sum(), "r60": s.rolling("60min").sum()})
        agg = d.groupby(d.index.floor("D")).agg(
            total_mm=("mm", "sum"), max_5min=("mm", "max"),
            max_15min=("r15", "max"), max_60min=("r60", "max"))
        agg["channel"] = channel
        out.append(agg.rename_axis("date").reset_index())
    return pd.concat(out, ignore_index=True)


def season_dates(year: int) -> pd.DatetimeIndex:
    (m0, d0), (m1, d1) = SEASON
    return pd.date_range(f"{year}-{m0:02d}-{d0:02d}", f"{year}-{m1:02d}-{d1:02d}", freq="D")


def active_gauge_seasons(tips: pd.DataFrame, min_tips: int = 20,
                         min_rel_total: float = 0.25) -> set[tuple[int, int]]:
    """(channel, year) pairs trusted as 'online'.

    The logs are sparse (rows only when it rained), so a dead gauge looks like a dry one.
    A gauge-season counts as online when it has >= `min_tips` records and a seasonal total
    of at least `min_rel_total` x the median gauge that year.
    """
    year = tips["ts"].dt.year
    stats = tips.groupby(["channel", year]).agg(n=("mm", "size"), total=("mm", "sum"))
    median_total = stats.groupby(level=1)["total"].median()
    out = set()
    for (ch, yr), row in stats.iterrows():
        if row["n"] >= min_tips and row["total"] >= min_rel_total * median_total.loc[yr]:
            out.add((int(ch), int(yr)))
    return out


def daily_matrix(gdaily: pd.DataFrame, active: set[tuple[int, int]], value: str = "total_mm") -> pd.DataFrame:
    """Dense (date x channel) matrix. 0 on online days with no tips, NaN when the gauge is offline."""
    years = sorted({y for _, y in active})
    dates = pd.DatetimeIndex(np.concatenate([season_dates(y).values for y in years]))
    channels = sorted({c for c, _ in active})
    online = pd.DataFrame(False, index=dates, columns=channels)
    for ch, yr in active:
        online.loc[season_dates(yr), ch] = True
    g = gdaily.pivot_table(index="date", columns="channel", values=value, aggfunc="max")
    g = g.reindex(index=online.index, columns=online.columns).fillna(0.0)  # no tips -> 0 mm
    return g.where(online)  # NaN while the gauge is offline


def idw(values: np.ndarray, dist_km: np.ndarray, power: float = 2.0, k: int | None = 6,
        eps_km: float = 0.5) -> np.ndarray:
    """Inverse-distance-weighted field. values (T, G) with NaN = offline; dist_km (C, G) -> (T, C)."""
    n_t, _ = values.shape
    w_all = 1.0 / np.maximum(dist_km, eps_km) ** power
    out = np.full((n_t, dist_km.shape[0]), np.nan)
    for t in range(n_t):
        v = values[t]
        ok = ~np.isnan(v)
        if not ok.any():
            continue
        w = w_all[:, ok]
        if k and w.shape[1] > k:
            cut = -np.partition(-w, k - 1, axis=1)[:, k - 1][:, None]
            w = np.where(w >= cut, w, 0.0)
        out[t] = (w * v[ok]).sum(axis=1) / w.sum(axis=1)
    return out


def community_fields(mats: dict[str, pd.DataFrame], gauges: pd.DataFrame, comm: pd.DataFrame, *,
                     power: float = 2.0, k: int | None = 6) -> dict[str, pd.DataFrame]:
    """Interpolate each (date x gauge) matrix to community centroids -> (date x comm_code).

    `gauges` is indexed by channel with lat/lon; `comm` has comm_code, lat, lon.
    """
    first = next(iter(mats.values()))
    channels = [c for c in first.columns if c in gauges.index]
    dist = distance_matrix_km(to_xy_km(comm["lat"], comm["lon"]),
                              to_xy_km(gauges.loc[channels, "lat"], gauges.loc[channels, "lon"]))
    return {name: pd.DataFrame(idw(m[channels].to_numpy(), dist, power, k), index=m.index,
                               columns=list(comm["comm_code"])) for name, m in mats.items()}


EVENT_COLS = ["event_id", "start", "window_end", "n_storm_days", "peak_gauge_mm",
              "n_gauges_hit", "city_mean_mm"]


def detect_storm_events(total_mm: pd.DataFrame, *, trigger_mm: float = 10.0, min_gauges: int = 2,
                        window_days: int = 3) -> pd.DataFrame:
    """Tile the season with fixed-length, non-overlapping response windows.

    A day is a storm day when at least `min_gauges` gauges recorded >= `trigger_mm`.
    The first storm day not already inside a window opens a new event whose window covers
    `window_days` days (storm day + lag). Storm days inside a window are absorbed, so a
    wet spell becomes consecutive events rather than one season-long cluster.
    """
    n_hit = (total_mm >= trigger_mm).sum(axis=1)
    rows: list[dict] = []
    until = None
    for d in n_hit.index[n_hit >= min_gauges]:
        if until is not None and d <= until:
            rows[-1]["n_storm_days"] += 1
            continue
        until = d + pd.Timedelta(days=window_days - 1)
        rows.append({"start": d, "window_end": until, "n_storm_days": 1})
    ev = pd.DataFrame(rows)
    if ev.empty:
        return pd.DataFrame(columns=EVENT_COLS)
    wins = [total_mm.loc[r.start:r.window_end] for r in ev.itertuples()]
    ev["event_id"] = ev["start"].dt.strftime("%Y-%m-%d")
    ev["peak_gauge_mm"] = [float(w.max().max()) for w in wins]
    ev["n_gauges_hit"] = [int((w.max() >= trigger_mm).sum()) for w in wins]
    ev["city_mean_mm"] = [float(w.sum().mean()) for w in wins]
    return ev[EVENT_COLS]
