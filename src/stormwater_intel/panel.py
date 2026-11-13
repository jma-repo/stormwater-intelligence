"""Event x community panel: post-storm complaint labels and leakage-free features.

One row per (storm event, community). Every feature uses information available before
the event window opens, except the rainfall features, which stand in for a rainfall
forecast / nowcast (i.e. the backtest assumes the storm itself is forecast well).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rainfall import distance_matrix_km, to_xy_km
from .terrain import TERRAIN_FEATURES

AUX_GROUPS = ["sewer", "manhole", "drainage_bylaw", "storm_pond", "erosion"]  # other 311 water-related requests
HEAVY_MM = 25.0   # events whose peak gauge reached this are "heavy"

RAIN_FEATURES = ["rain_win_mm", "rain_win_k3_mm", "rain_peak_day_mm", "rain_60min_mm", "rain_prev7_mm",
                 "rain_prev30_mm"]
EVENT_FEATURES = ["doy", "n_storm_days", "city_peak_gauge_mm", "city_n_gauges", "city_mean_mm",
                  "days_since_prev_event"]
HISTORY_FEATURES = ["h365", "hist_ev_mean", "hist_ev_share", "hist_ev_any", "hist_ev_n", "hist_ev_ewm",
                    "hist_heavy_share", "h365_per_dwell", "hist_per_dwell", "nbr_hist_share", "nbr_h365"]
AUX_FEATURES = [f"h365_{g}" for g in AUX_GROUPS]
STATIC_FEATURES = ["area_km2", "lat", "lon", "res_cnt", "dwell_cnt", "res_density", "share_single",
                   "share_multi", "share_apartment", "class_code", "sector_code", "srg_code",
                   "comm_structure_code"]
FEATURES = RAIN_FEATURES + EVENT_FEATURES + HISTORY_FEATURES + AUX_FEATURES + STATIC_FEATURES + TERRAIN_FEATURES

FEATURE_SETS = {
    "rain + storm only": RAIN_FEATURES + EVENT_FEATURES,
    "terrain only (30 m DEM)": TERRAIN_FEATURES,
    "community static only": STATIC_FEATURES,
    "history only": HISTORY_FEATURES + AUX_FEATURES,
    "history + rain": RAIN_FEATURES + EVENT_FEATURES + HISTORY_FEATURES + AUX_FEATURES,
    "history + rain + static": FEATURES[:len(FEATURES) - len(TERRAIN_FEATURES)],
    "all (incl. terrain)": FEATURES,
}


class HistoryTracker:
    """Running per-community complaint history over the events seen so far."""

    def __init__(self, n: int, alpha: float = 0.08, heavy_mm: float = HEAVY_MM):
        self.n, self.alpha, self.heavy_mm = n, alpha, heavy_mm
        self.sum_y = np.zeros(n)
        self.sum_any = np.zeros(n)
        self.heavy_sum = np.zeros(n)
        self.ewm = np.full(n, np.nan)
        self.n_prev = 0
        self.n_heavy = 0

    def update(self, y: np.ndarray, peak_gauge_mm: float) -> None:
        y = np.asarray(y, dtype=float)
        self.sum_y += y
        self.sum_any += (y > 0)
        self.ewm = y.copy() if self.n_prev == 0 else (1 - self.alpha) * self.ewm + self.alpha * y
        if peak_gauge_mm >= self.heavy_mm:
            self.heavy_sum += y
            self.n_heavy += 1
        self.n_prev += 1

    def features(self) -> dict[str, np.ndarray]:
        nan = np.full(self.n, np.nan)
        if self.n_prev == 0:
            return {"hist_ev_mean": nan, "hist_ev_share": nan, "hist_ev_any": nan,
                    "hist_ev_n": np.zeros(self.n), "hist_ev_ewm": nan, "hist_heavy_share": nan}
        total, heavy_total = self.sum_y.sum(), self.heavy_sum.sum()
        return {
            "hist_ev_mean": self.sum_y / self.n_prev,
            "hist_ev_share": self.sum_y / total if total > 0 else nan,
            "hist_ev_any": self.sum_any / self.n_prev,
            "hist_ev_n": np.full(self.n, float(self.n_prev)),
            "hist_ev_ewm": self.ewm.copy(),
            "hist_heavy_share": self.heavy_sum / heavy_total if heavy_total > 0 else nan,
        }


def per_dwelling(values: np.ndarray, dwell: np.ndarray) -> np.ndarray:
    dwell = np.asarray(dwell, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dwell > 0, np.asarray(values, dtype=float) / dwell, np.nan)


def knn_weights(lat, lon, k: int = 5) -> np.ndarray:
    """Row-normalised k-nearest-neighbour matrix (self excluded) on community centroids."""
    xy = to_xy_km(lat, lon)
    d = distance_matrix_km(xy, xy)
    np.fill_diagonal(d, np.inf)
    nearest = np.argsort(d, axis=1)[:, :k]
    w = np.zeros_like(d)
    w[np.arange(len(d))[:, None], nearest] = 1.0 / k
    return w


def build_event_panel(events: pd.DataFrame, y_daily: pd.DataFrame, rain_total: pd.DataFrame,
                      rain_60: pd.DataFrame, statics: pd.DataFrame,
                      aux_daily: dict[str, pd.DataFrame] | None = None,
                      rain_local: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assemble the panel.

    events      : output of rainfall.detect_storm_events
    y_daily     : (date x community) target complaint counts, dense daily index
    rain_total  : (season date x community) IDW daily rainfall, mm
    rain_60     : (season date x community) IDW of each gauge's max 60-min accumulation, mm
    statics     : community static features indexed by comm_code
    aux_daily   : {group: (date x community)} other 311 water-related requests (history only)
    rain_local  : sharper (3 nearest gauges, power 3) variant of rain_total
    """
    comms = list(y_daily.columns)
    static_block = statics.reindex(index=comms, columns=STATIC_FEATURES + TERRAIN_FEATURES).reset_index(drop=True)
    dwell = static_block["dwell_cnt"].to_numpy()
    nbr_w = knn_weights(static_block["lat"], static_block["lon"])
    rain_local = rain_total if rain_local is None else rain_local
    cum = y_daily.cumsum()
    aux_cum = {g: (aux_daily[g].reindex(columns=comms, fill_value=0).cumsum() if aux_daily and g in aux_daily else None)
               for g in AUX_GROUPS}
    first = y_daily.index[0]
    one = pd.Timedelta(days=1)

    def cum_at(frame: pd.DataFrame | None, date: pd.Timestamp) -> np.ndarray:
        if frame is None or date < first:
            return np.zeros(len(comms))
        return frame.loc[date].to_numpy()

    tracker = HistoryTracker(len(comms))
    frames = []
    prev_end = None
    for ev in events.sort_values("start").itertuples(index=False):
        s, e = ev.start, ev.window_end
        y = y_daily.loc[s:e].sum().to_numpy()
        win, win60 = rain_total.loc[s:e], rain_60.loc[s:e]
        h365 = cum_at(cum, s - one) - cum_at(cum, s - 366 * one)
        hist = tracker.features()
        feat = {
            "event_id": ev.event_id, "comm_code": comms, "start": s, "year": s.year, "y": y,
            "rain_win_mm": win.sum(min_count=1).to_numpy(),
            "rain_win_k3_mm": rain_local.loc[s:e].sum(min_count=1).to_numpy(),
            "rain_peak_day_mm": win.max().to_numpy(),
            "rain_60min_mm": win60.max().to_numpy(),
            "rain_prev7_mm": rain_total.loc[s - 7 * one:s - one].sum(min_count=1).to_numpy(),
            "rain_prev30_mm": rain_total.loc[s - 30 * one:s - one].sum(min_count=1).to_numpy(),
            "doy": s.day_of_year, "n_storm_days": ev.n_storm_days,
            "city_peak_gauge_mm": ev.peak_gauge_mm, "city_n_gauges": ev.n_gauges_hit,
            "city_mean_mm": ev.city_mean_mm,
            "days_since_prev_event": np.nan if prev_end is None else float((s - prev_end).days),
            "h365": h365, **hist,
            "h365_per_dwell": per_dwelling(h365, dwell),
            "hist_per_dwell": per_dwelling(hist["hist_ev_mean"], dwell),
            "nbr_hist_share": nbr_w @ np.nan_to_num(hist["hist_ev_share"]) if tracker.n_prev else np.nan,
            "nbr_h365": nbr_w @ h365,
        }
        for g in AUX_GROUPS:
            feat[f"h365_{g}"] = cum_at(aux_cum[g], s - one) - cum_at(aux_cum[g], s - 366 * one)
        frames.append(pd.concat([pd.DataFrame(feat), static_block], axis=1))
        tracker.update(y, ev.peak_gauge_mm)   # only AFTER the row is built: no label leakage
        prev_end = e
    return pd.concat(frames, ignore_index=True)
