"""Baseline rankings, the XGBoost count model and the walk-forward backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from . import metrics as M
from .panel import FEATURES

XGB_PARAMS = dict(
    objective="count:poisson", n_estimators=300, learning_rate=0.03, max_depth=3,
    min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0,
    tree_method="hist", random_state=0, n_jobs=4,
)

# Naive rankings the model has to beat. All use only information available before the event.
BASELINES = {
    "random": lambda d: np.zeros(len(d)),
    "population": lambda d: d["dwell_cnt"].fillna(0).to_numpy(),
    "history_365d": lambda d: d["h365"].fillna(0).to_numpy(),
    "history_events": lambda d: d["hist_ev_mean"].fillna(0).to_numpy(),
    "rain_only": lambda d: d["rain_win_mm"].fillna(0).to_numpy(),
    "history_x_rain": lambda d: (d["hist_ev_mean"].fillna(0) * d["rain_win_mm"].fillna(0)).to_numpy(),
}
MODEL_ORDER = ["random", "population", "rain_only", "history_365d", "history_events", "history_x_rain", "xgb"]


def add_baseline_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name, fn in BASELINES.items():
        out[f"score_{name}"] = fn(out)
    return out


def fit_xgb(train: pd.DataFrame, features: list[str] = FEATURES, params: dict | None = None) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**{**XGB_PARAMS, **(params or {})})
    model.fit(train[features], train["y"])
    return model


def walk_forward(panel: pd.DataFrame, test_years: list[int], *, train_years: int = 4,
                 features: list[str] = FEATURES, params: dict | None = None):
    """For each test year, train on the previous `train_years` seasons and score that year.

    Returns (scored test rows with baseline + xgb scores, {year: fitted model}).
    """
    outs, models = [], {}
    for ty in test_years:
        train = panel[(panel["year"] >= ty - train_years) & (panel["year"] < ty)]
        test = panel[panel["year"] == ty]
        if train.empty or test.empty:
            continue
        model = fit_xgb(train, features, params)
        scored = add_baseline_scores(test)
        scored["score_xgb"] = model.predict(test[features])
        outs.append(scored)
        models[ty] = model
    return pd.concat(outs, ignore_index=True), models


def summarize(scored: pd.DataFrame, *, frac: float = 0.10, models: list[str] | None = None,
              n_boot: int = 2000, min_total_macro: float = 10) -> pd.DataFrame:
    """One row per ranking: pooled top-`frac` capture with a bootstrap interval over events."""
    rows = []
    for name in models or [m for m in MODEL_ORDER if f"score_{m}" in scored]:
        per_event = M.per_event_capture(scored, f"score_{name}", frac=frac)
        lo, hi = M.bootstrap_pooled(per_event, n_boot=n_boot)
        pooled = M.pooled_capture(per_event)
        n_units = int(scored.groupby("event_id").size().iloc[0])
        rows.append({
            "model": name, "capture": pooled, "ci_lo": lo, "ci_hi": hi,
            "macro_capture": M.macro_capture(per_event, min_total_macro),
            "lift_vs_random": pooled / M.selected_fraction(n_units, frac),
            "n_events": int((per_event["total"] > 0).sum()), "n_complaints": int(per_event["total"].sum()),
        })
    return pd.DataFrame(rows)
