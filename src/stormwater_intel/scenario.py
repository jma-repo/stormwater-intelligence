"""Score a hypothetical design storm with the history available today."""
from __future__ import annotations

import pandas as pd

from . import config as C
from . import model as Mdl
from . import panel as P
from .pipeline import Inputs

MIN_DWELLINGS_FOR_RATE = 500   # 2019 census dwellings needed before a per-dwelling rate is shown

# Uniform storm: 30 mm in one day, 15 mm in the worst hour, a moderately wet lead-up, mid-July.
DEFAULT_STORM = dict(rain_mm=30.0, peak_60min_mm=15.0, prev7_mm=10.0, prev30_mm=40.0,
                     doy=196, days_since_prev_event=14.0, n_gauges_hit=20)


def design_storm_frame(inp: Inputs, **storm) -> pd.DataFrame:
    """Feature rows (one per community) for a synthetic storm starting the day after the data ends.

    History features come from the same code path as the backtest (the synthetic event is
    appended after all real events), so scenario scores are directly comparable to backtest scores.
    """
    p = {**DEFAULT_STORM, **storm}
    one = pd.Timedelta(days=1)
    cols = inp.y_daily.columns
    s = inp.y_daily.index[-1] + one
    e = s + (C.WINDOW_DAYS - 1) * one
    window = pd.date_range(s, e)

    def extend(frame: pd.DataFrame) -> pd.DataFrame:   # zero complaints for the synthetic window
        return pd.concat([frame, pd.DataFrame(0.0, index=window, columns=cols)])

    lead = pd.date_range(s - 30 * one, e)
    total = pd.DataFrame(0.0, index=lead, columns=cols)
    total.loc[s - 7 * one:s - one] = p["prev7_mm"] / 7
    total.loc[s - 30 * one:s - 8 * one] = (p["prev30_mm"] - p["prev7_mm"]) / 23
    total.loc[s] = p["rain_mm"]
    m60 = pd.DataFrame(0.0, index=lead, columns=cols)
    m60.loc[s] = p["peak_60min_mm"]

    design = pd.DataFrame([{
        "event_id": "design", "start": s, "window_end": e, "n_storm_days": 1,
        "peak_gauge_mm": p["rain_mm"], "n_gauges_hit": p["n_gauges_hit"], "city_mean_mm": p["rain_mm"],
    }])
    events = pd.concat([inp.events, design], ignore_index=True)
    aux = {g: extend(D) for g, D in _aux(inp).items()}
    panel = P.build_event_panel(
        events, extend(inp.y_daily), pd.concat([inp.fields["total"], total]),
        pd.concat([inp.fields["m60"], m60]), inp.statics, aux,
        pd.concat([inp.fields["k3"], total]) if "k3" in inp.fields else None)
    row = panel[panel["event_id"] == "design"].copy()
    row["doy"], row["days_since_prev_event"] = p["doy"], p["days_since_prev_event"]  # seasonal context, not data end date
    return row.reset_index(drop=True)


def _aux(inp: Inputs) -> dict[str, pd.DataFrame]:
    return inp.aux_daily


def score_design_storm(inp: Inputs, model, features: list[str] = P.FEATURES, **storm) -> pd.DataFrame:
    row = design_storm_frame(inp, **storm)
    out = row[["comm_code"]].copy()
    out["expected_complaints"] = model.predict(row[features])
    meta = inp.universe.set_index("comm_code")[["name", "sector", "comm_structure"]]
    out = out.join(meta, on="comm_code")
    out["risk_rank"] = out["expected_complaints"].rank(ascending=False, method="first").astype(int)
    out["risk_percentile"] = 100 * (1 - (out["risk_rank"] - 1) / len(out))
    out["expected_share_pct"] = 100 * out["expected_complaints"] / out["expected_complaints"].sum()
    # intensity view: big suburbs top the absolute ranking simply because they have more homes.
    # Blank below MIN_DWELLINGS_FOR_RATE (industrial areas, communities still being built in 2019).
    out["dwellings_2019"] = out["comm_code"].map(inp.statics["dwell_cnt"])
    out["expected_per_1000_dwellings"] = (
        1000 * out["expected_complaints"] / out["dwellings_2019"].where(out["dwellings_2019"] >= MIN_DWELLINGS_FOR_RATE))
    return out.sort_values("risk_rank").reset_index(drop=True)
