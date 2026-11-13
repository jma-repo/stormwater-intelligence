"""Raw downloads -> event panel -> backtest -> outputs."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import metrics as M
from . import model as Mdl
from . import panel as P
from . import rainfall as R
from .terrain import load_terrain

log = logging.getLogger(__name__)

PRIMARY_YEAR = 2025                          # test year of the headline backtest (train 2021-2024)
TRAIN_YEARS = 4
WALK_FORWARD_YEARS = list(range(2019, 2027))
TARGET_VARIANTS = {                          # robustness: what counts as a "drainage complaint"
    "catch basin only": ["catch_basin"],
    "catch basin + drainage (primary)": ["catch_basin", "drainage"],
    "all rain-sensitive groups": ["catch_basin", "drainage", "storm_pond", "manhole", "drainage_bylaw"],
}


@dataclass
class Inputs:
    events: pd.DataFrame
    panel: pd.DataFrame
    universe: gpd.GeoDataFrame        # community polygons in the ranking universe
    statics: pd.DataFrame
    y_daily: pd.DataFrame             # (date x community) target complaints
    aux_daily: dict[str, pd.DataFrame]
    fields: dict[str, pd.DataFrame]   # community-level rainfall: total, m60, k3
    mat_total: pd.DataFrame           # (date x gauge) daily totals
    gauges: pd.DataFrame
    complaints: pd.DataFrame          # every 311 drainage-related request: date, comm_code, group


def build_inputs(*, trigger_mm: float = C.TRIGGER_MM, min_gauges: int = C.MIN_GAUGES,
                 window_days: int = C.WINDOW_DAYS, target_groups: list[str] | None = None) -> Inputs:
    raw = C.RAW_DIR
    tips = R.load_tips([raw / "rain_5min_historical.csv.gz", raw / "rain_5min_current.csv.gz"],
                       min_year=C.FIRST_YEAR, max_tip_mm=C.MAX_TIP_MM)
    active = R.active_gauge_seasons(tips)
    gdaily = R.gauge_daily(tips)
    mat_total = R.daily_matrix(gdaily, active, "total_mm")
    mat_60 = R.daily_matrix(gdaily, active, "max_60min")
    log.info("rain: %d tips, %d gauge-seasons online, matrix %s", len(tips), len(active), mat_total.shape)

    comm = D.load_communities()
    census = D.load_census()
    gauges = D.load_gauges()
    complaints = D.load_complaints()
    universe = comm[comm["class"].isin(C.UNIVERSE_CLASSES)].reset_index(drop=True)
    codes = list(universe["comm_code"])
    start = f"{C.FIRST_YEAR}-01-01"

    y_daily = D.daily_counts(complaints, codes, target_groups or C.TARGET_GROUPS, start=start)
    aux_daily = {g: D.daily_counts(complaints, codes, [g], start=start) for g in P.AUX_GROUPS}
    statics = D.community_statics(comm, census).reindex(codes)
    statics = statics.join(load_terrain(C.RAW_DIR / "dem_cop30_calgary.tif", universe))
    fields = R.community_fields({"total": mat_total, "m60": mat_60}, gauges, universe)
    fields["k3"] = R.community_fields({"k3": mat_total}, gauges, universe, power=3.0, k=3)["k3"]
    events = R.detect_storm_events(mat_total, trigger_mm=trigger_mm, min_gauges=min_gauges, window_days=window_days)
    panel = P.build_event_panel(events, y_daily, fields["total"], fields["m60"], statics, aux_daily, fields["k3"])
    log.info("panel: %d events x %d communities = %d rows", len(events), len(codes), len(panel))
    return Inputs(events, panel, universe, statics, y_daily, aux_daily, fields, mat_total, gauges, complaints)


# ---------------------------------------------------------------------------------------------
def category_sensitivity(inp: Inputs) -> pd.DataFrame:
    """Citywide requests per day inside storm windows vs other rain-season days, per 311 group."""
    season = inp.mat_total.index
    in_window = pd.Series(False, index=season)
    for ev in inp.events.itertuples():
        in_window.loc[ev.start:ev.window_end] = True
    rows = []
    for group in C.SERVICE_GROUPS:
        daily = inp.complaints[inp.complaints["group"] == group].groupby("date").size().reindex(season, fill_value=0)
        rows.append({"group": group, "per_day_in_storm_window": daily[in_window].mean(),
                     "per_day_other_season_days": daily[~in_window].mean()})
    out = pd.DataFrame(rows)
    out["ratio"] = out["per_day_in_storm_window"] / out["per_day_other_season_days"]
    return out.sort_values("ratio", ascending=False).reset_index(drop=True)


def spatial_persistence(panel: pd.DataFrame, years=WALK_FORWARD_YEARS, back: int = TRAIN_YEARS) -> pd.DataFrame:
    """Rank correlation between each community's post-storm complaints in the previous `back` years and in year Y."""
    by_year = panel.pivot_table(index="comm_code", columns="year", values="y", aggfunc="sum", fill_value=0)
    rows = []
    for year in years:
        prev = by_year[[c for c in by_year.columns if year - back <= c < year]].sum(axis=1)
        rows.append({"year": year, "spearman_prev_years_vs_year": prev.corr(by_year[year], method="spearman"),
                     "n_complaints": int(by_year[year].sum())})
    return pd.DataFrame(rows)


def eccc_crosscheck(inp: Inputs) -> dict:
    """Compare the City gauge nearest to Calgary Intl A with the Environment Canada daily record."""
    ec = pd.read_csv(C.RAW_DIR / "eccc_yyc_daily.csv.gz")
    ec["date"] = pd.to_datetime(ec["LOCAL_DATE"]).dt.normalize()
    eccc = ec.set_index("date")["TOTAL_PRECIPITATION"].rename("eccc")
    g = inp.gauges.loc[[c for c in inp.mat_total.columns if c in inp.gauges.index]]
    dist = R.distance_matrix_km(R.to_xy_km([C.YYC_LAT], [C.YYC_LON]), R.to_xy_km(g["lat"], g["lon"]))[0]
    nearest = int(g.index[int(np.argmin(dist))])
    joined = pd.concat([eccc, inp.mat_total[nearest].rename("city")], axis=1, join="inner").dropna()
    return {"nearest_city_gauge": f"{g.loc[nearest, 'name']} (ch {nearest})",
            "distance_km": round(float(dist.min()), 1), "overlap_days": int(len(joined)),
            "pearson_r": round(float(joined["eccc"].corr(joined["city"])), 3),
            "days_ge_10mm_eccc": int((joined["eccc"] >= 10).sum()), "days_ge_10mm_city": int((joined["city"] >= 10).sum())}


def lag_profile(inp: Inputs, group: str = "catch_basin", offsets=range(-3, 9), first_year: int = 2018) -> pd.DataFrame:
    """Mean daily requests relative to the first storm day of each event, as a multiple of the quiet baseline."""
    idx = pd.date_range(f"{C.FIRST_YEAR}-01-01", inp.y_daily.index[-1], freq="D")
    daily = inp.complaints[inp.complaints["group"] == group].groupby("date").size().reindex(idx, fill_value=0)
    starts = [s for s in inp.events["start"] if s.year >= first_year]
    near = pd.Series(False, index=idx)
    for s in starts:
        near.loc[s - pd.Timedelta(days=3):s + pd.Timedelta(days=7)] = True
    quiet = daily[idx.isin(inp.mat_total.index) & ~near.to_numpy() & (idx.year >= first_year)].mean()
    rows = [{"day_offset": k, "ratio_to_quiet_baseline":
             float(np.mean([daily.get(s + pd.Timedelta(days=k), np.nan) for s in starts])) / quiet} for k in offsets]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
def _py(obj):
    """json default: numpy / pandas scalars -> python."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (pd.Timestamp, date)):
        return obj.strftime("%Y-%m-%d")
    raise TypeError(type(obj))


def _records(df: pd.DataFrame, nd: int = 4) -> list[dict]:
    return json.loads(json.dumps(df.round(nd).to_dict("records"), default=_py))


def _paired(scored: pd.DataFrame, baseline: str) -> dict:
    est, lo, hi = M.paired_bootstrap_diff(M.per_event_capture(scored, "score_xgb"),
                                          M.per_event_capture(scored, f"score_{baseline}"))
    return {"baseline": baseline, "diff_pts": round(100 * est, 2), "ci_lo_pts": round(100 * lo, 2),
            "ci_hi_pts": round(100 * hi, 2)}


def _best_naive(summary: pd.DataFrame) -> str:
    cand = summary[~summary["model"].isin(["xgb", "random"])]
    return str(cand.sort_values("capture", ascending=False).iloc[0]["model"])


def render_maps(out: Path, universe: gpd.GeoDataFrame, gauges: pd.DataFrame) -> None:
    """Interactive map + static figure from the CSVs/metrics already in `out` (no model run needed)."""
    from . import report as RP

    out = Path(out)
    biggest = json.loads((out / "metrics.json").read_text(encoding="utf-8"))["largest_2025_event"]["event_id"]
    ranking = pd.read_csv(out / "risk_ranking_design_storm_30mm.csv").set_index("comm_code")
    event = pd.read_csv(out / f"backtest_{biggest}_predicted_vs_actual.csv").set_index("comm_code")
    per_dwelling_pct = 100 * ranking["expected_per_1000_dwellings"].rank(pct=True)   # NaN stays NaN (grey)
    layers = {
        "Design storm (30 mm uniform): risk percentile": {
            "values": ranking["risk_percentile"], "kind": "percentile",
            "label": "Risk percentile (design storm)", "show": True},
        "Design storm (30 mm uniform): risk per 1,000 dwellings, percentile": {
            "values": per_dwelling_pct, "kind": "percentile",
            "label": "Risk percentile per 1,000 dwellings", "show": False},
        f"Backtest {biggest}: predicted risk percentile": {
            "values": event["pred_percentile"], "kind": "percentile",
            "label": "Predicted risk percentile", "show": False},
        f"Backtest {biggest}: actual complaints": {
            "values": event["y"], "kind": "count",
            "label": "Actual complaints (3-day window)", "show": False},
    }
    (out / "figures").mkdir(parents=True, exist_ok=True)
    RP.build_map(universe, gauges, layers, out / "risk_map.html")
    count_max = float(np.percentile(event["y"], 98))
    RP.plot_static_map(universe, gauges, [
        {"title": "Design storm (30 mm): risk percentile", "values": ranking["risk_percentile"],
         "cmap": "YlOrRd", "vmin": 0, "vmax": 100, "label": "risk percentile", "annotate": True},
        {"title": f"{biggest} storm: model's predicted risk", "values": event["pred_percentile"],
         "cmap": "YlOrRd", "vmin": 0, "vmax": 100, "label": "risk percentile"},
        {"title": f"{biggest} storm: actual complaints (3 days)", "values": event["y"],
         "cmap": "Blues", "vmin": 0, "vmax": max(count_max, 1), "label": "complaints"},
    ], out / "figures" / "risk_maps.png")


def run_poc(out_dir: Path | None = None, skip_ablation: bool = False) -> dict:
    """Run the whole analysis and write metrics, tables, figures and the map to `out_dir`."""
    from . import report as RP
    from . import scenario as SC

    out = Path(out_dir or C.OUTPUT_DIR)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    inp = build_inputs()
    panel, events = inp.panel, inp.events
    event_meta = events.set_index("event_id")

    # ---- backtests -------------------------------------------------------------------------
    scored, models = Mdl.walk_forward(panel, WALK_FORWARD_YEARS, train_years=TRAIN_YEARS)
    s25, s26 = scored[scored["year"] == PRIMARY_YEAR], scored[scored["year"] == 2026]
    summ = {"primary_2025": Mdl.summarize(s25), "forward_2026": Mdl.summarize(s26),
            "walk_forward": Mdl.summarize(scored)}
    paired = {"primary_2025": [_paired(s25, b) for b in ("history_x_rain", "history_events", "population")],
              "walk_forward": [_paired(scored, b) for b in ("history_x_rain", "history_events", "population")]}

    tier = pd.cut(scored["event_id"].map(event_meta["peak_gauge_mm"]), [0, 20, 30, 1000],
                  labels=["peak < 20 mm", "peak 20-30 mm", "peak > 30 mm"])
    by_tier = []
    for name, g in scored.groupby(tier, observed=True):
        row = {"tier": str(name), "n_events": int(g["event_id"].nunique()), "n_complaints": int(g["y"].sum())}
        for m in ("random", "history_events", "history_x_rain", "xgb"):
            row[m] = round(M.pooled_capture(M.per_event_capture(g, f"score_{m}")), 4)
        by_tier.append(row)

    ablation = sensitivity = None
    if not skip_ablation:
        rows = []
        for name, feats in P.FEATURE_SETS.items():
            sc, _ = Mdl.walk_forward(panel, WALK_FORWARD_YEARS, train_years=TRAIN_YEARS, features=feats)
            rows.append({"features": name, "n_features": len(feats),
                         "walk_forward": M.pooled_capture(M.per_event_capture(sc, "score_xgb")),
                         "y2025": M.pooled_capture(M.per_event_capture(sc[sc["year"] == PRIMARY_YEAR], "score_xgb"))})
        ablation = pd.DataFrame(rows)
        rows = []
        for ty in (2, 4, 6, 8):
            sc, _ = Mdl.walk_forward(panel, WALK_FORWARD_YEARS, train_years=ty)
            rows.append({"train_years": ty,
                         "walk_forward": M.pooled_capture(M.per_event_capture(sc, "score_xgb")),
                         "y2025": M.pooled_capture(M.per_event_capture(sc[sc["year"] == PRIMARY_YEAR], "score_xgb"))})
        sensitivity = pd.DataFrame(rows)

    target_sens = None
    if not skip_ablation:
        rows = []
        for label, groups in TARGET_VARIANTS.items():
            alt = build_inputs(target_groups=groups).panel
            sc, _ = Mdl.walk_forward(alt, WALK_FORWARD_YEARS, train_years=TRAIN_YEARS)
            row = {"target": label, "n_complaints_wf": int(sc["y"].sum())}
            for m in ("history_x_rain", "xgb"):
                row[f"{m}_wf"] = M.pooled_capture(M.per_event_capture(sc, f"score_{m}"))
                row[f"{m}_2025"] = M.pooled_capture(M.per_event_capture(sc[sc["year"] == PRIMARY_YEAR], f"score_{m}"))
            rows.append(row)
        target_sens = pd.DataFrame(rows)

    # ---- tables ----------------------------------------------------------------------------
    sens_cat, lag, persist = category_sensitivity(inp), lag_profile(inp), spatial_persistence(panel)
    sens_cat.round(3).to_csv(out / "category_rain_sensitivity.csv", index=False)
    lag.round(3).to_csv(out / "lag_profile_catch_basin.csv", index=False)
    persist.round(3).to_csv(out / "spatial_persistence.csv", index=False)
    raw_tips = R.load_tips([C.RAW_DIR / "rain_5min_historical.csv.gz", C.RAW_DIR / "rain_5min_current.csv.gz"],
                           min_year=C.FIRST_YEAR)
    data_quality = {"tips_total": int(len(raw_tips)), "tips_over_cap": int((raw_tips["mm"] > C.MAX_TIP_MM).sum()),
                    "cap_mm": C.MAX_TIP_MM, "largest_tip_mm": float(raw_tips["mm"].max())}
    for key, df in summ.items():
        df.round(4).to_csv(out / f"backtest_summary_{key}.csv", index=False)
    pe = {m: M.per_event_capture(s25, f"score_{m}").set_index("event_id") for m in ("random", "history_events", "xgb")}
    ev25 = pe["xgb"][["total"]].rename(columns={"total": "complaints"}).join(
        event_meta.loc[pe["xgb"].index, ["start", "peak_gauge_mm", "n_gauges_hit"]])
    for m, d in pe.items():
        ev25[f"capture_{m}"] = d["capture"]
    ev25 = ev25.reset_index()
    numeric = ev25.select_dtypes("number").columns
    ev25[numeric] = ev25[numeric].round(3)
    ev25.to_csv(out / "backtest_events_2025.csv", index=False)

    # ---- scenario ranking + largest 2025 storm --------------------------------------------
    final_model = Mdl.fit_xgb(panel[panel["year"] >= panel["year"].max() - (TRAIN_YEARS - 1)])
    ranking = SC.score_design_storm(inp, final_model)
    ranking.round(3).to_csv(out / "risk_ranking_design_storm_30mm.csv", index=False)

    biggest = s25.groupby("event_id")["y"].sum().idxmax()
    g = s25[s25["event_id"] == biggest].copy()
    g["pred_percentile"] = 100 * (1 - (g["score_xgb"].rank(ascending=False, method="first") - 1) / len(g))
    names = inp.universe.set_index("comm_code")["name"]
    g["name"] = g["comm_code"].map(names)
    g[["comm_code", "name", "y", "score_xgb", "pred_percentile"]].sort_values("score_xgb", ascending=False) \
        .round(3).to_csv(out / f"backtest_{biggest}_predicted_vs_actual.csv", index=False)

    # ---- figures ---------------------------------------------------------------------------
    RP.plot_gain_curves({f"{PRIMARY_YEAR} backtest (trained {PRIMARY_YEAR - TRAIN_YEARS}-{PRIMARY_YEAR - 1})": s25,
                         f"Walk-forward {WALK_FORWARD_YEARS[0]}-{WALK_FORWARD_YEARS[-1]}": scored},
                        out / "figures" / "gain_curves.png")
    RP.plot_dose_response(panel, events, out / "figures" / "dose_response.png")
    RP.plot_importance(models[PRIMARY_YEAR], out / "figures" / "feature_importance.png")
    if ablation is not None:
        RP.plot_ablation(ablation, out / "figures" / "ablation.png")

    # ---- metrics.json ----------------------------------------------------------------------
    n_comm = int(panel["comm_code"].nunique())
    metrics = {
        "generated": date.today().isoformat(),
        "config": {
            "target_311_groups": C.TARGET_GROUPS, "universe_classes": list(C.UNIVERSE_CLASSES),
            "n_communities": n_comm, "top_frac": C.TOP_FRAC, "k_selected": int(np.ceil(C.TOP_FRAC * n_comm)),
            "storm_trigger_mm": C.TRIGGER_MM, "storm_min_gauges": C.MIN_GAUGES, "window_days": C.WINDOW_DAYS,
            "train_years": TRAIN_YEARS, "primary_test_year": PRIMARY_YEAR,
            "n_features": len(P.FEATURES), "xgb_params": Mdl.XGB_PARAMS,
        },
        "data": {
            "n_events_total": int(len(events)),
            "events_per_year": {int(k): int(v) for k, v in events["start"].dt.year.value_counts().sort_index().items()},
            "complaints_last_date": inp.y_daily.index[-1].strftime("%Y-%m-%d"),
            "n_rain_gauges": int(inp.mat_total.shape[1]),
        },
        "primary_2025": _records(summ["primary_2025"]),
        "forward_2026": _records(summ["forward_2026"]),
        "walk_forward": _records(summ["walk_forward"]),
        "paired_vs_baselines": paired,
        "by_storm_severity_walk_forward": by_tier,
        "feature_ablation": None if ablation is None else _records(ablation),
        "training_window_sensitivity": None if sensitivity is None else _records(sensitivity),
        "target_definition_sensitivity": None if target_sens is None else _records(target_sens),
        "largest_2025_event": {"event_id": biggest, "complaints": int(g["y"].sum())},
        "spatial_persistence": _records(persist, 3),
        "gauge_data_quality": data_quality,
        "eccc_crosscheck": eccc_crosscheck(inp),
        "category_rain_sensitivity": _records(sens_cat, 3),
        "lag_profile_catch_basin": _records(lag, 3),
        "gauge_spacing_km_median_nearest": round(float(np.median(
            np.sort(R.distance_matrix_km(R.to_xy_km(inp.gauges["lat"], inp.gauges["lon"]),
                                         R.to_xy_km(inp.gauges["lat"], inp.gauges["lon"])), axis=1)[:, 1])), 2),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_py), encoding="utf-8")
    render_maps(out, inp.universe, inp.gauges)
    log.info("wrote outputs to %s", out)
    return metrics
