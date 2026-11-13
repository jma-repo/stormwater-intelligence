"""Figures and the interactive risk map."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import folium  # noqa: E402
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from branca.colormap import LinearColormap  # noqa: E402

COLORS = {"random": "#9aa0a6", "population": "#b07cc6", "rain_only": "#7fb2e5", "history_365d": "#e8c547",
          "history_events": "#f2a20c", "history_x_rain": "#2ca58d", "xgb": "#1a4fa0"}
LABELS = {"random": "Random", "population": "Most dwellings", "rain_only": "Local rainfall only",
          "history_365d": "Last-12-month complaints", "history_events": "Past post-storm complaints",
          "history_x_rain": "Past complaints x local rain", "xgb": "XGBoost model"}
CURVE_MODELS = ["random", "population", "history_events", "history_x_rain", "xgb"]


def gain_curve(scored: pd.DataFrame, score_col: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pooled cumulative capture vs share of communities selected (ties broken at random)."""
    rng = np.random.default_rng(seed)
    n = int(scored.groupby("event_id").size().iloc[0])
    num, tot = np.zeros(n), 0.0
    for _, g in scored.groupby("event_id"):
        y = g["y"].to_numpy(float)
        order = np.lexsort((rng.random(len(g)), -g[score_col].to_numpy(float)))
        num += np.cumsum(y[order])
        tot += y.sum()
    return np.arange(1, n + 1) / n, num / tot


def plot_gain_curves(panels: dict[str, pd.DataFrame], out_path: Path, frac: float = 0.10) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, scored) in zip(axes, panels.items()):
        for name in CURVE_MODELS:
            x, y = gain_curve(scored, f"score_{name}")
            ax.plot(x * 100, y * 100, color=COLORS[name], lw=2.6 if name == "xgb" else 1.6,
                    ls="--" if name == "random" else "-", label=LABELS[name])
            if name != "random":
                k = int(np.ceil(frac * len(x)))
                ax.plot([x[k - 1] * 100], [y[k - 1] * 100], "o", color=COLORS[name], ms=5)
        ax.axvline(frac * 100, color="#555", lw=0.8, ls=":")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Share of communities selected, highest risk first (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Share of post-storm drainage complaints captured (%)")
    axes[-1].legend(loc="lower right", fontsize=9, frameon=False)
    fig.suptitle("Where do post-storm complaints land? Cumulative gain by ranking (dot = top 10%)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dose_response(panel: pd.DataFrame, events: pd.DataFrame, out_path: Path) -> None:
    ev = events.set_index("event_id")
    tot = panel.groupby("event_id")["y"].sum().rename("complaints").to_frame().join(ev[["peak_gauge_mm", "start"]])
    tot = tot[tot["start"].dt.year >= 2013]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sc = ax.scatter(tot["peak_gauge_mm"], tot["complaints"].clip(lower=1), c=tot["start"].dt.year, cmap="viridis",
                    s=26, alpha=0.85)
    x, y = np.log(tot["peak_gauge_mm"]), np.log(tot["complaints"].clip(lower=1))
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(np.exp(xs), np.exp(a + b * xs), color="#c0392b", lw=1.8,
            label=f"complaints ~ rain^{b:.1f}  (r = {np.corrcoef(x, y)[0, 1]:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Peak gauge rainfall in the 3-day window (mm)")
    ax.set_ylabel("Drainage complaints in the window (city-wide)")
    ax.set_title("Storm severity drives 311 drainage complaints")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, which="both")
    fig.colorbar(sc, ax=ax, label="year")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_importance(model, out_path: Path, top: int = 15) -> None:
    imp = pd.Series(model.feature_importances_, index=model.feature_names_in_).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.barh(imp.index, imp.values, color="#1a4fa0")
    ax.set_xlabel("XGBoost importance (gain share)")
    ax.set_title(f"Top {top} features (model for the 2025 backtest)")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ablation(rows: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    y = np.arange(len(rows))
    ax.barh(y, rows["walk_forward"] * 100, color="#1a4fa0", height=0.6)
    ax.axvline(10.2, color="#9aa0a6", ls="--", lw=1)
    ax.text(10.5, len(rows) - 0.45, "random", color="#666", fontsize=8)
    ax.set_yticks(y, rows["features"])
    for yi, v in zip(y, rows["walk_forward"]):
        ax.text(v * 100 + 0.4, yi, f"{v * 100:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, max(rows["walk_forward"]) * 100 + 8)
    ax.set_xlabel("Top-10% capture, walk-forward 2019-2026 (%)")
    ax.set_title("What carries the signal? Feature-group ablation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_static_map(universe: gpd.GeoDataFrame, gauges: pd.DataFrame, panels: list[dict], out_path: Path,
                    annotate_top: int = 6) -> None:
    """Side-by-side choropleths. panel = {title, values: Series(comm_code), cmap, vmin, vmax, label, annotate: bool}."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 6.6))
    axes = np.atleast_1d(axes)
    geo = universe.set_index("comm_code")
    west, south, east, north = geo.total_bounds
    pad_x, pad_y = 0.02 * (east - west), 0.02 * (north - south)
    inside = gauges[gauges["lon"].between(west - pad_x, east + pad_x) & gauges["lat"].between(south - pad_y, north + pad_y)]
    for ax, spec in zip(axes, panels):
        g = geo.copy()
        g["v"] = spec["values"].reindex(g.index)
        g.plot(column="v", cmap=spec["cmap"], vmin=spec["vmin"], vmax=spec.get("vmax"), ax=ax, linewidth=0.25,
               edgecolor="#8a8a8a", missing_kwds={"color": "#ececec"}, legend=True,
               legend_kwds={"shrink": 0.55, "label": spec["label"]})
        if spec.get("annotate"):
            for code in g["v"].sort_values(ascending=False).head(annotate_top).index:
                pt = g.loc[code, "geometry"].representative_point()
                ax.annotate(g.loc[code, "name"].title(), (pt.x, pt.y), fontsize=6.5, ha="center", color="#222",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
        ax.scatter(inside["lon"], inside["lat"], s=6, color="#1a4fa0", alpha=0.75, zorder=3)
        ax.set_xlim(west - pad_x, east + pad_x)
        ax.set_ylim(south - pad_y, north + pad_y)
        ax.set_title(spec["title"], fontsize=10)
        ax.set_axis_off()
    fig.text(0.5, 0.01, "Blue dots: City of Calgary rain gauges. Community polygons: Open Calgary.", ha="center",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_map(universe: gpd.GeoDataFrame, gauges: pd.DataFrame, layers: dict[str, dict], out_html: Path) -> None:
    """layers: title -> {values: Series(comm_code), kind: 'percentile'|'count', label: str, show: bool}."""
    # No basemap tiles on purpose: hosted tile services now need API keys or a Referer header (a local
    # file:// page has none), and community polygons carry the whole message at city scale.
    m = folium.Map(location=[51.05, -114.07], zoom_start=10, tiles=None, control_scale=True)
    m.get_root().header.add_child(folium.Element("<style>.leaflet-container{background:#f2f2f2}</style>"))
    base = universe[["comm_code", "name", "geometry"]].copy()
    base["geometry"] = base.geometry.simplify(0.0002, preserve_topology=True)
    for title, spec in layers.items():
        vals = spec["values"].reindex(base["comm_code"]).to_numpy(dtype=float)
        if spec["kind"] == "percentile":
            cmap = LinearColormap(["#fff7bc", "#fec44f", "#d95f0e", "#7a2e0c"], vmin=0, vmax=100)
            cmap.caption = spec["label"]
        else:
            vmax = max(float(np.nanpercentile(vals, 97)), 1.0)
            cmap = LinearColormap(["#f7fbff", "#9ecae1", "#2171b5", "#08306b"], vmin=0, vmax=vmax)
            cmap.caption = spec["label"]
        layer = base.copy()
        layer["value"] = vals
        layer["shown"] = [("-" if not np.isfinite(v) else (f"{v:.0f}" if spec["kind"] == "count" else f"{v:.0f}"))
                          for v in vals]
        layer["rank"] = pd.Series(vals).rank(ascending=False, method="min").fillna(0).astype(int).astype(str).to_numpy()
        lookup = dict(zip(layer["comm_code"], layer["value"]))

        def style(feat, lookup=lookup, cmap=cmap):
            v = lookup.get(feat["properties"]["comm_code"])
            ok = v is not None and np.isfinite(v)
            return {"fillColor": cmap(v) if ok else "#e0e0e0", "color": "#6b6b6b", "weight": 0.4, "fillOpacity": 0.78}

        folium.GeoJson(
            layer[["comm_code", "name", "shown", "rank", "geometry"]].to_json(), name=title, style_function=style,
            highlight_function=lambda f: {"weight": 2, "color": "#000"},
            tooltip=folium.GeoJsonTooltip(fields=["name", "shown", "rank"],
                                          aliases=["Community", spec["label"], "Rank (1 = highest)"]),
            show=spec.get("show", False)).add_to(m)
        cmap.add_to(m)
    fg = folium.FeatureGroup(name="City rain gauges", show=False)
    for ch, r in gauges.iterrows():
        folium.CircleMarker([r["lat"], r["lon"]], radius=3, color="#1a4fa0", fill=True, weight=1,
                            tooltip=f"{r['name']} (ch {ch})").add_to(fg)
    fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        "<div style='position:fixed;top:10px;left:50px;z-index:9999;background:rgba(255,255,255,.92);"
        "padding:6px 10px;border:1px solid #bbb;border-radius:4px;font:13px sans-serif;max-width:360px'>"
        "<b>Stormwater Intelligence (PoC)</b><br>Post-storm 311 drainage-complaint risk by community. "
        "Complaint risk, not flood risk. Use the layer switcher (top right).</div>"))
    m.save(str(out_html))
