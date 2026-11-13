"""Ranking metrics: how many post-storm complaints do the top-k% communities capture?"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def selected_fraction(n_units: int, frac: float = 0.10) -> float:
    """Actual share of units selected once k = ceil(frac * n) is rounded up."""
    return max(1, math.ceil(frac * n_units)) / n_units


def capture_at_frac(scores, counts, frac: float = 0.10) -> float:
    """Share of `counts` that sits in the top `frac` of units ranked by `scores`.

    Ties at the cut-off are split proportionally (the expectation under random
    tie-breaking), so a constant score returns exactly the selected fraction.
    """
    s = np.asarray(scores, dtype=float)
    c = np.asarray(counts, dtype=float)
    n = len(s)
    total = c.sum()
    if n == 0 or total <= 0:
        return float("nan")
    k = max(1, math.ceil(frac * n))
    order = np.argsort(-s, kind="mergesort")
    s_sorted, c_sorted = s[order], c[order]
    cut = s_sorted[k - 1]
    above = s_sorted > cut
    tie = s_sorted == cut
    n_above, n_tie = int(above.sum()), int(tie.sum())
    captured = c_sorted[above].sum() + c_sorted[tie].sum() * (k - n_above) / n_tie
    return float(captured / total)


def per_event_capture(panel: pd.DataFrame, score_col: str, *, count_col: str = "y",
                      event_col: str = "event_id", frac: float = 0.10) -> pd.DataFrame:
    """One row per event: complaints in the event window and the share the top-k% captured."""
    rows = []
    for event, g in panel.groupby(event_col, sort=True):
        total = float(g[count_col].sum())
        cap = capture_at_frac(g[score_col].to_numpy(), g[count_col].to_numpy(), frac)
        sel = selected_fraction(len(g), frac)
        rows.append({event_col: event, "n_units": len(g), "total": total, "capture": cap,
                     "lift": cap / sel if total > 0 else float("nan")})
    return pd.DataFrame(rows)


def pooled_capture(per_event: pd.DataFrame) -> float:
    """Complaint-weighted capture across events (= total captured / total complaints)."""
    ok = per_event[per_event["total"] > 0]
    if ok.empty:
        return float("nan")
    return float((ok["capture"] * ok["total"]).sum() / ok["total"].sum())


def macro_capture(per_event: pd.DataFrame, min_total: float = 10) -> float:
    """Unweighted mean capture over events with at least `min_total` complaints."""
    ok = per_event[per_event["total"] >= min_total]
    return float(ok["capture"].mean()) if len(ok) else float("nan")


def paired_bootstrap_diff(a: pd.DataFrame, b: pd.DataFrame, event_col: str = "event_id",
                          n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Pooled-capture difference a - b with a 95% interval, resampling the SAME events for both."""
    m = a[[event_col, "total", "capture"]].merge(b[[event_col, "capture"]], on=event_col, suffixes=("_a", "_b"))
    m = m[m["total"] > 0].reset_index(drop=True)
    if len(m) < 2:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    tot, ca, cb = m["total"].to_numpy(), m["capture_a"].to_numpy(), m["capture_b"].to_numpy()
    idx = rng.integers(0, len(m), size=(n_boot, len(m)))
    diff = ((ca[idx] - cb[idx]) * tot[idx]).sum(axis=1) / tot[idx].sum(axis=1)
    est = float(((ca - cb) * tot).sum() / tot.sum())
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return est, float(lo), float(hi)


def bootstrap_pooled(per_event: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval of pooled capture, resampling whole events."""
    ok = per_event[per_event["total"] > 0].reset_index(drop=True)
    if len(ok) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    cap, tot = ok["capture"].to_numpy(), ok["total"].to_numpy()
    idx = rng.integers(0, len(ok), size=(n_boot, len(ok)))
    vals = (cap[idx] * tot[idx]).sum(axis=1) / tot[idx].sum(axis=1)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
