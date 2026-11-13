import math

import numpy as np
import pandas as pd
import pytest

from stormwater_intel import metrics as M


def test_capture_top_unit_holds_half_of_complaints():
    scores = np.arange(10, 0, -1)          # unit 0 has the highest score
    counts = np.array([5, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    assert M.capture_at_frac(scores, counts, 0.10) == pytest.approx(0.5)


def test_capture_ranks_by_score_not_by_position():
    scores = np.array([1, 9, 2, 3])
    counts = np.array([0, 4, 6, 0])        # score winner (unit 1) holds 4 of 10
    assert M.capture_at_frac(scores, counts, 0.25) == pytest.approx(0.4)


def test_constant_scores_give_selected_fraction():
    counts = np.random.default_rng(0).integers(0, 5, size=40)
    assert M.capture_at_frac(np.zeros(40), counts, 0.10) == pytest.approx(0.10)


def test_ties_at_cutoff_are_split_proportionally():
    # k = 2 of 4; one clear winner, three-way tie for the second slot
    scores = np.array([5.0, 1.0, 1.0, 1.0])
    counts = np.array([2.0, 3.0, 0.0, 0.0])
    # winner 2 + 1/3 of the tied group's 3 complaints = 3 of 5
    assert M.capture_at_frac(scores, counts, 0.5) == pytest.approx(3 / 5)


def test_no_complaints_is_nan():
    assert math.isnan(M.capture_at_frac([1, 2, 3], [0, 0, 0]))


def test_selected_fraction_rounds_up():
    assert M.selected_fraction(25, 0.10) == pytest.approx(3 / 25)
    assert M.selected_fraction(250, 0.10) == pytest.approx(0.10)


def test_pooled_capture_is_complaint_weighted():
    per_event = pd.DataFrame({"total": [90.0, 10.0, 0.0], "capture": [0.5, 1.0, np.nan]})
    assert M.pooled_capture(per_event) == pytest.approx((0.5 * 90 + 1.0 * 10) / 100)


def test_paired_bootstrap_diff_detects_a_better_ranking():
    rng = np.random.default_rng(2)
    rows = []
    for ev in range(30):
        y = rng.poisson(0.8, size=60)
        y[:3] += 6                                                 # three hot communities
        rows.append(pd.DataFrame({"event_id": ev, "good": np.r_[[9, 8, 7], rng.random(57)],
                                  "poor": rng.random(60), "y": y}))
    panel = pd.concat(rows)
    good, poor = M.per_event_capture(panel, "good"), M.per_event_capture(panel, "poor")
    est, lo, hi = M.paired_bootstrap_diff(good, poor, n_boot=500)
    assert est > 0.2 and lo > 0
    same, lo0, hi0 = M.paired_bootstrap_diff(good, good, n_boot=200)
    assert same == 0 and lo0 == 0 == hi0


def test_per_event_capture_and_bootstrap_interval_brackets_estimate():
    rng = np.random.default_rng(1)
    rows = []
    for ev in range(12):
        y = rng.poisson(1.0, size=50)
        y[0] += 8                             # a hot community the score finds
        score = np.r_[10.0, rng.random(49)]
        rows.append(pd.DataFrame({"event_id": ev, "score": score, "y": y}))
    panel = pd.concat(rows)
    pe = M.per_event_capture(panel, "score", frac=0.10)
    est = M.pooled_capture(pe)
    lo, hi = M.bootstrap_pooled(pe, n_boot=500, seed=3)
    assert lo <= est <= hi
    assert est > 0.10                          # beats random
