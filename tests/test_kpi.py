import numpy as np
import pandas as pd
import pytest
from helpers import make_log

from opsmind.kpi import (
    E2,
    KpiMeta,
    analyse_kpi,
    change_point,
    derive_daily_kpis,
    rolling_zscore,
    target_variance,
    trend,
    western_electric,
    xmr_limits,
)
from opsmind.mining import case_table, enrich


def test_xmr_limits_from_the_moving_range():
    lim = xmr_limits([1, 2, 3, 2, 1])
    assert lim.center == pytest.approx(1.8) and lim.mr_bar == pytest.approx(1)
    assert lim.sigma == pytest.approx(1 / 1.128)
    assert lim.ucl == pytest.approx(1.8 + E2) and lim.lcl == pytest.approx(1.8 - E2)
    assert lim.mr_ucl == pytest.approx(3.267)
    assert xmr_limits([5, 6, 7, 100], baseline=3).center == pytest.approx(6)  # later points do not move limits


def test_western_electric_rules_one_and_four():
    rules = western_electric([0, 0, 0, 4, 0], center=0, sigma=1)
    assert list(rules["rule1"]) == [False, False, False, True, False]
    run = western_electric([0.5] * 8, center=0, sigma=1)
    assert list(run["rule4"]) == [False] * 7 + [True]


def test_western_electric_rules_two_and_three_on_either_side():
    assert list(western_electric([0, 2.5, 0, 2.5], 0, 1)["rule2"]) == [False, False, False, True]
    assert list(western_electric([-2.5, 0, -2.5], 0, 1)["rule2"]) == [False, False, True]
    assert western_electric([1.5, 1.5, 0, 1.5, 1.5], 0, 1)["rule3"].iloc[-1]
    assert not western_electric([5, 5, 5], 0, 0).any().any()  # no spread: no signals rather than a crash


def test_rolling_zscore_flags_a_spike_and_survives_flat_data():
    z = rolling_zscore([10, 12] * 10 + [30], window=20)
    assert np.isnan(z[:20]).all() and z[-1] > 3
    assert np.isnan(rolling_zscore([7] * 30, window=20)).all()  # zero spread gives NaN, never inf


def test_trend_slope_and_significance():
    perfect = trend([2 + 0.5 * x for x in range(30)])
    assert perfect.slope == pytest.approx(0.5) and perfect.r2 == pytest.approx(1) and perfect.significant
    assert perfect.change == pytest.approx(14.5)
    flat = trend([3.0] * 10)
    assert flat.slope == 0 and flat.p_value == 1 and not flat.significant
    noise = trend(np.random.default_rng(1).normal(0, 1, 60))
    assert not noise.significant


def test_change_point_finds_a_step():
    values = np.r_[np.zeros(30), np.full(30, 5.0)] + np.random.default_rng(2).normal(0, 0.3, 60)
    cp = change_point(values)
    assert cp.index == 30 and cp.delta == pytest.approx(5, abs=0.3) and cp.clear
    assert change_point([1, 2, 3]) is None


def test_target_variance_respects_direction():
    lower = target_variance([5, 5, 5], target=4, higher_is_better=False)
    assert lower["gap"] == pytest.approx(1) and lower["on_target"] is False and lower["days_met"] == 0
    higher = target_variance([0.9, 0.9, 0.8], target=0.85, higher_is_better=True, recent=2)
    assert higher["recent"] == pytest.approx(0.85) and higher["on_target"] is True
    assert higher["days_met"] == pytest.approx(2 / 3)
    assert target_variance([1, 2], None, True)["on_target"] is None


def test_daily_kpis_derived_from_the_log_roll_weekends_forward():
    log = make_log(
        [
            ("1", "A", "2026-03-05 09:00", "2026-03-05 10:00", "r"),
            ("1", "B", "2026-03-06 09:00", "2026-03-06 10:00", "r"),  # Friday
            ("2", "A", "2026-03-06 09:00", "2026-03-06 10:00", "r"),
            ("2", "A", "2026-03-06 11:00", "2026-03-06 12:00", "r"),
            ("2", "B", "2026-03-07 09:00", "2026-03-07 10:00", "r"),  # Saturday: counts on Monday
            ("3", "A", "2026-03-09 09:00", "2026-03-09 10:00", "r"),  # never reaches B
        ]
    )
    k = derive_daily_kpis(case_table(enrich(log.events)), end_activity="B")
    assert list(k["date"].dt.strftime("%a")) == ["Fri", "Mon"]
    assert list(k["throughput"]) == [1, 1]
    assert list(k["first_pass_yield"]) == [1.0, 0.0]
    assert k["cycle_time_days"].iloc[0] == pytest.approx(25 / 24)


def test_analyse_kpi_bounds_limits_and_reads_direction():
    dates = pd.bdate_range("2026-01-05", periods=60)
    rising = np.r_[np.full(30, 3.0), np.full(30, 4.0)] + np.random.default_rng(3).normal(0, 0.2, 60)
    r = analyse_kpi("cycle_time_days", dates, rising, KpiMeta("Cycle time", "days", False, target=3.5),
                    baseline=25)
    assert r.trend_direction == "worsening" and r.rule_points > 10 and r.change.clear
    counts = analyse_kpi("throughput", dates, np.tile([0.0, 20.0], 30), KpiMeta("Throughput", "cases/day", True))
    assert counts.limits.lcl == 0  # a count cannot go below zero
    share = analyse_kpi("otif", dates, np.tile([0.5, 1.0], 30), KpiMeta("OTIF", "%", True, percent=True))
    assert share.limits.ucl == 1
