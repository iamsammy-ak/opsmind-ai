import numpy as np
import pandas as pd
import pytest
from helpers import ABC, REFERENCE, overloaded_log

from opsmind.mining import (
    END,
    START,
    activity_stats,
    case_table,
    conformance,
    dfg,
    enrich,
    handoff_stats,
    levenshtein,
    littles_law,
    mine,
    overload_table,
    rework_stats,
    trace_deviations,
    variants,
    weekend_hours,
)

EV = enrich(ABC.events)
CASES = case_table(EV)


def test_waiting_processing_and_throughput_times():
    waits = EV.set_index(["case_id", "pos"])["wait_h"]
    assert np.isnan(waits[("1", 0)])
    assert waits[("1", 1)] == pytest.approx(2) and waits[("2", 1)] == pytest.approx(4)
    assert waits[("2", 3)] == pytest.approx(0.5)
    assert list(CASES.loc[["1", "2", "3"], "throughput_h"]) == pytest.approx([6, 8.5, 3])
    assert list(CASES["has_rework"]) == [False, True, False]
    assert CASES.loc["2", "handoffs"] == 2


def test_variants_are_ranked_with_shares():
    v = variants(CASES)
    assert list(v["variant"]) == ["V1", "V2", "V3"]
    assert v["share"].sum() == pytest.approx(1) and v["cum_share"].iloc[-1] == pytest.approx(1)
    assert v["path"].iloc[0] == "A → B → B → C"  # ties broken alphabetically, so the order is stable


def test_directly_follows_graph_counts_and_mean_waits():
    edges = dfg(EV, len(CASES)).set_index(["source", "target"])
    assert edges.loc[("A", "B"), "count"] == 2 and edges.loc[("A", "B"), "mean_wait_h"] == pytest.approx(3)
    assert edges.loc[("B", "C"), "mean_wait_h"] == pytest.approx(0.75)
    assert edges.loc[("B", "B"), "count"] == 1
    assert edges.loc[(START, "A"), "count"] == 3 and edges.loc[("C", END), "count"] == 3
    assert edges.loc[("A", "C"), "share"] == pytest.approx(1 / 3)


def test_bottleneck_is_the_activity_with_most_total_waiting():
    stats = activity_stats(EV)
    assert stats["activity"].iloc[0] == "B"
    b = stats.set_index("activity").loc["B"]
    assert b["total_wait_h"] == pytest.approx(8) and b["mean_wait_h"] == pytest.approx(8 / 3)
    assert b["wait_share"] == pytest.approx(8 / 10.5)


def test_rework_per_activity_and_summary():
    table, summary = rework_stats(EV, CASES)
    b = table.set_index("activity").loc["B"]
    assert (b["cases"], b["repeat_cases"], b["extra_executions"]) == (2, 1, 1)
    assert b["rework_rate"] == pytest.approx(0.5) and b["extra_wait_h"] == pytest.approx(2)
    assert summary.case_share == pytest.approx(1 / 3) and summary.top_activity == "B"
    assert summary.extra_days == pytest.approx((8.5 - 4.5) / 24)


def test_conformance_fitting_share_fitness_and_deviations():
    result = conformance(CASES, REFERENCE)
    assert result.fitting_share == pytest.approx(1 / 3)
    assert result.mean_fitness == pytest.approx((1 + 0.75 + 2 / 3) / 3)
    found = {(k, d) for k, d in zip(result.deviations["kind"], result.deviations["detail"], strict=True)}
    assert found == {("Repeated", "B"), ("Skipped", "B")}


def test_trace_deviations_explain_order_and_extra_steps():
    assert trace_deviations(("A", "C", "B"), REFERENCE) == [("Out of order", "C before B")]
    # a step that jumps several places ahead is reported once, against the earliest step it skipped past
    assert trace_deviations(("A", "D", "B", "C"), ("A", "B", "C", "D")) == [("Out of order", "D before B")]
    assert ("Extra step", "X") in trace_deviations(("A", "X", "B", "C"), REFERENCE)
    assert conformance(CASES, ()).deviations.empty


def test_levenshtein_on_sequences():
    assert levenshtein(list("kitten"), list("sitting")) == 3
    assert levenshtein((), ("a", "b")) == 2
    assert levenshtein(("a",), ("a",)) == 0


def test_handoffs_compare_waits_after_a_change_of_hands():
    h = handoff_stats(EV, len(CASES))
    assert h.count == 4 and h.per_case == pytest.approx(4 / 3) and h.share == pytest.approx(4 / 6)
    assert h.wait_after_handoff_h == pytest.approx(7.5 / 4) and h.wait_same_person_h == pytest.approx(1.5)


def test_overloaded_person_is_flagged_but_peers_are_not():
    ov = overload_table(enrich(overloaded_log().events)).set_index("resource")
    assert bool(ov.loc["Ann", "overloaded"]) and ov.loc["Ann", "share"] == pytest.approx(0.75)
    assert ov.loc["Ann", "wait_ratio"] == pytest.approx(5)
    assert not ov.loc["Bob", "overloaded"] and not ov.loc["Cid", "overloaded"]


def test_weekend_hours_of_intervals():
    start = pd.Series(pd.to_datetime(["2026-03-06 16:00", "2026-03-07 10:00", "2026-03-09 09:00",
                                      "2026-03-06 20:00", None]))
    end = pd.Series(pd.to_datetime(["2026-03-09 10:00", "2026-03-07 12:00", "2026-03-09 17:00",
                                    "2026-03-08 06:00", "2026-03-09 10:00"]))
    assert list(weekend_hours(start, end)) == pytest.approx([48, 2, 0, 30, 0])


def test_littles_law_on_a_steady_flow():
    starts = pd.date_range("2026-01-01", periods=100, freq="D")
    cases = pd.DataFrame({"start": starts, "end": starts + pd.Timedelta(days=2), "throughput_d": 2.0})
    lc = littles_law(cases)
    assert lc.avg_wip == pytest.approx(2)
    assert lc.mean_days == pytest.approx(2)
    assert lc.consistent and abs(lc.gap) < 0.02
    assert littles_law(cases.head(1)) is None


def test_mine_bundles_everything_for_the_app():
    m = mine(ABC, REFERENCE)
    assert m.bottleneck == "B" and m.n_cases == 3
    assert m.throughput.flow_efficiency == pytest.approx(
        (3 + 2 + 2) / (6 + 8.5 + 3)  # processing hours ÷ throughput hours
    )
    # WIP is an end-of-day snapshot: all three cases arrive and finish on the same day
    assert list(m.wip[["arrivals", "completions", "wip"]].iloc[0]) == [3, 3, 0]
