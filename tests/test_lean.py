import pytest
from helpers import ABC, REFERENCE, make_log, overloaded_log

from opsmind.lean import WASTES, ImpactParams, lean_map
from opsmind.mining import mine


def test_every_waste_is_reported_in_a_fixed_order():
    lean = lean_map(mine(ABC, REFERENCE), has_start=True)
    assert [f.waste for f in lean.findings] == list(WASTES)
    assert not lean.get("Motion").detected  # never visible in an event log
    assert lean.get("Waiting").detected and "'B'" in lean.get("Waiting").evidence
    assert lean.get("Defects").detected and lean.get("Overproduction").detected is False


def test_impact_scales_with_the_assumptions():
    m = mine(ABC, REFERENCE)
    base = lean_map(m, True, ImpactParams(wait_cut=0.3, rework_cut=0.5))
    double = lean_map(m, True, ImpactParams(wait_cut=0.6, rework_cut=1.0))
    none = lean_map(m, True, ImpactParams(wait_cut=0.0, rework_cut=0.0))
    assert double.get("Waiting").lead_time_h == pytest.approx(2 * base.get("Waiting").lead_time_h)
    assert double.get("Defects").labour_h == pytest.approx(2 * base.get("Defects").labour_h)
    assert none.lead_time_h == 0 and none.labour_h == 0
    assert none.future_mean_days == pytest.approx(none.current_mean_days)
    assert base.future_mean_days < base.current_mean_days
    assert base.euros == pytest.approx(base.labour_h * 45)


def test_clean_flow_has_no_defects_or_abandoned_work():
    log = make_log(
        [
            ("1", "A", "2026-03-02 09:00", "2026-03-02 09:30", "r1"),
            ("1", "B", "2026-03-02 11:00", "2026-03-02 11:30", "r2"),
            ("2", "A", "2026-03-03 09:00", "2026-03-03 09:30", "r1"),
            ("2", "B", "2026-03-03 10:00", "2026-03-03 10:30", "r2"),
        ]
    )
    lean = lean_map(mine(log, ("A", "B")), True)
    for waste in ("Defects", "Overproduction", "Extra-processing", "Non-utilised talent"):
        assert not lean.get(waste).detected, waste
    assert lean.get("Defects").labour_h == 0


def test_overload_at_the_bottleneck_is_shown_but_not_double_counted():
    lean = lean_map(mine(overloaded_log(), ("Submit", "Approve")), True)
    talent = lean.get("Non-utilised talent")
    assert talent.detected and "Ann" in talent.evidence
    # Ann waits 10 h against 2 h for colleagues: (10 - 2) h × 30 cases spread over the log's months
    months = (39 + 10 / 24) / 30.44
    assert talent.lead_time_h == pytest.approx(8 * 30 / months, rel=0.01)
    assert not talent.in_total  # Approve is also the bottleneck, so Waiting already covers it
    assert lean.lead_time_h == pytest.approx(sum(f.lead_time_h for f in lean.findings if f.in_total))
