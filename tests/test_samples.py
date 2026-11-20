"""The synthetic samples: valid, reproducible, and the analytics find what the generator planted."""

import importlib.util
from functools import cache
from pathlib import Path

import pandas as pd
import pytest

from opsmind.demo import DEMO_DIR, load_samples, sample_dataset
from opsmind.pipeline import AnalysisParams, compute

ROOT = Path(__file__).resolve().parent.parent


@cache
def computed(sample_id: str):
    sample = next(s for s in load_samples() if s["id"] == sample_id)
    ds = sample_dataset(sample)
    return ds, compute(ds, AnalysisParams(reference=ds.reference))


def test_samples_are_valid_and_synthetic():
    samples = load_samples()
    assert len(samples) == 2
    for sample in samples:
        ds = sample_dataset(sample)
        assert ds.synthetic and ds.planted and 1000 <= ds.log.n_cases <= 1500
        assert ds.log.has_start and ds.log.has_resource and ds.log.dropped_rows == 0
        assert set(ds.reference) <= set(ds.log.activities)
        assert set(ds.kpis.columns) == {"date", "throughput", "cycle_time_days", "first_pass_yield", "otif",
                                        "cost_per_order"}
        assert (ds.kpis["date"].dt.day_of_week < 5).all()  # business days only


def test_valdora_planted_realities_are_found():
    ds, c = computed("valdora-order-fulfilment")
    m = c.mining
    assert m.bottleneck == "Approve order"
    flagged = m.overload[m.overload["overloaded"]]
    assert list(zip(flagged["resource"], flagged["activity"], strict=True)) == [("Marta Ilić", "Approve order")]
    devs = m.conformance.deviations.set_index(["kind", "detail"])["share"]
    assert 0.05 < devs[("Skipped", "Check credit")] < 0.10
    assert ("Out of order", "Send invoice before Pick goods") in devs.index
    assert m.rework_summary.case_share > 0.15 and m.little.consistent
    cycle = c.kpis["cycle_time_days"]
    assert cycle.trend_direction == "worsening" and cycle.change.clear
    assert cycle.data["date"].iloc[cycle.change.index] >= pd.Timestamp("2026-03-30")  # after the reorganisation
    spikes = set(c.kpis["cost_per_order"].data.loc[lambda d: d["anomaly"], "date"].dt.strftime("%Y-%m-%d"))
    assert {"2026-03-12", "2026-06-10"} <= spikes


def test_brenmoor_planted_realities_are_found():
    ds, c = computed("brenmoor-invoice-to-pay")
    m = c.mining
    assert m.bottleneck == "Approve invoice"
    flagged = m.overload[m.overload["overloaded"]]
    assert list(flagged["resource"]) == ["Inês Duarte"]
    devs = m.conformance.deviations.set_index(["kind", "detail"])["share"]
    assert devs[("Skipped", "Match to PO")] > 0.05 and devs[("Skipped", "Approve invoice")] > 0.02
    cycle = c.kpis["cycle_time_days"]
    assert cycle.change.delta > 0.5 and cycle.data["date"].iloc[cycle.change.index] >= pd.Timestamp("2026-03-23")
    assert c.kpis["otif"].rule_points > 10  # the March supplier problems


@pytest.mark.parametrize("sample_id", ["valdora-order-fulfilment"])
def test_generator_reproduces_the_committed_data(sample_id):
    spec = importlib.util.spec_from_file_location("make_data", ROOT / "scripts" / "make_data.py")
    make_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_data)
    sample = next(s for s in make_data.SAMPLES if s["id"] == sample_id)
    sim, _ = make_data.GENERATORS[sample_id]()
    regenerated = sim.frame(extract_only=True).to_csv(index=False, lineterminator="\n")
    committed = (DEMO_DIR / sample["events_file"]).read_text(encoding="utf-8")
    assert regenerated == committed
