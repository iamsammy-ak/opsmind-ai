import json
import re
from dataclasses import replace

import pandas as pd
from helpers import ABC
from test_llm_and_agents import DIAGNOSIS

from opsmind.demo import ANALYSES_DIR, find_saved_for_data, load_samples, sample_dataset, upload_dataset
from opsmind.pipeline import AnalysisParams, compute, data_fingerprint, facts, fingerprint
from opsmind.report import build_report
from opsmind.schemas import Diagnosis


def _upload():
    csv = "case_id,activity,timestamp,resource\n" + "\n".join(
        f"{r.case_id},{r.activity},{r.end:%Y-%m-%d %H:%M},{r.resource}" for r in ABC.events.itertuples()
    )
    return upload_dataset("orders.csv", csv.encode("utf-8"))


def test_fingerprints_separate_data_from_settings():
    ds = _upload()
    params = AnalysisParams(reference=ds.reference)
    assert fingerprint(ds, params) == fingerprint(ds, replace(params, wait_cut=0.30000000000000004))
    assert fingerprint(ds, params) != fingerprint(ds, replace(params, wait_cut=0.4))
    assert data_fingerprint(ds) == data_fingerprint(_upload())
    assert find_saved_for_data("0" * 16) is None


def test_upload_without_kpi_file_derives_kpis_and_a_reference_path():
    # first variant (ties sort alphabetically) is A → B → B → C; the default reference drops the repeat
    assert _upload().reference == ("A", "B", "C")
    days = pd.bdate_range("2026-03-02", periods=10)
    rows = ["case_id,activity,timestamp"]
    for i, day in enumerate(days):
        rows += [f"{i},A,{day:%Y-%m-%d} 09:00", f"{i},B,{day:%Y-%m-%d} 11:00"]
        rows += [f"{i},B,{day:%Y-%m-%d} 12:00"] if i % 2 else []
        rows += [f"{i},C,{day:%Y-%m-%d} 15:00"]
    ds = upload_dataset("tickets.csv", "\n".join(rows).encode("utf-8"))
    assert not ds.log.has_start and not ds.log.has_resource and not ds.synthetic
    c = compute(ds, AnalysisParams(reference=ds.reference))
    assert c.kpis_derived and set(c.kpis) == {"throughput", "cycle_time_days", "first_pass_yield"}
    assert c.kpis["first_pass_yield"].data["value"].mean() == 0.5
    assert c.mining.throughput.flow_efficiency is None  # unknown without start timestamps
    assert c.mining.resources.empty and c.mining.handoffs is None


def test_saved_analyses_match_samples_and_parse():
    samples = [sample_dataset(s) for s in load_samples()]
    fingerprints = {fingerprint(ds, AnalysisParams(reference=ds.reference)) for ds in samples}
    for path in ANALYSES_DIR.glob("*.json"):
        assert path.stem in fingerprints, f"{path.name} does not belong to any sample (stale fixture?)"
        Diagnosis.model_validate(json.loads(path.read_text(encoding="utf-8"))["diagnosis"])


def test_facts_and_report_carry_the_computed_figures():
    ds = sample_dataset(load_samples()[0])
    params = AnalysisParams(reference=ds.reference)
    c = compute(ds, params)
    sheet = facts(ds, c, params)
    text = json.dumps(sheet, ensure_ascii=False)
    assert not re.search(r"\bnan\b", text, flags=re.I)  # every figure was formatted
    assert sheet["waiting before each activity (top 4)"][0]["activity"] == "Approve order"
    report = build_report(ds, c, params, None)
    for heading in ("## Key findings", "## Flow", "## Daily KPIs", "## Lean waste map", "Approve order"):
        assert heading in report
    assert "Root-cause hypotheses" not in report
    analysis = {"model": "fake/model", "summary": "## Situation\nQueues.", "diagnosis": DIAGNOSIS}
    with_ai = build_report(ds, c, params, analysis)
    assert "## Root-cause hypotheses" in with_ai and "Add a backup approver" in with_ai
