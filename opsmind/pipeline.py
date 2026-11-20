"""Glue between the deterministic analytics and the AI agents, shared by the app and scripts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import pandas as pd

from .agents import SupportsLLM, diagnose, write_summary
from .demo import Dataset
from .eventlog import canonical_csv
from .formatting import days, duration, money, num, p_value, pct, signed
from .kpi import KpiResult, analyse_all, derive_daily_kpis
from .lean import ImpactParams, LeanMap, WasteFinding, lean_map
from .mining import Mining, mine
from .schemas import Diagnosis, PlanItem

EFFORT_WEIGHT = {"low": 1.0, "medium": 2.0, "high": 3.0}


@dataclass(frozen=True)
class AnalysisParams:
    reference: tuple[str, ...]
    wait_cut: float = 0.30
    rework_cut: float = 0.50
    cost_per_hour: float = 45.0
    baseline: int = 40  # business days used to set control limits
    z_window: int = 20
    z_threshold: float = 3.0


@dataclass(frozen=True)
class Computed:
    mining: Mining
    kpis: dict[str, KpiResult]
    kpis_derived: bool  # True when the KPI series was derived from the log (no KPI file)
    lean: LeanMap
    findings: list[str]  # Markdown bullets, every number computed


def compute(ds: Dataset, params: AnalysisParams) -> Computed:
    m = mine(ds.log, params.reference)
    if ds.kpis is not None:
        table, derived = ds.kpis, False
    else:
        table, derived = derive_daily_kpis(m.cases, params.reference[-1] if params.reference else None), True
    kpis = analyse_all(table, ds.kpi_overrides, params.baseline, params.z_window, params.z_threshold)
    lean = lean_map(m, ds.log.has_start, ImpactParams(params.wait_cut, params.rework_cut, params.cost_per_hour))
    return Computed(m, kpis, derived, lean, key_findings(m, kpis))


# ----------------------------------------------------------------------------- fingerprints
def _normalise(value):
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return value


def _digest(payload) -> str:
    text = payload if isinstance(payload, str) else json.dumps(_normalise(payload), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def data_fingerprint(ds: Dataset) -> str:
    """Identifies the data itself (log + KPI table), independent of the analysis settings."""
    kpi_text = "" if ds.kpis is None else ds.kpis.to_csv(index=False, float_format="%.6f", lineterminator="\n",
                                                          date_format="%Y-%m-%d")
    return _digest(canonical_csv(ds.log) + "\n#kpis\n" + kpi_text)


def fingerprint(ds: Dataset, params: AnalysisParams, data_fp: str | None = None) -> str:
    """Identifies data + settings: a saved AI analysis is valid for exactly this combination."""
    return _digest({"data": data_fp or data_fingerprint(ds), "params": asdict(params)})


# --------------------------------------------------------------------------------- findings
def _nan(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def worst_kpi(kpis: dict[str, KpiResult]) -> KpiResult | None:
    """The KPI that most needs attention: worsening first, then the most rule violations."""
    if not kpis:
        return None
    return max(kpis.values(), key=lambda r: (r.trend_direction == "worsening", r.rule_points))


def key_findings(m: Mining, kpis: dict[str, KpiResult]) -> list[str]:
    out = []
    b = m.bottleneck
    if b:
        row = m.activities.set_index("activity").loc[b]
        out.append(
            f"**{b}** is the main bottleneck: cases wait **{duration(row['mean_wait_h'])}** on average before it "
            f"(p90 {duration(row['p90_wait_h'])}), **{pct(row['wait_share'], 0)}** of all waiting time."
        )
    rs = m.rework_summary
    if rs.case_share > 0 and rs.top_activity:
        out.append(
            f"**{pct(rs.case_share, 0)}** of cases repeat a step, most often *{rs.top_activity}*; they take "
            f"**{signed(rs.extra_days)} days** longer than cases without rework."
        )
    c = m.conformance
    if not _nan(c.fitting_share):
        text = f"**{pct(c.fitting_share, 0)}** of cases follow the reference path exactly"
        if len(c.deviations):
            top = c.deviations.iloc[0]
            text += f"; the most common deviation is {top['kind'].lower()} *{top['detail']}* ({pct(top['share'])})"
        out.append(text + ".")
    if not m.overload.empty:
        for _, row in m.overload[m.overload["overloaded"]].head(2).iterrows():
            out.append(
                f"**{row['resource']}** handles **{pct(row['share'], 0)}** of *{row['activity']}* (even split "
                f"{pct(row['fair_share'], 0)}); cases wait **{row['wait_ratio']:.1f}×** longer for them."
            )
    wd = m.weekdays[m.weekdays["cases"] >= 5].dropna(subset=["mean_days"])
    if len(wd) >= 2:
        slow, fast = wd.loc[wd["mean_days"].idxmax()], wd.loc[wd["mean_days"].idxmin()]
        out.append(
            f"**{pct(m.weekend_wait_share, 0)}** of waiting time falls on weekends; cases started on {slow['weekday']} "
            f"take **{slow['mean_days']:.1f} days** against {fast['mean_days']:.1f} on {fast['weekday']}."
        )
    lc = m.little
    if lc is not None and not _nan(lc.gap):
        verdict = ("the log describes a stable flow" if lc.consistent
                   else "arrivals and completions are out of balance, or the log is incomplete")
        out.append(
            f"Little's Law: average WIP **{lc.avg_wip:.1f}** against λ×W = {lc.arrival_rate:.2f}/day × "
            f"{lc.mean_days:.2f} days = **{lc.predicted_wip:.1f}** ({signed(lc.gap * 100, 1, '%')}): {verdict}."
        )
    w = worst_kpi(kpis)
    if w is not None and (w.rule_points or w.trend_direction == "worsening"):
        text = f"**{w.meta.label}** breaks control-chart rules on **{w.rule_points}** days"
        if w.trend.significant:
            text += f" and is {w.trend_direction} ({w.meta.fmt(w.trend.slope, 3)} per day, {p_value(w.trend.p_value)})"
        out.append(text + ".")
    return out


# ------------------------------------------------------------------------------- AI facts
def _kpi_fact(r: KpiResult) -> dict:
    fact = {
        "recent level (last 10 days)": r.meta.fmt(r.variance.get("recent")),
        "target": r.meta.fmt(r.meta.target) if r.meta.target is not None else "none set",
        "days on control-chart rule violations": r.rule_points,
        "rolling z-score anomalies": r.anomalies,
        "trend": f"{r.trend_direction} ({r.meta.fmt(r.trend.slope, 3)} per business day, {p_value(r.trend.p_value)})",
    }
    if r.change is not None:
        date = r.data["date"].iloc[r.change.index].strftime("%d %b %Y")
        fact["largest level shift"] = (f"from {r.meta.fmt(r.change.mean_before)} to {r.meta.fmt(r.change.mean_after)} "
                                       f"around {date} ({p_value(r.change.p_value)})")
    return fact


def _waste_fact(f: WasteFinding) -> dict:
    return {"waste": f.waste, "evidence": f.evidence, "estimated impact": f.impact, "how estimated": f.formula}


def facts(ds: Dataset, c: Computed, params: AnalysisParams) -> dict:
    m = c.mining
    t0, t1 = ds.log.period
    acts = m.activities.head(4)
    devs = m.conformance.deviations.head(6)
    flagged = m.overload[m.overload["overloaded"]] if not m.overload.empty else pd.DataFrame()
    lean = c.lean
    lc = m.little
    return {
        "process": {
            "organisation": ds.company,
            "process": ds.process,
            "description": ds.description,
            "data": "synthetic sample data" if ds.synthetic else "uploaded event log",
            "period": f"{t0:%d %b %Y} to {t1:%d %b %Y}",
            "cases": num(m.n_cases),
            "events": num(ds.log.n_events),
            "variants": num(len(m.variants)),
            "reference path": " → ".join(params.reference),
        },
        "flow": {
            "mean throughput time": days(m.throughput.mean_days),
            "median throughput time": days(m.throughput.median_days),
            "90th percentile throughput time": days(m.throughput.p90_days),
            "flow efficiency (processing ÷ throughput time)": pct(m.throughput.flow_efficiency)
            if m.throughput.flow_efficiency is not None else "unknown (no start timestamps)",
            "average work in progress": f"{lc.avg_wip:.1f} cases" if lc else "—",
            "Little's Law check": (f"λ {lc.arrival_rate:.2f} cases/day × W {lc.mean_days:.2f} days = "
                                   f"{lc.predicted_wip:.1f} (gap {signed(lc.gap * 100, 1, '%')})") if lc else "—",
        },
        "waiting before each activity (top 4)": [
            {"activity": r["activity"], "mean wait": duration(r["mean_wait_h"]), "p90 wait": duration(r["p90_wait_h"]),
             "share of all waiting": pct(r["wait_share"], 0)} for _, r in acts.iterrows()
        ],
        "calendar": {
            "share of waiting time on weekends": pct(m.weekend_wait_share, 0),
            "mean throughput time by start weekday": {
                r["weekday"]: days(r["mean_days"]) for _, r in m.weekdays.iterrows() if r["cases"] >= 5
            },
        },
        "rework": {
            "cases with a repeated step": pct(m.rework_summary.case_share),
            "extra throughput time of those cases": f"{signed(m.rework_summary.extra_days)} days",
            "most repeated steps": [
                f"{r['activity']}: repeated in {pct(r['rework_rate'])} of its cases"
                for _, r in m.rework[m.rework["repeat_cases"] > 0].head(3).iterrows()
            ],
        },
        "conformance": {
            "cases following the reference path exactly": pct(m.conformance.fitting_share),
            "mean trace fitness (1 = identical)": f"{m.conformance.mean_fitness:.2f}",
            "most common deviations": [f"{r['kind']}: {r['detail']} ({pct(r['share'])} of cases)"
                                       for _, r in devs.iterrows()],
        },
        "workload": {
            "overloaded people": [
                f"{r['resource']} does {pct(r['share'], 0)} of {r['activity']} (even split {pct(r['fair_share'], 0)}); "
                f"cases wait {r['wait_ratio']:.1f}× longer for them" for _, r in flagged.iterrows()
            ] or ["none flagged"],
            "hand-offs per case": f"{m.handoffs.per_case:.1f}" if m.handoffs else "unknown (no resources)",
        },
        "daily KPIs": {r.meta.label: _kpi_fact(r) for r in c.kpis.values()},
        "Lean wastes detected": [_waste_fact(f) for f in lean.detected],
        "improvement scenario (computed)": {
            "assumptions": f"waiting at the bottleneck and hand-offs cut by {pct(params.wait_cut, 0)}; rework, extra "
                           f"steps and abandoned work cut by {pct(params.rework_cut, 0)}; labour at "
                           f"{money(params.cost_per_hour)} per hour",
            "mean throughput time": f"{days(lean.current_mean_days, 2)} → {days(lean.future_mean_days, 2)}",
            "case-hours of waiting removed per month": num(lean.lead_time_h),
            "labour hours freed per month": num(lean.labour_h),
            "labour value per month": money(lean.euros),
        },
    }


# ----------------------------------------------------------------------------- plan ranking
@dataclass(frozen=True)
class RankedAction:
    item: PlanItem
    finding: WasteFinding | None
    score: float


def rank_plan(plan: list[PlanItem], lean: LeanMap) -> list[RankedAction]:
    """Order actions by the computed monthly hours at stake for their waste ÷ the AI's effort rating."""
    ranked = []
    for item in plan:
        finding = lean.get(item.waste) if item.waste else None
        hours = finding.score if finding is not None and finding.detected else 0.0
        ranked.append(RankedAction(item, finding, hours / EFFORT_WEIGHT.get(item.effort, 2.0)))
    order = sorted(range(len(ranked)), key=lambda i: (-ranked[i].score, i))
    return [ranked[i] for i in order]


# ------------------------------------------------------------------------------------ AI
def run_ai(
    llm: SupportsLLM,
    ds: Dataset,
    c: Computed,
    params: AnalysisParams,
    on_step: Callable[[str], object] | None = None,
) -> dict:
    """Two model calls: structured diagnosis, then the executive summary. Returns a JSON-safe dict."""
    step = on_step or (lambda _msg: None)
    fact_sheet = facts(ds, c, params)
    step("Root-cause analyst: forming hypotheses from the computed evidence…")
    diagnosis: Diagnosis = diagnose(llm, fact_sheet)
    step("Summary writer: drafting the executive summary from the computed figures…")
    summary = write_summary(llm, fact_sheet, diagnosis)
    return {
        "fingerprint": fingerprint(ds, params),
        "data_fingerprint": data_fingerprint(ds),
        "params": asdict(params),
        "diagnosis": diagnosis.model_dump(),
        "summary": summary,
        "model": getattr(llm, "last_model", None),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
