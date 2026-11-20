"""Downloadable Markdown report: computed findings, tables and (when available) the AI diagnosis."""

from __future__ import annotations

from datetime import date

from .demo import Dataset
from .formatting import days, duration, money, num, p_value, pct
from .lean import ICONS
from .pipeline import AnalysisParams, Computed, rank_plan
from .schemas import Diagnosis


def _cell(text) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def build_report(ds: Dataset, c: Computed, params: AnalysisParams, analysis: dict | None) -> str:
    m = c.mining
    t0, t1 = ds.log.period
    model = analysis.get("model") if analysis else None
    lines = [
        f"# {ds.company}: {ds.process} · OpsMind report",
        "",
        f"_Generated {date.today().isoformat()}"
        + (f" · AI diagnosis by `{model}`" if model else " · computed analysis only, no AI diagnosis")
        + (" · synthetic sample data_" if ds.synthetic else "_"),
        "",
        f"{num(m.n_cases)} cases and {num(ds.log.n_events)} events from {t0:%d %b %Y} to {t1:%d %b %Y}. "
        f"Reference path: {' → '.join(params.reference) or 'not set'}.",
        "",
        "## Key findings",
        "",
    ]
    lines += [f"- {f}" for f in c.findings]

    if analysis:
        lines += ["", "## Executive summary", "", analysis.get("summary", "").strip()]

    lines += [
        "",
        "## Flow",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean / median / p90 throughput time | {days(m.throughput.mean_days)} / {days(m.throughput.median_days)} / "
        f"{days(m.throughput.p90_days)} |",
        f"| Flow efficiency | {pct(m.throughput.flow_efficiency)} |",
        f"| Cases following the reference path | {pct(m.conformance.fitting_share)} |",
        f"| Cases with rework | {pct(m.rework_summary.case_share)} |",
        f"| Variants | {num(len(m.variants))} |",
    ]
    if m.little:
        lc = m.little
        lines.append(f"| Average WIP vs λ×W (Little's Law) | {lc.avg_wip:.1f} vs {lc.predicted_wip:.1f} |")

    lines += ["", "## Waiting before each activity", "", "| Activity | Mean wait | p90 wait | Share of waiting |",
              "|---|---:|---:|---:|"]
    for _, r in m.activities.dropna(subset=["mean_wait_h"]).iterrows():
        lines.append(f"| {_cell(r['activity'])} | {duration(r['mean_wait_h'])} | {duration(r['p90_wait_h'])} | "
                     f"{pct(r['wait_share'])} |")

    lines += ["", "## Top deviations from the reference path", "", "| Deviation | Cases | Share |", "|---|---:|---:|"]
    for _, r in m.conformance.deviations.head(8).iterrows():
        lines.append(f"| {r['kind']}: {_cell(r['detail'])} | {num(r['cases'])} | {pct(r['share'])} |")

    lines += ["", "## Daily KPIs", "", "| KPI | Recent | Target | Rule violations | Anomalies | Trend |",
              "|---|---:|---:|---:|---:|---|"]
    for r in c.kpis.values():
        target = r.meta.fmt(r.meta.target) if r.meta.target is not None else "—"
        lines.append(f"| {r.meta.label} | {r.meta.fmt(r.variance.get('recent'))} | {target} | {r.rule_points} | "
                     f"{r.anomalies} | {r.trend_direction} ({p_value(r.trend.p_value)}) |")
    if c.kpis_derived:
        lines += ["", "_KPIs derived from the event log (no KPI file supplied)._"]

    lean = c.lean
    lines += [
        "",
        "## Lean waste map",
        "",
        f"Assumptions: waiting at the bottleneck and hand-offs cut by {pct(params.wait_cut, 0)}; rework, extra steps "
        f"and abandoned work cut by {pct(params.rework_cut, 0)}; labour at {money(params.cost_per_hour)} per hour.",
        "",
        "| Waste | Evidence | Estimated impact per month | Countermeasure |",
        "|---|---|---|---|",
    ]
    for f in lean.ranked():
        mark = "" if f.detected else " (not detected)"
        lines.append(f"| {ICONS[f.waste]} {f.waste}{mark} | {_cell(f.evidence)} | {_cell(f.impact)} | "
                     f"{_cell(f.countermeasure)} |")
    lines += [
        "",
        f"Combined (non-overlapping) estimate: {num(lean.lead_time_h)} case-hours of waiting and {num(lean.labour_h)} "
        f"labour hours a month ({money(lean.euros)}); mean throughput time {days(lean.current_mean_days, 2)} → "
        f"{days(lean.future_mean_days, 2)}.",
    ]

    if analysis:
        dx = Diagnosis.model_validate(analysis["diagnosis"])
        lines += ["", "## Root-cause hypotheses (AI, to be confirmed)", ""]
        for i, rc in enumerate(dx.root_causes, start=1):
            lines += [
                f"**{i}. {rc.hypothesis}** ({rc.confidence} confidence)",
                f"- Symptom: {rc.problem}",
                *[f"- Evidence: {e}" for e in rc.evidence],
                f"- Test: {rc.test}",
                "",
            ]
        lines += ["## Improvement plan", "", "| # | Action | Waste | Owner | Effort | Expected impact |",
                  "|---:|---|---|---|---|---|"]
        for i, ra in enumerate(rank_plan(dx.improvement_plan, lean), start=1):
            it = ra.item
            lines.append(f"| {i} | {_cell(it.action)} | {it.waste or '—'} | {_cell(it.owner_role)} | {it.effort} | "
                         f"{_cell(it.expected_impact)} |")
        if dx.data_gaps:
            lines += ["", "**What the log cannot show**", ""] + [f"- {g}" for g in dx.data_gaps]

    lines += [
        "",
        "---",
        "_Every figure above is computed from the event log and KPI series. AI-written hypotheses are starting "
        "points for investigation, not conclusions._",
    ]
    return "\n".join(lines)
