"""OpsMind AI: process mining, KPI control charts and Lean waste mapping, with an AI improvement planner."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict

import streamlit as st

from opsmind import charts
from opsmind.config import load_settings
from opsmind.demo import Dataset, find_saved_for_data, load_samples, load_saved_analysis, sample_dataset, upload_dataset
from opsmind.eventlog import EventLogError
from opsmind.formatting import days, duration, money, num, p_value, pct, signed
from opsmind.kpi import RULES
from opsmind.lean import ICONS
from opsmind.llm import LLMClient, LLMError
from opsmind.mining import END, START
from opsmind.pipeline import (
    AnalysisParams,
    Computed,
    compute,
    data_fingerprint,
    fingerprint,
    rank_plan,
    run_ai,
    worst_kpi,
)
from opsmind.processmap import filter_edges, to_dot
from opsmind.report import build_report
from opsmind.schemas import Diagnosis
from opsmind.ui import (
    card,
    context_line,
    hero,
    inject_css,
    num_cell,
    pill,
    severity,
    share_bar,
    status,
    table,
    trend_cell,
    waste_grid,
)

st.set_page_config(page_title="OpsMind AI", page_icon="⚙️", layout="wide")
inject_css()

SAMPLES = load_samples()
SAMPLE_NAMES = [s["name"] for s in SAMPLES]
UPLOAD = "Upload your own CSV"
TEMPLATE = (
    "case_id,activity,timestamp,resource,start_timestamp\n"
    "PO-1001,Create request,2026-03-02 09:14,Ana,2026-03-02 09:02\n"
    "PO-1001,Approve request,2026-03-03 15:40,Ben,2026-03-03 15:31\n"
    "PO-1001,Create order,2026-03-04 10:05,Cleo,2026-03-04 09:50\n"
)
ss = st.session_state


# ----------------------------------------------------------------------------- caching
@st.cache_data(show_spinner=False)
def _sample(sample_id: str) -> Dataset:
    return sample_dataset(next(s for s in SAMPLES if s["id"] == sample_id))


@st.cache_data(show_spinner=False)
def _upload(file_name: str, events: bytes, kpis: bytes | None) -> Dataset:
    return upload_dataset(file_name, events, kpis)


@st.cache_data(show_spinner=False)
def _data_fp(key: str, _ds: Dataset) -> str:
    return data_fingerprint(_ds)


@st.cache_data(show_spinner="Mining the event log…", max_entries=24)
def _compute(key: str, params_json: str, _ds: Dataset) -> Computed:
    raw = json.loads(params_json)
    return compute(_ds, AnalysisParams(**{**raw, "reference": tuple(raw["reference"])}))


# ----------------------------------------------------------------------------- sidebar
settings = load_settings()
ds: Dataset | None = None
upload_error = None
with st.sidebar:
    st.markdown("## ⚙️ OpsMind AI")
    if settings.live:
        fallbacks = len(settings.models) - 1
        st.success(f"**Live AI** via OpenRouter\n\n`{settings.models[0]}`\n\n+{fallbacks} fallback models")
    else:
        st.info(
            "**Demo mode**: all computed analytics work without a key.\n\n"
            "Add `OPENROUTER_API_KEY` to enable the AI root-cause analysis and improvement plan."
        )
    choice = st.selectbox("Dataset", SAMPLE_NAMES + [UPLOAD], key="dataset")
    if choice == UPLOAD:
        events_file = st.file_uploader("Event log (CSV)", type=["csv", "txt"], key="upload_events")
        kpi_file = st.file_uploader("Daily KPIs (CSV, optional)", type=["csv"], key="upload_kpis")
        with st.expander("Expected columns", expanded=events_file is None):
            st.markdown(
                "**Event log**: one row per executed step.\n\n"
                "- `case_id`, `activity`, `timestamp` (completion time): required\n"
                "- `resource`: optional, enables workload and hand-off analysis\n"
                "- `start_timestamp`: optional, separates waiting from processing time\n\n"
                "XES-style headers (`case:concept:name`, `concept:name`, `time:timestamp`, `org:resource`) also "
                "work.\n\n**KPIs**: a `date` column plus one numeric column per KPI. Without a KPI file, daily "
                "throughput, cycle time and first-pass yield are derived from the log."
            )
            st.download_button("Download a template", TEMPLATE, file_name="event_log_template.csv", mime="text/csv")
        if events_file is not None:
            try:
                ds = _upload(events_file.name, events_file.getvalue(), kpi_file.getvalue() if kpi_file else None)
            except (EventLogError, ValueError) as exc:
                upload_error = str(exc)
    else:
        ds = _sample(SAMPLES[SAMPLE_NAMES.index(choice)]["id"])
        st.caption("Synthetic data from a seeded generator (`scripts/make_data.py`). Companies and people are "
                   "fictional.")

    if ds is not None:
        dataset_key = ds.key if ds.key != "upload" else "upload:" + hashlib.sha256(
            events_file.getvalue() + (kpi_file.getvalue() if kpi_file else b"")).hexdigest()[:16]
        if ss.get("dataset_key") != dataset_key:
            ss["dataset_key"] = dataset_key
            ss["reference"] = list(ds.reference)
            ss.pop("kpi_choice", None)

        st.markdown("### Analysis settings")
        with st.expander("Reference path"):
            options = list(dict.fromkeys([*ds.reference, *ds.log.activities]))
            st.multiselect("Steps of the ideal path, in order", options, key="reference",
                           help="Conformance compares every case with this exact sequence.")
            st.caption("Select steps in the order they should happen.")
        with st.expander("Improvement assumptions", expanded=True):
            st.slider("Cut waiting at bottleneck and hand-offs (%)", 0, 80, 30, step=5, key="wait_cut")
            st.slider("Cut rework, extra steps, abandoned work (%)", 0, 100, 50, step=5, key="rework_cut")
            st.number_input("Labour cost per hour (€)", min_value=0.0, max_value=500.0, value=45.0, step=5.0,
                            key="cost_per_hour")
        with st.expander("Control charts"):
            st.slider("Baseline for limits (business days)", 10, 120, 40, step=5, key="baseline")
            st.slider("Rolling z-score window (days)", 5, 60, 20, step=5, key="z_window")
            st.slider("Anomaly threshold |z|", 2.0, 5.0, 3.0, step=0.5, key="z_threshold")

# ------------------------------------------------------------------------------ main
hero(
    "OpsMind AI",
    "Operations intelligence for business processes: process mining on event logs, KPI control charts and "
    "anomaly detection that pinpoint inefficiencies, mapped to Lean wastes with estimated savings.",
    ["Process mining", "Bottlenecks & rework", "Conformance", "XmR control charts", "Lean waste map",
     "AI improvement plan"],
)

if ds is None:
    if upload_error:
        st.error(f"That file could not be used: {upload_error}")
    st.info("Upload an event log in the sidebar (CSV with case_id, activity and timestamp), or pick one of the "
            "sample datasets.")
    st.stop()

params = AnalysisParams(
    reference=tuple(ss["reference"]),
    wait_cut=ss["wait_cut"] / 100,
    rework_cut=ss["rework_cut"] / 100,
    cost_per_hour=float(ss["cost_per_hour"]),
    baseline=int(ss["baseline"]),
    z_window=int(ss["z_window"]),
    z_threshold=float(ss["z_threshold"]),
)
c = _compute(ss["dataset_key"], json.dumps(asdict(params)), ds)
m = c.mining
dfp = _data_fp(ss["dataset_key"], ds)
fp = fingerprint(ds, params, dfp)
t0, t1 = ds.log.period
context_line(f"{ds.company} · {ds.process} · {num(m.n_cases)} cases · {num(ds.log.n_events)} events · "
             f"{t0:%d %b} – {t1:%d %b %Y}", ds.synthetic)
if ds.log.dropped_rows:
    st.caption(f"⚠️ {num(ds.log.dropped_rows)} rows were skipped (missing case id, activity or timestamp).")

# ------------------------------------------------------------------------ AI analysis
run_col, note_col = st.columns([1, 2.6], vertical_alignment="center")
can_run = settings.live and bool(params.reference)
clicked = run_col.button("✨ Run AI analysis", type="primary", disabled=not can_run, width="stretch")
if clicked:
    with st.status("Running the AI analysts…", expanded=True) as run_status:
        try:
            llm = LLMClient(settings, app_name="OpsMind AI")
            ss["analysis"] = run_ai(llm, ds, c, params, on_step=run_status.write)
            run_status.update(label=f"Analysis complete · {ss['analysis']['model']}", state="complete",
                              expanded=False)
        except LLMError as exc:
            run_status.update(label="The AI analysis failed", state="error")
            st.error(str(exc))

analysis, freshness = None, None
current = ss.get("analysis")
if current and current["fingerprint"] == fp:
    analysis, freshness = current, "fresh"
elif (saved := load_saved_analysis(fp)) is not None:
    analysis, freshness = saved, "saved"
elif current and current.get("data_fingerprint") == dfp:
    analysis, freshness = current, "settings_changed"
elif (other := find_saved_for_data(dfp)) is not None:
    analysis, freshness = other, "settings_changed"

if freshness == "saved":
    note_col.caption(f"Showing the saved AI analysis for this sample (`{analysis['model']}`, "
                     f"{analysis['generated_at'][:10]}). No model calls used.")
elif freshness == "fresh":
    note_col.caption(f"AI analysis by `{analysis['model']}`. Repeat runs are served from the cache.")
elif freshness == "settings_changed":
    note_col.caption("⚠️ Settings changed since the AI analysis: computed figures are current, the AI text refers "
                     "to the earlier settings.")
elif settings.live:
    note_col.caption("Two calls to free OpenRouter models, usually 30-90 seconds. Results are cached.")
else:
    note_col.caption("The AI step runs once an OpenRouter key is configured (README → Configuration). Every "
                     "computed figure below works without a key.")

# ------------------------------------------------------------------------- headline
lc, rs, conf = m.little, m.rework_summary, m.conformance
bottleneck = m.bottleneck
kpis = st.columns(6)
kpis[0].metric("Cases", num(m.n_cases), help=f"{num(ds.log.n_events)} events in {num(len(m.variants))} variants.")
kpis[1].metric("Median time", days(m.throughput.median_days),
               help="Median throughput time: first start to last completion of a case.")
kpis[2].metric("Fitting cases", pct(conf.fitting_share, 0),
               help="Cases whose activity sequence equals the reference path exactly.")
kpis[3].metric("Rework", pct(rs.case_share, 0), help="Cases in which at least one step runs more than once.")
kpis[4].metric("Avg WIP", f"{lc.avg_wip:.0f}" if lc else "—",
               help="Average number of open cases over the middle 80% of the log.")
flow = m.throughput.flow_efficiency
kpis[5].metric("Flow efficiency", pct(flow, 1) if flow is not None else "n/a",
               help="Processing time ÷ throughput time. The rest of a case's life is spent waiting.")
if bottleneck:
    share = m.activities.set_index("activity").loc[bottleneck, "wait_share"]
    pill(f"Main bottleneck: **{bottleneck}** · {pct(share, 0)} of all waiting time")
st.write("")

tabs = st.tabs(["📊 Overview", "🗺️ Process map", "🔁 Variants & rework", "⏳ Bottlenecks", "📉 KPI control",
                "♻️ Lean waste", "🧠 Improvement plan"])

# -------------------------------------------------------------------------- overview
with tabs[0]:
    left, right = st.columns([1.1, 1])
    with left:
        card("🔎 What the data shows", c.findings, markdown=True)
    right.plotly_chart(charts.throughput_hist(m.cases, m.throughput), width="stretch")
    left, right = st.columns([1.6, 1])
    left.plotly_chart(charts.wip_chart(m.wip, lc), width="stretch")
    if lc:
        right.markdown(
            f"""
**Little's Law check**

Average work in progress *L* should equal arrival rate *λ* × mean throughput time *W*.

- Measured L = **{lc.avg_wip:.1f}** open cases
- λ = **{lc.arrival_rate:.2f}** cases a day, W = **{lc.mean_days:.2f}** days
- λ × W = **{lc.predicted_wip:.1f}**, a gap of **{signed(lc.gap * 100, 1, '%')}**

{"A gap within ±10% means the log describes a stable flow and its timestamps are consistent."
 if lc.consistent else "A gap above 10% means WIP is building up or draining, or the log is incomplete."}
Measured on {lc.window_start:%d %b} – {lc.window_end:%d %b}, leaving out the first and last 10% of the log.
"""
        )
    st.markdown("##### Daily KPIs at a glance")
    rows = []
    for r in c.kpis.values():
        v = r.variance
        if v.get("on_target") is None:
            state = '<span class="om-muted">no target</span>'
        else:
            state = status("On target", "good") if v["on_target"] else status("Off target", "bad")
        rows.append([r.meta.label, num_cell(r.meta.fmt(v.get("recent"))),
                     num_cell(r.meta.fmt(r.meta.target)) if r.meta.target is not None else "—", state,
                     num_cell(str(r.rule_points)), num_cell(str(r.anomalies)),
                     trend_cell(r.trend_direction)])
    table(["KPI", "Last 10 days", "Target", "Status", "Rule violations", "Anomalies", "Trend"], rows,
          html_cols=(1, 2, 3, 4, 5, 6))
    if c.kpis_derived:
        st.caption("No KPI file supplied: throughput, cycle time and first-pass yield are derived from the log.")
    if ds.planted:
        with st.expander("🧪 What was planted in this synthetic data"):
            st.markdown("The generator built these realities into the data. Use them to check what the analysis "
                        "finds.\n\n" + "\n".join(f"- {p}" for p in ds.planted))

# ----------------------------------------------------------------------- process map
with tabs[1]:
    c1, c2, c3 = st.columns([2.2, 1.2, 1.2], vertical_alignment="bottom")
    min_share = c1.slider("Hide transitions taken by fewer than this share of cases (%)", 0.0, 20.0, 2.0,
                          step=0.5, key="edge_min")
    label_mode = c2.radio("Edge labels", ["Frequency", "Mean wait"], horizontal=True, key="edge_label")
    layout = c3.radio("Layout", ["Top-down", "Left-right"], horizontal=True, key="map_layout")
    dot = to_dot(m.dfg, m.activities, min_share=min_share / 100,
                 label="wait" if label_mode == "Mean wait" else "frequency", bottleneck=bottleneck,
                 rankdir="TB" if layout == "Top-down" else "LR")
    graph_col, side_col = st.columns([1.35, 1])
    graph_col.graphviz_chart(dot, width="stretch")
    hidden = len(m.dfg) - len(filter_edges(m.dfg, min_share / 100))
    with side_col:
        card("How to read the map", [
            "Each box is an activity: how often it ran and the mean wait before it started.",
            "Darker boxes have longer waits in front of them; the **red outline** marks the bottleneck.",
            "Arrows are direct successions; width follows how many times they happened.",
            f"**{hidden}** rare transitions are hidden by the slider.",
        ], markdown=True)
        st.markdown("##### Most frequent transitions")
        inner = m.dfg[(m.dfg["source"] != START) & (m.dfg["target"] != END)].head(10)
        table(["Transition", "Cases", "Mean wait"],
              [[f"{r['source']} → {r['target']}", num_cell(num(r["cases"])), num_cell(duration(r["mean_wait_h"]))]
               for _, r in inner.iterrows()], html_cols=(1, 2))

# ---------------------------------------------------------------- variants & rework
with tabs[2]:
    left, right = st.columns([1.7, 1])
    left.plotly_chart(charts.variant_pareto(m.variants), width="stretch")
    if conf.reference:
        right.plotly_chart(charts.conformance_donut(conf.fitting_share), width="stretch")
        right.caption(f"Mean trace fitness **{conf.mean_fitness:.2f}** (1 = identical to the reference path, "
                      "based on edit distance).")
    top_n = min(len(m.variants), 10)
    st.markdown(f"##### Top {top_n} of {num(len(m.variants))} variants")
    table(["Variant", "Cases", "Share", "Mean time", "Path"],
          [[r["variant"], num_cell(num(r["cases"])), share_bar(r["share"]), num_cell(days(r["mean_days"])),
            r["path"]] for _, r in m.variants.head(top_n).iterrows()], html_cols=(1, 2, 3))
    st.markdown("##### Deviations from the reference path")
    if conf.reference and len(conf.deviations):
        table(["Deviation", "Step", "Cases", "Share of cases"],
              [[r["kind"], r["detail"], num_cell(num(r["cases"])), share_bar(r["share"])]
               for _, r in conf.deviations.head(10).iterrows()], html_cols=(2, 3))
    else:
        st.caption("Set a reference path in the sidebar to check conformance.")

    st.markdown("##### Rework")
    left, right = st.columns([1.4, 1])
    left.plotly_chart(charts.rework_bar(m.rework), width="stretch")
    right.markdown(
        f"""
**{pct(rs.case_share)}** of cases repeat at least one step ({num(rs.extra_executions)} extra runs in total).

Cases with rework take **{days(rs.mean_days_with)}** on average against **{days(rs.mean_days_without)}**
without, **{signed(rs.extra_days)} days** longer.

Most repeated step: **{rs.top_activity or "none"}**.
"""
    )
    rw = m.rework[m.rework["repeat_cases"] > 0]
    if len(rw):
        table(["Activity", "Cases repeating it", "Rework rate", "Extra runs", "Extra work", "Extra waiting"],
              [[r["activity"], num_cell(num(r["repeat_cases"])), share_bar(r["rework_rate"]),
                num_cell(num(r["extra_executions"])), num_cell(duration(r["extra_proc_h"])),
                num_cell(duration(r["extra_wait_h"]))] for _, r in rw.iterrows()], html_cols=(1, 2, 3, 4, 5))

# ---------------------------------------------------------------------- bottlenecks
with tabs[3]:
    st.plotly_chart(charts.waiting_bar(m.activities, bottleneck), width="stretch")
    acts = m.activities.dropna(subset=["mean_wait_h"])
    table(["Activity", "Runs", "Mean wait", "Median", "p90", "Share of waiting", "Mean work time"],
          [[("🔴 " if r["activity"] == bottleneck else "") + r["activity"], num_cell(num(r["events"])),
            num_cell(duration(r["mean_wait_h"])), num_cell(duration(r["median_wait_h"])),
            num_cell(duration(r["p90_wait_h"])), share_bar(r["wait_share"]), num_cell(duration(r["mean_proc_h"]))]
           for _, r in acts.iterrows()], html_cols=(1, 2, 3, 4, 5, 6))
    if not ds.log.has_start:
        st.caption("No start timestamps: waiting time is the time between completions and includes the work itself.")

    st.markdown("##### Calendar effects")
    left, right = st.columns([1.5, 1])
    left.plotly_chart(charts.weekday_chart(m.weekdays), width="stretch")
    wd = m.weekdays[m.weekdays["cases"] >= 5].dropna(subset=["mean_days"])
    bullets = [f"**{pct(m.weekend_wait_share, 0)}** of all waiting time falls on a Saturday or Sunday."]
    if len(wd) >= 2:
        slow, fast = wd.loc[wd["mean_days"].idxmax()], wd.loc[wd["mean_days"].idxmin()]
        bullets.append(f"Cases started on **{slow['weekday']}** take **{slow['mean_days']:.1f} days**, against "
                       f"{fast['mean_days']:.1f} for **{fast['weekday']}** starts.")
    if m.handoffs:
        h = m.handoffs
        bullets.append(f"Cases change hands **{h.per_case:.1f}** times; waits after a hand-off average "
                       f"**{duration(h.wait_after_handoff_h)}** against {duration(h.wait_same_person_h)} when the "
                       "same person continues.")
    with right:
        card("📅 Weekends and hand-offs", bullets, markdown=True)

    if not m.resources.empty:
        st.markdown("##### Workload")
        left, right = st.columns([1, 1.25])
        left.plotly_chart(charts.resource_chart(m.resources), width="stretch")
        shared = m.overload.head(8)
        with right:
            st.caption("Steps done by several people: each person's share of the step and how long cases wait for "
                       "them compared with their colleagues.")
            table(["Person and step", "Share of step", "Wait vs others"],
                  [[("⚠️ " if r["overloaded"] else "") + f"{r['resource']} · {r['activity']}",
                    share_bar(r["share"]), num_cell(f"{r['wait_ratio']:.2f}×")]
                   for _, r in shared.iterrows()], html_cols=(1, 2))
            st.caption("⚠️ Overloaded: at least 1.5× an even share of the step, and cases wait at least 1.25× "
                       "longer for them.")

# ----------------------------------------------------------------------- KPI control
with tabs[4]:
    if not c.kpis:
        st.info("No KPI series with at least three values.")
    else:
        keys = list(c.kpis)
        if ss.get("kpi_choice") not in keys:
            ss["kpi_choice"] = worst_kpi(c.kpis).key
        kpi_key = st.selectbox("KPI", keys, format_func=lambda k: c.kpis[k].meta.label, key="kpi_choice")
        r = c.kpis[kpi_key]
        v, meta = r.variance, r.meta
        cols = st.columns(5)
        cols[0].metric("Last 10 days", meta.fmt(v.get("recent")))
        cols[1].metric("Target", meta.fmt(meta.target) if meta.target is not None else "—")
        gap = v.get("gap")
        cols[2].metric("Gap", "—" if gap is None or gap != gap else meta.fmt(gap) if not meta.percent
                       else f"{gap * 100:+.1f} pts")
        cols[3].metric("Rule violations", str(r.rule_points), help="Days that complete a Western Electric rule.")
        cols[4].metric("Anomalies", str(r.anomalies), help=f"|z| ≥ {params.z_threshold:g} against the previous "
                                                           f"{params.z_window} days.")
        st.plotly_chart(charts.xmr_chart(r), width="stretch")
        left, right = st.columns(2)
        tr = r.trend
        with left:
            card("📈 Trend (least squares)", [
                f"Slope **{meta.fmt(tr.slope, 4)}** per business day, **{meta.fmt(tr.change, 2)}** over the period.",
                f"R² **{tr.r2:.2f}**, {p_value(tr.p_value)}: **{r.trend_direction}**"
                + (" (significant at 5%)." if tr.significant else " (not significant at 5%)."),
            ], markdown=True)
        with right:
            cp = r.change
            if cp is None:
                card("🪜 Level shift", ["Too few points to test for a shift."])
            else:
                when = r.data["date"].iloc[cp.index]
                card("🪜 Level shift (best single split)", [
                    f"From **{meta.fmt(cp.mean_before)}** to **{meta.fmt(cp.mean_after)}** around "
                    f"**{when:%d %b %Y}**.",
                    f"Welch t-test {p_value(cp.p_value)}"
                    + (": a clear shift." if cp.clear else ": not a clear shift (needs p < 0.001 because the split "
                       "was chosen to fit)."),
                ], markdown=True)
        flagged = r.data[r.data["signal"] | r.data["anomaly"]].tail(12).iloc[::-1]
        st.markdown("##### Latest signals")
        if flagged.empty:
            st.caption("No rule violations or anomalies: the KPI behaves like a stable process.")
        else:
            table(["Date", "Value", "Rules broken", "z-score"],
                  [[f"{row['date']:%a %d %b %Y}", num_cell(meta.fmt(row["value"])),
                    ", ".join(RULES[k] for k in RULES if row[k]) or "—",
                    num_cell(f"{row['z']:+.1f}" if row["z"] == row["z"] else "—")]
                   for _, row in flagged.iterrows()], html_cols=(1, 3))
        st.caption(f"Limits use the first {r.limits.baseline_n} business days: centre ± 2.66 × the mean moving "
                   "range (σ ≈ MR̄ ÷ 1.128). Western Electric rules: 1 point beyond 3σ; 2 of 3 beyond 2σ; 4 of 5 "
                   "beyond 1σ; 8 in a row on one side.")

# ------------------------------------------------------------------------ Lean waste
with tabs[5]:
    lean = c.lean
    waste_grid([(ICONS[f.waste], f.waste, f.figure, f.detected) for f in lean.findings])
    cols = st.columns(4)
    cols[0].metric("Waiting removed", f"{num(lean.lead_time_h)} h/mo", help="Case-hours of waiting removed a month.")
    cols[1].metric("Labour freed", f"{num(lean.labour_h)} h/mo", help="Labour hours no longer needed a month.")
    cols[2].metric("Labour value", f"{money(lean.euros)}/mo", help="Labour hours × cost per hour.")
    cols[3].metric("Mean time", f"{lean.current_mean_days:.1f} → {lean.future_mean_days:.1f} d",
                   help="Mean throughput time today and after the cuts set in the sidebar.")
    left, right = st.columns(2)
    left.plotly_chart(charts.impact_waterfall(lean), width="stretch")
    right.plotly_chart(charts.waste_hours_bar(lean), width="stretch")
    right.caption("A light bar overlaps another estimate (for example rebalancing work at the bottleneck is one "
                  "way to cut its waiting), so it is left out of the totals.")
    table(["Waste", "Evidence (computed)", "Estimated impact a month", "Countermeasure"],
          [[f"{ICONS[f.waste]} {f.waste}" + ("" if f.detected else " · not detected"), f.evidence,
            f.impact + (f" ({f.formula})" if f.formula != "—" else ""), f.countermeasure]
           for f in lean.ranked()])
    st.caption(f"Assumptions from the sidebar: waiting at the bottleneck and hand-offs cut by "
               f"{pct(params.wait_cut, 0)}; "
               f"rework, extra steps and abandoned work cut by {pct(params.rework_cut, 0)}; labour at "
               f"{money(params.cost_per_hour)} an hour. Overlapping estimates are excluded from the totals.")

# ------------------------------------------------------------------ improvement plan
with tabs[6]:
    dx = Diagnosis.model_validate(analysis["diagnosis"]) if analysis else None
    if dx is None:
        if settings.live:
            st.info("Click **✨ Run AI analysis** above to add root-cause hypotheses, a prioritised improvement plan "
                    "and an executive summary built on the computed figures.")
        else:
            st.info("AI analysis runs once an OpenRouter key is configured: add `OPENROUTER_API_KEY` (see README → "
                    "Configuration), then click **✨ Run AI analysis**.")
        detected = [f.waste.lower() for f in lean.detected]
        card("🧠 What the AI step adds for this dataset", [
            f"Root-cause hypotheses for the main findings (starting with the queue before *{bottleneck}*), each with "
            "the computed evidence behind it and a test to confirm it." if bottleneck else
            "Root-cause hypotheses for the main findings, each with its evidence and a test to confirm it.",
            f"A prioritised improvement plan covering the {len(detected)} detected wastes "
            f"({', '.join(detected[:4])}{'…' if len(detected) > 4 else ''}), with owner roles and effort, ranked "
            "by the computed hours at stake.",
            "A one-page executive summary that quotes the computed figures, and the gaps this log cannot answer.",
        ], markdown=True)
        st.write("")
        st.markdown("##### Rule-based starting plan (computed, no AI)")
        table(["#", "Countermeasure", "Waste", "Estimated impact a month"],
              [[num_cell(str(i)), f.countermeasure, f"{ICONS[f.waste]} {f.waste}", f.impact]
               for i, f in enumerate(lean.detected, start=1) if f.countermeasure != "—"], html_cols=(0,))
    else:
        if freshness == "settings_changed":
            st.warning("This AI analysis was made with different settings. Computed figures elsewhere are current; "
                       "re-run the AI analysis to refresh this tab.")
        if dx.headline:
            st.markdown(f"#### {dx.headline}")
        summary = analysis.get("summary") or ""
        if summary:
            with st.container(border=True):
                st.markdown(re.sub(r"^#{1,3} ", "#### ", summary, flags=re.M))
        st.markdown("##### Root-cause hypotheses")
        st.caption("Model-generated hypotheses built on the computed evidence. Each comes with a test: confirm it "
                   "before acting.")
        table(["Symptom", "Hypothesis", "Evidence", "Test to confirm", "Confidence"],
              [[rc.problem, rc.hypothesis, "; ".join(rc.evidence), rc.test, severity(rc.confidence)]
               for rc in dx.root_causes], html_cols=(4,))
        st.markdown("##### Prioritised improvement plan")
        st.caption("Ranked by the computed monthly hours at stake for each action's waste, divided by the effort "
                   "the model assigned.")
        table(["#", "Action", "Waste", "Owner", "Effort", "Computed impact", "Expected impact (AI)"],
              [[num_cell(str(i)), ra.item.action, ra.item.waste or "—", ra.item.owner_role,
                severity(ra.item.effort), ra.finding.impact if ra.finding and ra.finding.detected else "—",
                ra.item.expected_impact]
               for i, ra in enumerate(rank_plan(dx.improvement_plan, lean), start=1)], html_cols=(0, 4))
        if dx.data_gaps:
            card("🔭 What this event log cannot show", dx.data_gaps)
    st.write("")
    report = build_report(ds, c, params, analysis if freshness in ("fresh", "saved") else None)
    st.download_button("⬇️ Download full report (Markdown)", report,
                       file_name=f"{ds.company.replace(' ', '_')}-opsmind-report.md", mime="text/markdown",
                       type="primary")

st.divider()
st.caption(
    "OpsMind AI is a decision-support tool. Every figure is computed from the event log and KPI series; "
    "AI-generated hypotheses and plans can be wrong and should be checked against the evidence."
)
