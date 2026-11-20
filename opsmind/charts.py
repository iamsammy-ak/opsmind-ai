"""Plotly figures used by the app. Colours follow the OpsMind teal palette."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .formatting import days, duration, pct
from .kpi import RULES, KpiResult
from .lean import LeanMap
from .mining import LittleCheck, ThroughputStats

ACCENT = "#0891B2"
ACCENT_SOFT = "#A5E8F5"
ACCENT_MID = "#22B8D6"
ACCENT_DARK = "#164E63"
GOOD = "#059669"
BAD = "#DC2626"
WARN = "#D97706"
MUTED = "#94A3B8"
FONT = "Inter, 'Segoe UI', system-ui, sans-serif"


def _style(fig: go.Figure, height: int = 340, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0, font=dict(size=15)) if title else None,
        margin=dict(l=8, r=8, t=48 if title else 16, b=8),
        font=dict(family=FONT, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_family=FONT),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.25)", zeroline=False)
    return fig


# ------------------------------------------------------------------------------- overview
def throughput_hist(cases: pd.DataFrame, stats: ThroughputStats) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=cases["throughput_d"],
            xbins=dict(size=1),
            marker=dict(color=ACCENT, line=dict(color="white", width=1)),
            hovertemplate="%{x} days: %{y} cases<extra></extra>",
            name="Cases",
        )
    )
    for value, name, color, pos in ((stats.median_days, "median", ACCENT_DARK, "top left"),
                                    (stats.p90_days, "p90", WARN, "top right")):
        fig.add_vline(x=value, line=dict(color=color, width=2, dash="dash"),
                      annotation_text=f"{name} {days(value)}", annotation_position=pos,
                      annotation_font=dict(color=color, size=12))
    fig.update_xaxes(title="Throughput time (days)")
    fig.update_yaxes(title="Cases")
    return _style(fig, height=430, title="How long cases take")


def wip_chart(wip: pd.DataFrame, little: LittleCheck | None) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=wip["date"], y=wip["wip"], mode="lines", line=dict(color=ACCENT, width=2), fill="tozeroy",
        fillcolor="rgba(8,145,178,0.12)", name="Open cases", hovertemplate="%{x|%d %b}: %{y} open<extra></extra>",
    )
    if little is not None:
        fig.add_vrect(x0=little.window_start, x1=little.window_end, fillcolor="rgba(148,163,184,0.10)", line_width=0)
        fig.add_hline(y=little.avg_wip, line=dict(color=ACCENT_DARK, dash="dash", width=1.5))
        fig.add_annotation(x=little.window_start, y=1.0, yref="paper", xanchor="left", yanchor="top",
                           showarrow=False, font=dict(size=12, color=ACCENT_DARK),
                           text=f"average {little.avg_wip:.1f} open cases (dashed)")
    fig.update_yaxes(title="Open cases at day end")
    return _style(fig, title="Work in progress (shaded: Little's Law window)")


# ------------------------------------------------------------------------------- variants
def variant_pareto(variants: pd.DataFrame, top: int = 12) -> go.Figure:
    data = variants.head(top)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(
        x=data["variant"], y=data["share"], name="Share of cases",
        marker_color=[ACCENT if not rw else ACCENT_SOFT for rw in data["rework"]],
        customdata=data[["cases", "mean_days", "steps"]].to_numpy(),
        hovertemplate="%{x}: %{y:.1%} of cases (%{customdata[0]})<br>%{customdata[2]} steps · mean "
                      "%{customdata[1]:.1f} days<extra></extra>",
    )
    fig.add_scatter(x=data["variant"], y=data["cum_share"], name="Cumulative share", mode="lines+markers",
                    line=dict(color=ACCENT_DARK, width=2), marker=dict(size=6), secondary_y=True,
                    hovertemplate="%{x}: %{y:.1%} cumulative<extra></extra>")
    fig.update_yaxes(tickformat=".0%", title="Share of cases", secondary_y=False)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.04], dtick=0.25, showgrid=False, secondary_y=True)
    _style(fig, title="Variant Pareto (light bars contain rework)")
    return fig


def conformance_donut(fitting_share: float) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=["Follow the reference path", "Deviate"],
            values=[fitting_share, 1 - fitting_share],
            hole=0.68,
            marker=dict(colors=[ACCENT, "#E2E8F0"]),
            textinfo="none",
            sort=False,
            hovertemplate="%{label}: %{percent}<extra></extra>",
        )
    )
    fig.add_annotation(text=f"<b>{pct(fitting_share, 0)}</b><br><span style='font-size:12px'>fitting</span>",
                       showarrow=False, font=dict(size=24, color=ACCENT_DARK))
    _style(fig, height=280)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"))
    return fig


def rework_bar(rework: pd.DataFrame) -> go.Figure:
    data = rework[rework["repeat_cases"] > 0].sort_values("rework_rate")
    fig = go.Figure(
        go.Bar(
            y=data["activity"], x=data["rework_rate"], orientation="h", marker_color=WARN,
            text=[pct(v) for v in data["rework_rate"]], textposition="outside", cliponaxis=False,
            customdata=data[["repeat_cases", "extra_executions"]].to_numpy(),
            hovertemplate="%{y}: repeated in %{x:.1%} of its cases (%{customdata[0]})<br>%{customdata[1]} extra "
                          "runs<extra></extra>",
        )
    )
    fig.update_xaxes(tickformat=".0%", title="Cases where the step runs more than once")
    height = 140 + 34 * max(len(data), 1)
    _style(fig, height=height, title="Rework by activity")
    fig.update_layout(margin=dict(l=8, r=48, t=48, b=8))
    return fig


# ------------------------------------------------------------------------------ bottlenecks
def waiting_bar(activities: pd.DataFrame, bottleneck: str | None) -> go.Figure:
    data = activities.dropna(subset=["mean_wait_h"]).sort_values("mean_wait_h")
    colors = [BAD if a == bottleneck else ACCENT for a in data["activity"]]
    fig = go.Figure()
    fig.add_bar(
        y=data["activity"], x=data["mean_wait_h"], orientation="h", marker_color=colors, name="Mean wait",
        text=[duration(v) for v in data["mean_wait_h"]], textposition="outside", cliponaxis=False,
        customdata=data[["median_wait_h", "p90_wait_h", "wait_share"]].to_numpy(),
        hovertemplate="%{y}<br>mean %{x:.1f} h · median %{customdata[0]:.1f} h · p90 %{customdata[1]:.1f} h"
                      "<br>%{customdata[2]:.0%} of all waiting<extra></extra>",
    )
    fig.add_scatter(
        y=data["activity"], x=data["p90_wait_h"], mode="markers", name="p90 wait",
        marker=dict(symbol="line-ns-open", size=16, color=ACCENT_DARK, line=dict(width=2)),
        hovertemplate="%{y}: p90 %{x:.1f} h<extra></extra>",
    )
    fig.update_xaxes(title="Hours waiting before the activity starts")
    _style(fig, height=150 + 36 * len(data), title="Waiting time before each activity")
    fig.update_layout(margin=dict(l=8, r=56, t=48, b=8))
    return fig


def weekday_chart(weekdays: pd.DataFrame) -> go.Figure:
    data = weekdays[weekdays["cases"] > 0]
    fig = go.Figure(
        go.Bar(
            x=data["weekday"], y=data["mean_days"], marker_color=ACCENT,
            text=[f"{v:.1f} d" for v in data["mean_days"]], textposition="outside", cliponaxis=False,
            customdata=data["cases"], hovertemplate="Started %{x}: mean %{y:.2f} days (%{customdata} cases)"
                                                    "<extra></extra>",
        )
    )
    fig.update_yaxes(title="Mean throughput time (days)", rangemode="tozero")
    return _style(fig, height=320, title="Throughput time by the weekday a case starts")


def resource_chart(resources: pd.DataFrame) -> go.Figure:
    data = resources.sort_values("events")
    colors = [WARN if flag else ACCENT for flag in data["overloaded"]]
    fig = go.Figure(
        go.Bar(
            y=data["resource"], x=data["events"], orientation="h", marker_color=colors,
            customdata=data[["main_activity", "mean_wait_h"]].to_numpy(),
            hovertemplate="%{y}: %{x} steps<br>mostly %{customdata[0]} · cases wait %{customdata[1]:.1f} h"
                          "<extra></extra>",
        )
    )
    fig.update_xaxes(title="Steps performed")
    return _style(fig, height=140 + 24 * len(data), title="Workload by person (amber: overloaded)")


# ---------------------------------------------------------------------------------- KPIs
def xmr_chart(r: KpiResult) -> go.Figure:
    d, lim, meta = r.data, r.limits, r.meta
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.06)
    if pd.notna(lim.sigma):
        for k, alpha in ((2, 0.07), (1, 0.07)):
            fig.add_hrect(y0=lim.center - k * lim.sigma, y1=lim.center + k * lim.sigma, row=1, col=1,
                          fillcolor=f"rgba(8,145,178,{alpha})", line_width=0)
    fig.add_scatter(x=d["date"], y=d["value"], mode="lines+markers", name=meta.label,
                    line=dict(color=ACCENT, width=1.6), marker=dict(size=4, color=ACCENT),
                    hovertemplate="%{x|%a %d %b}: %{y:.3~f}<extra></extra>", row=1, col=1)
    sig = d[d["signal"]]
    if len(sig):
        reasons = [", ".join(RULES[k] for k in RULES if row[k]) for _, row in sig.iterrows()]
        fig.add_scatter(x=sig["date"], y=sig["value"], mode="markers", name="Rule violation",
                        marker=dict(size=8, color=BAD, line=dict(color="white", width=1)), customdata=reasons,
                        hovertemplate="%{x|%d %b}: %{y:.3~f}<br>%{customdata}<extra></extra>", row=1, col=1)
    anom = d[d["anomaly"]]
    if len(anom):
        fig.add_scatter(x=anom["date"], y=anom["value"], mode="markers", name="z-score anomaly",
                        marker=dict(size=13, symbol="diamond-open", color=WARN, line=dict(width=2)),
                        customdata=anom["z"], hovertemplate="%{x|%d %b}: z = %{customdata:.1f}<extra></extra>",
                        row=1, col=1)
    for y, name, color, dash in ((lim.center, "CL", ACCENT_DARK, "solid"), (lim.ucl, "UCL", BAD, "dash"),
                                 (lim.lcl, "LCL", BAD, "dash")):
        if pd.notna(y):
            fig.add_hline(y=y, line=dict(color=color, width=1.2, dash=dash), row=1, col=1,
                          annotation_text=f"{name} {meta.fmt(y, 2)}", annotation_position="right",
                          annotation_font=dict(size=11, color=color))
    if meta.target is not None:
        fig.add_hline(y=meta.target, line=dict(color=GOOD, width=1.5, dash="dot"), row=1, col=1,
                      annotation_text=f"target {meta.fmt(meta.target, 2)}", annotation_position="bottom left",
                      annotation_font=dict(size=11, color=GOOD))
    if r.change is not None and r.change.clear:
        fig.add_vline(x=d["date"].iloc[r.change.index], line=dict(color=ACCENT_DARK, width=1, dash="dot"),
                      row=1, col=1)
    baseline_end = d["date"].iloc[min(lim.baseline_n, len(d)) - 1]
    fig.add_vrect(x0=d["date"].iloc[0], x1=baseline_end, fillcolor="rgba(148,163,184,0.10)", line_width=0,
                  row=1, col=1, annotation_text="baseline", annotation_position="bottom left",
                  annotation_font=dict(size=11, color="#64748B"))
    fig.add_bar(x=d["date"], y=d["mr"], name="Moving range", marker_color=ACCENT_SOFT, row=2, col=1,
                hovertemplate="%{x|%d %b}: MR %{y:.3~f}<extra></extra>", showlegend=False)
    if pd.notna(lim.mr_ucl):
        fig.add_hline(y=lim.mr_ucl, line=dict(color=BAD, width=1, dash="dash"), row=2, col=1)
    if meta.percent:
        fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="MR", row=2, col=1)
    _style(fig, height=470, title=f"{meta.label}: individuals (XmR) chart")
    fig.update_layout(margin=dict(l=8, r=78, t=60, b=8))
    return fig


# ---------------------------------------------------------------------------------- Lean
def impact_waterfall(lean: LeanMap) -> go.Figure:
    steps = [(w, -d) for w, d in lean.deltas.items() if d > 0.001]
    labels = ["Today"] + [f"−{w}" for w, _ in steps] + ["After"]
    values = [lean.current_mean_days] + [v for _, v in steps] + [lean.future_mean_days]
    fig = go.Figure(
        go.Waterfall(
            x=labels, y=values, measure=["absolute"] + ["relative"] * len(steps) + ["total"],
            text=[days(v, 2) if i in (0, len(values) - 1) else f"{v:+.2f} d" for i, v in enumerate(values)],
            textposition="outside", cliponaxis=False,
            connector=dict(line=dict(color=MUTED, width=1, dash="dot")),
            increasing=dict(marker=dict(color=BAD)), decreasing=dict(marker=dict(color=GOOD)),
            totals=dict(marker=dict(color=ACCENT)),
            hovertemplate="%{x}: %{y:.2f} days<extra></extra>",
        )
    )
    fig.update_yaxes(title="Mean throughput time (days)", rangemode="tozero")
    return _style(fig, height=330, title="Mean throughput time if the wastes are cut")


def waste_hours_bar(lean: LeanMap) -> go.Figure:
    data = [f for f in lean.ranked() if f.detected and f.lead_time_h > 0][::-1]
    fig = go.Figure(
        go.Bar(
            y=[f.waste for f in data], x=[f.lead_time_h for f in data], orientation="h",
            marker_color=[ACCENT if f.in_total else ACCENT_SOFT for f in data],
            text=[f"{f.lead_time_h:,.0f}" for f in data], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:,.0f} case-hours a month<extra></extra>",
        )
    )
    fig.update_xaxes(title="Case-hours of waiting removed a month")
    _style(fig, height=150 + 40 * max(len(data), 1), title="Where the waiting time can be won back")
    fig.update_layout(margin=dict(l=8, r=56, t=48, b=8))
    return fig
