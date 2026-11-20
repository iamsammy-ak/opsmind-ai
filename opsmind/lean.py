"""Map process-mining findings to the eight Lean wastes and estimate what fixing them is worth.

Each rule reads computed evidence (waiting, rework, deviations, workload, hand-offs, WIP) and
returns the evidence, a monthly impact estimate with the formula behind it, and a standard
Lean countermeasure. The estimates are deliberately simple so they can be checked by hand:

- lead-time hours = case-hours of waiting removed per month (cases finish sooner)
- labour hours = hours of work no longer needed per month (people get time back)

Estimates that overlap another one (for example rebalancing an overloaded approver at the
bottleneck, which is one way of cutting the bottleneck's waiting) are shown but left out of
the totals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .formatting import duration, money, num, pct
from .mining import Mining

WASTES = (
    "Defects",
    "Overproduction",
    "Waiting",
    "Non-utilised talent",
    "Transportation",
    "Inventory",
    "Motion",
    "Extra-processing",
)
ICONS = {
    "Defects": "🔧",
    "Overproduction": "📦",
    "Waiting": "⏳",
    "Non-utilised talent": "🧑‍💼",
    "Transportation": "🔀",
    "Inventory": "🗃️",
    "Motion": "🚶",
    "Extra-processing": "➕",
}
DEFAULT_STEP_MINUTES = 15  # assumed work per step when the log has no start timestamps


@dataclass(frozen=True)
class ImpactParams:
    wait_cut: float = 0.30  # share of waiting removed at the bottleneck and at hand-offs
    rework_cut: float = 0.50  # share of rework, extra steps and abandoned work removed
    cost_per_hour: float = 45.0  # loaded labour cost


@dataclass(frozen=True)
class WasteFinding:
    waste: str
    detected: bool
    evidence: str
    figure: str  # short headline figure for the tile
    impact: str  # the estimate in words
    formula: str
    countermeasure: str
    lead_time_h: float = 0.0  # case-hours of waiting removed per month
    labour_h: float = 0.0  # labour hours freed per month
    in_total: bool = True

    @property
    def score(self) -> float:
        return self.lead_time_h + self.labour_h


@dataclass(frozen=True)
class LeanMap:
    findings: list[WasteFinding]
    cost_per_hour: float
    current_mean_days: float
    future_mean_days: float
    current_wip: float | None
    future_wip: float | None
    deltas: dict[str, float] = field(default_factory=dict)  # mean throughput days removed, by waste

    def get(self, waste: str) -> WasteFinding:
        return next(f for f in self.findings if f.waste == waste)

    def ranked(self) -> list[WasteFinding]:
        return sorted(self.findings, key=lambda f: (not f.detected, -f.score, WASTES.index(f.waste)))

    @property
    def detected(self) -> list[WasteFinding]:
        return [f for f in self.ranked() if f.detected]

    @property
    def labour_h(self) -> float:
        return sum(f.labour_h for f in self.findings if f.in_total)

    @property
    def lead_time_h(self) -> float:
        return sum(f.lead_time_h for f in self.findings if f.in_total)

    @property
    def euros(self) -> float:
        return self.labour_h * self.cost_per_hour


def _work_hours(events: pd.DataFrame, has_start: bool) -> float:
    return float(events["proc_h"].sum()) if has_start else len(events) * DEFAULT_STEP_MINUTES / 60


def _waiting(m: Mining, p: ImpactParams) -> tuple[WasteFinding, float]:
    b = m.bottleneck
    if b is None:
        return WasteFinding("Waiting", False, "No measurable waiting between steps.", "No waiting", "—", "—",
                            "—"), 0.0
    row = m.activities.set_index("activity").loc[b]
    ev = m.events
    first_runs = ev[(ev["activity"] == b) & (ev["occurrence"] == 1)]
    monthly = first_runs["wait_h"].sum() / m.months
    saved = monthly * p.wait_cut
    per_case_days = first_runs["wait_h"].sum() * p.wait_cut / max(m.n_cases, 1) / 24
    if "approv" in b.lower():
        fix = ("Add a backup approver, delegate approval below a value threshold and clear the queue daily "
               "against a visible service level.")
    else:
        fix = (f"Add capacity at '{b}' at peak times, work the queue first-in-first-out and set a service level "
               "for picking up cases.")
    evidence = (f"Cases wait {duration(row['mean_wait_h'])} on average before '{b}' (p90 "
                f"{duration(row['p90_wait_h'])}), {pct(row['wait_share'], 0)} of all waiting in the process. "
                f"{pct(m.weekend_wait_share, 0)} of all waiting time falls on weekends.")
    finding = WasteFinding(
        "Waiting",
        bool(row["mean_wait_h"] >= 1),
        evidence,
        f"{duration(row['mean_wait_h'])} queue",
        f"−{num(saved)} case-hours a month; mean throughput time −{per_case_days:.2f} days",
        f"{num(monthly)} waiting hours a month before first runs of '{b}' × {pct(p.wait_cut, 0)} cut",
        fix,
        lead_time_h=saved,
    )
    return finding, per_case_days


def _defects(m: Mining, p: ImpactParams, has_start: bool) -> tuple[WasteFinding, float]:
    rs = m.rework_summary
    ref = set(m.conformance.reference) or set(m.events["activity"])
    repeats = m.events[(m.events["occurrence"] > 1) & m.events["activity"].isin(ref)]
    if repeats.empty or rs.case_share < 0.01:
        return WasteFinding("Defects", False, "No repeated steps found.", "No rework", "—", "—", "—"), 0.0
    labour = _work_hours(repeats, has_start) / m.months * p.rework_cut
    top = repeats["activity"].value_counts().index[0]
    top_row = m.rework.set_index("activity").loc[top]
    extra_days = rs.extra_days if pd.notna(rs.extra_days) else 0.0
    rework_cases = rs.case_share * m.n_cases / m.months
    lead = max(extra_days, 0.0) * 24 * rework_cases * p.rework_cut
    per_case_days = max(extra_days, 0.0) * rs.case_share * p.rework_cut
    evidence = (f"{pct(rs.case_share)} of cases repeat at least one step; '{top}' is repeated in "
                f"{pct(top_row['rework_rate'])} of the cases that reach it. Cases with rework take "
                f"{extra_days:.1f} days longer.")
    finding = WasteFinding(
        "Defects",
        True,
        evidence,
        f"{pct(rs.case_share, 0)} redo work",
        f"{num(labour)} labour hours a month ({money(labour * p.cost_per_hour)}); −{num(lead)} case-hours of delay",
        f"{num(len(repeats) / m.months)} repeated steps a month × work time × {pct(p.rework_cut, 0)} cut",
        f"Error-proof the inputs to '{top}': validate data at entry, use a checklist and track the top reasons "
        "for repeats on a Pareto chart.",
        lead_time_h=lead,
        labour_h=labour,
    )
    return finding, per_case_days


def _extra_processing(m: Mining, p: ImpactParams, has_start: bool) -> WasteFinding:
    ref = m.conformance.reference
    if not ref:
        return WasteFinding("Extra-processing", False, "Set a reference path to detect extra steps.", "—", "—",
                            "—", "—")
    ev = m.events
    completed = m.cases.index[m.cases["trace"].map(lambda t: ref[-1] in t)]
    extra = ev[~ev["activity"].isin(ref) & ev["case_id"].isin(completed)]
    if extra.empty:
        return WasteFinding("Extra-processing", False, "Completed cases only use reference steps.", "None", "—",
                            "—", "—")
    top = extra["activity"].value_counts().index[0]
    case_share = extra["case_id"].nunique() / max(m.n_cases, 1)
    labour = _work_hours(extra, has_start) / m.months * p.rework_cut
    return WasteFinding(
        "Extra-processing",
        case_share >= 0.01,
        f"{num(len(extra))} executions of steps outside the reference path in completed cases "
        f"({pct(case_share)} of cases), most often '{top}'.",
        f"{pct(case_share, 0)} take extra steps",
        f"{num(labour)} labour hours a month ({money(labour * p.cost_per_hour)})",
        f"{num(len(extra) / m.months)} extra steps a month × work time × {pct(p.rework_cut, 0)} cut",
        f"Ask why '{top}' is needed: remove its upstream cause or automate it, then update the standard work.",
        labour_h=labour,
    )


def _overproduction(m: Mining, p: ImpactParams, has_start: bool) -> WasteFinding:
    ref = m.conformance.reference
    if not ref:
        return WasteFinding("Overproduction", False, "Set a reference path to detect abandoned cases.", "—", "—",
                            "—", "—")
    abandoned = m.cases[~m.cases["trace"].map(lambda t: ref[-1] in t)]
    share = len(abandoned) / max(m.n_cases, 1)
    if abandoned.empty:
        return WasteFinding("Overproduction", False, f"Every case reaches '{ref[-1]}'.", "None", "—", "—", "—")
    work = m.events[m.events["case_id"].isin(abandoned.index)]
    monthly = _work_hours(work, has_start) / m.months
    last = abandoned["last_activity"].value_counts().index[0]
    return WasteFinding(
        "Overproduction",
        share >= 0.01,
        f"{pct(share)} of cases never reach '{ref[-1]}' (most end with '{last}'); {num(monthly, 1)} hours of "
        "work a month go into them.",
        f"{pct(share, 1)} abandoned",
        f"{num(monthly * p.rework_cut, 1)} labour hours a month ({money(monthly * p.rework_cut * p.cost_per_hour)})",
        f"{num(monthly, 1)} hours a month on abandoned cases × {pct(p.rework_cut, 0)} avoided",
        "Screen earlier: move the check that stops these cases to the start of the process so less work is done "
        "before a case is dropped.",
        labour_h=monthly * p.rework_cut,
    )


def _talent(m: Mining, p: ImpactParams) -> WasteFinding:
    ov = m.overload
    flagged = ov[ov["overloaded"]] if not ov.empty else ov
    if flagged.empty:
        msg = "No resource data in the log." if ov.empty else "Work is spread evenly across people."
        return WasteFinding("Non-utilised talent", False, msg, "Balanced", "—", "—", "—")
    top = flagged.sort_values("events", ascending=False).iloc[0]
    excess = (top["mean_wait_h"] - top["others_wait_h"]) * top["events"] / m.months
    peers = round(1 / top["fair_share"]) - 1
    overlaps = top["activity"] == m.bottleneck
    note = " This is one cause of the bottleneck's waiting, so it is not added to the totals." if overlaps else ""
    return WasteFinding(
        "Non-utilised talent",
        True,
        f"{top['resource']} does {pct(top['share'], 0)} of '{top['activity']}' (an even split would be "
        f"{pct(top['fair_share'], 0)}); cases wait {top['wait_ratio']:.1f}× longer for them than for their "
        f"{peers} colleague{'s' if peers != 1 else ''}.{note}",
        f"{top['resource']}: {pct(top['share'], 0)} of load",
        f"−{num(excess)} case-hours a month if the work were shared evenly",
        f"({top['mean_wait_h']:.1f} h − {top['others_wait_h']:.1f} h) × {num(top['events'] / m.months)} "
        "cases a month",
        f"Route '{top['activity']}' round-robin, cross-train colleagues and give them the authority to act, so one "
        "person is not the default.",
        lead_time_h=max(excess, 0.0),
        in_total=not overlaps,
    )


def _transportation(m: Mining, p: ImpactParams) -> tuple[WasteFinding, float]:
    ev = m.events
    steps = ev[ev["prev_resource"].notna() & (ev["activity"] != m.bottleneck)]
    steps = steps[(steps["resource"] != "(unknown)") & (steps["prev_resource"] != "(unknown)")]
    if steps.empty or m.handoffs is None:
        return WasteFinding("Transportation", False, "No resource data in the log.", "—", "—", "—", "—"), 0.0
    hand = steps["resource"] != steps["prev_resource"]
    after, same = steps.loc[hand, "wait_h"].mean(), steps.loc[~hand, "wait_h"].mean()
    extra = (after - same) if pd.notna(after) and pd.notna(same) else 0.0
    monthly = max(extra, 0.0) * hand.sum() / m.months
    saved = monthly * p.wait_cut
    per_case_days = max(extra, 0.0) * hand.sum() * p.wait_cut / max(m.n_cases, 1) / 24
    h = m.handoffs
    return WasteFinding(
        "Transportation",
        bool(h.per_case >= 1 and extra > 0),
        f"Cases change hands {h.per_case:.1f} times on average. Away from the bottleneck, waits after a hand-off "
        f"average {duration(after)} against {duration(same)} when the same person continues.",
        f"{h.per_case:.1f} hand-offs per case",
        f"−{num(saved)} case-hours a month",
        f"({duration(after)} − {duration(same)}) × {num(hand.sum() / m.months)} hand-offs a month × "
        f"{pct(p.wait_cut, 0)} cut",
        "Give one person ownership of consecutive steps where skills allow, and pass work on with complete "
        "information so it does not queue at each hand-off.",
        lead_time_h=saved,
    ), per_case_days


def _inventory(m: Mining, future_days: float) -> tuple[WasteFinding, float | None]:
    lc = m.little
    if lc is None or not lc.avg_wip >= 1:
        return WasteFinding("Inventory", False, "Too few overlapping cases to measure work in progress.", "—", "—",
                            "—", "—"), None
    future = lc.arrival_rate * future_days
    return WasteFinding(
        "Inventory",
        True,
        f"On average {lc.avg_wip:.0f} cases are open at once. Little's Law: {lc.arrival_rate:.1f} new cases a day × "
        f"{lc.mean_days:.1f} days ≈ {lc.predicted_wip:.0f}.",
        f"{lc.avg_wip:.0f} open cases",
        f"About {future:.0f} open cases once throughput time falls ({future - lc.avg_wip:+.0f}). Follows from the "
        "waiting and rework estimates, so it is not counted again.",
        f"λ {lc.arrival_rate:.2f} cases/day × future W {future_days:.2f} days",
        "Cap work in progress and release new cases only when the bottleneck has capacity (kanban); review ageing "
        "cases weekly.",
        in_total=False,
    ), future


MOTION = WasteFinding(
    "Motion",
    False,
    "Not visible in an event log: searching for information, switching between systems and re-keying data happen "
    "inside a step.",
    "Not in the log",
    "—",
    "—",
    "Walk the process (gemba) and time a few cases end to end; look for searching, re-keying and switching between "
    "systems.",
)


def lean_map(m: Mining, has_start: bool, p: ImpactParams | None = None) -> LeanMap:
    p = p or ImpactParams()
    waiting, d_wait = _waiting(m, p)
    defects, d_rework = _defects(m, p, has_start)
    transport, d_hand = _transportation(m, p)
    current = m.throughput.mean_days
    future = max(current - d_wait - d_rework - d_hand, 0.0)
    inventory, future_wip = _inventory(m, future)
    findings = [
        defects,
        _overproduction(m, p, has_start),
        waiting,
        _talent(m, p),
        transport,
        inventory,
        MOTION,
        _extra_processing(m, p, has_start),
    ]
    return LeanMap(
        findings=findings,
        cost_per_hour=p.cost_per_hour,
        current_mean_days=current,
        future_mean_days=future,
        current_wip=m.little.avg_wip if m.little else None,
        future_wip=future_wip,
        deltas={"Waiting": d_wait, "Defects": d_rework, "Transportation": d_hand},
    )
