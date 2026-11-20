"""Process mining on an event log.

Directly-follows graph, variants, throughput times, waiting time before each activity
(bottlenecks), rework, resource workload, hand-offs, weekday effects, conformance to a
reference path and a Little's Law consistency check. Pure pandas and numpy: every process
figure the app reports comes from this module.

Definitions used throughout:

- waiting time before an event = its start minus the completion of the previous event in the
  same case (0 for parallel, overlapping work). Without start timestamps it is the time between
  two completions.
- processing time = completion minus start of the same event (0 without start timestamps).
- throughput time of a case = last completion minus first start.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .eventlog import UNKNOWN_RESOURCE, EventLog

START, END = "▶ Start", "■ End"
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


# ------------------------------------------------------------------------------ per event / case
def enrich(events: pd.DataFrame) -> pd.DataFrame:
    """Add position, predecessor, waiting and processing time to each event (log must be sorted)."""
    ev = events.copy()
    g = ev.groupby("case_id", sort=False)
    ev["pos"] = g.cumcount()
    ev["is_last"] = g.cumcount(ascending=False) == 0
    ev["prev_activity"] = g["activity"].shift()
    ev["prev_resource"] = g["resource"].shift()
    ev["prev_end"] = g["end"].shift()
    ev["wait_h"] = ((ev["start"] - ev["prev_end"]).dt.total_seconds() / 3600).clip(lower=0)
    ev["proc_h"] = (ev["end"] - ev["start"]).dt.total_seconds() / 3600
    ev["occurrence"] = ev.groupby(["case_id", "activity"], sort=False).cumcount() + 1
    return ev


def case_table(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per case: start, end, trace, throughput time, rework and hand-offs."""
    g = ev.groupby("case_id", sort=False)
    handoff = (ev["prev_resource"].notna() & (ev["resource"] != ev["prev_resource"])).groupby(ev["case_id"]).sum()
    cases = pd.DataFrame(
        {
            "start": g["start"].min(),
            "end": g["end"].max(),
            "events": g.size(),
            "trace": g["activity"].agg(tuple),
            "repeats": (ev["occurrence"] > 1).groupby(ev["case_id"], sort=False).sum(),
            "proc_h": g["proc_h"].sum(),
            "wait_h": g["wait_h"].sum(),
            "handoffs": handoff,
            "last_activity": g["activity"].last(),
        }
    )
    cases["path"] = cases["trace"].map(" → ".join)
    cases["throughput_h"] = (cases["end"] - cases["start"]).dt.total_seconds() / 3600
    cases["throughput_d"] = cases["throughput_h"] / 24
    cases["has_rework"] = cases["repeats"] > 0
    return cases


# ------------------------------------------------------------------------------------- variants
def variants(cases: pd.DataFrame) -> pd.DataFrame:
    """Distinct activity sequences with their share and throughput time, most frequent first."""
    grp = cases.groupby("path", sort=False)
    out = pd.DataFrame(
        {
            "cases": grp.size(),
            "mean_days": grp["throughput_d"].mean(),
            "median_days": grp["throughput_d"].median(),
            "steps": grp["events"].first(),
            "rework": grp["has_rework"].first(),
        }
    ).reset_index()
    out = out.sort_values(["cases", "path"], ascending=[False, True], kind="stable").reset_index(drop=True)
    out["share"] = out["cases"] / out["cases"].sum()
    out["cum_share"] = out["share"].cumsum()
    out.insert(0, "variant", [f"V{i}" for i in range(1, len(out) + 1)])
    return out


# ------------------------------------------------------------------------ directly-follows graph
def dfg(ev: pd.DataFrame, n_cases: int) -> pd.DataFrame:
    """Directly-follows edges (including start and end) with frequency and mean waiting time."""
    inner = ev[ev["prev_activity"].notna()]
    edges = (
        inner.groupby(["prev_activity", "activity"])
        .agg(
            count=("case_id", "size"),
            cases=("case_id", "nunique"),
            mean_wait_h=("wait_h", "mean"),
            median_wait_h=("wait_h", "median"),
        )
        .reset_index()
        .rename(columns={"prev_activity": "source", "activity": "target"})
    )
    first = ev[ev["pos"] == 0].groupby("activity").size()
    last = ev[ev["is_last"]].groupby("activity").size()
    starts = pd.DataFrame({"source": START, "target": first.index, "count": first.values, "cases": first.values})
    ends = pd.DataFrame({"source": last.index, "target": END, "count": last.values, "cases": last.values})
    out = pd.concat([edges, starts, ends], ignore_index=True)
    out["share"] = out["cases"] / max(n_cases, 1)
    return out.sort_values(["count", "source", "target"], ascending=[False, True, True], kind="stable").reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------------- bottlenecks
def activity_stats(ev: pd.DataFrame) -> pd.DataFrame:
    """Waiting time before each activity and its processing time. Sorted by total waiting hours."""
    g = ev.groupby("activity")
    stats = pd.DataFrame(
        {
            "events": g.size(),
            "cases": g["case_id"].nunique(),
            "mean_wait_h": g["wait_h"].mean(),
            "median_wait_h": g["wait_h"].median(),
            "p90_wait_h": g["wait_h"].quantile(0.9),
            "total_wait_h": g["wait_h"].sum(),
            "mean_proc_h": g["proc_h"].mean(),
            "total_proc_h": g["proc_h"].sum(),
        }
    )
    total = stats["total_wait_h"].sum()
    stats["wait_share"] = stats["total_wait_h"] / total if total > 0 else 0.0
    return stats.sort_values(["total_wait_h", "events"], ascending=False, kind="stable").reset_index()


def bottleneck(activities: pd.DataFrame) -> str | None:
    """The activity with the most total waiting time in front of it."""
    if activities.empty or not activities["total_wait_h"].iloc[0] > 0:
        return None
    return str(activities["activity"].iloc[0])


# --------------------------------------------------------------------------------------- rework
@dataclass(frozen=True)
class ReworkSummary:
    case_share: float  # share of cases with at least one repeated activity
    mean_days_with: float
    mean_days_without: float
    extra_executions: int
    top_activity: str | None

    @property
    def extra_days(self) -> float:
        if np.isnan(self.mean_days_with) or np.isnan(self.mean_days_without):
            return float("nan")
        return self.mean_days_with - self.mean_days_without


def rework_stats(ev: pd.DataFrame, cases: pd.DataFrame) -> tuple[pd.DataFrame, ReworkSummary]:
    """Per activity: cases where it ran more than once, extra executions and the hours they took."""
    per_case = ev.groupby(["activity", "case_id"]).size()
    by_activity = per_case.groupby(level=0)
    out = pd.DataFrame(
        {
            "cases": by_activity.size(),
            "repeat_cases": (per_case > 1).groupby(level=0).sum(),
            "extra_executions": (per_case - 1).groupby(level=0).sum(),
        }
    )
    repeats = ev[ev["occurrence"] > 1]
    out["extra_proc_h"] = repeats.groupby("activity")["proc_h"].sum().reindex(out.index, fill_value=0.0)
    out["extra_wait_h"] = repeats.groupby("activity")["wait_h"].sum().reindex(out.index, fill_value=0.0)
    out["rework_rate"] = out["repeat_cases"] / out["cases"]
    out = out.sort_values(["repeat_cases", "cases"], ascending=False, kind="stable").reset_index()
    out = out.rename(columns={"index": "activity"})

    with_rw = cases.loc[cases["has_rework"], "throughput_d"]
    without = cases.loc[~cases["has_rework"], "throughput_d"]
    top = out.loc[out["repeat_cases"] > 0, "activity"]
    summary = ReworkSummary(
        case_share=float(cases["has_rework"].mean()) if len(cases) else 0.0,
        mean_days_with=float(with_rw.mean()) if len(with_rw) else float("nan"),
        mean_days_without=float(without.mean()) if len(without) else float("nan"),
        extra_executions=int(out["extra_executions"].sum()),
        top_activity=str(top.iloc[0]) if len(top) else None,
    )
    return out, summary


# ---------------------------------------------------------------------------------- conformance
def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance between two activity sequences (insertions, deletions, substitutions)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def trace_fitness(trace: Sequence[str], reference: Sequence[str]) -> float:
    """1 = identical to the reference path, 0 = nothing in common (normalised edit distance)."""
    longest = max(len(trace), len(reference))
    return 1.0 if longest == 0 else 1.0 - levenshtein(trace, reference) / longest


def trace_deviations(trace: Sequence[str], reference: Sequence[str]) -> list[tuple[str, str]]:
    """Explain how a trace departs from the reference: skipped, extra, repeated, out-of-order steps."""
    ref_set = set(reference)
    first: dict[str, int] = {}
    for i, activity in enumerate(trace):
        first.setdefault(activity, i)
    devs = [("Skipped", a) for a in reference if a not in first]
    devs += [("Extra step", a) for a in first if a not in ref_set]
    devs += [("Repeated", a) for a in dict.fromkeys(reference) if list(trace).count(a) > 1]
    # one entry per step that ran too early, naming the earliest reference step it jumped ahead of
    present = [a for a in dict.fromkeys(reference) if a in first]
    for j, b in enumerate(present):
        jumped = [a for a in present[:j] if first[b] < first[a]]
        if jumped:
            devs.append(("Out of order", f"{b} before {jumped[0]}"))
    return devs


@dataclass(frozen=True)
class Conformance:
    reference: tuple[str, ...]
    fitting_share: float  # cases whose trace equals the reference exactly
    mean_fitness: float  # case-weighted mean trace fitness
    deviations: pd.DataFrame  # kind, detail, cases, share


def conformance(cases: pd.DataFrame, reference: Sequence[str]) -> Conformance:
    reference = tuple(reference)
    columns = ["kind", "detail", "cases", "share"]
    n = len(cases)
    if not reference or n == 0:
        return Conformance(reference, float("nan"), float("nan"), pd.DataFrame(columns=columns))
    per_path = cases.groupby("path", sort=False).agg(n=("events", "size"), trace=("trace", "first"))
    rows, fitting, fitness_sum = [], 0, 0.0
    for count, trace in zip(per_path["n"], per_path["trace"], strict=True):
        fitness_sum += count * trace_fitness(trace, reference)
        if tuple(trace) == reference:
            fitting += count
        rows += [(kind, detail, count) for kind, detail in trace_deviations(trace, reference)]
    if rows:
        devs = pd.DataFrame(rows, columns=["kind", "detail", "cases"]).groupby(["kind", "detail"], as_index=False).sum()
        devs["share"] = devs["cases"] / n
        devs = devs.sort_values(["cases", "kind", "detail"], ascending=[False, True, True], kind="stable")
        devs = devs.reset_index(drop=True)
    else:
        devs = pd.DataFrame(columns=columns)
    return Conformance(reference, fitting / n, fitness_sum / n, devs)


def default_reference(variant_table: pd.DataFrame) -> list[str]:
    """Most frequent variant, without repeats: a sensible starting point for an unknown process."""
    if variant_table.empty:
        return []
    return list(dict.fromkeys(variant_table["path"].iloc[0].split(" → ")))


# ------------------------------------------------------------------------------------ resources
def overload_table(ev: pd.DataFrame, min_events: int = 20) -> pd.DataFrame:
    """For activities done by several people: each person's share of the work and the wait in front of them.

    A person is flagged as overloaded when they do at least 1.5× their fair share of an activity
    and cases wait at least 1.25× longer for them than for their colleagues.
    """
    rows = []
    known = ev[ev["resource"] != UNKNOWN_RESOURCE]
    for activity, grp in known.groupby("activity"):
        counts = grp["resource"].value_counts()
        if len(counts) < 2:
            continue
        fair = 1 / len(counts)
        for resource, count in counts.items():
            mine = grp["resource"] == resource
            own, others = grp.loc[mine, "wait_h"].mean(), grp.loc[~mine, "wait_h"].mean()
            ratio = own / others if others and others > 0 else float("nan")
            share = count / counts.sum()
            rows.append(
                {
                    "activity": activity,
                    "resource": resource,
                    "events": int(count),
                    "share": share,
                    "fair_share": fair,
                    "mean_wait_h": own,
                    "others_wait_h": others,
                    "wait_ratio": ratio,
                    "overloaded": bool(count >= min_events and share >= 1.5 * fair and ratio >= 1.25),
                }
            )
    cols = ["activity", "resource", "events", "share", "fair_share", "mean_wait_h", "others_wait_h", "wait_ratio",
            "overloaded"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["overloaded", "share"], ascending=False, kind="stable").reset_index(drop=True)


def resource_stats(ev: pd.DataFrame, has_start: bool, overload: pd.DataFrame) -> pd.DataFrame:
    """Workload per resource: volume, active days, busy hours and the main activity they perform."""
    known = ev[ev["resource"] != UNKNOWN_RESOURCE].copy()
    if known.empty:
        return pd.DataFrame()
    known["day"] = known["end"].dt.normalize()
    g = known.groupby("resource")
    out = pd.DataFrame(
        {
            "events": g.size(),
            "cases": g["case_id"].nunique(),
            "active_days": g["day"].nunique(),
            "proc_h": g["proc_h"].sum(),
            "mean_wait_h": g["wait_h"].mean(),
        }
    )
    pairs = known.groupby(["resource", "activity"]).size()
    main = pairs.groupby(level=0).idxmax().map(lambda key: key[1])
    out["main_activity"] = main
    activity_totals = known.groupby("activity").size()
    out["share_of_main"] = [pairs[(r, a)] / activity_totals[a] for r, a in zip(out.index, out["main_activity"],
                                                                                strict=True)]
    out["share"] = out["events"] / out["events"].sum()
    out["events_per_day"] = out["events"] / out["active_days"]
    out["busy_h_per_day"] = out["proc_h"] / out["active_days"] if has_start else float("nan")
    flagged = set(overload.loc[overload["overloaded"], "resource"]) if not overload.empty else set()
    out["overloaded"] = out.index.isin(flagged)
    return out.sort_values("events", ascending=False).reset_index(names="resource")


@dataclass(frozen=True)
class HandoffStats:
    per_case: float
    share: float  # share of transitions where a different person picks up the case
    wait_after_handoff_h: float
    wait_same_person_h: float
    count: int


def handoff_stats(ev: pd.DataFrame, n_cases: int) -> HandoffStats | None:
    steps = ev[ev["prev_resource"].notna()]
    steps = steps[(steps["resource"] != UNKNOWN_RESOURCE) & (steps["prev_resource"] != UNKNOWN_RESOURCE)]
    if steps.empty:
        return None
    hand = steps["resource"] != steps["prev_resource"]
    return HandoffStats(
        per_case=float(hand.sum() / max(n_cases, 1)),
        share=float(hand.mean()),
        wait_after_handoff_h=float(steps.loc[hand, "wait_h"].mean()) if hand.any() else float("nan"),
        wait_same_person_h=float(steps.loc[~hand, "wait_h"].mean()) if (~hand).any() else float("nan"),
        count=int(hand.sum()),
    )


# ------------------------------------------------------------------------------ calendar effects
def weekend_hours(start: pd.Series, end: pd.Series) -> np.ndarray:
    """Hours of each [start, end] interval that fall on a Saturday or Sunday."""
    s = pd.to_datetime(start).to_numpy("datetime64[ns]")
    e = pd.to_datetime(end).to_numpy("datetime64[ns]")
    valid = ~(np.isnat(s) | np.isnat(e)) & (e > s)
    out = np.zeros(len(s))
    if not valid.any():
        return out
    s, e = s[valid], e[valid]
    d0, d1 = s.astype("datetime64[D]"), e.astype("datetime64[D]")
    hour, one_day = np.timedelta64(1, "h"), np.timedelta64(1, "D")
    we0 = ~np.is_busday(d0, weekmask="1111100")
    we1 = ~np.is_busday(d1, weekmask="1111100")
    same_day = d0 == d1
    full = np.where(d1 > d0, np.busday_count(d0 + one_day, np.maximum(d1, d0 + one_day), weekmask="0000011"), 0)
    first = np.where(we0, (np.minimum(e, (d0 + one_day).astype("datetime64[ns]")) - s) / hour, 0.0)
    last = np.where(we1 & ~same_day, (e - d1.astype("datetime64[ns]")) / hour, 0.0)
    out[valid] = first + last + 24.0 * full
    return out


def weekday_profile(cases: pd.DataFrame) -> pd.DataFrame:
    """Cases started per weekday and their throughput time."""
    wd = cases["start"].dt.day_of_week
    grp = cases.groupby(wd)
    out = pd.DataFrame(
        {"cases": grp.size(), "mean_days": grp["throughput_d"].mean(), "median_days": grp["throughput_d"].median()}
    ).reindex(range(7))
    out["cases"] = out["cases"].fillna(0).astype(int)
    out.insert(0, "weekday", WEEKDAYS)
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------- flow and WIP
@dataclass(frozen=True)
class ThroughputStats:
    mean_days: float
    median_days: float
    p90_days: float
    max_days: float
    flow_efficiency: float | None  # processing time ÷ throughput time; None without start timestamps


def throughput_stats(cases: pd.DataFrame, has_start: bool) -> ThroughputStats:
    d = cases["throughput_d"]
    total = cases["throughput_h"].sum()
    flow = float(cases["proc_h"].sum() / total) if has_start and total > 0 else None
    return ThroughputStats(float(d.mean()), float(d.median()), float(d.quantile(0.9)), float(d.max()), flow)


@dataclass(frozen=True)
class LittleCheck:
    """Little's Law: average work in progress L = arrival rate λ × mean throughput time W."""

    window_start: pd.Timestamp
    window_end: pd.Timestamp
    window_days: float
    avg_wip: float  # measured L (time-averaged open cases)
    arrival_rate: float  # λ, cases started per day
    completion_rate: float  # cases finished per day
    mean_days: float  # W, for cases started in the window
    predicted_wip: float  # λ × W

    @property
    def gap(self) -> float:
        return (self.avg_wip - self.predicted_wip) / self.predicted_wip if self.predicted_wip > 0 else float("nan")

    @property
    def consistent(self) -> bool:
        return abs(self.gap) <= 0.10


def littles_law(cases: pd.DataFrame, trim: float = 0.1) -> LittleCheck | None:
    """Check L = λW on the middle of the log (the first and last ``trim`` share are warm-up/cool-down)."""
    if len(cases) < 2:
        return None
    t0, t1 = cases["start"].min(), cases["end"].max()
    span = t1 - t0
    if span <= pd.Timedelta(0):
        return None
    w0, w1 = t0 + trim * span, t1 - trim * span
    window_days = (w1 - w0) / pd.Timedelta(days=1)
    overlap = (cases["end"].clip(upper=w1) - cases["start"].clip(lower=w0)) / pd.Timedelta(days=1)
    avg_wip = float(overlap.clip(lower=0).sum() / window_days)
    started = cases[(cases["start"] >= w0) & (cases["start"] < w1)]
    finished = int(((cases["end"] >= w0) & (cases["end"] < w1)).sum())
    lam = len(started) / window_days
    w = float(started["throughput_d"].mean()) if len(started) else float("nan")
    return LittleCheck(w0, w1, window_days, avg_wip, lam, finished / window_days, w, lam * w)


def wip_series(cases: pd.DataFrame) -> pd.DataFrame:
    """Open cases at the end of each calendar day, plus daily arrivals and completions."""
    first, last = cases["start"].min().normalize(), cases["end"].max().normalize()
    days = pd.date_range(first, last, freq="D")
    cutoff = (days + pd.Timedelta(days=1)).to_numpy("datetime64[ns]")
    starts = np.sort(cases["start"].to_numpy("datetime64[ns]"))
    ends = np.sort(cases["end"].to_numpy("datetime64[ns]"))
    wip = np.searchsorted(starts, cutoff, side="left") - np.searchsorted(ends, cutoff, side="left")
    arrivals = cases.groupby(cases["start"].dt.normalize()).size().reindex(days, fill_value=0)
    completions = cases.groupby(cases["end"].dt.normalize()).size().reindex(days, fill_value=0)
    return pd.DataFrame({"date": days, "wip": wip, "arrivals": arrivals.values, "completions": completions.values})


# ------------------------------------------------------------------------------------- bundle
@dataclass(frozen=True)
class Mining:
    events: pd.DataFrame
    cases: pd.DataFrame
    variants: pd.DataFrame
    dfg: pd.DataFrame
    activities: pd.DataFrame
    rework: pd.DataFrame
    rework_summary: ReworkSummary
    conformance: Conformance
    overload: pd.DataFrame
    resources: pd.DataFrame
    handoffs: HandoffStats | None
    weekdays: pd.DataFrame
    weekend_wait_share: float
    throughput: ThroughputStats
    little: LittleCheck | None
    wip: pd.DataFrame
    months: float  # length of the log in months, used to express impacts per month

    @property
    def bottleneck(self) -> str | None:
        return bottleneck(self.activities)

    @property
    def n_cases(self) -> int:
        return len(self.cases)


def mine(log: EventLog, reference: Sequence[str]) -> Mining:
    ev = enrich(log.events)
    cases = case_table(ev)
    variant_table = variants(cases)
    rework, rework_summary = rework_stats(ev, cases)
    overload = overload_table(ev) if log.has_resource else pd.DataFrame()
    resources = resource_stats(ev, log.has_start, overload) if log.has_resource else pd.DataFrame()
    wait_total = ev["wait_h"].sum()
    weekend = weekend_hours(ev["prev_end"], ev["start"]).sum()
    t0, t1 = log.period
    months = max((t1 - t0) / pd.Timedelta(days=1) / 30.44, 1 / 30.44)
    return Mining(
        events=ev,
        cases=cases,
        variants=variant_table,
        dfg=dfg(ev, len(cases)),
        activities=activity_stats(ev),
        rework=rework,
        rework_summary=rework_summary,
        conformance=conformance(cases, reference),
        overload=overload,
        resources=resources,
        handoffs=handoff_stats(ev, len(cases)) if log.has_resource else None,
        weekdays=weekday_profile(cases),
        weekend_wait_share=float(weekend / wait_total) if wait_total > 0 else 0.0,
        throughput=throughput_stats(cases, log.has_start),
        little=littles_law(cases),
        wip=wip_series(cases),
        months=float(months),
    )
