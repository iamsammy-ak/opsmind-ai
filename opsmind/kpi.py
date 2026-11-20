"""KPI analytics: individuals (XmR) control charts, Western Electric rules, rolling z-score
anomalies, trend regression with a significance test, a single change-point search and
target-versus-actual variance.

KPI series are daily (business days). Percentages are stored as fractions (0.93 = 93%).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import stats

# Shewhart constants for individuals charts with a moving range of two points
D2 = 1.128  # sigma ≈ mean moving range / d2
E2 = 3 / D2  # 2.66: natural process limits = centre ± E2 × mean moving range
D4 = 3.267  # upper limit of the moving-range chart


@dataclass(frozen=True)
class KpiMeta:
    label: str
    unit: str
    higher_is_better: bool | None  # None = no preferred direction
    percent: bool = False
    target: float | None = None

    def fmt(self, value: float | None, digits: int | None = None) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "—"
        if self.percent:
            return f"{value * 100:.{1 if digits is None else digits}f}%"
        if self.unit == "€":
            return f"€{value:,.{2 if digits is None else digits}f}"
        text = f"{value:,.{1 if digits is None else digits}f}"
        return f"{text} {self.unit}".strip() if self.unit and self.unit != "cases/day" else text


KPI_META: dict[str, KpiMeta] = {
    "throughput": KpiMeta("Throughput", "cases/day", True),
    "cycle_time_days": KpiMeta("Cycle time", "days", False),
    "first_pass_yield": KpiMeta("First-pass yield", "%", True, percent=True),
    "otif": KpiMeta("On-time in-full", "%", True, percent=True),
    "cost_per_order": KpiMeta("Cost per order", "€", False),
}

RULES = {
    "rule1": "1 point beyond 3σ",
    "rule2": "2 of 3 beyond 2σ",
    "rule3": "4 of 5 beyond 1σ",
    "rule4": "8 in a row on one side",
}


def meta_for(key: str, overrides: dict | None = None) -> KpiMeta:
    base = KPI_META.get(key, KpiMeta(key.replace("_", " ").capitalize(), "", None))
    if not overrides:
        return base
    return KpiMeta(
        overrides.get("label", base.label),
        overrides.get("unit", base.unit),
        overrides.get("higher_is_better", base.higher_is_better),
        overrides.get("percent", base.percent),
        overrides.get("target", base.target),
    )


# --------------------------------------------------------------------------------- loading
def from_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate a KPI table: a ``date`` column plus one numeric column per KPI."""
    cols = {str(c).strip().lower(): c for c in raw.columns}
    date_col = cols.get("date") or cols.get("day")
    if date_col is None:
        raise ValueError("The KPI file needs a 'date' column.")
    df = raw.rename(columns={date_col: "date"}).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").astype("datetime64[ns]")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    numeric = {}
    for col in df.columns:
        if col == "date":
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() >= 3:
            numeric[str(col).strip().lower()] = values.astype(float)
    if not numeric:
        raise ValueError("The KPI file has no numeric columns with at least three values.")
    return pd.DataFrame({"date": df["date"].values, **{k: v.values for k, v in numeric.items()}})


def derive_daily_kpis(cases: pd.DataFrame, end_activity: str | None = None) -> pd.DataFrame:
    """Daily throughput, cycle time and first-pass yield from the event log (business days).

    A case counts as completed on the business day of its last event (weekend completions roll
    forward to Monday). With ``end_activity``, only cases that reached it count as completed.
    """
    done = cases if end_activity is None else cases[cases["trace"].map(lambda t: end_activity in t)]
    if done.empty:
        return pd.DataFrame(columns=["date", "throughput", "cycle_time_days", "first_pass_yield"])
    day = np.busday_offset(done["end"].to_numpy("datetime64[D]"), 0, roll="forward")
    day = pd.to_datetime(day).astype("datetime64[ns]")
    grp = done.groupby(day)
    days = pd.bdate_range(day.min(), day.max())
    return pd.DataFrame(
        {
            "date": days,
            "throughput": grp.size().reindex(days, fill_value=0).astype(float).values,
            "cycle_time_days": grp["throughput_d"].mean().reindex(days).values,
            "first_pass_yield": (1 - grp["has_rework"].mean()).reindex(days).values,
        }
    )


# --------------------------------------------------------------------------- control charts
@dataclass(frozen=True)
class ControlLimits:
    center: float
    mr_bar: float
    sigma: float
    ucl: float
    lcl: float
    mr_ucl: float
    baseline_n: int


def xmr_limits(values, baseline: int | None = None) -> ControlLimits:
    """Individuals-chart limits from the average moving range of the first ``baseline`` points."""
    x = pd.Series(values, dtype=float).dropna().to_numpy()
    if baseline:
        x = x[:baseline]
    if len(x) < 2:
        nan = float("nan")
        return ControlLimits(float(x.mean()) if len(x) else nan, nan, nan, nan, nan, nan, len(x))
    center = float(x.mean())
    mr_bar = float(np.abs(np.diff(x)).mean())
    return ControlLimits(center, mr_bar, mr_bar / D2, center + E2 * mr_bar, center - E2 * mr_bar, D4 * mr_bar, len(x))


def moving_range(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return np.concatenate([[np.nan], np.abs(np.diff(x))]) if len(x) else x


def _window_count(flags: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(flags.astype(float)).rolling(window, min_periods=window).sum().fillna(0).to_numpy()


def western_electric(values, center: float, sigma: float) -> pd.DataFrame:
    """Flag each point that completes a Western Electric rule.

    rule1: the point is beyond 3σ. rule2: it is beyond 2σ and so are 2 of the last 3 points on the
    same side. rule3: it is beyond 1σ and so are 4 of the last 5 on the same side. rule4: it and
    the previous 7 points all sit on the same side of the centre line.
    """
    x = np.asarray(values, dtype=float)
    out = pd.DataFrame(False, index=range(len(x)), columns=list(RULES))
    if not len(x) or not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(center):
        return out
    z = np.nan_to_num((x - center) / sigma, nan=0.0)
    out["rule1"] = np.abs(z) > 3
    for sign in (1, -1):
        side = sign * z
        beyond2, beyond1 = side > 2, side > 1
        out["rule2"] |= beyond2 & (_window_count(beyond2, 3) >= 2)
        out["rule3"] |= beyond1 & (_window_count(beyond1, 5) >= 4)
        out["rule4"] |= _window_count(side > 0, 8) >= 8
    return out


def rolling_zscore(values, window: int = 20) -> np.ndarray:
    """z of each point against the mean and standard deviation of the ``window`` points before it."""
    s = pd.Series(values, dtype=float)
    past = s.shift(1).rolling(window, min_periods=window)
    mean, std = past.mean(), past.std(ddof=1)
    z = (s - mean) / std.where(std > 0)
    return z.to_numpy()


# ------------------------------------------------------------------------- trend and shifts
@dataclass(frozen=True)
class Trend:
    slope: float  # change per observation (business day)
    intercept: float
    r2: float
    p_value: float
    n: int

    @property
    def change(self) -> float:
        """Fitted change from the first to the last observation."""
        return self.slope * max(self.n - 1, 0)

    @property
    def significant(self) -> bool:
        return bool(self.p_value < 0.05)


def trend(values) -> Trend:
    """Ordinary least squares on the observation index, with a two-sided t-test on the slope."""
    y = pd.Series(values, dtype=float).dropna()
    n = len(y)
    if n < 3 or np.ptp(y.to_numpy()) == 0:
        return Trend(0.0, float(y.mean()) if n else float("nan"), 0.0, 1.0, n)
    x = np.arange(n, dtype=float)
    fit = stats.linregress(x, y.to_numpy())
    p = float(fit.pvalue) if np.isfinite(fit.pvalue) else 1.0
    return Trend(float(fit.slope), float(fit.intercept), float(fit.rvalue**2), p, n)


@dataclass(frozen=True)
class ChangePoint:
    index: int  # first observation of the new level
    mean_before: float
    mean_after: float
    p_value: float  # Welch t-test; optimistic because the split was chosen to maximise the gap

    @property
    def delta(self) -> float:
        return self.mean_after - self.mean_before

    @property
    def clear(self) -> bool:
        return bool(self.p_value < 0.001)


def change_point(values, min_size: int = 10) -> ChangePoint | None:
    """The single split that minimises the within-segment squared error (a step change in level)."""
    y = pd.Series(values, dtype=float).dropna().to_numpy()
    n = len(y)
    if n < 2 * min_size:
        return None
    csum, csq = np.cumsum(y), np.cumsum(y**2)
    best_k, best_sse = None, np.inf
    for k in range(min_size, n - min_size + 1):
        left_n, right_n = k, n - k
        left_sse = csq[k - 1] - csum[k - 1] ** 2 / left_n
        right_sum, right_sq = csum[-1] - csum[k - 1], csq[-1] - csq[k - 1]
        sse = left_sse + right_sq - right_sum**2 / right_n
        if sse < best_sse - 1e-12:
            best_k, best_sse = k, sse
    if best_k is None:
        return None
    before, after = y[:best_k], y[best_k:]
    if np.ptp(y) == 0:
        p = 1.0
    else:
        p = float(stats.ttest_ind(before, after, equal_var=False).pvalue)
        p = p if np.isfinite(p) else 1.0
    return ChangePoint(best_k, float(before.mean()), float(after.mean()), p)


def target_variance(values, target: float | None, higher_is_better: bool | None, recent: int = 10) -> dict:
    """Recent level against the target, and the share of days that met it."""
    y = pd.Series(values, dtype=float).dropna()
    level = float(y.tail(recent).mean()) if len(y) else float("nan")
    if target is None or higher_is_better is None or not len(y):
        return {"recent": level, "target": target, "gap": float("nan"), "on_target": None, "days_met": float("nan")}
    met = y >= target if higher_is_better else y <= target
    on_target = level >= target if higher_is_better else level <= target
    return {"recent": level, "target": target, "gap": level - target, "on_target": bool(on_target),
            "days_met": float(met.mean())}


# --------------------------------------------------------------------------------- bundle
@dataclass(frozen=True)
class KpiResult:
    key: str
    meta: KpiMeta
    data: pd.DataFrame  # date, value, mr, z, rule1..rule4, signal, anomaly
    limits: ControlLimits
    trend: Trend
    change: ChangePoint | None
    variance: dict = field(default_factory=dict)

    @property
    def rule_points(self) -> int:
        return int(self.data["signal"].sum())

    @property
    def anomalies(self) -> int:
        return int(self.data["anomaly"].sum())

    @property
    def trend_direction(self) -> str:
        """'improving', 'worsening' or 'stable' (stable = not significant at 5%)."""
        if not self.trend.significant:
            return "stable"
        rising = self.trend.slope > 0
        if self.meta.higher_is_better is None:
            return "rising" if rising else "falling"
        return "improving" if rising == self.meta.higher_is_better else "worsening"


def analyse_kpi(
    key: str,
    dates,
    values,
    meta: KpiMeta,
    baseline: int = 40,
    z_window: int = 20,
    z_threshold: float = 3.0,
) -> KpiResult:
    df = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates)).values, "value": pd.Series(values, dtype=float).values})
    df = df.dropna().reset_index(drop=True)
    baseline = max(2, min(baseline, len(df)))
    limits = xmr_limits(df["value"], baseline)
    # natural bounds: a count or cost cannot go below zero, a percentage cannot exceed 100%
    if len(df) and df["value"].min() >= 0 and limits.lcl < 0:
        limits = replace(limits, lcl=0.0)
    if meta.percent and limits.ucl > 1:
        limits = replace(limits, ucl=1.0)
    df["mr"] = moving_range(df["value"])
    df["z"] = rolling_zscore(df["value"], z_window)
    rules = western_electric(df["value"], limits.center, limits.sigma)
    df = pd.concat([df, rules], axis=1)
    df["signal"] = rules.any(axis=1)
    df["anomaly"] = np.abs(df["z"].fillna(0)) >= z_threshold
    return KpiResult(
        key=key,
        meta=meta,
        data=df,
        limits=limits,
        trend=trend(df["value"]),
        change=change_point(df["value"]),
        variance=target_variance(df["value"], meta.target, meta.higher_is_better),
    )


def analyse_all(
    kpis: pd.DataFrame,
    overrides: dict | None = None,
    baseline: int = 40,
    z_window: int = 20,
    z_threshold: float = 3.0,
) -> dict[str, KpiResult]:
    overrides = overrides or {}
    results = {}
    for key in [c for c in kpis.columns if c != "date"]:
        if kpis[key].notna().sum() < 3:
            continue
        results[key] = analyse_kpi(
            key, kpis["date"], kpis[key], meta_for(key, overrides.get(key)), baseline, z_window, z_threshold
        )
    return results
