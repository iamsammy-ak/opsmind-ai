"""Human-friendly number formatting shared by the UI, the AI facts and the report."""

from __future__ import annotations

import math


def _missing(value: float | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def duration(hours: float | None) -> str:
    """45 min · 18.4 h · 3.2 days."""
    if _missing(hours):
        return "—"
    if math.isinf(hours):
        return "∞"
    if abs(hours) < 1:
        return f"{hours * 60:.0f} min"
    if abs(hours) < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days"


def days(value: float | None, digits: int = 1) -> str:
    if _missing(value):
        return "—"
    return f"{value:.{digits}f} days"


def pct(value: float | None, digits: int = 1) -> str:
    if _missing(value):
        return "—"
    return f"{value * 100:.{digits}f}%"


def num(value: float | None, digits: int = 0) -> str:
    if _missing(value):
        return "—"
    return f"{value:,.{digits}f}"


def signed(value: float | None, digits: int = 1, suffix: str = "") -> str:
    if _missing(value):
        return "—"
    return f"{value:+,.{digits}f}{suffix}"


def money(value: float | None, currency: str = "€") -> str:
    if _missing(value):
        return "—"
    if math.isinf(value):
        return "∞"
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1e6:
        body = f"{v / 1e6:.1f}M"
    elif v >= 1e4:
        body = f"{v / 1e3:.0f}k"
    elif v >= 1e3:
        body = f"{v / 1e3:.1f}k"
    elif v >= 100:
        body = f"{v:,.0f}"
    else:
        body = f"{v:,.2f}"
    return f"{sign}{currency}{body}"


def p_value(p: float | None) -> str:
    if _missing(p):
        return "—"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
