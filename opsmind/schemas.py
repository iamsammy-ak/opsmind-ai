"""Pydantic models for the AI diagnosis: root-cause hypotheses and an improvement plan.

Free models do not always follow a schema exactly, so validators coerce common variations
("High effort", "Waiting / queues", objects where strings were expected) instead of
rejecting the reply.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .lean import WASTES

LEVELS = ("low", "medium", "high")

_WASTE_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Defects", ("defect", "rework", "error", "correction", "quality")),
    ("Overproduction", ("overproduc", "abandon", "cancel")),
    ("Non-utilised talent", ("talent", "skill", "workload", "overburden", "overload", "people", "unused")),
    ("Transportation", ("transport", "handoff", "hand-off", "hand off")),
    ("Inventory", ("inventory", "wip", "work in progress", "work-in-progress", "backlog")),
    ("Motion", ("motion", "movement", "searching")),
    ("Extra-processing", ("extra", "over-process", "overprocess", "redundant", "unnecessary step")),
    ("Waiting", ("wait", "queue", "delay", "idle", "bottleneck")),
)


def _as_text(item: Any) -> str:
    if isinstance(item, dict):
        return " — ".join(str(v) for v in item.values() if v not in (None, ""))
    return str(item).strip()


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [text for text in (_as_text(v) for v in value) if text]


def _level(value: Any, default: str = "medium") -> str:
    text = str(value or default).lower()
    for level in ("high", "medium", "low"):
        if level in text:
            return level
    if "moderate" in text or "mid" in text:
        return "medium"
    return default


def coerce_waste(value: Any) -> str:
    """Map free text to one of the eight Lean wastes, or '' when nothing matches."""
    text = str(value or "").strip()
    for waste in WASTES:
        if text.lower() == waste.lower():
            return waste
    low = text.lower()
    for waste, words in _WASTE_WORDS:
        if any(word in low for word in words):
            return waste
    return ""


class RootCause(BaseModel):
    problem: str = Field(description="The symptom, quoting a computed figure")
    hypothesis: str = Field(description="A plausible cause of the symptom")
    evidence: list[str] = Field(default_factory=list, description="Computed figures that support it, quoted exactly")
    test: str = Field("", description="A concrete check that would confirm or reject the hypothesis")
    confidence: str = Field("medium", description="low, medium or high")

    @field_validator("evidence", mode="before")
    @classmethod
    def _evidence(cls, value: Any) -> list[str]:
        return _as_text_list(value)

    @field_validator("problem", "hypothesis", "test", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        return _as_text(value) if value is not None else ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, value: Any) -> str:
        return _level(value)


class PlanItem(BaseModel):
    action: str
    waste: str = Field("", description="One of the eight Lean wastes")
    owner_role: str = Field("", description="A role, not a person's name")
    effort: str = Field("medium", description="low, medium or high")
    expected_impact: str = Field("", description="Quote the computed estimate for this waste where there is one")
    timeframe: str = ""

    @field_validator("waste", mode="before")
    @classmethod
    def _waste(cls, value: Any) -> str:
        return coerce_waste(value)

    @field_validator("effort", mode="before")
    @classmethod
    def _effort(cls, value: Any) -> str:
        return _level(value)

    @field_validator("action", "owner_role", "expected_impact", "timeframe", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        return _as_text(value) if value is not None else ""


class Diagnosis(BaseModel):
    headline: str = Field("", description="The main finding in one sentence")
    root_causes: list[RootCause] = []
    improvement_plan: list[PlanItem] = []
    data_gaps: list[str] = []

    @field_validator("data_gaps", mode="before")
    @classmethod
    def _gaps(cls, value: Any) -> list[str]:
        return _as_text_list(value)
