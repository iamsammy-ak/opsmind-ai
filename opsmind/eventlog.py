"""Load and validate event logs.

An event log has one row per executed activity. Required columns:

- ``case_id``: the process instance (order, invoice, ticket…)
- ``activity``: the step that was executed
- ``timestamp``: when the step was completed

Optional columns:

- ``resource``: who (or which system) executed the step
- ``start_timestamp``: when work on the step started. With it, waiting time and processing time
  can be separated; without it, the time between two completions counts as waiting.

Common alternative headers (XES names such as ``case:concept:name`` and ``time:timestamp``, ``case``,
``user``, ``start``…) are recognised automatically.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED = ("case_id", "activity", "timestamp")
OPTIONAL = ("resource", "start_timestamp")

SYNONYMS: dict[str, tuple[str, ...]] = {
    "case_id": ("case_id", "case", "caseid", "case:concept:name", "case_concept_name", "trace_id", "process_id"),
    "activity": ("activity", "activity_name", "concept:name", "concept_name", "event", "event_name", "task", "step"),
    "timestamp": (
        "timestamp",
        "time:timestamp",
        "time_timestamp",
        "complete_timestamp",
        "end_timestamp",
        "end_time",
        "end",
        "completed_at",
        "time",
    ),
    "resource": ("resource", "org:resource", "org_resource", "user", "employee", "performer", "agent"),
    "start_timestamp": ("start_timestamp", "start_time", "start", "started_at", "start:timestamp"),
}

UNKNOWN_RESOURCE = "(unknown)"


class EventLogError(ValueError):
    """The file cannot be used as an event log; the message says why."""


@dataclass(frozen=True)
class EventLog:
    """A cleaned event log. ``events`` has the columns case_id, activity, resource, start, end."""

    events: pd.DataFrame
    has_start: bool
    has_resource: bool
    dropped_rows: int = 0
    fixed_starts: int = 0

    @property
    def n_cases(self) -> int:
        return int(self.events["case_id"].nunique())

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def activities(self) -> list[str]:
        """Activities ordered by their typical position in a case."""
        ev = self.events
        position = ev.groupby("case_id", sort=False).cumcount()
        order = position.groupby(ev["activity"]).median().sort_values(kind="stable")
        return list(order.index)

    @property
    def period(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.events["start"].min(), self.events["end"].max()


def _normalise_header(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def map_columns(columns: list[str]) -> dict[str, str]:
    """Map raw headers to canonical names. Unrecognised columns are left out."""
    normalised = {_normalise_header(c): c for c in columns}
    mapping: dict[str, str] = {}
    for canonical, names in SYNONYMS.items():
        for name in names:
            if name in normalised and normalised[name] not in mapping:
                mapping[normalised[name]] = canonical
                break
    return mapping


def _parse_times(values: pd.Series) -> pd.Series:
    """Parse timestamps; timezone-aware values are converted to UTC and made naive."""
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.isna().mean() > 0.05 and values.notna().any():
        parsed = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
    return parsed.dt.tz_localize(None).astype("datetime64[ns]")


def from_frame(raw: pd.DataFrame) -> EventLog:
    """Validate a raw table and return a sorted, typed event log."""
    mapping = map_columns(list(raw.columns))
    missing = [c for c in REQUIRED if c not in mapping.values()]
    if missing:
        raise EventLogError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Found: {', '.join(map(str, raw.columns)) or 'none'}. "
            "An event log needs case_id, activity and timestamp."
        )
    df = raw.rename(columns=mapping)[list(dict.fromkeys(mapping.values()))].copy()
    has_start = "start_timestamp" in df.columns
    has_resource = "resource" in df.columns

    out = pd.DataFrame(
        {
            "case_id": df["case_id"].astype("string").str.strip(),
            "activity": df["activity"].astype("string").str.strip(),
            "end": _parse_times(df["timestamp"]),
        }
    )
    out["start"] = _parse_times(df["start_timestamp"]) if has_start else out["end"]
    if has_resource:
        out["resource"] = df["resource"].astype("string").str.strip().fillna(UNKNOWN_RESOURCE)
        out.loc[out["resource"] == "", "resource"] = UNKNOWN_RESOURCE
    else:
        out["resource"] = UNKNOWN_RESOURCE

    valid = (out["case_id"].fillna("") != "") & (out["activity"].fillna("") != "") & out["end"].notna()
    valid = valid.astype(bool)
    dropped = int((~valid).sum())
    out = out[valid].copy()
    if out.empty:
        raise EventLogError("No usable rows: every row is missing a case id, an activity or a valid timestamp.")

    # a missing or later-than-completion start time falls back to the completion time
    bad_start = out["start"].isna() | (out["start"] > out["end"])
    fixed = int(bad_start.sum()) if has_start else 0
    out.loc[bad_start, "start"] = out.loc[bad_start, "end"]

    out["_row"] = range(len(out))
    out = out.sort_values(["case_id", "end", "start", "_row"], kind="stable").drop(columns="_row")
    out = out[["case_id", "activity", "resource", "start", "end"]].astype(
        {"case_id": str, "activity": str, "resource": str}
    )
    return EventLog(out.reset_index(drop=True), has_start, has_resource, dropped, fixed)


def read_csv(source) -> EventLog:
    """Read an event log from a path, a file-like object or raw bytes."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        if hasattr(source, "read"):
            data = source.read()
            text = data.decode("utf-8-sig") if isinstance(data, bytes) else str(data)
        else:
            text = Path(source).read_text(encoding="utf-8-sig")
        header = text.split("\n", 1)[0]
        sep = max((",", ";", "\t", "|"), key=header.count)
        raw = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
    except Exception as exc:  # malformed CSV, wrong encoding…
        raise EventLogError(f"Could not read the file as CSV: {exc}") from exc
    return from_frame(raw)


def canonical_csv(log: EventLog) -> str:
    """Stable text form of the log, used to fingerprint a dataset across machines and pandas versions."""
    ev = log.events
    fmt = "%Y-%m-%dT%H:%M:%S"
    return pd.DataFrame(
        {
            "c": ev["case_id"],
            "a": ev["activity"],
            "r": ev["resource"],
            "s": ev["start"].dt.strftime(fmt),
            "e": ev["end"].dt.strftime(fmt),
        }
    ).to_csv(index=False, lineterminator="\n")
