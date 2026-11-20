"""Sample datasets and saved AI analyses, so the app works without an API key."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import eventlog, kpi
from .mining import case_table, default_reference, enrich, variants

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
ANALYSES_DIR = DEMO_DIR / "analyses"


@dataclass(frozen=True)
class Dataset:
    key: str
    name: str
    company: str
    process: str
    description: str
    synthetic: bool
    log: eventlog.EventLog
    kpis: pd.DataFrame | None  # daily KPI table; None = derive from the log
    kpi_overrides: dict
    reference: tuple[str, ...]
    planted: tuple[str, ...] = ()
    industry: str = ""


def load_samples() -> list[dict]:
    return json.loads((DEMO_DIR / "samples.json").read_text(encoding="utf-8"))


def sample_dataset(sample: dict) -> Dataset:
    log = eventlog.read_csv(DEMO_DIR / sample["events_file"])
    kpis = kpi.from_frame(pd.read_csv(DEMO_DIR / sample["kpis_file"])) if sample.get("kpis_file") else None
    return Dataset(
        key=sample["id"],
        name=sample["name"],
        company=sample["company"],
        process=sample["process"],
        description=sample["description"],
        synthetic=True,
        log=log,
        kpis=kpis,
        kpi_overrides=sample.get("kpis", {}),
        reference=tuple(sample["reference_path"]),
        planted=tuple(sample.get("planted", [])),
        industry=sample.get("industry", ""),
    )


def upload_dataset(file_name: str, events: bytes, kpis: bytes | None = None) -> Dataset:
    """Build a dataset from uploaded CSV bytes. Raises EventLogError / ValueError with a readable message."""
    log = eventlog.read_csv(events)
    kpi_table = kpi.from_frame(pd.read_csv(io.BytesIO(kpis))) if kpis else None
    reference = default_reference(variants(case_table(enrich(log.events))))
    stem = Path(file_name).stem or "Uploaded log"
    return Dataset(
        key="upload",
        name=stem,
        company=stem,
        process="Uploaded event log",
        description=f"{log.n_cases:,} cases uploaded from {file_name}.",
        synthetic=False,
        log=log,
        kpis=kpi_table,
        kpi_overrides={},
        reference=tuple(reference),
    )


def load_saved_analysis(fingerprint: str) -> dict | None:
    path = ANALYSES_DIR / f"{fingerprint}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_saved_for_data(data_fingerprint: str) -> dict | None:
    """A saved analysis of the same data made with different settings (or None)."""
    for path in sorted(ANALYSES_DIR.glob("*.json")):
        analysis = json.loads(path.read_text(encoding="utf-8"))
        if analysis.get("data_fingerprint") == data_fingerprint:
            return analysis
    return None


def save_analysis(analysis: dict) -> Path:
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    path = ANALYSES_DIR / f"{analysis['fingerprint']}.json"
    path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
