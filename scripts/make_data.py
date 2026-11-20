"""Generate the synthetic demo datasets: demo/data/*.csv and demo/samples.json.

Two fictional companies, each with one process simulated case by case. The event log is an
extract of the cases created between 5 Jan and 26 Jun 2026 (25 weeks), all run to completion.
The simulation starts five weeks earlier and runs two weeks longer so the daily KPI series has
no warm-up or drain effects at its edges. Everything is seeded, so re-running reproduces the
same files.

    python scripts/make_data.py

Simulation assumptions (shared):
- Work happens Monday to Friday, 08:00-17:00. A step that becomes ready outside those hours
  starts at the next opening time plus a short start-up delay, which is how weekend effects arise.
- Waiting and work times are lognormal around the medians below; people with a heavier load
  get proportionally longer queues.
- Daily KPIs are business-day series. Throughput, cycle time and first-pass yield are derived
  from the generated log with the same code the app uses; on-time in-full and cost per case are
  simulated per case and averaged per completion day.

The realities planted in each process are listed in PLANTED below and shown in the app.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opsmind.eventlog import from_frame  # noqa: E402
from opsmind.kpi import derive_daily_kpis  # noqa: E402
from opsmind.mining import case_table, enrich  # noqa: E402

DATA_DIR = ROOT / "demo" / "data"
SIM_START = datetime(2025, 12, 1)  # warm-up starts (a Monday)
LOG_START = datetime(2026, 1, 5)  # first day of the extract (a Monday)
LOG_END = datetime(2026, 6, 27)  # extract covers cases created before this day
SIM_END = datetime(2026, 7, 11)  # no new cases after this day
OPEN, CLOSE = 8, 17


class Sim:
    """Minimal case-by-case simulator with a working calendar."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.rows: list[tuple] = []
        self.created: dict[str, datetime] = {}

    def ln(self, median: float, sigma: float = 0.6) -> float:
        return float(median * np.exp(sigma * self.rng.standard_normal()))

    def jitter(self, max_minutes: float) -> timedelta:
        return timedelta(minutes=float(self.rng.uniform(0, max_minutes)))

    def work_time(self, t: datetime) -> datetime:
        """Move t into working hours; a moved start gets a short start-up delay."""
        moved = False
        while True:
            midnight = datetime(t.year, t.month, t.day)
            if t.weekday() >= 5:
                t, moved = midnight + timedelta(days=7 - t.weekday(), hours=OPEN), True
            elif t.hour < OPEN:
                t, moved = midnight + timedelta(hours=OPEN), True
            elif t.hour >= CLOSE:
                t, moved = midnight + timedelta(days=1, hours=OPEN), True
            else:
                break
        return t + self.jitter(50) if moved else t

    def step(self, case: str, activity: str, resource: str, ready: datetime, wait_h: float, work_min: float):
        start = self.work_time(ready + timedelta(hours=wait_h))
        end = start + timedelta(minutes=work_min)
        self.rows.append((case, activity, resource, start, end))
        return end

    def pick(self, weights: dict[str, float] | list[str]) -> str:
        if isinstance(weights, list):
            weights = dict.fromkeys(weights, 1.0)
        names = list(weights)
        p = np.array(list(weights.values()), dtype=float)
        return names[int(self.rng.choice(len(names), p=p / p.sum()))]

    def arrivals(self, weekday_rates: list[float], weekend_rate: float) -> list[datetime]:
        out = []
        for offset in range((SIM_END - SIM_START).days):
            day = SIM_START + timedelta(days=offset)
            if day.weekday() < 5:
                hours = self.rng.uniform(OPEN, CLOSE, self.rng.poisson(weekday_rates[day.weekday()]))
            else:  # web orders / e-mailed invoices that arrive at the weekend
                hours = self.rng.uniform(0, 24, self.rng.poisson(weekend_rate))
            out += [day + timedelta(hours=float(h)) for h in sorted(hours)]
        return out

    def cases(self, prefix: str, start_number: int, arrivals: list[datetime]):
        """Yield (case id, arrival); the first case of the extract gets `start_number`."""
        warm_up = sum(a < LOG_START for a in arrivals)
        for i, arrival in enumerate(arrivals):
            case = f"{prefix}-{start_number + i - warm_up}"
            self.created[case] = arrival
            yield case, arrival

    def frame(self, extract_only: bool) -> pd.DataFrame:
        rows = self.rows
        if extract_only:
            rows = [r for r in rows if LOG_START <= self.created[r[0]] < LOG_END]
        df = pd.DataFrame(rows, columns=["case_id", "activity", "resource", "start_timestamp", "timestamp"])
        fmt = "%Y-%m-%d %H:%M:%S"
        df["start_timestamp"] = pd.to_datetime(df["start_timestamp"]).dt.strftime(fmt)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime(fmt)
        return df


def next_slot(t: datetime, weekdays: set[int], hour: float, skip: set[date] = frozenset()) -> datetime:
    """First time at `hour` on an allowed weekday that is not before t."""
    slot = datetime(t.year, t.month, t.day) + timedelta(hours=hour)
    if slot < t:
        slot += timedelta(days=1)
    while slot.weekday() not in weekdays or slot.date() in skip:
        slot += timedelta(days=1)
    return slot


BUSINESS_DAYS = {0, 1, 2, 3, 4}


# --------------------------------------------------------------------------- Valdora Supply
def valdora(seed: int = 7) -> tuple[Sim, dict[str, dict]]:
    sim = Sim(seed)
    clerks = ["Lucia Ferro", "Tomás Vidal", "Grace Okoro"]
    credit = ["Henrik Salo", "Ava Brennan"]
    approvers_before = {"Marta Ilić": 0.56, "Jonas Berg": 0.24, "Priya Raman": 0.20}
    approvers_after = {"Marta Ilić": 0.80, "Priya Raman": 0.20}
    warehouse = ["Dev Patel", "Sofia Marin", "Kenji Ito", "Olu Adebayo"]
    shipping = ["Rosa Klein", "Bram de Vries"]
    billing = ["Nina Petrov", "Elias Moreau"]
    reorg = datetime(2026, 3, 30)  # Jonas Berg moves to key accounts
    price_list = datetime(2026, 5, 4)  # new price list: more invoice errors
    strike = {date(2026, 4, 22)}  # carrier strike: no pickup that day
    shortage = (datetime(2026, 5, 18), datetime(2026, 6, 6))  # supplier shortage: fewer in-full deliveries
    cost_spikes = {date(2026, 3, 12), date(2026, 6, 10)}  # emergency courier days

    info: dict[str, dict] = {}
    for case, arrival in sim.cases("SO", 10001, sim.arrivals([13, 10.5, 10.5, 10, 8.5], 1.0)):
        clerk, biller = sim.pick(clerks), sim.pick(billing)
        first_row = len(sim.rows)
        proforma = sim.rng.random() < 0.03
        t = sim.step(case, "Receive order", clerk, arrival, 0, sim.ln(12, 0.4))
        if sim.rng.random() >= 0.07:  # 7% skip the credit check ("known customer")
            t = sim.step(case, "Check credit", sim.pick(credit), t, sim.ln(2.5), sim.ln(18, 0.5))
            if sim.rng.random() < 0.03:
                sim.step(case, "Cancel order", clerk, t, sim.ln(3), sim.ln(8, 0.4))
                continue
        rejections = 0
        while True:
            approver = sim.pick(approvers_before if t < reorg else approvers_after)
            load = 1.0 if approver != "Marta Ilić" else (2.3 if t < reorg else 3.6)
            t = sim.step(case, "Approve order", approver, t, sim.ln(9 * load, 0.7), sim.ln(10, 0.5))
            if rejections < 2 and sim.rng.random() < (0.12 if rejections == 0 else 0.30):
                rejections += 1
                t = sim.step(case, "Update order", clerk, t, sim.ln(5), sim.ln(25, 0.5))
                continue
            break
        if proforma:  # pro-forma invoice for prepayment, sent before picking
            t = sim.step(case, "Send invoice", biller, t, sim.ln(2), sim.ln(8, 0.4))
        picker = sim.pick(warehouse)
        t = sim.step(case, "Pick goods", picker, t, sim.ln(5), sim.ln(35, 0.5))
        packer = picker if sim.rng.random() < 0.7 else sim.pick(warehouse)
        t = sim.step(case, "Pack goods", packer, t, sim.ln(0.8), sim.ln(20, 0.4))
        pickup = next_slot(t + timedelta(minutes=30), BUSINESS_DAYS, 15, strike)
        shipped = sim.step(case, "Ship goods", sim.pick(shipping), pickup + sim.jitter(20), 0, sim.ln(12, 0.4))
        t = shipped
        if not proforma:
            t = sim.step(case, "Send invoice", biller, t, sim.ln(3), sim.ln(8, 0.4))
            if sim.rng.random() < (0.08 if t < price_list else 0.17):
                t = sim.step(case, "Correct invoice", biller, t, sim.ln(14), sim.ln(25, 0.5))
                t = sim.step(case, "Send invoice", biller, t, sim.ln(1), sim.ln(8, 0.4))
        on_time = np.busday_count(arrival.date(), shipped.date()) <= 3
        in_full = sim.rng.random() < (0.86 if shortage[0] <= shipped < shortage[1] else 0.975)
        work_h = sum((r[4] - r[3]).total_seconds() for r in sim.rows[first_row:]) / 3600
        cost = work_h * 38 + 14 * sim.ln(1, 0.1) + (22 if not on_time and sim.rng.random() < 0.5 else 0)
        info[case] = {"otif": float(on_time and in_full), "cost": cost, "spikes": cost_spikes, "spike": 35.0}
    return sim, info


# ------------------------------------------------------------------------- Brenmoor Foods
def brenmoor(seed: int = 11) -> tuple[Sim, dict[str, dict]]:
    sim = Sim(seed)
    clerks = ["Noah Park", "Elif Şahin", "Ruth Adeyemi", "Sam Lindqvist"]
    specialists = {"Inês Duarte": 0.80, "Tom Becker": 0.20}
    holders = {"Owen Hale": 0.47, "Freya Collins": 0.30, "Dario Russo": 0.23}
    policy = datetime(2026, 3, 23)  # two budget holders move to one approval session a week
    price_update = datetime(2026, 5, 11)  # supplier price changes: more match exceptions
    shortage = (datetime(2026, 3, 2), datetime(2026, 3, 21))  # supplier delivery problems
    cost_spikes = {date(2026, 2, 18), date(2026, 6, 3)}  # one-off agency costs posted in a lump

    def approval_slot(holder: str, t: datetime) -> datetime:
        if holder == "Owen Hale":
            return next_slot(t, {1, 3} if t < policy else {3}, 14)
        if holder == "Freya Collins":
            return next_slot(t, {0, 2, 4} if t < policy else {0}, 11)
        return next_slot(t, BUSINESS_DAYS, 16)

    info: dict[str, dict] = {}
    for case, arrival in sim.cases("INV", 24001, sim.arrivals([12, 9, 9, 9, 8], 2.0)):
        clerk = sim.pick(clerks)
        first_row = len(sim.rows)
        t = sim.step(case, "Receive invoice", clerk, arrival, 0, sim.ln(4, 0.4))
        t = sim.step(case, "Scan and capture", clerk, t, sim.ln(3), sim.ln(7, 0.4))
        if sim.rng.random() < 0.02:  # duplicate invoice, cancelled after capture
            sim.step(case, "Cancel invoice", clerk, t, sim.ln(6), sim.ln(5, 0.4))
            continue
        if sim.rng.random() >= 0.06:  # 6% are non-PO invoices that skip matching
            t = sim.step(case, "Match to PO", clerk, t, sim.ln(2.5), sim.ln(10, 0.5))
            exception_rate, loops = (0.15 if t < price_update else 0.30), 0
            while loops < 2 and sim.rng.random() < exception_rate:
                specialist = sim.pick(specialists)
                load = 2.4 if specialist == "Inês Duarte" else 1.0
                t = sim.step(case, "Resolve discrepancy", specialist, t, sim.ln(10 * load), sim.ln(35, 0.5))
                t = sim.step(case, "Match to PO", clerk, t, sim.ln(1.5), sim.ln(8, 0.4))
                exception_rate, loops = 0.15, loops + 1
        if sim.rng.random() >= 0.03:  # 3% are paid without approval (a control breach)
            holder, rounds = sim.pick(holders), 0
            while True:
                slot = approval_slot(holder, t + timedelta(hours=sim.ln(1, 0.5)))
                t = sim.step(case, "Approve invoice", holder, slot + sim.jitter(90), 0, sim.ln(4, 0.5))
                if rounds == 0 and sim.rng.random() < 0.06:
                    t = sim.step(case, "Request clarification", clerk, t, sim.ln(4), sim.ln(15, 0.5))
                    rounds += 1
                    continue
                break
        t = sim.step(case, "Schedule payment", "Lena Vogt", next_slot(t, BUSINESS_DAYS, 13) + sim.jitter(30), 0,
                     sim.ln(6, 0.3))
        paid = sim.step(case, "Pay invoice", "Bank interface", next_slot(t, BUSINESS_DAYS, 16.5) + sim.jitter(20),
                        0, sim.ln(2, 0.3))
        in_full = sim.rng.random() < (0.80 if shortage[0] <= paid < shortage[1] else 0.945)
        work_h = sum((r[4] - r[3]).total_seconds() for r in sim.rows[first_row:]) / 3600
        info[case] = {"otif": float(in_full), "cost": work_h * 36 + 2.4 * sim.ln(1, 0.1),
                      "spikes": cost_spikes, "spike": 9.0}
    return sim, info


# ------------------------------------------------------------------------------- KPIs
def daily_kpis(events: pd.DataFrame, info: dict[str, dict], end_activity: str) -> pd.DataFrame:
    """KPIs from every simulated case, kept for the business days of the extract window."""
    log = from_frame(events)
    cases = case_table(enrich(log.events))
    kpis = derive_daily_kpis(cases, end_activity)
    done = cases[cases["trace"].map(lambda t: end_activity in t)]
    day = pd.to_datetime(np.busday_offset(done["end"].to_numpy("datetime64[D]"), 0, roll="forward"))
    per_case = pd.DataFrame({"day": day, "otif": [info[c]["otif"] for c in done.index],
                             "cost": [info[c]["cost"] + (info[c]["spike"] if d.date() in info[c]["spikes"] else 0)
                                      for c, d in zip(done.index, day, strict=True)]})
    grp = per_case.groupby("day")
    kpis["otif"] = grp["otif"].mean().reindex(kpis["date"]).values
    kpis["cost_per_order"] = grp["cost"].mean().reindex(kpis["date"]).values
    kpis = kpis[(kpis["date"] >= LOG_START) & (kpis["date"] < LOG_END)].copy()
    kpis["date"] = kpis["date"].dt.strftime("%Y-%m-%d")
    kpis["throughput"] = kpis["throughput"].astype(int)
    return kpis.round({"cycle_time_days": 3, "first_pass_yield": 4, "otif": 4, "cost_per_order": 2})


SAMPLES = [
    {
        "id": "valdora-order-fulfilment",
        "name": "Valdora Supply · Order fulfilment",
        "company": "Valdora Supply Co.",
        "industry": "B2B distributor of industrial parts",
        "process": "Order fulfilment (order-to-invoice)",
        "description": "Sales orders from receipt through credit check, manager approval, picking, packing and "
                       "shipping to the invoice.",
        "events_file": "data/valdora_events.csv",
        "kpis_file": "data/valdora_kpis.csv",
        "reference_path": ["Receive order", "Check credit", "Approve order", "Pick goods", "Pack goods",
                           "Ship goods", "Send invoice"],
        "kpis": {
            "throughput": {"label": "Orders invoiced / day", "target": 10},
            "cycle_time_days": {"label": "Order cycle time", "target": 4.0},
            "first_pass_yield": {"target": 0.85},
            "otif": {"target": 0.95},
            "cost_per_order": {"label": "Cost per order", "target": 95},
        },
        "planted": [
            "Approval bottleneck: three sales managers approve orders, but Marta Ilić receives 56% of them and "
            "her queue is about 2.3 times longer than her colleagues'.",
            "From 30 Mar 2026 Jonas Berg moves to key accounts: Marta's share rises to 80% and her queue grows "
            "again, which shifts cycle time upwards.",
            "Rework: 12% of orders are rejected at approval and updated; 8% of invoices need correcting, rising to "
            "17% after a new price list on 4 May 2026.",
            "Skipped and out-of-order steps: 7% of orders skip the credit check; 3% get a pro-forma invoice before "
            "picking.",
            "3% of credit-checked orders are cancelled.",
            "Calendar: no work at weekends; the carrier collects at 15:00 on business days, and a strike cancels "
            "the pickup on 22 Apr 2026.",
            "KPIs: in-full deliveries drop during a supplier shortage (18 May to 5 Jun 2026); emergency couriers "
            "add two one-day cost spikes (12 Mar and 10 Jun 2026).",
        ],
    },
    {
        "id": "brenmoor-invoice-to-pay",
        "name": "Brenmoor Foods · Invoice-to-pay",
        "company": "Brenmoor Foods Ltd",
        "industry": "Food manufacturer",
        "process": "Procure-to-pay: supplier invoice handling",
        "description": "Supplier invoices from receipt through capture, three-way match, budget-holder approval and "
                       "the daily payment run.",
        "events_file": "data/brenmoor_events.csv",
        "kpis_file": "data/brenmoor_kpis.csv",
        "reference_path": ["Receive invoice", "Scan and capture", "Match to PO", "Approve invoice",
                           "Schedule payment", "Pay invoice"],
        "kpis": {
            "throughput": {"label": "Invoices paid / day", "target": 10},
            "cycle_time_days": {"label": "Invoice cycle time", "target": 3.5},
            "first_pass_yield": {"target": 0.85},
            "otif": {"label": "Supplier OTIF", "target": 0.95},
            "cost_per_order": {"label": "Cost per invoice", "target": 28},
        },
        "planted": [
            "Approval batching: budget holders approve in fixed sessions. Owen Hale (47% of invoices) approves on "
            "Tuesday and Thursday afternoons only, Freya Collins on Monday, Wednesday and Friday mornings.",
            "From 23 Mar 2026, to save meeting time, Owen Hale approves on Thursdays only and Freya Collins on "
            "Mondays only; Dario Russo keeps approving daily. This shifts cycle time upwards.",
            "Overloaded specialist: Inês Duarte resolves 80% of price discrepancies and her queue is about 2.4 "
            "times longer than Tom Becker's.",
            "Rework: 15% of matches fail and are re-matched after the discrepancy is resolved, rising to 30% after "
            "supplier price changes on 11 May 2026; 6% of approvals need a clarification first.",
            "Skipped steps: 6% are non-PO invoices that skip matching; 3% are paid without approval.",
            "2% of invoices are duplicates, cancelled after capture.",
            "Calendar: no work at weekends; one payment run a day at 13:00, released to the bank at 16:30.",
            "KPIs: supplier OTIF drops during delivery problems (2 to 20 Mar 2026); two one-day cost spikes "
            "(18 Feb and 3 Jun 2026).",
        ],
    },
]

GENERATORS = {"valdora-order-fulfilment": valdora, "brenmoor-invoice-to-pay": brenmoor}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for sample in SAMPLES:
        sim, info = GENERATORS[sample["id"]]()
        events = sim.frame(extract_only=True)
        events.to_csv(ROOT / "demo" / sample["events_file"], index=False, lineterminator="\n")
        kpis = daily_kpis(sim.frame(extract_only=False), info, sample["reference_path"][-1])
        kpis.to_csv(ROOT / "demo" / sample["kpis_file"], index=False, lineterminator="\n")
        print(f"{sample['id']}: {events['case_id'].nunique():,} cases, {len(events):,} events, {len(kpis)} KPI days")
    (ROOT / "demo" / "samples.json").write_text(json.dumps(SAMPLES, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    print("wrote demo/samples.json")


if __name__ == "__main__":
    main()
