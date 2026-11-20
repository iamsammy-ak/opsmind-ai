import pandas as pd
import pytest

from opsmind.eventlog import UNKNOWN_RESOURCE, EventLogError, from_frame, read_csv


def test_xes_headers_are_recognised_and_resource_is_optional():
    raw = pd.DataFrame(
        {
            "case:concept:name": ["x", "x"],
            "concept:name": ["Open", "Close"],
            "time:timestamp": ["2026-03-02T10:00:00+01:00", "2026-03-02T12:00:00+01:00"],
        }
    )
    log = from_frame(raw)
    assert list(log.events.columns) == ["case_id", "activity", "resource", "start", "end"]
    assert not log.has_start and not log.has_resource
    assert (log.events["resource"] == UNKNOWN_RESOURCE).all()
    # timezone-aware input is converted to UTC; without start times, start equals completion
    assert log.events["end"].iloc[0] == pd.Timestamp("2026-03-02 09:00")
    assert (log.events["start"] == log.events["end"]).all()


def test_missing_required_column_names_it():
    with pytest.raises(EventLogError, match="timestamp"):
        from_frame(pd.DataFrame({"case_id": ["1"], "activity": ["A"]}))


def test_bad_rows_are_dropped_and_late_starts_fixed():
    raw = pd.DataFrame(
        {
            "case_id": ["1", "1", "", "2"],
            "activity": ["A", "B", "A", "A"],
            "timestamp": ["2026-03-02 10:00", "not a date", "2026-03-02 11:00", "2026-03-02 12:00"],
            "start_timestamp": ["2026-03-02 09:00", "2026-03-02 09:00", "2026-03-02 10:00", "2026-03-02 13:00"],
        }
    )
    log = from_frame(raw)
    assert log.dropped_rows == 2 and log.n_events == 2 and log.n_cases == 2
    assert log.fixed_starts == 1  # case 2 started after it finished: start falls back to completion
    case2 = log.events[log.events["case_id"] == "2"].iloc[0]
    assert case2["start"] == case2["end"]


def test_read_csv_detects_semicolons_and_orders_activities():
    text = (
        "Case;Activity;Timestamp;User\n"
        "1;Ship;2026-03-02 15:00;bo\n"
        "1;Pick;2026-03-02 10:00;al\n"
        "2;Pick;2026-03-03 10:00;al\n"
        "2;Ship;2026-03-03 16:00;bo\n"
    )
    log = read_csv(text.encode("utf-8"))
    assert log.has_resource and log.n_cases == 2
    assert log.activities == ["Pick", "Ship"]  # ordered by typical position, not by file order
    assert list(log.events["activity"]) == ["Pick", "Ship", "Pick", "Ship"]


def test_unreadable_file_raises_a_readable_error():
    with pytest.raises(EventLogError):
        read_csv(b"just one column\nno timestamps here\n")
