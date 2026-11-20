"""End-to-end UI test: runs the real Streamlit script headlessly (no browser, no model calls)."""

import pytest
from streamlit.testing.v1 import AppTest

from opsmind.demo import load_samples
from opsmind.pipeline import AnalysisParams

TABS = 7


@pytest.fixture(autouse=True)
def demo_mode(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")


def _app() -> AppTest:
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.secrets["OPENROUTER_API_KEY"] = ""  # demo mode even if a global secrets file exists
    at.run()
    assert not at.exception, at.exception
    return at


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_app_renders_headline_metrics_tabs_and_demo_notice():
    at = _app()
    labels = [m.label for m in at.metric]
    for label in ("Cases", "Median time", "Fitting cases", "Rework", "Avg WIP", "Flow efficiency"):
        assert label in labels
    assert len(at.tabs) == TABS
    assert any("Demo mode" in i.value for i in at.sidebar.info)
    assert any("OPENROUTER_API_KEY" in i.value for i in at.info)  # the AI tab explains the setup step
    assert at.button[0].disabled  # no model calls without a key
    # the app's default settings equal AnalysisParams' defaults, so scripts/build_demo.py fixtures match
    ss = at.session_state
    in_app = AnalysisParams(tuple(ss["reference"]), ss["wait_cut"] / 100, ss["rework_cut"] / 100,
                            float(ss["cost_per_hour"]), int(ss["baseline"]), int(ss["z_window"]),
                            float(ss["z_threshold"]))
    assert in_app == AnalysisParams(reference=tuple(load_samples()[0]["reference_path"]))


def test_every_sample_renders():
    at = _app()
    for sample in load_samples():
        at.selectbox(key="dataset").set_value(sample["name"]).run()
        assert not at.exception, (sample["name"], at.exception)
        assert at.multiselect(key="reference").value == sample["reference_path"]


def test_controls_recompute_the_analysis():
    at = _app()
    before = _metric(at, "Waiting removed")
    at.slider(key="wait_cut").set_value(60).run()
    assert _metric(at, "Waiting removed") != before
    fitting = _metric(at, "Fitting cases")
    at.multiselect(key="reference").set_value(["Receive order", "Approve order", "Pick goods", "Pack goods",
                                               "Ship goods", "Send invoice"]).run()
    assert _metric(at, "Fitting cases") != fitting
    for key, value in (("edge_label", "Mean wait"), ("map_layout", "Left-right")):
        at.radio(key=key).set_value(value).run()
        assert not at.exception
    at.slider(key="edge_min").set_value(10.0).run()
    at.selectbox(key="kpi_choice").set_value("cost_per_order").run()
    assert not at.exception and _metric(at, "Target") == "€95.00"


def test_upload_mode_without_a_file_shows_instructions():
    at = _app()
    at.selectbox(key="dataset").set_value("Upload your own CSV").run()
    assert not at.exception
    assert any("Upload an event log" in i.value for i in at.info)
