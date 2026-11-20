"""LLM parsing and agent tests. No network: a fake client stands in for OpenRouter."""

import json

import pytest
from helpers import ABC, REFERENCE

from opsmind.agents import diagnose, write_summary
from opsmind.config import Settings
from opsmind.demo import Dataset
from opsmind.lean import lean_map
from opsmind.llm import LLMClient, LLMError, clean_text, extract_json
from opsmind.mining import mine
from opsmind.pipeline import AnalysisParams, compute, data_fingerprint, fingerprint, rank_plan, run_ai
from opsmind.schemas import Diagnosis, PlanItem, coerce_waste

DIAGNOSIS = {
    "headline": "Approvals queue for two days.",
    "root_causes": [
        {
            "problem": "Cases wait 42.1 h before Approve order",
            "hypothesis": "One approver gets most of the work",
            "evidence": [{"figure": "67% of approvals"}, "2.1× longer waits"],
            "test": "Compare queue length per approver for two weeks",
            "confidence": "Moderate",
        }
    ],
    "improvement_plan": [
        {"action": "Error-proof order entry", "waste": "rework loops", "owner_role": "Sales ops lead",
         "effort": "High effort", "expected_impact": "fewer repeats"},
        {"action": "Add a backup approver", "waste": "Waiting / queues", "owner_role": "Sales manager",
         "effort": "low", "expected_impact": "−2,733 case-hours a month"},
        {"action": "Celebrate wins", "waste": "morale", "owner_role": "Team lead", "effort": "low"},
    ],
    "data_gaps": "Order value per case",
}


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.last_model = "fake/model"
        self.prompts = []

    def chat(self, system, user, **kwargs):
        self.prompts.append(user)
        return self.replies.pop(0)

    def complete_json(self, system, user, schema, **kwargs):
        return schema.model_validate(extract_json(self.chat(system, user)))


def _dataset() -> Dataset:
    return Dataset(key="abc", name="ABC", company="Test Co", process="Test process", description="tiny", synthetic=True,
                   log=ABC, kpis=None, kpi_overrides={}, reference=REFERENCE)


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('<think>hmm {"no": 1}</think>Sure! {"a": 3} hope that helps') == {"a": 3}
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        extract_json("[1, 2]")
    assert clean_text("<think>plan</think>\n## Situation\nok") == "## Situation\nok"


def test_schema_coercion_absorbs_free_model_quirks():
    d = Diagnosis.model_validate(DIAGNOSIS)
    rc = d.root_causes[0]
    assert rc.confidence == "medium" and rc.evidence == ["67% of approvals", "2.1× longer waits"]
    plan = d.improvement_plan
    assert [p.waste for p in plan] == ["Defects", "Waiting", ""]
    assert [p.effort for p in plan] == ["high", "low", "low"]
    assert d.data_gaps == ["Order value per case"]
    assert coerce_waste("hand-offs between teams") == "Transportation"
    assert coerce_waste("Inventory") == "Inventory" and coerce_waste("WIP build-up") == "Inventory"


def test_plan_is_ranked_by_computed_hours_over_effort():
    lean = lean_map(mine(ABC, REFERENCE), has_start=True)
    plan = Diagnosis.model_validate(DIAGNOSIS).improvement_plan
    ranked = rank_plan(plan, lean)
    assert ranked[0].item.action == "Add a backup approver"  # most hours at stake, lowest effort
    assert ranked[-1].item.action == "Celebrate wins" and ranked[-1].finding is None and ranked[-1].score == 0
    hours = lean.get("Defects").score
    assert next(r for r in ranked if r.item.waste == "Defects").score == pytest.approx(hours / 3)
    assert rank_plan([PlanItem(action="x", waste="Motion")], lean)[0].score == 0  # Motion is never detected


def test_agents_receive_computed_facts_only():
    ds = _dataset()
    params = AnalysisParams(reference=REFERENCE)
    fake = FakeLLM([json.dumps(DIAGNOSIS), "<think>x</think>## Situation\nQueues."])
    result = run_ai(fake, ds, compute(ds, params), params, on_step=lambda _msg: None)
    json.dumps(result)
    assert result["summary"] == "## Situation\nQueues."
    assert result["fingerprint"] == fingerprint(ds, params)
    assert result["data_fingerprint"] == data_fingerprint(ds)
    facts = json.loads(fake.prompts[0].split("\n", 1)[1].split("\n\nReturn:")[0])
    assert facts["process"]["cases"] == "3" and "Lean wastes detected" in facts
    assert "2026-03-02 09:00" not in fake.prompts[0]  # no raw events in the prompt


def test_diagnose_and_summary_directly():
    fake = FakeLLM([json.dumps(DIAGNOSIS), "```markdown\n## Situation\nok\n```"])
    d = diagnose(fake, {"flow": {"cases": "3"}})
    assert d.headline.startswith("Approvals")
    assert write_summary(fake, {"flow": {}}, d) == "## Situation\nok"


def test_client_requires_key():
    with pytest.raises(LLMError):
        LLMClient(Settings(api_key=None))


def test_complete_json_repairs_once(tmp_path, monkeypatch):
    client = LLMClient(Settings(api_key="test", models=("fake/model",)), cache_dir=tmp_path)
    replies = iter(["not json at all", json.dumps(DIAGNOSIS)])
    monkeypatch.setattr(client, "_call_with_fallback", lambda *args, **kw: (next(replies), "fake/model"))
    result = client.complete_json("sys", "user", Diagnosis)
    assert result.headline.startswith("Approvals")
    assert len(list(tmp_path.glob("*.json"))) == 1  # only the valid repaired answer stays cached


def test_chat_uses_cache(tmp_path, monkeypatch):
    client = LLMClient(Settings(api_key="test", models=("fake/model",)), cache_dir=tmp_path)
    calls = []

    def fake_call(*args, **kw):
        calls.append(1)
        return "hello", "fake/model"

    monkeypatch.setattr(client, "_call_with_fallback", fake_call)
    assert client.chat("s", "u") == "hello" and client.chat("s", "u") == "hello"
    assert len(calls) == 1
