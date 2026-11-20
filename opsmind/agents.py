"""The two AI agents: a root-cause analyst that returns structured hypotheses and an
improvement plan, and a writer that turns the computed facts plus that diagnosis into an
executive summary. Neither sees the raw event log; both receive only computed facts."""

from __future__ import annotations

import json
from typing import Protocol

from .lean import WASTES
from .llm import clean_text
from .schemas import Diagnosis


class SupportsLLM(Protocol):
    def complete_json(self, system: str, user: str, schema: type, **kwargs): ...
    def chat(self, system: str, user: str, **kwargs) -> str: ...


ANALYST_SYSTEM = """You are a Lean Six Sigma Black Belt reviewing a process-mining study of one business process.
You receive computed facts only. Explain them; never invent data, people or figures.
Quote figures exactly as written in the facts. When the facts cannot tell two explanations apart,
say which extra data would. Owners are roles (for example "AP team lead"), never personal names."""

SUMMARY_SYSTEM = """You write one-page executive summaries of operations reviews for a chief operating officer.
Use Markdown with exactly these sections: ## Situation, ## What the data shows, ## Recommended actions,
## Expected impact, ## Next steps.
Use only the figures provided and write them exactly as given; never invent numbers or sources.
Be direct and specific. 250-400 words. Return the summary only, without a title line."""


def diagnose(llm: SupportsLLM, facts: dict) -> Diagnosis:
    user = (
        "Computed facts about the process (authoritative):\n"
        + json.dumps(facts, indent=2, ensure_ascii=False)
        + f"""

Return:
- headline: the most important finding in one sentence.
- root_causes: 3-5 items. problem = the symptom, quoting a computed figure; hypothesis = a plausible
  cause; evidence = 2-3 computed figures that support it, quoted exactly; test = one concrete check
  that would confirm or reject it (a data cut, a time study or a gemba observation);
  confidence = low, medium or high.
- improvement_plan: 4-7 actions, most valuable first. waste = exactly one of: {", ".join(WASTES)};
  owner_role = a role; effort = low, medium or high; expected_impact = quote the computed estimate for
  that waste when there is one, otherwise describe the effect without numbers; timeframe = e.g. "2 weeks".
- data_gaps: 2-4 things this event log cannot show that would sharpen the analysis."""
    )
    return llm.complete_json(ANALYST_SYSTEM, user, Diagnosis)


def write_summary(llm: SupportsLLM, facts: dict, diagnosis: Diagnosis) -> str:
    user = (
        "Computed facts (authoritative, use exactly as written):\n"
        + json.dumps(facts, indent=2, ensure_ascii=False)
        + "\n\nRoot-cause analysis and improvement plan:\n"
        + diagnosis.model_dump_json(indent=2)
    )
    return clean_text(llm.chat(SUMMARY_SYSTEM, user, temperature=0.4, max_tokens=1600))
