"""Citation-constrained OpenAI reasoning for contextual questions and synthesis."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from models import DynamicQuestionSet, NotebookRoutingPlan, QueryNotebookRoute, ScientificSynthesis, SearchQuery, SpecialistNotebook


SYSTEM_CONTRACT = """You are NeuroAthena's scientific reasoning layer. Numerical fields are model-inferred,
not independently measured biological mechanisms. Use only the supplied artifacts, verbatim expert responses,
and retrieved evidence. Never invent a source, anatomical orientation, result, or citation. Cite literature only
with record IDs present in the context. Use calibrated language and clearly separate observation, expert
interpretation, fluid-dynamics reasoning, biological/anatomical reasoning, and literature evidence."""

DISALLOWED_QUESTION_PATTERNS = (
    r"\bcompute\b", r"\bcalculate\b", r"\brun (?:the|a|these)\b", r"\bretrain\b",
    r"\bprovide (?:the )?exact\b", r"\bsupply\b", r"\bdeliver\b", r"\breturn (?:a |the )?(?:map|plot|correlation)",
    r"\bper-voxel\b", r"\bloss decomposition\b", r"\bhyperparameters?\b", r"\bablation\b",
)


def question_is_immediately_answerable(question: str) -> bool:
    """Reject analysis assignments masquerading as expert questions."""
    text = " ".join(question.split())
    return len(text) <= 500 and not any(re.search(pattern, text, re.IGNORECASE) for pattern in DISALLOWED_QUESTION_PATTERNS)


def _fallback_followup(state: Any) -> str:
    latest = state.human_interpretations[-1]["verbatim_text"] if state.human_interpretations else "the current interpretation"
    return (
        f"Considering your explanation “{' '.join(latest.split())[:220]}”, which uncertainty or alternative "
        "explanation would most change your interpretation of the displayed results?"
    )


def _client() -> Any:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for NeuroAthena V3; use --no-openai for an offline run")
    from openai import OpenAI

    return OpenAI()


def _context(state: Any) -> dict[str, Any]:
    return {
        "goal": state.requested_goal,
        "dataset_metadata": state.dataset_metadata,
        "artifacts": [
            {
                "artifact_id": item["artifact_id"],
                "tool_name": item["tool_name"],
                "parameters": item["parameters"],
                "numerical_summary": item["numerical_summary"],
            }
            for item in state.artifacts
        ],
        "questions": state.questions,
        "verbatim_human_interpretations": state.human_interpretations,
        "hypotheses": state.hypotheses,
        "literature_records": state.literature_records,
        "notebooklm_answers": state.notebooklm_responses,
        "evidence_assessments": state.evidence_assessments,
        "deferred_analysis_requests": [
            question for call in state.llm_calls for question in call.get("rejected_questions", [])
        ],
    }


def generate_dynamic_questions(state: Any, round_number: int) -> tuple[DynamicQuestionSet, dict[str, Any]]:
    response = _client().responses.parse(
        model=state.openai_model,
        input=[
            {"role": "system", "content": SYSTEM_CONTRACT},
            {
                "role": "user",
                "content": (
                    f"Generate one or at most two concise expert questions for HITL round {round_number}. "
                    "They must directly follow from unresolved tensions or gaps in the supplied expert answers and "
                    "retrieved evidence. Each question must be answerable immediately from the expert's present "
                    "knowledge and the displayed figures. Never ask the expert to compute statistics or maps, run "
                    "code, retrain a model, perform an ablation, provide per-voxel outputs, or supply unavailable "
                    "training records. Put such work only in unresolved_uncertainties for the final report. Do not "
                    "repeat the fixed round-one questions.\n\n"
                    + json.dumps(_context(state), default=str)
                ),
            },
        ],
        text_format=DynamicQuestionSet,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned no structured follow-up questions")
    rejected = [question for question in parsed.questions if not question_is_immediately_answerable(question)]
    accepted = [question for question in parsed.questions if question_is_immediately_answerable(question)]
    if not accepted:
        accepted = [_fallback_followup(state)]
    constrained = DynamicQuestionSet(
        questions=accepted[:2],
        unresolved_uncertainties=parsed.unresolved_uncertainties + rejected,
    )
    call = _call_record(response, "dynamic_questions", state.openai_model)
    call["rejected_questions"] = rejected
    return constrained, call


def route_notebooks_with_openai(
    state: Any, queries: list[SearchQuery], notebooks: list[SpecialistNotebook]
) -> tuple[list[QueryNotebookRoute], dict[str, Any]]:
    response = _client().responses.parse(
        model=state.openai_model,
        input=[
            {"role": "system", "content": SYSTEM_CONTRACT},
            {
                "role": "user",
                "content": (
                    "Route every search query to one or at most two of the allow-listed specialist notebooks. "
                    "Choose the smallest sufficient set based on domain fit. Use only exact query IDs and notebook "
                    "IDs supplied below. Do not omit a query.\n\n"
                    + json.dumps({"queries": [item.model_dump() for item in queries],
                                  "notebooks": [item.model_dump() for item in notebooks],
                                  "expert_interpretations": state.human_interpretations}, default=str)
                ),
            },
        ],
        text_format=NotebookRoutingPlan,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned no structured notebook routing plan")
    query_ids = {item.query_id for item in queries}
    notebook_ids = {item.notebook_id for item in notebooks}
    routed_query_ids = {item.query_id for item in parsed.routes}
    if routed_query_ids != query_ids or len(parsed.routes) != len(queries):
        raise ValueError("Notebook routing must cover every query exactly once")
    if any(not set(item.notebook_ids).issubset(notebook_ids) for item in parsed.routes):
        raise ValueError("Notebook routing referenced a notebook outside the allow-list")
    return parsed.routes, _call_record(response, "notebook_routing", state.openai_model)


def synthesize_report(state: Any) -> tuple[ScientificSynthesis, dict[str, Any]]:
    response = _client().responses.parse(
        model=state.openai_model,
        input=[
            {"role": "system", "content": SYSTEM_CONTRACT},
            {
                "role": "user",
                "content": (
                    "Prepare a concise evidence-grounded synthesis. Every citation_record_id must exactly match a "
                    "record_id in literature_records. Put the relevant record ID in square brackets immediately "
                    "after each literature-supported statement, and include every such ID in citation_record_ids. "
                    "If evidence is absent, say so and return no citation IDs. "
                    "Do not strengthen a conclusion beyond the weakest link in the data-to-evidence chain.\n\n"
                    + json.dumps(_context(state), default=str)
                ),
            },
        ],
        text_format=ScientificSynthesis,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned no structured scientific synthesis")
    allowed = {item["record_id"] for item in state.literature_records}
    if not set(parsed.citation_record_ids).issubset(allowed):
        raise ValueError("OpenAI synthesis referenced an unknown literature record")
    return parsed, _call_record(response, "scientific_synthesis", state.openai_model)


def _call_record(response: Any, purpose: str, model: str) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    return {
        "purpose": purpose,
        "model": model,
        "response_id": getattr(response, "id", None),
        "usage": usage.model_dump() if usage and hasattr(usage, "model_dump") else {},
    }
