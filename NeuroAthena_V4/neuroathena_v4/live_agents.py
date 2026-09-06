"""OpenAI agents grounded in specialist NotebookLM resources."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, Field

from .models import (
    AnalysisProposal, AnalysisResult, CandidateFunction, CandidateQuestion, CriticReview, RunState,
    Score, SelectedQuestion, Verdict,
)
from .providers.notebooklm_client import ask_notebook, notebook_context
from .providers.openai_gateway import create_text, parse_structured


SYSTEM = """You are an agent in NeuroAthena V4, an autonomous scientific investigation of a trained
MR-AIV/PINN result derived from mouse-brain DCE-MRI. Velocity and permeability are model-inferred fields.
Never invent data, atlas labels, orientation, analyses, citations, or causal biological mechanisms. Distinguish
facts in Shared Analyzed Data, limitations, NotebookLM background, proposed questions, and interpretations."""


class ProposedItem(BaseModel):
    """One schema-validated question and its originating agent's self-score."""
    text: str = Field(min_length=15, max_length=500)
    feasibility: float = Field(ge=0, le=1)
    novelty: float = Field(ge=0, le=1)
    importance: float = Field(ge=0, le=1)
    feasibility_reason: str
    novelty_reason: str
    importance_reason: str


class ProposalSet(BaseModel):
    """Exactly three questions returned by one live QG agent."""
    questions: list[ProposedItem] = Field(min_length=3, max_length=3)


class GradeItem(BaseModel):
    """One anonymous candidate's three scores from one QG perspective."""
    candidate_id: str
    feasibility: float = Field(ge=0, le=1)
    novelty: float = Field(ge=0, le=1)
    importance: float = Field(ge=0, le=1)


class GradeSet(BaseModel):
    """Structured cross-grades returned by one QG agent."""
    grades: list[GradeItem]


class AnalyzerPlan(BaseModel):
    """Structured plan selecting a trusted function or authoring a candidate."""
    mode: Literal["trusted", "author"] = "author"
    function_name: str
    parameters_json: str = "{}"
    intended_answer: str
    function_version: str = "0.1.0"
    function_description: str = ""
    function_source: str | None = None


class ReviewOutput(BaseModel):
    """Structured critic decision for a proposal or executed result."""
    verdict: Verdict
    question_answered: bool
    method_valid: bool
    result_supported: bool
    issues: list[str] = Field(default_factory=list)
    reanalysis_feasible: bool = False
    revision_request: str | None = None
    unanswerable_reason: str | None = None


def _parse(model: str, schema: type[BaseModel], prompt: str) -> BaseModel:
    """Delegate structured model access to the provider gateway."""
    return parse_structured(model, SYSTEM, prompt, schema)


def _snapshot(state: RunState) -> dict[str, Any]:
    return {
        "step": state.step,
        "fixed_metadata": state.fixed_metadata,
        "limitations": state.limitations,
        "qa_pool": [asdict(item) for item in state.qa_pool],
        "analyzed_data": [asdict(item) for item in state.analyzed_data],
        "lessons": state.lessons,
    }


def _notebook_snapshot(state: RunState) -> dict[str, Any]:
    """Small domain context suitable for NotebookLM's question-size limit."""
    observed = state.fixed_metadata.get("observed", {})
    acquisition = state.fixed_metadata.get("acquisition", {})
    return {
        "step": state.step,
        "data_kind": state.fixed_metadata.get("data_kind"),
        "sample_count": state.fixed_metadata.get("sample_count"),
        "available_fields": state.fixed_metadata.get("available_fields", []),
        "acquisition": acquisition,
        "observed": observed,
        "limitations": state.limitations,
        "recent_questions": [item.question.text for item in state.qa_pool[-5:]],
        "analyzed_quantities": [item.name for item in state.analyzed_data[-10:]],
    }


class NotebookOpenAIQG:
    """Generate and grade questions using OpenAI plus private NotebookLM context.

    NotebookLM supplies domain background only. Proposed questions go directly
    to the coordinator, which anonymizes them before cross-grading.
    """
    def __init__(self, agent_id: str, perspective: str, notebook_id: str, *, profile: str, model: str):
        self.agent_id, self.perspective, self.notebook_id = agent_id, perspective, notebook_id
        self.profile, self.model = profile, model

    def _ground(self, task: str, state: RunState) -> dict[str, Any]:
        print(f"NeuroAthena · {self.perspective} NotebookLM consultation…", flush=True)
        prompt = (
            f"Use only this notebook's sources. Advise a {self.perspective} question-generator. {task}\n"
            f"Compact state: {json.dumps(_notebook_snapshot(state), default=str, separators=(',', ':'))}"
        )
        try:
            payload = ask_notebook(self.notebook_id, prompt[:7000], profile=self.profile)
            return notebook_context(payload)
        except Exception as error:
            # V3 preserves individual retrieval failures and continues with the
            # evidence that remains. Live QGs follow the same partial-mode rule.
            print(f"NeuroAthena · {self.perspective} NotebookLM unavailable; continuing with recorded limitation.", flush=True)
            return {
                "answer": "",
                "references": [],
                "notebook_id": self.notebook_id,
                "failure": f"{type(error).__name__}: {error}",
            }

    def propose(self, snapshot: RunState, step: int) -> list[CandidateQuestion]:
        """Consult the assigned notebook, then generate exactly three questions."""
        evidence = self._ground(
            "Identify scientifically valuable, feasible, and novel questions answerable through deterministic analysis "
            "of the available fields. Avoid questions already answered. Provide cautious source-grounded guidance.", snapshot,
        )
        parsed = _parse(
            self.model, ProposalSet,
            "Generate exactly three distinct questions from the specified perspective. Questions may request a new "
            "deterministic analysis. Score each from 0 to 1. Feasibility must be low when required data or metadata are "
            "unavailable. Novelty must account for the entire Q&A pool and analyzed data.\n"
            f"Perspective: {self.perspective}\nShared state: {json.dumps(_snapshot(snapshot), default=str)}\n"
            f"NotebookLM guidance: {json.dumps(evidence, default=str)}",
        )
        return [
            CandidateQuestion(
                candidate_id=f"s{step}-{self.agent_id}-{index}", step=step, text=item.text,
                author=self.agent_id,
                rationale={
                    "feasibility": item.feasibility_reason,
                    "novelty": item.novelty_reason,
                    "importance": item.importance_reason,
                    "self_score": json.dumps({"feasibility": item.feasibility, "novelty": item.novelty, "importance": item.importance}),
                },
                evidence={**evidence, "notebook_id": self.notebook_id},
            )
            for index, item in enumerate(parsed.questions, 1)
        ]

    def grade(self, snapshot: RunState, anonymized: list[CandidateQuestion]) -> dict[str, Score]:
        """Grade every coordinator-anonymized candidate from this perspective."""
        questions = [{"candidate_id": item.candidate_id, "text": item.text} for item in anonymized]
        evidence = self._ground(
            "Provide concise domain criteria for judging candidate questions for feasibility, novelty, and "
            "importance given the available data. Do not enumerate or rewrite candidate questions.", snapshot,
        )
        parsed = _parse(
            self.model, GradeSet,
            "Grade every anonymous candidate exactly once. Use only exact candidate IDs. Do not infer authorship. "
            "Scores are 0 to 1.\n"
            f"Perspective: {self.perspective}\nCandidates: {json.dumps(questions)}\n"
            f"Shared state: {json.dumps(_snapshot(snapshot), default=str)}\nNotebookLM guidance: {json.dumps(evidence, default=str)}",
        )
        result = {item.candidate_id: Score(item.feasibility, item.novelty, item.importance) for item in parsed.grades}
        expected = {item.candidate_id for item in anonymized}
        if set(result) != expected:
            raise ValueError("OpenAI grader must grade every candidate exactly once using exact IDs")
        return result


class OpenAIAnalyzer:
    """Select a trusted function or author a deterministic candidate function."""
    def __init__(self, capabilities: dict[str, dict[str, Any]], model: str):
        self.capabilities, self.model = capabilities, model

    def propose(self, question: SelectedQuestion, snapshot: RunState, attempt: int, feedback: str | None) -> AnalysisProposal:
        """Return a trusted call or analyzer-authored candidate computation."""
        parsed = _parse(
            self.model, AnalyzerPlan,
            "Answer the selected question with deterministic analysis. Use mode=trusted only when an existing "
            "capability directly answers it. Otherwise use mode=author and provide one complete Python function "
            "named analyze(data, parameters). The function may use np (already provided), must perform no imports, "
            "I/O, randomness, mutation, networking, or dynamic code execution, and must return exactly a dictionary "
            "with value (JSON-compatible), units (string or null), and answer (non-empty evidence-linked string). "
            "Use ordinary NumPy array/statistics operations only; file I/O and non-scientific NumPy attributes are blocked. "
            "MATLAB arrays are available by field name in data. Put parameters in parameters_json as an object. "
            "Use a unique descriptive public function_name and version 0.1.0 for a new candidate.\n"
            f"Question: {json.dumps(asdict(question), default=str)}\nCapabilities: {json.dumps(self.capabilities)}\n"
            f"Critic feedback: {feedback}\nState: {json.dumps(_snapshot(snapshot), default=str)}",
        )
        if parsed.mode == "trusted" and parsed.function_name not in self.capabilities:
            raise ValueError(f"Analyzer selected unknown function: {parsed.function_name}")
        if parsed.mode == "author" and not parsed.function_source:
            raise ValueError("Analyzer authoring plan omitted function_source")
        try:
            parameters = json.loads(parsed.parameters_json)
        except json.JSONDecodeError as error:
            raise ValueError("Analyzer parameters_json is not valid JSON") from error
        if not isinstance(parameters, dict):
            raise ValueError("Analyzer parameters_json must decode to an object")
        candidate = None
        if parsed.mode == "author":
            candidate = CandidateFunction(
                name=parsed.function_name,
                version=parsed.function_version,
                description=parsed.function_description,
                source=parsed.function_source or "",
            )
        return AnalysisProposal(
            f"{question.question_id}-attempt-{attempt}", question.question_id,
            parsed.function_name, parameters, parsed.intended_answer, candidate,
        )


class OpenAICritic:
    """Review analysis proposals and results with structured OpenAI verdicts."""
    def __init__(self, capabilities: dict[str, dict[str, Any]], model: str):
        self.capabilities, self.model = capabilities, model

    def _review(self, question: SelectedQuestion, analysis_id: str, payload: dict[str, Any], state: RunState) -> CriticReview:
        parsed = _parse(
            self.model, ReviewOutput,
            "Audit the method and its answer conservatively. Use revise only when deterministic re-analysis with "
            "available data/functions can fix the issue. Never accept unsupported biological or anatomical claims.\n"
            f"Question: {json.dumps(asdict(question), default=str)}\nSubmission: {json.dumps(payload, default=str)}\n"
            f"Capabilities: {json.dumps(self.capabilities)}\nState: {json.dumps(_snapshot(state), default=str)}",
        )
        return CriticReview(question.question_id, analysis_id, **parsed.model_dump())

    def review_proposal(self, question: SelectedQuestion, proposal: AnalysisProposal, snapshot: RunState) -> CriticReview:
        """Check method feasibility and validity before deterministic execution."""
        return self._review(question, proposal.analysis_id, {"stage": "proposal", **asdict(proposal)}, snapshot)

    def review_result(self, question: SelectedQuestion, result: AnalysisResult, snapshot: RunState) -> CriticReview:
        """Check whether the executed result validly answers the question."""
        return self._review(question, result.analysis_id, {"stage": "result", **asdict(result)}, snapshot)


class OpenAISynthesizer:
    """Create the final report after the coordinator terminates the loop."""
    def __init__(self, model: str):
        self.model = model

    def synthesize(self, state: RunState) -> str:
        """Render accepted evidence, failures, limitations, and termination."""
        prompt = (
            "Write a self-contained Markdown scientific report from this completed state. Clearly separate "
            "validated deterministic results, agent-selected questions, limitations, failed/unanswerable "
            "analyses, and interpretations. Cite NotebookLM source titles/excerpts when present. Never claim "
            "causality or anatomical identity unsupported by metadata. State the termination reason.\n"
            + json.dumps(_snapshot(state), default=str)
        )
        return create_text(self.model, SYSTEM, prompt)
