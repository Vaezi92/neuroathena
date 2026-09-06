"""Replaceable agent and deterministic-analysis interfaces."""

from __future__ import annotations

from typing import Any, Protocol

from .models import AnalysisProposal, AnalysisResult, CandidateFunction, CandidateQuestion, CriticReview, RunState, Score, SelectedQuestion


class QuestionGenerator(Protocol):
    """Interface for a domain agent that proposes and cross-grades questions."""
    agent_id: str
    perspective: str

    def propose(self, snapshot: RunState, step: int) -> list[CandidateQuestion]:
        """Return exactly three candidate questions for the step."""
        ...
    def grade(self, snapshot: RunState, anonymized: list[CandidateQuestion]) -> dict[str, Score]:
        """Return one score for every anonymous candidate ID."""
        ...


class Analyzer(Protocol):
    """Interface for planning one deterministic analysis attempt."""
    def propose(self, question: SelectedQuestion, snapshot: RunState, attempt: int, feedback: str | None) -> AnalysisProposal:
        """Plan one function execution, optionally responding to critic feedback."""
        ...


class Critic(Protocol):
    """Interface for pre-execution and post-execution scientific review."""
    def review_proposal(self, question: SelectedQuestion, proposal: AnalysisProposal, snapshot: RunState) -> CriticReview:
        """Return a pre-execution validity verdict."""
        ...
    def review_result(self, question: SelectedQuestion, result: AnalysisResult, snapshot: RunState) -> CriticReview:
        """Return a post-execution answer-quality verdict."""
        ...


class Synthesizer(Protocol):
    """Interface for final report creation after loop termination."""
    def synthesize(self, state: RunState) -> str:
        """Return the final report for a terminated state."""
        ...


class DeterministicFunctionRegistry(Protocol):
    """Interface for describing and executing trusted computations."""
    def execute(self, proposal: AnalysisProposal, state: RunState) -> AnalysisResult:
        """Execute one allow-listed proposal and return provenance."""
        ...
    def capabilities(self) -> dict[str, dict[str, Any]]:
        """Describe the currently allow-listed functions."""
        ...
    def promote(self, function: CandidateFunction, result: AnalysisResult) -> bool:
        """Promote an accepted, validated candidate according to policy."""
        ...
