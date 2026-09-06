"""Typed, dependency-free contracts for NeuroAthena V4."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


PERSPECTIVES = ("mri", "fluid_dynamics", "applied_mathematics", "neuroscience")
ASPECTS = ("feasibility", "novelty", "importance")


class Verdict(str, Enum):
    """Permitted critic outcomes for an analysis attempt."""
    ACCEPT = "accept"
    REVISE = "revise"
    PARTIAL = "partially_answered"
    UNANSWERABLE = "unanswerable_from_data"
    IMPLEMENTATION_FAILED = "implementation_failed"
    INVALID_QUESTION = "invalid_question"
    REJECTED = "critic_rejected"


@dataclass(frozen=True)
class Score:
    """Feasibility, novelty, and importance values constrained to [0, 1]."""
    feasibility: float
    novelty: float
    importance: float

    def __post_init__(self) -> None:
        for name in ASPECTS:
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass
class CandidateQuestion:
    """A proposed question with authorship, grades, aggregate, and audit status."""
    candidate_id: str
    step: int
    text: str
    author: str
    rationale: dict[str, str]
    grades: dict[str, Score] = field(default_factory=dict)
    aggregate: Score | None = None
    overall_score: float | None = None
    status: str = "proposed"
    duplicate_of: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectedQuestion:
    """The single coordinator-selected question for one investigation step."""
    question_id: str
    step: int
    text: str
    aggregate: Score
    overall_score: float
    synthesized_description: str
    source_candidate_id: str


@dataclass(frozen=True)
class CandidateFunction:
    """Analyzer-authored deterministic function awaiting validation and trust."""
    name: str
    version: str
    description: str
    source: str


@dataclass
class AnalysisProposal:
    """An analyzer's request to execute a trusted or candidate function."""
    analysis_id: str
    question_id: str
    function_name: str
    parameters: dict[str, Any]
    intended_answer: str
    candidate_function: CandidateFunction | None = None


@dataclass
class AnalysisResult:
    """A deterministic function result with units and reproducibility metadata."""
    analysis_id: str
    question_id: str
    function_name: str
    function_version: str
    parameters: dict[str, Any]
    value: Any
    units: str | None
    answer: str
    provenance: dict[str, Any]
    validation: dict[str, Any] = field(default_factory=dict)


@dataclass
class CriticReview:
    """Critic verdict stating validity, completeness, and possible revision."""
    question_id: str
    analysis_id: str
    verdict: Verdict
    question_answered: bool
    method_valid: bool
    result_supported: bool
    issues: list[str] = field(default_factory=list)
    reanalysis_feasible: bool = False
    revision_request: str | None = None
    unanswerable_reason: str | None = None


@dataclass
class QARecord:
    """Shared Q&A entry containing selection, attempts, and terminal answer."""
    question: SelectedQuestion
    attempts: list[dict[str, Any]] = field(default_factory=list)
    final_status: str = "selected"
    final_answer: str | None = None


@dataclass
class AnalyzedQuantity:
    """Critic-accepted quantity stored in trusted Shared Analyzed Data."""
    quantity_id: str
    question_id: str
    name: str
    value: Any
    units: str | None
    analysis_id: str
    function_name: str
    function_version: str
    parameters: dict[str, Any]
    provenance: dict[str, Any]
    validation: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    """User-configurable scoring, duplicate, retry, and stopping policies."""
    maximum_steps: int = 10
    low_score_threshold: float = 0.25
    consecutive_low_score_limit: int = 5
    maximum_analysis_attempts: int = 3
    duplicate_similarity_threshold: float = 0.85
    aspect_weights: dict[str, float] = field(default_factory=lambda: {name: 1.0 for name in ASPECTS})
    grader_weights: dict[str, float] = field(default_factory=lambda: {name: 1.0 for name in PERSPECTIVES})

    def __post_init__(self) -> None:
        if min(self.maximum_steps, self.consecutive_low_score_limit, self.maximum_analysis_attempts) < 1:
            raise ValueError("Loop and attempt limits must be positive")
        if not 0 <= self.low_score_threshold <= 1:
            raise ValueError("low_score_threshold must be between 0 and 1")
        if not 0 <= self.duplicate_similarity_threshold <= 1:
            raise ValueError("duplicate_similarity_threshold must be between 0 and 1")
        for required, weights in ((ASPECTS, self.aspect_weights), (PERSPECTIVES, self.grader_weights)):
            if any(weights.get(name, 0) < 0 for name in required) or sum(weights.get(name, 0) for name in required) <= 0:
                raise ValueError("Weights must be non-negative and contain positive total weight")


@dataclass
class RunState:
    """Complete persistent shared state for one autonomous investigation."""
    run_id: str
    step: int = 0
    consecutive_low_score_steps: int = 0
    candidates: list[CandidateQuestion] = field(default_factory=list)
    qa_pool: list[QARecord] = field(default_factory=list)
    analyzed_data: list[AnalyzedQuantity] = field(default_factory=list)
    fixed_metadata: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    lessons: list[dict[str, Any]] = field(default_factory=list)
    termination_reason: str | None = None
    final_report: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the nested dataclass state into JSON-compatible structures."""
        return asdict(self)
