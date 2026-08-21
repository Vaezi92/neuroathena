"""Strict data contracts for NeuroAthena V3."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator


class CoordinateROI(BaseModel):
    """A reproducible spatial ROI; it is not an anatomical label."""

    name: str = Field(min_length=1)
    x_mm: tuple[float, float]
    y_mm: tuple[float, float]
    z_mm: tuple[float, float]

    @model_validator(mode="after")
    def ordered_bounds(self) -> "CoordinateROI":
        if any(low > high for low, high in (self.x_mm, self.y_mm, self.z_mm)):
            raise ValueError("ROI lower bounds must not exceed upper bounds")
        return self


class ToolCall(BaseModel):
    tool_name: Literal[
        "mraiv.get_metadata",
        "mraiv.get_roi_statistics",
        "mraiv.plot_velocity_slice",
        "mraiv.plot_permeability_slice",
        "mraiv.plot_distribution",
        "mraiv.compare_rois",
        "mraiv.retrieve_artifact",
    ]
    parameters: dict[str, Any] = Field(default_factory=dict)


class Artifact(BaseModel):
    artifact_id: str
    run_id: str
    tool_name: str
    tool_version: str
    source_file: str
    source_sha256: str
    parameters: dict[str, Any]
    numerical_summary: dict[str, Any]
    units: dict[str, str] = Field(default_factory=dict)
    figure_path: str | None = None
    figure_sha256: str | None = None


class SkillInvocation(BaseModel):
    skill_name: str
    version: str
    reason: str
    tool_calls: list[ToolCall]
    cautions: list[str]


class QuantitativeObservation(BaseModel):
    claim_id: str
    text: str
    source: Literal["computed_statistics"] = "computed_statistics"
    related_artifact_ids: list[str]
    numerical_evidence: dict[str, Any]
    confidence: Literal["low", "moderate", "high"] = "high"


class VisualObservation(BaseModel):
    claim_id: str
    text: str
    source: Literal["figure"] = "figure"
    related_artifact_ids: list[str] = Field(min_length=1)
    confidence: Literal["low", "moderate", "high"] = "moderate"


class ExpertQuestion(BaseModel):
    question_id: str
    round: int = Field(ge=1, le=3)
    text: str
    perspective: Literal["scientific_explanation", "fluid_dynamics", "biological_anatomical", "contextual_followup"]
    related_artifact_ids: list[str]


class HumanInterpretation(BaseModel):
    interpretation_id: str
    round: int = Field(ge=1, le=3)
    verbatim_text: str
    related_question_id: str


class DynamicQuestionSet(BaseModel):
    questions: list[Annotated[str, Field(min_length=10, max_length=500)]] = Field(min_length=1, max_length=2)
    unresolved_uncertainties: list[str] = Field(default_factory=list)


class ScientificSynthesis(BaseModel):
    executive_summary: str
    integrated_interpretation: str
    fluid_dynamics_interpretation: str
    biological_anatomical_interpretation: str
    alternative_explanations: list[str]
    limitations: list[str]
    recommended_next_experiments: list[str]
    citation_record_ids: list[str]


class AgentHypothesis(BaseModel):
    claim_id: str
    text: str
    source: Literal["agent_hypothesis"] = "agent_hypothesis"
    related_artifact_ids: list[str]
    human_interpretation_ids: list[str]
    confidence: Literal["low", "moderate"] = "low"


class SearchQuery(BaseModel):
    query_id: str
    claim_id: str
    intent: Literal["support", "contradiction", "alternative", "limitation"]
    query: str


class SpecialistNotebook(BaseModel):
    notebook_id: str
    title: str
    specialty: Literal["mri", "applied_mathematics", "neuroscience", "pinns", "fluid_dynamics"]
    description: str


class QueryNotebookRoute(BaseModel):
    query_id: str
    notebook_ids: list[str] = Field(min_length=1, max_length=2)
    reason: str


class NotebookRoutingPlan(BaseModel):
    routes: list[QueryNotebookRoute]


class LiteratureRecord(BaseModel):
    record_id: str
    query_id: str
    title: str
    authors: str = ""
    year: int | None = None
    journal: str = ""
    doi: str | None = None
    url: str
    peer_reviewed: bool | None = None
    provider: Literal["europe_pmc", "notebooklm"] = "europe_pmc"
    source_id: str | None = None
    excerpt: str | None = None


class EvidenceAssessment(BaseModel):
    claim_id: str
    direct_artifact_ids: list[str]
    supporting_record_ids: list[str]
    contradicting_record_ids: list[str]
    alternative_record_ids: list[str]
    limitation_record_ids: list[str]
    strength: Literal["insufficient", "weak", "moderate", "strong"]
    rationale: str


class GuardrailResult(BaseModel):
    guardrail: str
    passed: bool
    message: str


class NeuroAthenaState(BaseModel):
    run_id: str
    source_file: str
    requested_goal: str
    online_literature: bool = False
    notebooklm_notebook: str | None = None
    notebooklm_profile: str = "default"
    notebook_registry: list[dict[str, Any]] = Field(default_factory=list)
    notebook_routes: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_failures: list[dict[str, Any]] = Field(default_factory=list)
    notebook_concurrency: int = Field(default=3, ge=1, le=5)
    use_openai: bool = False
    openai_model: str = "gpt-5-mini"
    show_progress: bool = False
    dataset_metadata: dict[str, Any] = Field(default_factory=dict)
    selected_skills: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    quantitative_observations: list[dict[str, Any]] = Field(default_factory=list)
    visual_observations: list[dict[str, Any]] = Field(default_factory=list)
    questions: list[dict[str, Any]] = Field(default_factory=list)
    human_interpretations: list[dict[str, Any]] = Field(default_factory=list)
    hitl_round: int = 0
    uncertainty_resolved: bool = False
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    search_queries: list[dict[str, Any]] = Field(default_factory=list)
    literature_records: list[dict[str, Any]] = Field(default_factory=list)
    notebooklm_responses: list[dict[str, Any]] = Field(default_factory=list)
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)
    evidence_assessments: list[dict[str, Any]] = Field(default_factory=list)
    guardrail_results: list[dict[str, Any]] = Field(default_factory=list)
    final_summary: str | None = None
    final_synthesis: dict[str, Any] = Field(default_factory=dict)
    report_path: str | None = None
