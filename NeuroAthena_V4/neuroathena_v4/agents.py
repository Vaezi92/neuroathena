"""Deterministic reference agents used by the demo and tests."""

from __future__ import annotations

from dataclasses import replace

from .models import (
    AnalysisProposal, AnalysisResult, CandidateQuestion, CriticReview, RunState,
    Score, SelectedQuestion, Verdict,
)


class RuleBasedQG:
    """Deterministic QG used only for offline orchestration tests."""
    def __init__(self, agent_id: str, perspective: str):
        self.agent_id = agent_id
        self.perspective = perspective

    def propose(self, snapshot: RunState, step: int) -> list[CandidateQuestion]:
        """Return three predefined questions for the requested test step."""
        investigations = (
            (
                "Which available metadata most strongly constrains interpretation of these inferred fields",
                "Which coordinate bounds and sampling characteristics define the analyzable brain domain",
                "Which missing acquisition metadata creates the largest interpretation risk",
            ),
            (
                "What are the robust distribution summaries of velocity magnitude and its components",
                "How strongly do extreme velocity values influence the mean relative to the median",
                "Which velocity values should be flagged as low-reliability or potential outliers",
            ),
            (
                "What are the robust distribution summaries of inferred permeability",
                "How many orders of magnitude does permeability span across valid samples",
                "Where do permeability estimates approach numerical or physical plausibility limits",
            ),
            (
                "How heterogeneous is velocity across coordinate-bounded spatial regions",
                "Which spatial regions contain the largest velocity gradients",
                "How stable are regional velocity summaries under small boundary changes",
            ),
            (
                "How heterogeneous is permeability across coordinate-bounded spatial regions",
                "Which spatial regions contain the largest permeability gradients",
                "How stable are regional permeability summaries under small boundary changes",
            ),
            (
                "What spatial association exists between velocity magnitude and inferred permeability",
                "Does the velocity-permeability association persist after robust rank transformation",
                "Which regions contribute most strongly to the velocity-permeability association",
            ),
            (
                "How large are the available concentration prediction errors",
                "Where is concentration uncertainty greatest relative to predicted concentration",
                "Which inferred-field conclusions overlap regions of elevated concentration error",
            ),
            (
                "How large are the available advection-diffusion residual errors",
                "How large are the available mass-conservation residual errors",
                "Which spatial samples satisfy both residual reliability criteria",
            ),
            (
                "How sensitive are central-slice patterns to the chosen slice location",
                "Which apparent spatial structures persist across neighboring slices",
                "Could partial-volume or boundary effects explain the strongest displayed patterns",
            ),
            (
                "Which findings are supported by multiple independent deterministic analyses",
                "Which current conclusions remain underdetermined by the available result file",
                "Which next experiment would most reduce uncertainty in the integrated interpretation",
            ),
        )
        topics = investigations[min(step, len(investigations)) - 1]
        return [
            CandidateQuestion(
                candidate_id=f"s{step}-{self.agent_id}-{index}", step=step,
                text=f"{topic} from the {self.perspective} perspective?",
                author=self.agent_id,
                rationale={
                    "feasibility": "Uses accessible shared state.",
                    "novelty": "Targets the current investigation step.",
                    "importance": "Clarifies interpretation boundaries.",
                },
            )
            for index, topic in enumerate(topics, 1)
        ]

    def grade(self, snapshot: RunState, anonymized: list[CandidateQuestion]) -> dict[str, Score]:
        """Assign deterministic scores without OpenAI or NotebookLM calls."""
        output = {}
        for item in anonymized:
            already_has_results = bool(snapshot.analyzed_data)
            output[item.candidate_id] = Score(
                feasibility=0.9,
                novelty=0.55 if not already_has_results else 0.2,
                importance=0.7,
            )
        return output


class RuleBasedAnalyzer:
    """Offline analyzer that selects one of the two placeholder functions."""
    def propose(self, question: SelectedQuestion, snapshot: RunState, attempt: int, feedback: str | None) -> AnalysisProposal:
        """Choose metadata summary or analyzed-quantity count for a test."""
        function = "metadata_summary" if "metadata" in question.text.lower() or "limitation" in question.text.lower() else "analyzed_quantity_count"
        return AnalysisProposal(
            analysis_id=f"{question.question_id}-attempt-{attempt}", question_id=question.question_id,
            function_name=function, parameters={}, intended_answer=feedback or "Answer using trusted shared state.",
        )


class RuleBasedCritic:
    """Offline critic that checks allow-list membership and nonempty results."""
    def __init__(self, allowed_functions: set[str]):
        self.allowed_functions = allowed_functions

    def review_proposal(self, question: SelectedQuestion, proposal: AnalysisProposal, snapshot: RunState) -> CriticReview:
        """Accept only proposals naming functions in the supplied allow-list."""
        valid = proposal.function_name in self.allowed_functions
        return CriticReview(
            question.question_id, proposal.analysis_id,
            Verdict.ACCEPT if valid else Verdict.REVISE,
            question_answered=False, method_valid=valid, result_supported=False,
            issues=[] if valid else ["The proposed function is not allow-listed."],
            reanalysis_feasible=not valid,
            revision_request=None if valid else "Choose an allow-listed deterministic function.",
        )

    def review_result(self, question: SelectedQuestion, result: AnalysisResult, snapshot: RunState) -> CriticReview:
        """Accept a result only when it contains a value and written answer."""
        supported = bool(result.answer.strip()) and result.value is not None
        return CriticReview(
            question.question_id, result.analysis_id,
            Verdict.ACCEPT if supported else Verdict.REVISE,
            question_answered=supported, method_valid=True, result_supported=supported,
            issues=[] if supported else ["The result lacks a supported answer."],
            reanalysis_feasible=not supported,
            revision_request=None if supported else "Return a value and an evidence-linked answer.",
        )


class MarkdownSynthesizer:
    """Offline deterministic final-report renderer used by tests and demos."""
    def synthesize(self, state: RunState) -> str:
        """Render Q&A outcomes and limitations into a Markdown report."""
        rows = []
        for record in state.qa_pool:
            rows.append(
                f"## Step {record.question.step}: {record.question.text}\n\n"
                f"Score: {record.question.overall_score:.3f}. Status: `{record.final_status}`.\n\n"
                f"{record.final_answer or 'No supported answer was produced.'}"
            )
        limitations = "\n".join(f"- {item}" for item in state.limitations) or "- None declared."
        return (
            f"# NeuroAthena V4 Report\n\nRun: `{state.run_id}`\n\n"
            f"Termination: `{state.termination_reason}` after {state.step} step(s).\n\n"
            f"# Fixed limitations\n\n{limitations}\n\n# Investigation\n\n" + "\n\n".join(rows)
        )
