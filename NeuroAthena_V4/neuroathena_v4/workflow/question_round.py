"""Generate, deduplicate, cross-grade, and select one scientific question."""

from __future__ import annotations

import copy

from ..models import CandidateQuestion, QARecord, RunConfig, RunState, SelectedQuestion
from ..protocols import QuestionGenerator
from ..scoring import aggregate_candidate, find_duplicate, select_best


class NoValidCandidateError(RuntimeError):
    """Raised when duplicate filtering leaves no selectable question."""

    def __init__(self, candidates: list[CandidateQuestion]):
        super().__init__("No valid candidate questions remain")
        self.candidates = candidates


class QuestionRound:
    """Run the QG portion of one investigation step.

    Every agent receives the same immutable snapshot, proposes three questions,
    and grades an author-anonymized candidate list. This object does not mutate
    shared run state; it returns a new Q&A record and the complete candidate log.
    """

    def __init__(self, agents: list[QuestionGenerator], config: RunConfig):
        if not agents:
            raise ValueError("At least one QG agent is required")
        if len({agent.agent_id for agent in agents}) != len(agents):
            raise ValueError("QG agent IDs must be unique")
        self.agents = agents
        self.config = config

    def run(self, snapshot: RunState, step: int) -> tuple[QARecord, list[CandidateQuestion]]:
        """Return the selected question record and all audited candidates."""
        candidates = [item for agent in self.agents for item in agent.propose(snapshot, step)]
        self._require_three_per_agent(candidates)
        self._mark_duplicates(snapshot, candidates)
        valid = [candidate for candidate in candidates if candidate.status == "proposed"]
        if not valid:
            raise NoValidCandidateError(candidates)
        anonymous = self._anonymize(valid)
        self._collect_grades(snapshot, valid, anonymous)
        for candidate in valid:
            aggregate_candidate(candidate, self.config)
        winner = select_best(valid)
        winner.status = "selected"
        selected = SelectedQuestion(
            question_id=f"question-{step}",
            step=step,
            text=winner.text,
            aggregate=winner.aggregate,
            overall_score=winner.overall_score,
            synthesized_description=self._describe_selection(winner),
            source_candidate_id=winner.candidate_id,
        )
        return QARecord(question=selected), candidates

    def _require_three_per_agent(self, candidates: list[CandidateQuestion]) -> None:
        for agent in self.agents:
            count = sum(candidate.author == agent.agent_id for candidate in candidates)
            if count != 3:
                raise ValueError(f"QG agent {agent.agent_id} proposed {count} questions; expected 3")

    def _mark_duplicates(self, snapshot: RunState, candidates: list[CandidateQuestion]) -> None:
        known = {candidate.candidate_id: candidate.text for candidate in snapshot.candidates}
        known.update({record.question.question_id: record.question.text for record in snapshot.qa_pool})
        known.update({f"quantity:{item.quantity_id}": item.name for item in snapshot.analyzed_data})
        accepted: dict[str, str] = {}
        for candidate in candidates:
            match = find_duplicate(
                candidate.text,
                {**known, **accepted},
                self.config.duplicate_similarity_threshold,
            )
            if match:
                candidate.status = "duplicate"
                candidate.duplicate_of = match
            else:
                accepted[candidate.candidate_id] = candidate.text

    @staticmethod
    def _anonymize(candidates: list[CandidateQuestion]) -> list[CandidateQuestion]:
        anonymous = copy.deepcopy(candidates)
        for candidate in anonymous:
            candidate.author = "anonymous"
        return anonymous

    def _collect_grades(
        self,
        snapshot: RunState,
        candidates: list[CandidateQuestion],
        anonymous: list[CandidateQuestion],
    ) -> None:
        expected_ids = {candidate.candidate_id for candidate in candidates}
        for agent in self.agents:
            grades = agent.grade(snapshot, anonymous)
            if set(grades) != expected_ids:
                missing = sorted(expected_ids - set(grades))
                extra = sorted(set(grades) - expected_ids)
                raise ValueError(f"Agent {agent.agent_id} returned invalid grades; missing={missing}, extra={extra}")
            for candidate in candidates:
                candidate.grades[agent.agent_id] = grades[candidate.candidate_id]

    @staticmethod
    def _describe_selection(candidate: CandidateQuestion) -> str:
        reasons = "; ".join(f"{name}: {reason}" for name, reason in candidate.rationale.items())
        score = candidate.aggregate
        return (
            f"Selected after anonymous cross-grading with feasibility={score.feasibility:.3f}, "
            f"novelty={score.novelty:.3f}, importance={score.importance:.3f}. {reasons}"
        )
