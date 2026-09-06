"""Bounded analyzer, deterministic execution, and critic repair stage."""

from __future__ import annotations

import copy
from dataclasses import asdict

from ..models import AnalyzedQuantity, CriticReview, QARecord, RunState, Verdict
from ..protocols import Analyzer, Critic, DeterministicFunctionRegistry


FINAL_VERDICTS = {
    Verdict.ACCEPT,
    Verdict.PARTIAL,
    Verdict.UNANSWERABLE,
    Verdict.IMPLEMENTATION_FAILED,
    Verdict.INVALID_QUESTION,
    Verdict.REJECTED,
}


class AnalysisRound:
    """Attempt to answer one selected question with critic-controlled retries.

    The analyzer proposes a registered function. The critic reviews before and
    after execution. Only an accepted result is appended to Shared Analyzed
    Data. All attempts remain attached to the Q&A record for auditability.
    """

    def __init__(
        self,
        analyzer: Analyzer,
        critic: Critic,
        registry: DeterministicFunctionRegistry,
        maximum_attempts: int,
    ) -> None:
        self.analyzer = analyzer
        self.critic = critic
        self.registry = registry
        self.maximum_attempts = maximum_attempts

    def run(self, state: RunState, record: QARecord) -> None:
        """Mutate the Q&A record and commit an accepted result when available."""
        feedback: str | None = None
        last_review: CriticReview | None = None
        for attempt_number in range(1, self.maximum_attempts + 1):
            proposal = self.analyzer.propose(
                record.question,
                copy.deepcopy(state),
                attempt_number,
                feedback,
            )
            pre_review = self.critic.review_proposal(
                record.question,
                proposal,
                copy.deepcopy(state),
            )
            attempt = {
                "attempt": attempt_number,
                "proposal": asdict(proposal),
                "pre_review": asdict(pre_review),
                "result": None,
                "post_review": None,
            }
            if pre_review.verdict == Verdict.REVISE:
                record.attempts.append(attempt)
                last_review = pre_review
                feedback = pre_review.revision_request
                continue
            if pre_review.verdict != Verdict.ACCEPT:
                record.attempts.append(attempt)
                last_review = pre_review
                self._record_terminal_review(record, pre_review)
                break
            result, post_review = self._execute_and_review(state, record, proposal, attempt_number)
            attempt["result"] = asdict(result) if result else None
            attempt["post_review"] = asdict(post_review)
            record.attempts.append(attempt)
            last_review = post_review
            if post_review.verdict == Verdict.ACCEPT and result is not None:
                self._accept(state, record, proposal, result)
                break
            if post_review.verdict in FINAL_VERDICTS:
                self._record_terminal_review(record, post_review)
                break
            feedback = post_review.revision_request
        if record.final_status == "selected":
            record.final_status = "analysis_failed"
            record.final_answer = "Analyzer–critic revision limit exhausted."
        if last_review and last_review.issues:
            state.lessons.append({
                "question_id": record.question.question_id,
                "status": record.final_status,
                "issues": last_review.issues,
            })

    def _execute_and_review(self, state, record, proposal, attempt_number):
        try:
            result = self.registry.execute(proposal, state)
            review = self.critic.review_result(record.question, result, copy.deepcopy(state))
            return result, review
        except Exception as error:
            review = CriticReview(
                question_id=record.question.question_id,
                analysis_id=proposal.analysis_id,
                verdict=Verdict.REVISE,
                question_answered=False,
                method_valid=False,
                result_supported=False,
                issues=[str(error)],
                reanalysis_feasible=attempt_number < self.maximum_attempts,
                revision_request="Correct the deterministic analysis failure.",
                unanswerable_reason=str(error),
            )
            return None, review

    @staticmethod
    def _record_terminal_review(record: QARecord, review: CriticReview) -> None:
        record.final_status = review.verdict.value
        record.final_answer = review.unanswerable_reason or "; ".join(review.issues) or None

    def _accept(self, state: RunState, record: QARecord, proposal, result) -> None:
        """Commit an accepted result and promote its candidate when policy permits."""
        promoted = False
        if proposal.candidate_function is not None:
            promoted = self.registry.promote(proposal.candidate_function, result)
            result.validation["promoted"] = promoted
        record.final_status = "answered"
        record.final_answer = result.answer
        state.analyzed_data.append(AnalyzedQuantity(
            quantity_id=f"quantity-{record.question.step}",
            question_id=record.question.question_id,
            name=record.question.text,
            value=result.value,
            units=result.units,
            analysis_id=result.analysis_id,
            function_name=result.function_name,
            function_version=result.function_version,
            parameters=result.parameters,
            provenance=result.provenance,
            validation=result.validation,
        ))
