"""Readable top-level orchestration for NeuroAthena V4."""

from __future__ import annotations

import copy
from dataclasses import asdict

from .models import RunConfig, RunState
from .protocols import Analyzer, Critic, DeterministicFunctionRegistry, QuestionGenerator, Synthesizer
from .store import RunStore
from .workflow import AnalysisRound, QuestionRound, StoppingPolicy
from .workflow.question_round import NoValidCandidateError


class Coordinator:
    """Run the investigation while delegating each workflow responsibility.

    `QuestionRound` owns generation, anonymization, grading, and selection.
    `AnalysisRound` owns analyzer/critic retries and trusted-data admission.
    `StoppingPolicy` owns termination rules. The coordinator only sequences
    these stages, records events, checkpoints state, and requests the report.
    """

    def __init__(
        self,
        config: RunConfig,
        qg_agents: list[QuestionGenerator],
        analyzer: Analyzer,
        critic: Critic,
        registry: DeterministicFunctionRegistry,
        synthesizer: Synthesizer,
        store: RunStore,
    ) -> None:
        self.config = config
        self.question_round = QuestionRound(qg_agents, config)
        self.analysis_round = AnalysisRound(
            analyzer,
            critic,
            registry,
            maximum_attempts=config.maximum_analysis_attempts,
        )
        self.stopping_policy = StoppingPolicy(config)
        self.synthesizer = synthesizer
        self.store = store

    def run(self, state: RunState) -> RunState:
        """Run complete steps until a stopping rule is satisfied, then report."""
        self.store.append("run_started", {"run_id": state.run_id, "config": asdict(self.config)})
        while state.termination_reason is None:
            self._run_step(state)
            self.store.checkpoint(state)
        self._write_final_report(state)
        return state

    def _run_step(self, state: RunState) -> None:
        """Sequence one question round, analysis round, and stopping decision."""
        state.step += 1
        snapshot = copy.deepcopy(state)
        try:
            record, candidates = self.question_round.run(snapshot, state.step)
        except NoValidCandidateError as error:
            state.candidates.extend(error.candidates)
            state.termination_reason = "no_valid_candidates"
            self.store.append("no_valid_candidates", {"step": state.step, "candidates": error.candidates})
            return
        state.candidates.extend(candidates)
        state.qa_pool.append(record)
        self.store.append("question_selected", record)
        self.analysis_round.run(state, record)
        self.store.append("analysis_terminal", record)
        self.stopping_policy.evaluate(state, record.question)

    def _write_final_report(self, state: RunState) -> None:
        """Synthesize, save, audit, and checkpoint the terminal report."""
        state.final_report = self.synthesizer.synthesize(state)
        report_path = self.store.run_dir / "report.md"
        report_path.write_text(state.final_report, encoding="utf-8")
        self.store.append("run_finished", {"reason": state.termination_reason, "report": str(report_path)})
        self.store.checkpoint(state)
