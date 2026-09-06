"""Behavioral requirements for deterministic function authoring and promotion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neuroathena_v4.agents import MarkdownSynthesizer
from neuroathena_v4.coordinator import Coordinator
from neuroathena_v4.dataset import initialize_dataset
from neuroathena_v4.deterministic_execution import CandidateFunctionValidator
from neuroathena_v4.live_agents import AnalyzerPlan, OpenAIAnalyzer
from neuroathena_v4.models import (
    AnalysisProposal, CandidateFunction, CandidateQuestion, CriticReview,
    RunConfig, RunState, Score, Verdict,
)
from neuroathena_v4.registry import default_registry
from neuroathena_v4.store import RunStore


MEAN_VELOCITY_SOURCE = """def analyze(data, parameters):
    values = np.asarray(data["u_mm_s"], dtype=float).reshape(-1)
    mean_velocity = float(np.mean(values) * 1000.0)
    return {
        "value": mean_velocity,
        "units": "um/s",
        "answer": "The deterministic mean x-velocity is %.3f um/s." % mean_velocity,
    }
"""


class ThreeQuestionGenerator:
    """Deterministic question generator used by vertical-slice tests."""

    agent_id = "fluid_dynamics"
    perspective = "fluid_dynamics"

    def propose(self, snapshot: RunState, step: int) -> list[CandidateQuestion]:
        """Propose exactly three distinct velocity questions."""
        return [
            CandidateQuestion(
                f"candidate-{index}", step,
                f"What deterministic x velocity statistic number {index} is supported?",
                self.agent_id,
                {"feasibility": "field exists", "novelty": "not computed", "importance": "tests flow"},
            )
            for index in range(1, 4)
        ]

    def grade(self, snapshot: RunState, anonymized: list[CandidateQuestion]) -> dict[str, Score]:
        """Give every anonymous candidate a complete grade."""
        if any(candidate.author != "anonymous" for candidate in anonymized):
            raise AssertionError("Candidate authorship was exposed")
        return {candidate.candidate_id: Score(0.95, 0.9, 0.85) for candidate in anonymized}


class AuthoredFunctionAnalyzer:
    """Offline analyzer that authors one deterministic mean computation."""

    def propose(self, question, snapshot, attempt, feedback):
        """Return a candidate function rather than a placeholder selection."""
        function = CandidateFunction(
            "mean_x_velocity", "0.1.0", "Compute mean inferred x-velocity.", MEAN_VELOCITY_SOURCE,
        )
        return AnalysisProposal(
            f"{question.question_id}-attempt-{attempt}", question.question_id,
            function.name, {}, "Compute the mean inferred x-velocity.", function,
        )


class CodeAwareCritic:
    """Test critic that separately reviews candidate code and its result."""

    def review_proposal(self, question, proposal, snapshot):
        """Accept only a proposal containing reviewable candidate source."""
        valid = proposal.candidate_function is not None and "def analyze" in proposal.candidate_function.source
        return CriticReview(
            question.question_id, proposal.analysis_id,
            Verdict.ACCEPT if valid else Verdict.REJECTED,
            False, valid, False, [] if valid else ["Candidate source is missing."],
        )

    def review_result(self, question, result, snapshot):
        """Accept only a deterministic isolated result in the expected units."""
        valid = (
            result.units == "um/s"
            and result.validation.get("deterministic") is True
            and result.provenance.get("execution_mode") == "isolated_subprocess"
        )
        return CriticReview(
            question.question_id, result.analysis_id,
            Verdict.ACCEPT if valid else Verdict.REJECTED,
            valid, True, valid, [] if valid else ["Result validation failed."],
        )


def initialized_state(dataset_path: Path, run_id: str) -> RunState:
    """Build shared state from a real temporary MATLAB file inspection."""
    metadata, limitations = initialize_dataset(dataset_path)
    return RunState(run_id, fixed_metadata=metadata, limitations=limitations)


class FunctionAuthoringTests(unittest.TestCase):
    """Verify the question-to-accepted-result vertical slice."""

    def create_dataset(self, directory: str) -> Path:
        """Create the smallest schema-valid deterministic fixture dataset."""
        import numpy as np
        from scipy.io import savemat

        path = Path(directory) / "fixture.mat"
        coordinates = np.array([[240, 0.0, 0.0, 0.0], [240, 0.1, 0.1, 0.1]])
        savemat(path, {
            "txyz_smm": coordinates,
            "u_mm_s": np.array([[0.001], [0.003]]),
            "v_mm_s": np.zeros((2, 1)),
            "w_mm_s": np.zeros((2, 1)),
            "K_m2": np.ones((2, 1)),
        })
        return path

    def run_vertical_slice(self, directory: str, analyzer) -> tuple[RunState, object]:
        """Run one complete question, authoring, review, execution, and admission step."""
        registry = default_registry()
        coordinator = Coordinator(
            RunConfig(maximum_steps=1), [ThreeQuestionGenerator()], analyzer,
            CodeAwareCritic(), registry, MarkdownSynthesizer(), RunStore(Path(directory) / "run"),
        )
        state = coordinator.run(initialized_state(self.create_dataset(directory), "vertical-slice"))
        return state, registry

    def test_offline_candidate_is_reviewed_executed_accepted_and_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state, registry = self.run_vertical_slice(directory, AuthoredFunctionAnalyzer())

        self.assertEqual(state.qa_pool[0].final_status, "answered")
        self.assertEqual(state.analyzed_data[0].value, 2.0)
        self.assertTrue(state.analyzed_data[0].validation["promoted"])
        self.assertEqual(registry.capabilities()["mean_x_velocity"]["trust"], "promoted")
        self.assertIn("candidate_function", state.qa_pool[0].attempts[0]["proposal"])
        self.assertIsNotNone(state.qa_pool[0].attempts[0]["pre_review"])

    @patch("neuroathena_v4.live_agents._parse")
    def test_mocked_openai_plan_authors_the_same_accepted_vertical_slice(self, parse) -> None:
        parse.return_value = AnalyzerPlan(
            mode="author", function_name="mean_x_velocity", function_version="0.1.0",
            function_description="Compute mean inferred x-velocity.",
            function_source=MEAN_VELOCITY_SOURCE, parameters_json="{}",
            intended_answer="Compute the mean inferred x-velocity.",
        )
        with tempfile.TemporaryDirectory() as directory:
            state, _ = self.run_vertical_slice(directory, OpenAIAnalyzer(default_registry().capabilities(), "mock-model"))

        self.assertEqual(parse.call_count, 1)
        self.assertEqual(state.qa_pool[0].final_status, "answered")
        self.assertEqual(len(state.analyzed_data), 1)

    def test_candidate_imports_are_rejected_before_execution(self) -> None:
        function = CandidateFunction(
            "unsafe", "0.1.0", "Unsafe candidate.",
            "def analyze(data, parameters):\n    import os\n    return {'value': 1, 'units': None, 'answer': 'x'}\n",
        )
        with self.assertRaisesRegex(ValueError, "prohibited syntax: Import"):
            CandidateFunctionValidator().validate(function)

    def test_numpy_file_io_is_rejected_before_execution(self) -> None:
        function = CandidateFunction(
            "unsafe_numpy", "0.1.0", "Unsafe NumPy candidate.",
            "def analyze(data, parameters):\n    np.save('leak.npy', data['u_mm_s'])\n    return {'value': 1, 'units': None, 'answer': 'x'}\n",
        )
        with self.assertRaisesRegex(ValueError, "non-allow-listed attribute: save"):
            CandidateFunctionValidator().validate(function)


if __name__ == "__main__":
    unittest.main()
