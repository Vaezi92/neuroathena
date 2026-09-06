from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neuroathena_v4.agents import MarkdownSynthesizer, RuleBasedAnalyzer, RuleBasedCritic
from neuroathena_v4.coordinator import Coordinator
from neuroathena_v4.dataset import initialize_dataset
from neuroathena_v4.models import CandidateQuestion, CriticReview, PERSPECTIVES, RunConfig, RunState, Score, Verdict
from neuroathena_v4.live_agents import GradeItem, GradeSet, NotebookOpenAIQG, ProposalSet, ProposedItem
from neuroathena_v4.registry import default_registry
from neuroathena_v4.scoring import aggregate_candidate, find_duplicate, select_best
from neuroathena_v4.store import RunStore


class FixedQG:
    def __init__(self, agent_id: str, scores: Score):
        self.agent_id = agent_id
        self.perspective = agent_id
        self.scores = scores

    def propose(self, snapshot: RunState, step: int) -> list[CandidateQuestion]:
        return [
            CandidateQuestion(
                candidate_id=f"{step}-{self.agent_id}-{index}", step=step,
                text=f"Investigate unique quantity {step} {self.agent_id} {index}", author=self.agent_id,
                rationale={"feasibility": "f", "novelty": "n", "importance": "i"},
            )
            for index in range(3)
        ]

    def grade(self, snapshot: RunState, anonymized: list[CandidateQuestion]) -> dict[str, Score]:
        if any(item.author != "anonymous" for item in anonymized):
            raise AssertionError("Authorship was not anonymized")
        return {item.candidate_id: self.scores for item in anonymized}


class ScoringTests(unittest.TestCase):
    def test_two_stage_weighted_averaging(self) -> None:
        candidate = CandidateQuestion("c", 1, "question", "mri", {})
        candidate.grades = {
            "mri": Score(1, 0, 0.5),
            "fluid_dynamics": Score(0, 1, 0.5),
        }
        config = RunConfig(
            aspect_weights={"feasibility": 2, "novelty": 1, "importance": 1},
            grader_weights={"mri": 3, "fluid_dynamics": 1, "applied_mathematics": 0, "neuroscience": 0},
        )
        aggregate_candidate(candidate, config)
        self.assertEqual(candidate.aggregate, Score(0.75, 0.25, 0.5))
        self.assertEqual(candidate.overall_score, 0.5625)

    def test_deterministic_tie_break_prefers_importance(self) -> None:
        a = CandidateQuestion("a", 1, "a", "mri", {}, aggregate=Score(0.5, 0.5, 0.8), overall_score=0.6)
        b = CandidateQuestion("b", 1, "b", "mri", {}, aggregate=Score(0.7, 0.5, 0.6), overall_score=0.6)
        self.assertIs(select_best([a, b]), a)

    def test_duplicate_detection(self) -> None:
        known = {"q1": "What is the spatial correlation between velocity and permeability?"}
        duplicate = "What is the correlation between permeability and spatial velocity?"
        self.assertEqual(find_duplicate(duplicate, known, 0.75), "q1")

    def test_similar_templates_for_different_fields_are_not_duplicates(self) -> None:
        known = {"q1": "How heterogeneous is velocity across coordinate bounded spatial regions?"}
        candidate = "How heterogeneous is permeability across coordinate bounded spatial regions?"
        self.assertIsNone(find_duplicate(candidate, known, 0.75))


class DatasetTests(unittest.TestCase):
    def test_dataset_initialization_records_declared_and_observed_metadata(self) -> None:
        try:
            import numpy as np
            from scipy.io import savemat
        except ImportError:
            self.skipTest("numpy/scipy not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.mat"
            coordinates = np.array([[240, 0.0, 0.0, 0.0], [240, 0.1, 0.1, 0.1]])
            fields = {name: np.ones((2, 1)) for name in ("u_mm_s", "v_mm_s", "w_mm_s", "K_m2")}
            savemat(path, {"txyz_smm": coordinates, **fields})
            metadata, limitations = initialize_dataset(path)
        self.assertEqual(metadata["observed"]["time_point_count"], 1)
        self.assertTrue(metadata["observed"]["spatial_resolution_matches_declaration"])
        self.assertEqual(metadata["acquisition"]["declared_temporal_resolution_minutes"], 1.0)
        self.assertTrue(any("one unique time" in item for item in limitations))


class LiveAgentWiringTests(unittest.TestCase):
    @patch("neuroathena_v4.live_agents._parse")
    @patch("neuroathena_v4.live_agents.ask_notebook")
    def test_qg_uses_notebook_for_proposal_and_anonymous_grading(self, ask, parse) -> None:
        ask.return_value = {"answer": "Grounded guidance", "references": [{"source_id": "s1"}]}
        proposed = ProposedItem(
            text="What deterministic quantity would test the inferred velocity pattern?",
            feasibility=0.8, novelty=0.7, importance=0.9,
            feasibility_reason="Available", novelty_reason="Not analyzed", importance_reason="Material",
        )
        parse.side_effect = [
            ProposalSet(questions=[proposed, proposed.model_copy(update={"text": proposed.text + " A"}), proposed.model_copy(update={"text": proposed.text + " B"})]),
            GradeSet(grades=[GradeItem(candidate_id=f"s1-mri-{index}", feasibility=.8, novelty=.7, importance=.9) for index in range(1, 4)]),
        ]
        agent = NotebookOpenAIQG("mri", "mri", "notebook-1", profile="default", model="test-model")
        candidates = agent.propose(RunState("run"), 1)
        anonymous = [CandidateQuestion(**{**item.__dict__, "author": "anonymous"}) for item in candidates]
        grades = agent.grade(RunState("run"), anonymous)
        self.assertEqual(ask.call_count, 2)
        self.assertEqual(len(candidates), 3)
        self.assertEqual(len(grades), 3)
        self.assertEqual(candidates[0].evidence["notebook_id"], "notebook-1")


class CoordinatorTests(unittest.TestCase):
    def build(self, directory: str, score: Score, **config):
        registry = default_registry()
        qgs = [FixedQG(name, score) for name in PERSPECTIVES]
        return Coordinator(
            RunConfig(**config), qgs, RuleBasedAnalyzer(), RuleBasedCritic(set(registry.capabilities())),
            registry, MarkdownSynthesizer(), RunStore(Path(directory)),
        )

    def test_stops_after_five_consecutive_low_score_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coordinator = self.build(
                directory, Score(0.1, 0.2, 0.24), maximum_steps=10,
                consecutive_low_score_limit=5, duplicate_similarity_threshold=1.0,
            )
            state = coordinator.run(RunState("low-score-run"))
            self.assertEqual(state.termination_reason, "scientific_saturation")
            self.assertEqual(state.step, 5)
            self.assertEqual(state.consecutive_low_score_steps, 5)

    def test_maximum_step_stop_and_persistent_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coordinator = self.build(
                directory, Score(0.9, 0.8, 0.7), maximum_steps=2,
                duplicate_similarity_threshold=1.0,
            )
            state = coordinator.run(RunState("maximum-run", fixed_metadata={"kind": "test"}))
            self.assertEqual(state.termination_reason, "maximum_steps")
            self.assertEqual(len(state.qa_pool), 2)
            self.assertTrue(all(record.final_status == "answered" for record in state.qa_pool))
            self.assertEqual(len(state.analyzed_data), 2)
            self.assertTrue((Path(directory) / "state.json").is_file())
            self.assertTrue((Path(directory) / "events.jsonl").is_file())
            self.assertIn("Termination: `maximum_steps`", (Path(directory) / "report.md").read_text())

    def test_all_graders_contribute_to_selected_question(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coordinator = self.build(directory, Score(0.6, 0.7, 0.8), maximum_steps=1, duplicate_similarity_threshold=1.0)
            state = coordinator.run(RunState("grades-run"))
            selected = next(item for item in state.candidates if item.status == "selected")
            self.assertEqual(set(selected.grades), set(PERSPECTIVES))
            self.assertAlmostEqual(selected.overall_score, 0.7)

    def test_critic_can_request_bounded_reanalysis(self) -> None:
        class ReviseOnceCritic(RuleBasedCritic):
            def review_result(self, question, result, snapshot):
                if result.analysis_id.endswith("attempt-1"):
                    return CriticReview(
                        question.question_id, result.analysis_id, Verdict.REVISE,
                        False, True, False, ["First result is incomplete."], True,
                        "Run the analysis again with the critic feedback.", None,
                    )
                return super().review_result(question, result, snapshot)

        with tempfile.TemporaryDirectory() as directory:
            registry = default_registry()
            qgs = [FixedQG(name, Score(0.8, 0.8, 0.8)) for name in PERSPECTIVES]
            coordinator = Coordinator(
                RunConfig(maximum_steps=1), qgs, RuleBasedAnalyzer(),
                ReviseOnceCritic(set(registry.capabilities())), registry,
                MarkdownSynthesizer(), RunStore(Path(directory)),
            )
            state = coordinator.run(RunState("revision-run"))
            self.assertEqual(len(state.qa_pool[0].attempts), 2)
            self.assertEqual(state.qa_pool[0].final_status, "answered")


if __name__ == "__main__":
    unittest.main()
