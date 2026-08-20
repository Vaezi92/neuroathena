"""Unit and end-to-end tests for NeuroAthena V2."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langgraph.types import Command

from analysis import AnalysisService, get_metadata
from models import CoordinateROI, NeuroAthenaState, SearchQuery
from notebooklm_research import READ_ONLY_TOOLS, records_from_payload
from skills import route_skills
from workflow import DEFAULT_RESULT, build_graph, format_hitl_prompt, show_figures


class V2UnitTests(unittest.TestCase):
    def test_roi_rejects_reversed_bounds(self) -> None:
        with self.assertRaises(ValueError):
            CoordinateROI(name="invalid", x_mm=(1, 0), y_mm=(0, 1), z_mm=(0, 1))

    def test_skill_router_is_allow_listed(self) -> None:
        selected = route_skills("Inspect velocity and permeability")
        self.assertEqual({item.skill_name for item in selected}, {"inspect_velocity_field", "inspect_permeability_field"})
        self.assertTrue(all(call.tool_name.startswith("mraiv.") for item in selected for call in item.tool_calls))

    def test_default_hitl_prompt_hides_structured_logs(self) -> None:
        payload = {
            "round": 2,
            "maximum_rounds": 3,
            "question": "What remains uncertain?",
            "figures": [{"figure_path": "/private/result.png"}],
        }
        concise = format_hitl_prompt(payload)
        self.assertIn("round 2 of 3", concise)
        self.assertIn("What remains uncertain?", concise)
        self.assertNotIn("figure_path", concise)
        self.assertNotIn("/private/result.png", concise)
        self.assertIn("figure_path", format_hitl_prompt(payload, verbose=True))

    def test_notebooklm_adapter_is_read_only(self) -> None:
        self.assertEqual(READ_ONLY_TOOLS, {"server_info", "notebook_list", "source_list", "chat_ask"})

    def test_notebooklm_references_require_source_identity(self) -> None:
        query = SearchQuery(query_id="query-1", claim_id="claim-1", intent="support", query="brain flow")
        payload = {
            "references": [
                {"source_id": "source-1", "source_title": "MR-AIV paper", "cited_text": "Relevant result."},
                {"source_title": "Missing source identity", "cited_text": "Excluded."},
            ]
        }
        records = records_from_payload(query, "notebook-1", payload)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].provider, "notebooklm")
        self.assertEqual(records[0].source_id, "source-1")
        self.assertEqual(records[0].excerpt, "Relevant result.")

    def test_show_figures_opens_saved_images_non_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            figure = Path(directory) / "figure.png"
            figure.touch()
            payload = {"figures": [{"figure_path": str(figure)}, {"figure_path": "/missing.png"}]}
            with patch("workflow.subprocess.Popen") as process:
                self.assertEqual(show_figures(payload), 1)
            process.assert_called_once()

    @unittest.skipUnless(DEFAULT_RESULT.is_file(), "Local trained .mat result is not available")
    def test_metadata_has_source_provenance(self) -> None:
        metadata = get_metadata(DEFAULT_RESULT)
        self.assertEqual(len(metadata["source_sha256"]), 64)
        self.assertFalse(metadata["atlas_roi_labels_available"])
        self.assertIsNone(metadata["orientation"])

    @unittest.skipUnless(DEFAULT_RESULT.is_file(), "Local trained .mat result is not available")
    def test_unknown_tool_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = AnalysisService(DEFAULT_RESULT, "unit", Path(directory))
            with self.assertRaises(ValueError):
                service.execute("python.exec", {})


class V2WorkflowTests(unittest.TestCase):
    @unittest.skipUnless(DEFAULT_RESULT.is_file(), "Local trained .mat result is not available")
    def test_complete_offline_workflow(self) -> None:
        graph = build_graph()
        config = {"configurable": {"thread_id": "v2-unit-test"}}
        initial = NeuroAthenaState(
            run_id="v2-unit-test",
            source_file=str(DEFAULT_RESULT),
            requested_goal="Inspect velocity and permeability heterogeneity.",
        )
        paused = graph.invoke(initial, config=config)
        self.assertIn("__interrupt__", paused)
        finished = graph.invoke(Command(resume="Review is sufficient and resolved."), config=config)
        self.assertEqual(finished["hitl_round"], 1)
        self.assertEqual(len(finished["artifacts"]), 5)
        self.assertEqual(len(finished["search_queries"]), 4)
        self.assertEqual(finished["literature_records"], [])
        self.assertTrue(all(item["passed"] for item in finished["guardrail_results"]))
        self.assertTrue(Path(finished["report_path"]).is_file())
        self.assertTrue(Path(finished["report_path"]).with_suffix(".md").is_file())

    @unittest.skipUnless(DEFAULT_RESULT.is_file(), "Local trained .mat result is not available")
    def test_hitl_stops_after_three_rounds(self) -> None:
        graph = build_graph()
        config = {"configurable": {"thread_id": "v2-three-round-test"}}
        result = graph.invoke(
            NeuroAthenaState(
                run_id="v2-three-round-test",
                source_file=str(DEFAULT_RESULT),
                requested_goal="Inspect velocity.",
            ),
            config=config,
        )
        for answer in ("Uncertain.", "More review is needed.", "Still unresolved."):
            self.assertIn("__interrupt__", result)
            result = graph.invoke(Command(resume=answer), config=config)
        self.assertNotIn("__interrupt__", result)
        self.assertEqual(result["hitl_round"], 3)


if __name__ == "__main__":
    unittest.main()
