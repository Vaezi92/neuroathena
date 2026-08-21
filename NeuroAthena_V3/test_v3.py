"""Unit and end-to-end tests for NeuroAthena V3."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.types import Command

from analysis import AnalysisService, get_metadata
from models import CoordinateROI, NeuroAthenaState, SearchQuery
from notebooklm_research import NotebookLMEvidence, READ_ONLY_TOOLS, records_from_payload, retrieve_notebooks_parallel
from notebook_router import deterministic_routes, load_registry
from llm_synthesis import question_is_immediately_answerable
from reporting import render_html_report
from skills import route_skills
from workflow import DEFAULT_RESULT, build_graph, format_hitl_prompt, show_figures
from workflow import DEFAULT_NOTEBOOK_REGISTRY


class V3UnitTests(unittest.TestCase):
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
            "questions": [{"text": "What remains uncertain?"}],
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

    def test_specialist_registry_and_fallback_routing_are_allow_listed(self) -> None:
        notebooks = load_registry(DEFAULT_NOTEBOOK_REGISTRY)
        self.assertEqual(len(notebooks), 5)
        query = SearchQuery(query_id="query-flow", claim_id="claim-1", intent="alternative", query="velocity boundary and permeability transport")
        route = deterministic_routes([query], notebooks)[0]
        allowed = {item.notebook_id for item in notebooks}
        self.assertTrue(set(route.notebook_ids).issubset(allowed))
        selected_specialties = {item.specialty for item in notebooks if item.notebook_id in route.notebook_ids}
        self.assertIn("fluid_dynamics", selected_specialties)

    def test_dynamic_question_policy_rejects_analysis_assignments(self) -> None:
        self.assertTrue(question_is_immediately_answerable("Which explanation seems most plausible to you?"))
        self.assertFalse(question_is_immediately_answerable("Compute per-voxel maps and retrain the model."))

    def test_notebook_workers_run_with_bounded_parallelism(self) -> None:
        active = 0
        maximum_active = 0

        async def fake_retrieve(notebook, profile, queries, progress_callback=None, maximum_attempts=2):
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            return NotebookLMEvidence(records=[], responses=[{"notebook": notebook}], failures=[])

        query = SearchQuery(query_id="query-1", claim_id="claim-1", intent="support", query="test")
        assignments = {f"notebook-{index}": [query] for index in range(4)}
        with patch("notebooklm_research._retrieve", side_effect=fake_retrieve):
            evidence = retrieve_notebooks_parallel(assignments, "default", maximum_concurrency=2)
        self.assertEqual(len(evidence.responses), 4)
        self.assertEqual(maximum_active, 2)

    def test_notebook_session_failure_is_preserved_not_raised(self) -> None:
        query = SearchQuery(query_id="query-fail", claim_id="claim-1", intent="alternative", query="test")
        with patch("notebooklm_research._retrieve", side_effect=RuntimeError("temporary failure")):
            evidence = retrieve_notebooks_parallel({"notebook-1": [query]}, "default")
        self.assertEqual(evidence.records, [])
        self.assertEqual(evidence.failures[0]["query_id"], "query-fail")
        self.assertIn("temporary failure", evidence.failures[0]["error"])

    def test_show_figures_opens_saved_images_non_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            figure = Path(directory) / "figure.png"
            figure.touch()
            payload = {"figures": [{"figure_path": str(figure)}, {"figure_path": "/missing.png"}]}
            with patch("workflow.subprocess.Popen") as process:
                self.assertEqual(show_figures(payload), 1)
            process.assert_called_once()

    def test_html_report_escapes_user_text_and_links_notebook_citations(self) -> None:
        record_id = "literature-1"
        state = SimpleNamespace(
            run_id="html-test", requested_goal="Inspect <unsafe>", dataset_metadata={"source_sha256": "abc"},
            artifacts=[], human_interpretations=[{"round": 1, "verbatim_text": "<script>alert(1)</script>", "related_question_id": "q1"}],
            final_synthesis={"executive_summary": "Supported finding.", "integrated_interpretation": "Interpretation.",
                "fluid_dynamics_interpretation": "Flow.", "biological_anatomical_interpretation": "Biology.",
                "alternative_explanations": [], "limitations": [], "recommended_next_experiments": [],
                "citation_record_ids": [record_id]},
            literature_records=[{"record_id": record_id, "title": "Curated paper", "excerpt": "Relevant passage.",
                "url": "notebooklm://notebook/source", "source_id": "source-1"}],
            notebooklm_responses=[{"intent": "support", "answer": "Notebook result.", "query_id": "query-1",
                "citation_count": 1, "record_ids": [record_id]}], evidence_assessments=[], guardrail_results=[],
            notebook_registry=[{"notebook_id": "notebook-1", "title": "MRI agent"}],
            notebook_routes=[{"query_id": "query-1", "notebook_ids": ["notebook-1"], "reason": "MRI evidence."}],
            retrieval_failures=[],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = render_html_report(state, Path(directory) / "report.html")
            document = output.read_text(encoding="utf-8")
        self.assertNotIn("<script>", document)
        self.assertIn("&lt;script&gt;", document)
        self.assertIn("Notebook result.", document)
        self.assertIn("href='#literature-1'>[1]</a>", document)
        self.assertIn("Curated paper", document)
        self.assertIn("MRI agent", document)

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


class V3WorkflowTests(unittest.TestCase):
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
        first_prompt = paused["__interrupt__"][0].value
        self.assertEqual(len(first_prompt["questions"]), 3)
        self.assertEqual(
            [item["perspective"] for item in first_prompt["questions"]],
            ["scientific_explanation", "fluid_dynamics", "biological_anatomical"],
        )
        finished = graph.invoke(
            Command(resume={"answers": ["Pattern explanation.", "Fluid explanation.", "Biological explanation; resolved."]}),
            config=config,
        )
        self.assertEqual(finished["hitl_round"], 1)
        self.assertEqual(len(finished["human_interpretations"]), 3)
        self.assertEqual(len(finished["artifacts"]), 5)
        self.assertEqual(len(finished["search_queries"]), 12)
        self.assertEqual(finished["literature_records"], [])
        self.assertTrue(all(item["passed"] for item in finished["guardrail_results"]))
        self.assertTrue(Path(finished["report_path"]).is_file())
        self.assertEqual(Path(finished["report_path"]).suffix, ".html")
        report = Path(finished["report_path"]).read_text(encoding="utf-8")
        self.assertIn("Pattern explanation.", report)
        self.assertIn("data:image/png;base64,", report)

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
            prompt = result["__interrupt__"][0].value
            result = graph.invoke(Command(resume={"answers": [answer] * len(prompt["questions"])}), config=config)
        self.assertNotIn("__interrupt__", result)
        self.assertEqual(result["hitl_round"], 3)


if __name__ == "__main__":
    unittest.main()
