"""End-to-end smoke test for the NeuroAthena V1 LangGraph workflow."""

import unittest
from pathlib import Path

from langgraph.types import Command

from langgraph_workflow import BASE_DIR, DEFAULT_RESULT, NeuroAthenaState, build_graph


class WorkflowTest(unittest.TestCase):
    @unittest.skipUnless(DEFAULT_RESULT.is_file(), "Local trained .mat result is not available")
    def test_complete_hitl_workflow(self) -> None:
        graph = build_graph()
        config = {"configurable": {"thread_id": "unit-test"}}
        state = NeuroAthenaState(
            run_id="unit-test",
            source_file=str(DEFAULT_RESULT),
            requested_goal="Generate all four trained-result diagnostics.",
        )
        paused = graph.invoke(state, config=config)
        self.assertIn("__interrupt__", paused)
        finished = graph.invoke(Command(resume="Figures reviewed."), config=config)
        self.assertEqual(finished["hitl_round"], 1)
        self.assertEqual(len(finished["artifacts"]), 4)
        self.assertTrue(Path(finished["report_path"]).is_file())
        for artifact in finished["artifacts"]:
            self.assertTrue(Path(artifact["figure_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
