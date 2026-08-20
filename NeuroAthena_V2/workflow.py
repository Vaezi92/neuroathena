"""NeuroAthena V2 LangGraph: skills, MCP contract, bounded HITL, and evidence."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from analysis import AnalysisService, BASE_DIR, REPOSITORY_ROOT, get_metadata
from guardrails import evaluate_guardrails
from literature import build_queries, retrieve_all
from notebooklm_research import retrieve_notebooklm
from models import (
    AgentHypothesis,
    EvidenceAssessment,
    ExpertQuestion,
    HumanInterpretation,
    NeuroAthenaState,
    QuantitativeObservation,
    SearchQuery,
    ToolCall,
    VisualObservation,
)
from skills import artifact_review_skill, route_skills


DEFAULT_RESULT = REPOSITORY_ROOT / "Brain" / "Real" / "10ROI_perm_FT_thr25to75_concentration_WTM1-Train:0.5_mass:1steady_fields_high.mat"
GRAPH_VERSION = "2.0.0"
MAX_HITL_ROUNDS = 3
_SERVICES: dict[str, AnalysisService] = {}


def load_validate(state: NeuroAthenaState) -> dict[str, Any]:
    path = Path(state.source_file).resolve()
    metadata = get_metadata(path)
    if metadata["sample_count"] <= 0:
        raise ValueError("Dataset contains no valid samples")
    expected = {"velocity": "um/s", "permeability": "mm^2"}
    for field, unit in expected.items():
        if metadata["units"].get(field) != unit:
            raise ValueError(f"Unexpected {field} unit: {metadata['units'].get(field)}")
    return {"source_file": str(path), "dataset_metadata": metadata}


def analysis_planner(state: NeuroAthenaState) -> dict[str, Any]:
    return {"dataset_metadata": {**state.dataset_metadata, "planned_goal": state.requested_goal}}


def skill_router(state: NeuroAthenaState) -> dict[str, Any]:
    skills = route_skills(state.requested_goal)
    skills.append(artifact_review_skill())
    calls: list[ToolCall] = []
    seen: set[str] = set()
    for skill in skills:
        for call in skill.tool_calls:
            signature = json.dumps(call.model_dump(), sort_keys=True)
            if signature not in seen:
                calls.append(call)
                seen.add(signature)
    if len(calls) > 8:
        raise ValueError("Skill routing exceeded the maximum of eight tool calls")
    return {
        "selected_skills": [item.model_dump() for item in skills],
        "tool_calls": [item.model_dump() for item in calls],
    }


def mcp_tool_calls(state: NeuroAthenaState) -> dict[str, Any]:
    """Execute the exact interface exposed by mcp_server; no arbitrary dispatch."""
    service = AnalysisService(Path(state.source_file), state.run_id, BASE_DIR / "artifacts" / state.run_id)
    _SERVICES[state.run_id] = service
    artifacts = [service.execute(ToolCall.model_validate(raw).tool_name, ToolCall.model_validate(raw).parameters) for raw in state.tool_calls]
    return {"artifacts": [item.model_dump() for item in artifacts]}


def artifact_registry(state: NeuroAthenaState) -> dict[str, Any]:
    path = _SERVICES[state.run_id].write_registry()
    return {"dataset_metadata": {**state.dataset_metadata, "artifact_registry": str(path), "graph_version": GRAPH_VERSION}}


def observation_agent(state: NeuroAthenaState) -> dict[str, Any]:
    quantitative: list[QuantitativeObservation] = []
    visual: list[VisualObservation] = []
    for artifact in state.artifacts:
        summary = artifact["numerical_summary"]
        statistics = summary.get("statistics")
        if statistics:
            quantitative.append(
                QuantitativeObservation(
                    claim_id=f"observation-{uuid.uuid4().hex[:10]}",
                    text=f"The model output summarized by {artifact['artifact_id']} has the stored deterministic statistics shown in the evidence record.",
                    related_artifact_ids=[artifact["artifact_id"]],
                    numerical_evidence=statistics,
                )
            )
        if artifact.get("figure_path"):
            visual.append(
                VisualObservation(
                    claim_id=f"visual-{uuid.uuid4().hex[:10]}",
                    text=f"Figure {artifact['artifact_id']} visualizes the model-predicted field using the recorded parameters; biological meaning is not assigned.",
                    related_artifact_ids=[artifact["artifact_id"]],
                )
            )
    return {
        "quantitative_observations": [item.model_dump() for item in quantitative],
        "visual_observations": [item.model_dump() for item in visual],
    }


def question_generator(state: NeuroAthenaState) -> dict[str, Any]:
    round_number = state.hitl_round + 1
    figure_ids = [item["artifact_id"] for item in state.artifacts if item.get("figure_path")]
    if round_number == 1:
        text = "Which model-predicted spatial patterns appear scientifically meaningful, and which may be artifacts?"
    elif round_number == 2:
        text = "What alternative biological or preprocessing explanation should be prioritized for the unresolved pattern?"
    else:
        text = "What single unresolved issue would most change the interpretation or next experiment?"
    question = ExpertQuestion(
        question_id=f"question-{uuid.uuid4().hex[:10]}",
        round=round_number,
        text=text,
        related_artifact_ids=figure_ids,
    )
    return {"questions": state.questions + [question.model_dump()]}


def hitl_round(state: NeuroAthenaState) -> dict[str, Any]:
    question = ExpertQuestion.model_validate(state.questions[-1])
    answer = interrupt(
        {
            "round": question.round,
            "maximum_rounds": MAX_HITL_ROUNDS,
            "question": question.text,
            "related_artifact_ids": question.related_artifact_ids,
            "figures": [
                {"artifact_id": item["artifact_id"], "tool_name": item["tool_name"], "figure_path": item["figure_path"]}
                for item in state.artifacts
                if item.get("figure_path")
            ],
        }
    )
    human = HumanInterpretation(
        interpretation_id=f"human-{uuid.uuid4().hex[:10]}",
        round=question.round,
        verbatim_text=str(answer).strip() or "No interpretation provided.",
        related_question_id=question.question_id,
    )
    return {"human_interpretations": state.human_interpretations + [human.model_dump()], "hitl_round": question.round}


def uncertainty_assessment(state: NeuroAthenaState) -> dict[str, Any]:
    answer = state.human_interpretations[-1]["verbatim_text"].lower()
    explicit_resolution = any(phrase in answer for phrase in ("resolved", "sufficient", "no further", "finish", "stop"))
    return {"uncertainty_resolved": explicit_resolution or state.hitl_round >= MAX_HITL_ROUNDS}


def route_after_uncertainty(state: NeuroAthenaState) -> Literal["QuestionGenerator", "ClaimExtractor"]:
    return "ClaimExtractor" if state.uncertainty_resolved else "QuestionGenerator"


def claim_extractor(state: NeuroAthenaState) -> dict[str, Any]:
    figure_ids = [item["artifact_id"] for item in state.artifacts if item.get("figure_path")]
    hypotheses: list[AgentHypothesis] = []
    for interpretation in state.human_interpretations:
        hypotheses.append(
            AgentHypothesis(
                claim_id=f"hypothesis-{uuid.uuid4().hex[:10]}",
                text=(
                    "A possible explanation is consistent with the expert's recorded interpretation, "
                    "but the model output alone cannot establish mechanism or causality: "
                    f"{interpretation['verbatim_text']}"
                ),
                related_artifact_ids=figure_ids,
                human_interpretation_ids=[interpretation["interpretation_id"]],
            )
        )
    return {"hypotheses": [item.model_dump() for item in hypotheses]}


def literature_research(state: NeuroAthenaState) -> dict[str, Any]:
    queries: list[SearchQuery] = []
    for hypothesis, interpretation in zip(state.hypotheses, state.human_interpretations):
        queries.extend(build_queries(hypothesis["claim_id"], interpretation["verbatim_text"]))
    notebooklm_responses: list[dict[str, Any]] = []
    if state.notebooklm_notebook:
        evidence = retrieve_notebooklm(state.notebooklm_notebook, state.notebooklm_profile, queries)
        records = evidence.records
        notebooklm_responses = evidence.responses
    else:
        records = retrieve_all(queries, online=state.online_literature)
    return {
        "search_queries": [item.model_dump() for item in queries],
        "literature_records": [item.model_dump() for item in records],
        "notebooklm_responses": notebooklm_responses,
    }


def evidence_critic(state: NeuroAthenaState) -> dict[str, Any]:
    query_by_id = {item["query_id"]: item for item in state.search_queries}
    assessments: list[EvidenceAssessment] = []
    for hypothesis in state.hypotheses:
        grouped = {intent: [] for intent in ("support", "contradiction", "alternative", "limitation")}
        for record in state.literature_records:
            query = query_by_id.get(record["query_id"])
            if query and query["claim_id"] == hypothesis["claim_id"]:
                grouped[query["intent"]].append(record["record_id"])
        total = sum(len(items) for items in grouped.values())
        complete = all(grouped.values())
        strength = "moderate" if complete and total >= 8 else "weak" if total else "insufficient"
        rationale = (
            "All four retrieval intents returned candidate records; relevance still requires expert full-text review."
            if complete
            else "Evidence is incomplete; no biological conclusion should be drawn from the current retrieval."
        )
        assessments.append(
            EvidenceAssessment(
                claim_id=hypothesis["claim_id"],
                direct_artifact_ids=hypothesis["related_artifact_ids"],
                supporting_record_ids=grouped["support"],
                contradicting_record_ids=grouped["contradiction"],
                alternative_record_ids=grouped["alternative"],
                limitation_record_ids=grouped["limitation"],
                strength=strength,
                rationale=rationale,
            )
        )
    return {"evidence_assessments": [item.model_dump() for item in assessments]}


def guardrail_layer(state: NeuroAthenaState) -> dict[str, Any]:
    return {"guardrail_results": [item.model_dump() for item in evaluate_guardrails(state)]}


def scientific_summary(state: NeuroAthenaState) -> dict[str, Any]:
    passed = sum(item["passed"] for item in state.guardrail_results)
    summary = (
        f"NeuroAthena V2 run {state.run_id} executed {len(state.artifacts)} allow-listed analyses through "
        f"the MR-AIV MCP tool contract, completed {state.hitl_round} bounded HITL round(s), extracted "
        f"{len(state.hypotheses)} calibrated hypothesis record(s), and retrieved {len(state.literature_records)} "
        f"literature candidate(s). Guardrails passed: {passed}/{len(state.guardrail_results)}. "
        "All fields are model outputs; the report does not establish biological mechanism or causality."
    )
    run_dir = BASE_DIR / "artifacts" / state.run_id
    report_path = run_dir / "report.json"
    payload = state.model_dump()
    payload.update({"final_summary": summary, "graph_version": GRAPH_VERSION})
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    evidence_rows = [
        f"| {item['claim_id']} | {item['strength']} | {len(item['supporting_record_ids'])} | {len(item['contradicting_record_ids'])} | {len(item['alternative_record_ids'])} | {len(item['limitation_record_ids'])} |"
        for item in state.evidence_assessments
    ]
    markdown = "\n".join(
        [
            "# NeuroAthena V2 scientific report",
            "",
            summary,
            "",
            "## Evidence table",
            "",
            "| Claim | Strength | Support | Contradiction | Alternatives | Limitations |",
            "|---|---:|---:|---:|---:|---:|",
            *evidence_rows,
            "",
            "## Guardrails",
            "",
            *[f"- {'PASS' if item['passed'] else 'FAIL'} — {item['guardrail']}: {item['message']}" for item in state.guardrail_results],
        ]
    )
    (run_dir / "report.md").write_text(markdown, encoding="utf-8")
    return {"final_summary": summary, "report_path": str(report_path)}


def build_graph():
    builder = StateGraph(NeuroAthenaState)
    nodes = (
        ("LoadValidate", load_validate),
        ("AnalysisPlanner", analysis_planner),
        ("SkillRouter", skill_router),
        ("MCPToolCalls", mcp_tool_calls),
        ("ArtifactRegistry", artifact_registry),
        ("ObservationAgent", observation_agent),
        ("QuestionGenerator", question_generator),
        ("HITLRound", hitl_round),
        ("UncertaintyAssessment", uncertainty_assessment),
        ("ClaimExtractor", claim_extractor),
        ("LiteratureResearch", literature_research),
        ("EvidenceCritic", evidence_critic),
        ("Guardrails", guardrail_layer),
        ("ScientificSummary", scientific_summary),
    )
    for name, function in nodes:
        builder.add_node(name, function)
    linear = [name for name, _ in nodes[:9]]
    builder.add_edge(START, linear[0])
    for current, following in zip(linear, linear[1:]):
        builder.add_edge(current, following)
    builder.add_conditional_edges("UncertaintyAssessment", route_after_uncertainty)
    tail = [name for name, _ in nodes[9:]]
    for current, following in zip(tail, tail[1:]):
        builder.add_edge(current, following)
    builder.add_edge("ScientificSummary", END)
    return builder.compile(checkpointer=InMemorySaver())


def format_hitl_prompt(payload: dict[str, Any], *, verbose: bool = False) -> str:
    """Render a calm human-facing prompt; retain raw state only in verbose mode."""
    if verbose:
        return json.dumps(payload, indent=2)
    return (
        f"\nExpert review — round {payload['round']} of {payload['maximum_rounds']}\n"
        f"{payload['question']}"
    )


def show_figures(payload: dict[str, Any]) -> int:
    """Open saved figures with the desktop viewer without blocking the workflow."""
    paths = [
        str(path)
        for item in payload.get("figures", [])
        if (path := Path(item.get("figure_path", ""))).is_file()
    ]
    if not paths:
        return 0
    command = ["open", *paths] if sys.platform == "darwin" else ["xdg-open", paths[0]]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return len(paths)


def main() -> None:
    parser = ArgumentParser(description="Run the NeuroAthena V2 scientific workflow.")
    parser.add_argument("mat_file", nargs="?", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--goal", default="Inspect velocity and permeability heterogeneity in the trained PINN result.")
    parser.add_argument("--auto-review", action="append", help="Automated HITL response; repeat up to three times.")
    parser.add_argument("--online-literature", action="store_true", help="Query Europe PMC after HITL (requires network).")
    parser.add_argument("--notebooklm-notebook", help="Canonical NotebookLM notebook ID for read-only MCP retrieval.")
    parser.add_argument("--notebooklm-profile", default="default", help="Local NotebookLM authentication profile.")
    parser.add_argument(
        "--no-show-figures",
        action="store_true",
        help="Do not open generated figures (useful for automated or headless runs).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print complete HITL JSON payloads and artifact paths.")
    args = parser.parse_args()
    run_id = uuid.uuid4().hex[:12]
    graph = build_graph()
    config = {"configurable": {"thread_id": run_id}}
    result = graph.invoke(
        NeuroAthenaState(
            run_id=run_id,
            source_file=str(args.mat_file),
            requested_goal=args.goal,
            online_literature=args.online_literature,
            notebooklm_notebook=args.notebooklm_notebook,
            notebooklm_profile=args.notebooklm_profile,
        ),
        config=config,
    )
    auto_answers = iter(args.auto_review or [])
    figures_shown = False
    while "__interrupt__" in result:
        prompt = result["__interrupt__"][0].value
        if not args.no_show_figures and not figures_shown:
            try:
                figures_shown = show_figures(prompt) > 0
            except OSError as error:
                print(f"Could not open figures automatically: {error}")
        print(format_hitl_prompt(prompt, verbose=args.verbose))
        try:
            answer = next(auto_answers)
        except StopIteration:
            answer = input("Your interpretation (type 'resolved' when done): ")
        result = graph.invoke(Command(resume=answer), config=config)
    print("\n" + result["final_summary"])
    print(f"Report: {result['report_path']}")


if __name__ == "__main__":
    main()
