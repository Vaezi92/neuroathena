"""NeuroAthena V3: contextual HITL, evidence retrieval, synthesis, and HTML reporting."""

from __future__ import annotations

import json
import subprocess
import sys
import time
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
from notebooklm_research import retrieve_notebooks_parallel
from llm_synthesis import generate_dynamic_questions, route_notebooks_with_openai, synthesize_report
from notebook_router import deterministic_routes, load_registry
from reporting import render_html_report
from models import (
    AgentHypothesis,
    EvidenceAssessment,
    ExpertQuestion,
    HumanInterpretation,
    NeuroAthenaState,
    QuantitativeObservation,
    SearchQuery,
    SpecialistNotebook,
    ToolCall,
    VisualObservation,
)
from skills import artifact_review_skill, route_skills


DEFAULT_RESULT = REPOSITORY_ROOT / "Brain" / "Real" / "10ROI_perm_FT_thr25to75_concentration_WTM1-Train:0.5_mass:1steady_fields_high.mat"
DEFAULT_NOTEBOOK_REGISTRY = BASE_DIR / "notebooks.json"
GRAPH_VERSION = "3.0.0"
MAX_HITL_ROUNDS = 3
_SERVICES: dict[str, AnalysisService] = {}


def progress(state: NeuroAthenaState, message: str) -> None:
    if state.show_progress:
        print(f"NeuroAthena · {message}", flush=True)


def load_validate(state: NeuroAthenaState) -> dict[str, Any]:
    progress(state, "Validating the MR-AIV dataset…")
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
    artifacts = []
    for index, raw in enumerate(state.tool_calls, 1):
        call = ToolCall.model_validate(raw)
        progress(state, f"Running deterministic analysis {index}/{len(state.tool_calls)}: {call.tool_name}")
        artifacts.append(service.execute(call.tool_name, call.parameters))
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
        prompts = [
            ("scientific_explanation", "Please explain what you observe in these model-predicted results and which patterns you consider scientifically meaningful or potentially artifactual."),
            ("fluid_dynamics", "From a fluid-dynamics perspective, what additional explanation or justification do you have for the observed velocity and permeability patterns?"),
            ("biological_anatomical", "From a biological or anatomical perspective, how do you interpret these patterns, while accounting for the available anatomical metadata and its limitations?"),
        ]
    else:
        if state.use_openai:
            progress(state, f"Synthesizing contextual question(s) for review round {round_number}…")
            generated, call = generate_dynamic_questions(state, round_number)
            texts = generated.questions
            llm_calls = state.llm_calls + [call]
        else:
            latest = state.human_interpretations[-1]["verbatim_text"][:180]
            evidence_count = len(state.literature_records)
            texts = [f"Considering your statement “{latest}” and the {evidence_count} retrieved evidence record(s), what unresolved assumption would most change the interpretation or next experiment?"]
            llm_calls = state.llm_calls
        prompts = [("contextual_followup", text) for text in texts]
    questions = [
        ExpertQuestion(
            question_id=f"question-{uuid.uuid4().hex[:10]}",
            round=round_number,
            text=text,
            perspective=perspective,
            related_artifact_ids=figure_ids,
        ).model_dump()
        for perspective, text in prompts
    ]
    result: dict[str, Any] = {"questions": state.questions + questions}
    if round_number > 1:
        result["llm_calls"] = llm_calls
    return result


def hitl_round(state: NeuroAthenaState) -> dict[str, Any]:
    round_number = state.hitl_round + 1
    questions = [ExpertQuestion.model_validate(item) for item in state.questions if item["round"] == round_number]
    answer = interrupt(
        {
            "round": round_number,
            "maximum_rounds": MAX_HITL_ROUNDS,
            "questions": [item.model_dump() for item in questions],
            "related_artifact_ids": questions[0].related_artifact_ids,
            "figures": [
                {"artifact_id": item["artifact_id"], "tool_name": item["tool_name"], "figure_path": item["figure_path"]}
                for item in state.artifacts
                if item.get("figure_path")
            ],
        }
    )
    raw_answers = answer.get("answers", []) if isinstance(answer, dict) else [str(answer)]
    raw_answers = list(raw_answers) + ["No interpretation provided."] * max(0, len(questions) - len(raw_answers))
    humans = [
        HumanInterpretation(
            interpretation_id=f"human-{uuid.uuid4().hex[:10]}",
            round=round_number,
            verbatim_text=str(raw).strip() or "No interpretation provided.",
            related_question_id=question.question_id,
        ).model_dump()
        for question, raw in zip(questions, raw_answers)
    ]
    return {"human_interpretations": state.human_interpretations + humans, "hitl_round": round_number}


def uncertainty_assessment(state: NeuroAthenaState) -> dict[str, Any]:
    answer = " ".join(item["verbatim_text"].lower() for item in state.human_interpretations if item["round"] == state.hitl_round)
    explicit_resolution = any(phrase in answer for phrase in ("resolved", "sufficient", "no further", "no more", "finish", "stop"))
    return {"uncertainty_resolved": explicit_resolution or state.hitl_round >= MAX_HITL_ROUNDS}


def route_after_uncertainty(state: NeuroAthenaState) -> Literal["QuestionGenerator", "ClaimExtractor"]:
    return "ClaimExtractor" if state.uncertainty_resolved else "QuestionGenerator"


def claim_extractor(state: NeuroAthenaState) -> list[AgentHypothesis]:
    figure_ids = [item["artifact_id"] for item in state.artifacts if item.get("figure_path")]
    hypotheses = [AgentHypothesis.model_validate(item) for item in state.hypotheses]
    processed = {item for hypothesis in hypotheses for item in hypothesis.human_interpretation_ids}
    for interpretation in state.human_interpretations:
        if interpretation["interpretation_id"] in processed:
            continue
        normalized = interpretation["verbatim_text"].strip().lower().rstrip(".")
        if normalized in {"", "no interpretation provided", "resolved", "sufficient", "no more explanations", "no further explanation"}:
            continue
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
    return hypotheses


def context_research(state: NeuroAthenaState) -> dict[str, Any]:
    hypotheses = claim_extractor(state)
    previous_claim_ids = {item["claim_id"] for item in state.hypotheses}
    new_hypotheses = [item for item in hypotheses if item.claim_id not in previous_claim_ids]
    interpretation_by_id = {item["interpretation_id"]: item for item in state.human_interpretations}
    queries: list[SearchQuery] = []
    for hypothesis in new_hypotheses:
        interpretation = interpretation_by_id[hypothesis.human_interpretation_ids[0]]
        queries.extend(build_queries(hypothesis.claim_id, interpretation["verbatim_text"]))
    notebooklm_responses: list[dict[str, Any]] = []
    routes = []
    llm_calls = state.llm_calls
    if state.notebooklm_notebook:
        routes = [{"query_id": item.query_id, "notebook_ids": [state.notebooklm_notebook], "reason": "Explicit single-notebook override."} for item in queries]
    elif state.notebook_registry and queries:
        notebooks = [SpecialistNotebook.model_validate(item) for item in state.notebook_registry]
        if state.use_openai:
            progress(state, f"Routing {len(queries)} new evidence queries across specialist notebooks…")
            planned_routes, call = route_notebooks_with_openai(state, queries, notebooks)
            llm_calls = state.llm_calls + [call]
        else:
            planned_routes = deterministic_routes(queries, notebooks)
        routes = [item.model_dump() for item in planned_routes]
    if routes:
        query_by_id = {item.query_id: item for item in queries}
        grouped: dict[str, list[SearchQuery]] = {}
        for route in routes:
            for notebook_id in route["notebook_ids"]:
                grouped.setdefault(notebook_id, []).append(query_by_id[route["query_id"]])
        records = []
        title_by_id = {item["notebook_id"]: item["title"] for item in state.notebook_registry}
        progress(state, f"Searching {len(grouped)} specialist notebook(s) in parallel (maximum {state.notebook_concurrency})…")
        started = time.monotonic()
        evidence = retrieve_notebooks_parallel(
            grouped,
            state.notebooklm_profile,
            maximum_concurrency=state.notebook_concurrency,
            progress_callback=lambda notebook_id, query_index, query_total, query: progress(
                state,
                f"{title_by_id.get(notebook_id, notebook_id)}: query {query_index}/{query_total} ({query.intent})…",
            ),
        )
        records.extend(evidence.records)
        notebooklm_responses.extend(evidence.responses)
        progress(
            state,
            f"Parallel retrieval finished in {time.monotonic() - started:.0f}s: "
            f"{len(evidence.records)} cited passage(s), {len(evidence.failures)} failed query/queries.",
        )
    else:
        records = retrieve_all(queries, online=state.online_literature)
        evidence = None
    partial = {
        "hypotheses": [item.model_dump() for item in hypotheses],
        "search_queries": state.search_queries + [item.model_dump() for item in queries],
        "literature_records": state.literature_records + [item.model_dump() for item in records],
        "notebooklm_responses": state.notebooklm_responses + notebooklm_responses,
        "notebook_routes": state.notebook_routes + routes,
        "llm_calls": llm_calls,
        "retrieval_failures": state.retrieval_failures + (evidence.failures if evidence else []),
    }
    refreshed = state.model_copy(update=partial)
    partial["evidence_assessments"] = evidence_critic(refreshed)["evidence_assessments"]
    return partial


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


def scientific_synthesizer(state: NeuroAthenaState) -> dict[str, Any]:
    if state.use_openai:
        progress(state, "Synthesizing the evidence-grounded scientific interpretation…")
        synthesis, call = synthesize_report(state)
        return {"final_synthesis": synthesis.model_dump(), "llm_calls": state.llm_calls + [call]}
    return {
        "final_synthesis": {
            "executive_summary": "Offline run: expert explanations and evidence are recorded, but no OpenAI synthesis was requested.",
            "integrated_interpretation": "Review the verbatim expert explanations and cited evidence below.",
            "fluid_dynamics_interpretation": "Not synthesized in offline mode.",
            "biological_anatomical_interpretation": "Not synthesized in offline mode.",
            "alternative_explanations": [], "limitations": ["OpenAI synthesis disabled."],
            "recommended_next_experiments": [], "citation_record_ids": [],
        }
    }


def scientific_report(state: NeuroAthenaState) -> dict[str, Any]:
    progress(state, "Rendering the self-contained HTML report…")
    passed = sum(item["passed"] for item in state.guardrail_results)
    summary = (
        f"NeuroAthena V3 run {state.run_id} executed {len(state.artifacts)} allow-listed analyses through "
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
    html_path = render_html_report(state, run_dir / "report.html")
    return {"final_summary": summary, "report_path": str(html_path)}


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
        ("ContextResearch", context_research),
        ("ScientificSynthesizer", scientific_synthesizer),
        ("Guardrails", guardrail_layer),
        ("ScientificReport", scientific_report),
    )
    for name, function in nodes:
        builder.add_node(name, function)
    linear = [name for name, _ in nodes[:8]]
    builder.add_edge(START, linear[0])
    for current, following in zip(linear, linear[1:]):
        builder.add_edge(current, following)
    builder.add_edge("HITLRound", "ContextResearch")
    builder.add_edge("ContextResearch", "UncertaintyAssessment")
    builder.add_conditional_edges("UncertaintyAssessment", route_after_uncertainty, {"QuestionGenerator": "QuestionGenerator", "ClaimExtractor": "ScientificSynthesizer"})
    tail = ["ScientificSynthesizer", "Guardrails", "ScientificReport"]
    for current, following in zip(tail, tail[1:]):
        builder.add_edge(current, following)
    builder.add_edge("ScientificReport", END)
    return builder.compile(checkpointer=InMemorySaver())


def format_hitl_prompt(payload: dict[str, Any], *, verbose: bool = False, include_questions: bool = True) -> str:
    """Render a calm human-facing prompt; retain raw state only in verbose mode."""
    if verbose:
        return json.dumps(payload, indent=2)
    header = f"\nExpert review — round {payload['round']} of {payload['maximum_rounds']}"
    if not include_questions:
        count = len(payload["questions"])
        return f"{header}\n{count} question{'s' if count != 1 else ''} will be asked one at a time."
    return header + "\n" + "\n".join(f"{index}. {item['text']}" for index, item in enumerate(payload['questions'], 1))


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


def open_html_report(path: str) -> None:
    """Open the completed report in the default browser without blocking Python."""
    command = ["open", path] if sys.platform == "darwin" else ["xdg-open", path]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    parser = ArgumentParser(description="Run the NeuroAthena V3 scientific workflow.")
    parser.add_argument("mat_file", nargs="?", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--goal", default="Inspect velocity and permeability heterogeneity in the trained PINN result.")
    parser.add_argument("--auto-review", action="append", help="Automated HITL response; repeat up to three times.")
    parser.add_argument("--online-literature", action="store_true", help="Query Europe PMC after HITL (requires network).")
    parser.add_argument("--notebooklm-notebook", help="Canonical NotebookLM notebook ID for read-only MCP retrieval.")
    parser.add_argument("--notebooklm-profile", default="default", help="Local NotebookLM authentication profile.")
    parser.add_argument("--notebook-registry", type=Path, default=DEFAULT_NOTEBOOK_REGISTRY, help="Allow-listed specialist notebook registry.")
    parser.add_argument("--no-notebooklm", action="store_true", help="Disable specialist NotebookLM retrieval.")
    parser.add_argument("--notebook-concurrency", type=int, choices=range(1, 6), default=3, help="Maximum specialist notebooks searched concurrently (1–5).")
    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI model used for dynamic questions and synthesis.")
    parser.add_argument("--no-openai", action="store_true", help="Disable OpenAI dynamic questions and synthesis.")
    parser.add_argument("--no-open-report", action="store_true", help="Do not open report.html after completion.")
    parser.add_argument(
        "--no-show-figures",
        action="store_true",
        help="Do not open generated figures (useful for automated or headless runs).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print complete HITL JSON payloads and artifact paths.")
    args = parser.parse_args()
    notebook_registry = [] if args.no_notebooklm else [item.model_dump() for item in load_registry(args.notebook_registry)]
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
            notebook_registry=notebook_registry,
            notebook_concurrency=args.notebook_concurrency,
            use_openai=not args.no_openai,
            openai_model=args.model,
            show_progress=True,
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
        print(format_hitl_prompt(prompt, verbose=args.verbose, include_questions=args.verbose))
        answers = []
        for question in prompt["questions"]:
            try:
                answer = next(auto_answers)
            except StopIteration:
                print("\n" + question["text"])
                answer = input("Your explanation (include 'resolved' when sufficient): ")
            answers.append(answer)
        result = graph.invoke(Command(resume={"answers": answers}), config=config)
    print("\n" + result["final_summary"])
    print(f"Report: {result['report_path']}")
    if not args.no_open_report:
        try:
            open_html_report(result["report_path"])
        except OSError as error:
            print(f"Could not open the HTML report automatically: {error}")


if __name__ == "__main__":
    main()
