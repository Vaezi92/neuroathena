"""NeuroAthena V1: deterministic PINN-result analysis with LangGraph and HITL."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from deterministic_tools import (
    TOOL_VERSION,
    dataset_metadata,
    plot_permeability_distribution,
    plot_permeability_slices,
    plot_speed_slices,
    plot_velocity_distributions,
)


BASE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = BASE_DIR.parent
DEFAULT_RESULT = REPOSITORY_ROOT / "Brain" / "Real" / "10ROI_perm_FT_thr25to75_concentration_WTM1-Train:0.5_mass:1steady_fields_high.mat"
ToolName = Literal[
    "plot_velocity_distributions",
    "plot_permeability_distribution",
    "plot_speed_slices",
    "plot_permeability_slices",
]
ALLOWED_TOOLS = {
    "plot_velocity_distributions": plot_velocity_distributions,
    "plot_permeability_distribution": plot_permeability_distribution,
    "plot_speed_slices": plot_speed_slices,
    "plot_permeability_slices": plot_permeability_slices,
}


class ToolCall(BaseModel):
    tool_name: ToolName
    parameters: dict[str, Any] = Field(default_factory=dict)


class Artifact(BaseModel):
    artifact_id: str
    tool_name: ToolName
    tool_version: str
    source_file: str
    figure_path: str
    parameters: dict[str, Any]
    numerical_summary: dict[str, Any]


class Observation(BaseModel):
    text: str
    artifact_ids: list[str]
    kind: Literal["observation"] = "observation"


class HumanInterpretation(BaseModel):
    text: str
    kind: Literal["human_interpretation"] = "human_interpretation"


class NeuroAthenaState(BaseModel):
    run_id: str
    source_file: str
    requested_goal: str
    use_openai: bool = False
    dataset_metadata: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    human_interpretations: list[dict[str, Any]] = Field(default_factory=list)
    hitl_round: int = 0
    final_summary: str | None = None
    report_path: str | None = None


class LLMObservation(BaseModel):
    observation: str
    review_question: str


def load_data(state: NeuroAthenaState) -> dict[str, Any]:
    path = Path(state.source_file).resolve()
    metadata = dataset_metadata(path)
    return {"source_file": str(path), "dataset_metadata": metadata}


def validate_data(state: NeuroAthenaState) -> dict[str, Any]:
    metadata = state.dataset_metadata
    errors: list[str] = []
    if metadata.get("sample_count", 0) <= 0:
        errors.append("Dataset contains no valid samples")
    if metadata.get("units", {}).get("velocity") != "um/s":
        errors.append("Velocity conversion did not produce um/s")
    if metadata.get("units", {}).get("permeability") != "mm^2":
        errors.append("Permeability conversion did not produce mm^2")
    if errors:
        raise ValueError("; ".join(errors))
    return {"validation_errors": []}


def analysis_planner(state: NeuroAthenaState) -> dict[str, Any]:
    # V1 always runs the complete, fixed diagnostic suite requested by the user.
    calls = [ToolCall(tool_name=name) for name in ALLOWED_TOOLS]
    return {"tool_calls": [call.model_dump() for call in calls]}


def tool_executor(state: NeuroAthenaState) -> dict[str, Any]:
    if len(state.tool_calls) != len(ALLOWED_TOOLS):
        raise ValueError(f"Expected exactly {len(ALLOWED_TOOLS)} allow-listed tool calls")
    artifact_dir = BASE_DIR / "artifacts" / state.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for raw_call in state.tool_calls:
        call = ToolCall.model_validate(raw_call)
        function = ALLOWED_TOOLS.get(call.tool_name)
        if function is None:
            raise ValueError(f"Tool is not allowed: {call.tool_name}")
        output = artifact_dir / f"{call.tool_name}.png"
        numerical_summary = function(Path(state.source_file), output, **call.parameters)
        artifact = Artifact(
            artifact_id=f"artifact-{uuid.uuid4().hex[:10]}",
            tool_name=call.tool_name,
            tool_version=TOOL_VERSION,
            source_file=state.source_file,
            figure_path=str(output.resolve()),
            parameters=call.parameters,
            numerical_summary=numerical_summary,
        )
        artifacts.append(artifact.model_dump())
    return {"artifacts": artifacts}


def _deterministic_observation(artifacts: list[Artifact]) -> LLMObservation:
    by_tool = {artifact.tool_name: artifact for artifact in artifacts}
    velocity = by_tool["plot_velocity_distributions"].numerical_summary["statistics"]
    permeability = by_tool["plot_permeability_distribution"].numerical_summary["statistics"]
    speed_planes = by_tool["plot_speed_slices"].numerical_summary["plane_locations_mm"]
    permeability_planes = by_tool["plot_permeability_slices"].numerical_summary["plane_locations_mm"]
    ids = ", ".join(artifact.artifact_id for artifact in artifacts)
    return LLMObservation(
        observation=(
            f"Artifacts {ids} report the trained PINN fields. Median speed is "
            f"{velocity['speed']['median']:.4g} µm/s and median permeability is "
            f"{permeability['median']:.4g} mm². Speed slices use central planes "
            f"{speed_planes}; permeability slices use central planes {permeability_planes}. "
            "These are numerical and visual observations only; no biological interpretation is assigned."
        ),
        review_question=(
            "Do the four figures—velocity distributions, permeability distribution, speed slices, "
            "and permeability slices—look scientifically and anatomically reasonable?"
        ),
    )


def observation_node(state: NeuroAthenaState) -> dict[str, Any]:
    artifacts = [Artifact.model_validate(item) for item in state.artifacts]
    drafted = _deterministic_observation(artifacts)
    if state.use_openai:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("--use-openai requires OPENAI_API_KEY")
        from openai import OpenAI

        response = OpenAI().responses.parse(
            model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            input=[
                {
                    "role": "system",
                    "content": (
                        "Write one neutral quantitative observation and one review question. Reference the "
                        "artifact IDs. Do not infer biology, causality, anatomy, or experimental validity."
                    ),
                },
                {"role": "user", "content": json.dumps([item.model_dump() for item in artifacts])},
            ],
            text_format=LLMObservation,
        )
        if response.output_parsed is None:
            raise RuntimeError("OpenAI returned no structured observation")
        drafted = response.output_parsed
    observation = Observation(text=drafted.observation, artifact_ids=[item.artifact_id for item in artifacts])
    return {
        "observations": [observation.model_dump()],
        "dataset_metadata": {**state.dataset_metadata, "review_question": drafted.review_question},
    }


def human_review(state: NeuroAthenaState) -> dict[str, Any]:
    answer = interrupt(
        {
            "question": state.dataset_metadata["review_question"],
            "figures": [
                {"tool_name": item["tool_name"], "artifact_id": item["artifact_id"], "figure_path": item["figure_path"]}
                for item in state.artifacts
            ],
            "observation": state.observations[0]["text"],
        }
    )
    interpretation = HumanInterpretation(text=str(answer).strip() or "No interpretation provided.")
    return {"human_interpretations": [interpretation.model_dump()], "hitl_round": 1}


def final_summary(state: NeuroAthenaState) -> dict[str, Any]:
    observation = Observation.model_validate(state.observations[0])
    human = HumanInterpretation.model_validate(state.human_interpretations[0])
    summary = (
        f"Run {state.run_id} completed all four allow-listed PINN diagnostics. "
        f"Observation: {observation.text} Human interpretation: {human.text} "
        "The figures describe model outputs and do not establish biological causality."
    )
    report_path = BASE_DIR / "artifacts" / state.run_id / "report.json"
    report = {
        "run_id": state.run_id,
        "goal": state.requested_goal,
        "dataset_metadata": state.dataset_metadata,
        "artifacts": state.artifacts,
        "observations": state.observations,
        "human_interpretations": state.human_interpretations,
        "final_summary": summary,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"final_summary": summary, "report_path": str(report_path)}


def build_graph():
    builder = StateGraph(NeuroAthenaState)
    for name, node in (
        ("LoadData", load_data),
        ("ValidateData", validate_data),
        ("AnalysisPlanner", analysis_planner),
        ("ToolExecutor", tool_executor),
        ("ObservationNode", observation_node),
        ("HumanReview", human_review),
        ("FinalSummary", final_summary),
    ):
        builder.add_node(name, node)
    builder.add_edge(START, "LoadData")
    builder.add_edge("LoadData", "ValidateData")
    builder.add_edge("ValidateData", "AnalysisPlanner")
    builder.add_edge("AnalysisPlanner", "ToolExecutor")
    builder.add_edge("ToolExecutor", "ObservationNode")
    builder.add_edge("ObservationNode", "HumanReview")
    builder.add_edge("HumanReview", "FinalSummary")
    builder.add_edge("FinalSummary", END)
    return builder.compile(checkpointer=InMemorySaver())


def open_figure(figure_path: str) -> bool:
    path = Path(figure_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sys.platform == "darwin":
        commands = [["open", "-a", "Preview", str(path)], ["open", str(path)]]
    elif os.name == "nt":
        commands = [["cmd", "/c", "start", "", str(path)]]
    else:
        commands = [["xdg-open", str(path)]]
    return any(subprocess.run(command, capture_output=True, text=True, check=False).returncode == 0 for command in commands)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("mat_file", nargs="?", type=Path, default=DEFAULT_RESULT)
    parser.add_argument(
        "--goal",
        default="Inspect trained PINN velocity and permeability distributions and central spatial slices.",
    )
    parser.add_argument("--auto-review", help="Resume HITL automatically with this answer (useful for tests).")
    parser.add_argument("--use-openai", action="store_true", help="Use an OpenAI structured-output observation node.")
    parser.add_argument("--no-open", action="store_true", help="Do not open generated figures in an image viewer.")
    args = parser.parse_args()

    run_id = uuid.uuid4().hex[:12]
    graph = build_graph()
    config = {"configurable": {"thread_id": run_id}}
    initial = NeuroAthenaState(
        run_id=run_id,
        source_file=str(args.mat_file),
        requested_goal=args.goal,
        use_openai=args.use_openai,
    )
    result = graph.invoke(initial, config=config)
    payload = result["__interrupt__"][0].value
    print(json.dumps(payload, indent=2))
    if not args.no_open:
        opened = 0
        for figure in payload["figures"]:
            opened += int(open_figure(figure["figure_path"]))
        if opened:
            print(f"Opened {opened} of {len(payload['figures'])} figures in an image viewer.")
        else:
            print("Could not launch an image viewer; open the printed PNG paths directly.")
    answer = args.auto_review if args.auto_review is not None else input("Your review: ")
    final = graph.invoke(Command(resume=answer), config=config)
    print("\n" + final["final_summary"])
    for artifact in final["artifacts"]:
        print(f"{artifact['tool_name']}: {artifact['figure_path']}")
    print(f"Report: {final['report_path']}")


if __name__ == "__main__":
    main()
