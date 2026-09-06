"""Run a local orchestration demonstration."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from .agents import MarkdownSynthesizer, RuleBasedAnalyzer, RuleBasedCritic, RuleBasedQG
from .coordinator import Coordinator
from .credentials import openai_api_key_available
from .dataset import initialize_dataset
from .models import PERSPECTIVES, RunConfig, RunState
from .live_agents import NotebookOpenAIQG, OpenAIAnalyzer, OpenAICritic, OpenAISynthesizer
from .registry import default_registry
from .store import RunStore


def main() -> None:
    """Configure agents, initialize the dataset, and start one V4 run."""
    parser = argparse.ArgumentParser(description="Run the NeuroAthena V4 orchestration demo")
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--maximum-steps", type=int, default=10)
    parser.add_argument("--mat-file", type=Path, required=True)
    parser.add_argument("--temporal-resolution-minutes", type=float, default=1.0)
    parser.add_argument("--spatial-resolution-mm", type=float, default=0.1)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--notebooklm-profile", default="default")
    parser.add_argument("--notebook-registry", type=Path, default=Path(__file__).resolve().parent.parent / "notebooks.json")
    parser.add_argument(
        "--qg-specialties", nargs="+", choices=PERSPECTIVES,
        default=["fluid_dynamics", "neuroscience"],
        help="Active QG specialties (temporary default: fluid_dynamics neuroscience)",
    )
    parser.add_argument("--offline-demo", action="store_true", help="Use deterministic placeholder agents without OpenAI or NotebookLM")
    args = parser.parse_args()
    registry = default_registry()
    if args.offline_demo:
        agents = [RuleBasedQG(name, name) for name in args.qg_specialties]
        analyzer = RuleBasedAnalyzer()
        critic = RuleBasedCritic(set(registry.capabilities()))
        synthesizer = MarkdownSynthesizer()
    else:
        if not openai_api_key_available():
            parser.error("OpenAI key is unavailable; set OPENAI_API_KEY, create api.txt, or use --offline-demo")
        payload = json.loads(args.notebook_registry.read_text(encoding="utf-8"))
        configured = payload.get("notebooks", [])
        by_specialty = {item["specialty"]: item for item in configured}
        missing = set(args.qg_specialties) - set(by_specialty)
        if missing:
            parser.error(f"Notebook registry lacks specialties: {sorted(missing)}")
        agents = [
            NotebookOpenAIQG(
                agent_id=specialty, perspective=specialty,
                notebook_id=by_specialty[specialty]["notebook_id"],
                profile=args.notebooklm_profile, model=args.model,
            )
            for specialty in args.qg_specialties
        ]
        analyzer = OpenAIAnalyzer(registry.capabilities(), args.model)
        critic = OpenAICritic(registry.capabilities(), args.model)
        synthesizer = OpenAISynthesizer(args.model)
    coordinator = Coordinator(
        RunConfig(maximum_steps=args.maximum_steps), agents, analyzer,
        critic, registry, synthesizer, RunStore(args.run_dir),
    )
    metadata, limitations = initialize_dataset(
        args.mat_file,
        declared_temporal_resolution_minutes=args.temporal_resolution_minutes,
        declared_spatial_resolution_mm=args.spatial_resolution_mm,
    )
    state = coordinator.run(RunState(
        run_id=uuid.uuid4().hex[:12],
        fixed_metadata=metadata,
        limitations=limitations,
    ))
    print(args.run_dir / "report.md")
    print(f"Stopped: {state.termination_reason}")


if __name__ == "__main__":
    main()
