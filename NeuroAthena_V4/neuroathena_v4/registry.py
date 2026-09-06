"""Allow-listed deterministic functions and their provenance."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable

from .deterministic_execution import IsolatedFunctionExecutor
from .models import AnalysisProposal, AnalysisResult, CandidateFunction, RunState
from .promotion import AcceptedDeterministicPromotionPolicy


@dataclass(frozen=True)
class FunctionSpec:
    """Metadata and callable for one trusted deterministic computation."""
    name: str
    version: str
    function: Callable[[RunState, dict[str, Any]], tuple[Any, str | None, str]]
    description: str


class FunctionRegistry:
    """Allow-list and execute trusted deterministic scientific functions.

    Registration is explicit. Model output cannot execute arbitrary Python.
    Each result records the registered version and a source-code hash.
    """
    def __init__(self, candidate_executor: IsolatedFunctionExecutor | None = None) -> None:
        self._functions: dict[str, FunctionSpec] = {}
        self._promoted: dict[str, CandidateFunction] = {}
        self._candidate_executor = candidate_executor or IsolatedFunctionExecutor()
        self._promotion_policy = AcceptedDeterministicPromotionPolicy()

    def register(self, spec: FunctionSpec) -> None:
        """Add one uniquely named trusted function to the allow-list."""
        if spec.name in self._functions:
            raise ValueError(f"Function already registered: {spec.name}")
        self._functions[spec.name] = spec

    def capabilities(self) -> dict[str, dict[str, Any]]:
        """Describe callable functions without exposing implementation control."""
        builtins = {
            name: {"version": spec.version, "description": spec.description, "trust": "built_in"}
            for name, spec in self._functions.items()
        }
        promoted = {
            name: {"version": item.version, "description": item.description, "trust": "promoted"}
            for name, item in self._promoted.items()
        }
        return {**builtins, **promoted}

    def execute(self, proposal: AnalysisProposal, state: RunState) -> AnalysisResult:
        """Execute an allow-listed proposal and attach function provenance."""
        if proposal.candidate_function is not None:
            if proposal.candidate_function.name in self.capabilities():
                raise ValueError(f"Candidate function name is already registered: {proposal.candidate_function.name}")
            return self._candidate_executor.execute(proposal, state)
        if proposal.function_name in self._promoted:
            promoted = self._promoted[proposal.function_name]
            authored_proposal = AnalysisProposal(
                analysis_id=proposal.analysis_id,
                question_id=proposal.question_id,
                function_name=proposal.function_name,
                parameters=proposal.parameters,
                intended_answer=proposal.intended_answer,
                candidate_function=promoted,
            )
            return self._candidate_executor.execute(authored_proposal, state)
        try:
            spec = self._functions[proposal.function_name]
        except KeyError as error:
            raise ValueError(f"Function is not allow-listed: {proposal.function_name}") from error
        value, units, answer = spec.function(state, proposal.parameters)
        source = inspect.getsource(spec.function).encode("utf-8")
        return AnalysisResult(
            analysis_id=proposal.analysis_id,
            question_id=proposal.question_id,
            function_name=spec.name,
            function_version=spec.version,
            parameters=proposal.parameters,
            value=value,
            units=units,
            answer=answer,
            provenance={"function_sha256": hashlib.sha256(source).hexdigest(), "run_id": state.run_id},
        )

    def promote(self, function: CandidateFunction, result: AnalysisResult) -> bool:
        """Promote a validated candidate when the explicit trust policy permits."""
        if function.name in self.capabilities() or not self._promotion_policy.permits(function, result):
            return False
        self._promoted[function.name] = function
        return True


def metadata_summary(state: RunState, parameters: dict[str, Any]) -> tuple[Any, str | None, str]:
    """Return dataset metadata and declared limitations without array analysis."""
    value = {"metadata": state.fixed_metadata, "limitations": state.limitations}
    return value, None, "The available metadata and declared limitations were summarized deterministically."


def analyzed_quantity_count(state: RunState, parameters: dict[str, Any]) -> tuple[Any, str | None, str]:
    """Count quantities previously accepted into Shared Analyzed Data."""
    count = len(state.analyzed_data)
    return count, "count", f"Shared Analyzed Data currently contains {count} critic-approved quantities."


def default_registry() -> FunctionRegistry:
    """Build the current minimal registry of placeholder deterministic tools."""
    registry = FunctionRegistry()
    registry.register(FunctionSpec("metadata_summary", "1.0.0", metadata_summary, "Return fixed metadata and limitations."))
    registry.register(FunctionSpec("analyzed_quantity_count", "1.0.0", analyzed_quantity_count, "Count trusted analyzed quantities."))
    return registry
