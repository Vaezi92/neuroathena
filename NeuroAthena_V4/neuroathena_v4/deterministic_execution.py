"""Validation and isolated execution for analyzer-authored computations."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .dataset import file_sha256
from .models import AnalysisProposal, AnalysisResult, CandidateFunction, RunState


class CandidateFunctionValidator:
    """Reject candidate source that escapes the deterministic function contract."""

    _forbidden_nodes = (
        ast.AsyncFunctionDef, ast.ClassDef, ast.Delete, ast.Global, ast.Import,
        ast.ImportFrom, ast.Lambda, ast.Nonlocal, ast.Raise, ast.Try, ast.With,
        ast.AsyncWith, ast.Yield, ast.YieldFrom,
    )
    _forbidden_calls = {
        "breakpoint", "compile", "eval", "exec", "getattr", "globals", "help",
        "input", "locals", "open", "setattr", "vars", "__import__",
    }
    _allowed_attributes = {
        "abs", "all", "any", "argmax", "argmin", "array", "asarray", "astype",
        "clip", "concatenate", "corrcoef", "count_nonzero", "dtype", "flatten",
        "float64", "hypot", "isfinite", "isnan", "max", "mean", "median", "min",
        "nanmax", "nanmean", "nanmedian", "nanmin", "nanpercentile", "nanstd",
        "nansum", "ndim", "percentile", "quantile", "ravel", "reshape", "shape",
        "size", "sqrt", "square", "std", "sum", "tolist", "transpose", "unique",
        "where",
    }

    def validate(self, function: CandidateFunction) -> None:
        """Validate naming, syntax, entry point, and prohibited Python features."""
        if not function.name.isidentifier() or function.name.startswith("_"):
            raise ValueError("Candidate function name must be a public Python identifier")
        try:
            tree = ast.parse(function.source)
        except SyntaxError as error:
            raise ValueError(f"Candidate function has invalid syntax: {error.msg}") from error
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if len(tree.body) != 1 or len(definitions) != 1 or definitions[0].name != "analyze":
            raise ValueError("Candidate source must contain only def analyze(data, parameters)")
        arguments = definitions[0].args
        if [item.arg for item in arguments.args] != ["data", "parameters"] or arguments.vararg or arguments.kwarg:
            raise ValueError("Candidate entry point must be analyze(data, parameters)")
        for node in ast.walk(tree):
            if isinstance(node, self._forbidden_nodes):
                raise ValueError(f"Candidate source uses prohibited syntax: {type(node).__name__}")
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise ValueError("Candidate source cannot access private attributes")
            if isinstance(node, ast.Attribute) and node.attr not in self._allowed_attributes:
                raise ValueError(f"Candidate source uses non-allow-listed attribute: {node.attr}")
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise ValueError("Candidate source cannot access private names")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in self._forbidden_calls:
                raise ValueError(f"Candidate source calls prohibited function: {node.func.id}")


class IsolatedFunctionExecutor:
    """Execute validated candidate source in a credential-free subprocess."""

    def __init__(self, timeout_seconds: float = 60.0) -> None:
        self.timeout_seconds = timeout_seconds
        self.validator = CandidateFunctionValidator()
        self.runner = Path(__file__).with_name("isolated_runner.py")

    def execute(self, proposal: AnalysisProposal, state: RunState) -> AnalysisResult:
        """Validate and replay a candidate twice, accepting identical JSON results."""
        function = proposal.candidate_function
        if function is None:
            raise ValueError("An authored execution requires candidate_function")
        self.validator.validate(function)
        source_file = state.fixed_metadata.get("source_file")
        source_hash = state.fixed_metadata.get("source_sha256")
        if not source_file or not source_hash:
            raise ValueError("Dataset source path and hash are required for authored analysis")
        if file_sha256(Path(source_file)) != source_hash:
            raise ValueError("Dataset hash changed after shared-state initialization")
        request = {
            "source": function.source,
            "dataset_path": source_file,
            "parameters": proposal.parameters,
        }
        first = self._run(request)
        second = self._run(request)
        if self._canonical(first) != self._canonical(second):
            raise ValueError("Candidate function failed deterministic replay validation")
        return AnalysisResult(
            analysis_id=proposal.analysis_id,
            question_id=proposal.question_id,
            function_name=function.name,
            function_version=function.version,
            parameters=proposal.parameters,
            value=first["value"],
            units=first.get("units"),
            answer=first["answer"],
            provenance={
                "function_sha256": hashlib.sha256(function.source.encode("utf-8")).hexdigest(),
                "dataset_sha256": source_hash,
                "run_id": state.run_id,
                "execution_mode": "isolated_subprocess",
            },
            validation={"static_validation": True, "deterministic_replays": 2, "deterministic": True},
        )

    def _run(self, request: dict[str, Any]) -> dict[str, Any]:
        environment = {key: value for key, value in os.environ.items() if key in {"PATH", "LANG", "LC_ALL"}}
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(self.runner)],
                input=json.dumps(request), text=True, capture_output=True,
                timeout=self.timeout_seconds, env=environment, check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Candidate function exceeded the isolated execution timeout") from error
        if completed.returncode != 0:
            raise RuntimeError("Candidate function failed in isolated execution")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("Candidate function returned invalid isolated output") from error
        if not isinstance(payload, dict) or set(payload) != {"value", "units", "answer"}:
            raise ValueError("Candidate result must contain exactly value, units, and answer")
        if not isinstance(payload["answer"], str) or not payload["answer"].strip():
            raise ValueError("Candidate result answer must be a non-empty string")
        return payload

    @staticmethod
    def _canonical(payload: dict[str, Any]) -> str:
        return json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
