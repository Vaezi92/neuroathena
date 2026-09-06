"""Private subprocess entry point for deterministic candidate functions."""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
from scipy.io import loadmat


def _json_value(value: Any) -> Any:
    """Convert NumPy results to strict JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def main() -> None:
    """Load the dataset, execute the candidate, and emit one JSON result."""
    request = json.load(sys.stdin)
    data = {key: value for key, value in loadmat(request["dataset_path"]).items() if not key.startswith("__")}
    safe_builtins = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "float": float, "int": int, "len": len,
        "list": list, "max": max, "min": min, "range": range, "round": round,
        "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
        "zip": zip,
    }
    namespace: dict[str, Any] = {"__builtins__": safe_builtins, "np": np}
    exec(compile(request["source"], "<candidate_function>", "exec"), namespace, namespace)
    result = namespace["analyze"](data, request["parameters"])
    if not isinstance(result, dict):
        raise TypeError("analyze must return a dictionary")
    if set(result) != {"value", "units", "answer"}:
        raise ValueError("analyze must return exactly value, units, and answer")
    payload = {key: _json_value(result.get(key)) for key in ("value", "units", "answer")}
    sys.stdout.write(json.dumps(payload, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
