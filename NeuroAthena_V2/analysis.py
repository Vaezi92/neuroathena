"""Allow-listed deterministic analysis API used locally and through MCP."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

from models import Artifact, CoordinateROI


BASE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = BASE_DIR.parent
V1_TOOLS = REPOSITORY_ROOT / "NeuroAthena_V1" / "deterministic_tools.py"
TOOL_VERSION = "2.0.0"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))


def _load_v1_module():
    spec = importlib.util.spec_from_file_location("neuroathena_v1_tools", V1_TOOLS)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load deterministic tools from {V1_TOOLS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v1 = _load_v1_module()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_metadata(path: Path) -> dict[str, Any]:
    metadata = v1.dataset_metadata(path)
    metadata.update(
        {
            "source_sha256": sha256_file(path.resolve()),
            "orientation": None,
            "atlas_roi_labels_available": False,
            "roi_policy": "Coordinate-bounded ROIs only; do not assign anatomical names.",
        }
    )
    return metadata


def _roi_mask(data: Any, roi: CoordinateROI) -> np.ndarray:
    return (
        (data.x_mm >= roi.x_mm[0]) & (data.x_mm <= roi.x_mm[1])
        & (data.y_mm >= roi.y_mm[0]) & (data.y_mm <= roi.y_mm[1])
        & (data.z_mm >= roi.z_mm[0]) & (data.z_mm <= roi.z_mm[1])
    )


def _stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("ROI contains no samples")
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "p01": float(np.percentile(values, 1)),
        "p99": float(np.percentile(values, 99)),
    }


def get_roi_statistics(path: Path, roi: CoordinateROI) -> dict[str, Any]:
    data = v1.load_pinn_result(path)
    mask = _roi_mask(data, roi)
    if not mask.any():
        raise ValueError(f"Coordinate ROI {roi.name!r} contains no samples")
    return {
        "roi": roi.model_dump(),
        "sample_count": int(mask.sum()),
        "statistics": {
            "speed_um_s": _stats(data.speed_um_s[mask]),
            "permeability_mm2": _stats(data.permeability_mm2[mask]),
        },
        "units": {"speed": "um/s", "permeability": "mm^2", "coordinates": "mm"},
        "anatomical_label": None,
    }


def compare_rois(path: Path, first: CoordinateROI, second: CoordinateROI) -> dict[str, Any]:
    a = get_roi_statistics(path, first)
    b = get_roi_statistics(path, second)
    output: dict[str, Any] = {"first": a, "second": b, "ratios": {}}
    for field in ("speed_um_s", "permeability_mm2"):
        denominator = b["statistics"][field]["median"]
        output["ratios"][f"median_{field}"] = (
            a["statistics"][field]["median"] / denominator if denominator else None
        )
    return output


class AnalysisService:
    """Stable source-of-truth interface behind local and MCP transports."""

    def __init__(self, source_file: Path, run_id: str, artifact_dir: Path):
        self.source_file = source_file.resolve()
        self.run_id = run_id
        self.artifact_dir = artifact_dir.resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.source_sha256 = sha256_file(self.source_file)
        self.registry: dict[str, Artifact] = {}

    def _register(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        summary: dict[str, Any],
        figure_path: Path | None = None,
    ) -> Artifact:
        raw_units = summary.get("units", {})
        units = raw_units if isinstance(raw_units, dict) else {"reported_field": str(raw_units)}
        artifact = Artifact(
            artifact_id=f"artifact-{uuid.uuid4().hex[:12]}",
            run_id=self.run_id,
            tool_name=tool_name,
            tool_version=TOOL_VERSION,
            source_file=str(self.source_file),
            source_sha256=self.source_sha256,
            parameters=parameters,
            numerical_summary=summary,
            units=units,
            figure_path=str(figure_path) if figure_path else None,
            figure_sha256=sha256_file(figure_path) if figure_path else None,
        )
        self.registry[artifact.artifact_id] = artifact
        return artifact

    def execute(self, tool_name: str, parameters: dict[str, Any]) -> Artifact:
        if tool_name == "mraiv.get_metadata":
            return self._register(tool_name, parameters, get_metadata(self.source_file))
        if tool_name == "mraiv.get_roi_statistics":
            roi = CoordinateROI.model_validate(parameters["roi"])
            return self._register(tool_name, parameters, get_roi_statistics(self.source_file, roi))
        if tool_name == "mraiv.compare_rois":
            first = CoordinateROI.model_validate(parameters["first"])
            second = CoordinateROI.model_validate(parameters["second"])
            return self._register(tool_name, parameters, compare_rois(self.source_file, first, second))
        plotters: dict[str, Callable[..., dict[str, Any]]] = {
            "mraiv.plot_velocity_slice": v1.plot_speed_slices,
            "mraiv.plot_permeability_slice": v1.plot_permeability_slices,
        }
        if tool_name == "mraiv.plot_distribution":
            field = parameters.get("field")
            if field not in {"velocity", "permeability"}:
                raise ValueError("Distribution field must be velocity or permeability")
            function = v1.plot_velocity_distributions if field == "velocity" else v1.plot_permeability_distribution
        elif tool_name in plotters:
            function = plotters[tool_name]
        elif tool_name == "mraiv.retrieve_artifact":
            artifact_id = parameters["artifact_id"]
            if artifact_id not in self.registry:
                raise KeyError(f"Unknown artifact ID: {artifact_id}")
            return self.registry[artifact_id]
        else:
            raise ValueError(f"Tool is not allow-listed: {tool_name}")
        output = self.artifact_dir / f"{tool_name.replace('.', '_')}-{uuid.uuid4().hex[:6]}.png"
        forwarded = {key: value for key, value in parameters.items() if key != "field"}
        summary = function(self.source_file, output, **forwarded)
        return self._register(tool_name, parameters, summary, output)

    def write_registry(self) -> Path:
        path = self.artifact_dir / "artifact_registry.json"
        path.write_text(
            json.dumps([item.model_dump() for item in self.registry.values()], indent=2),
            encoding="utf-8",
        )
        return path
