"""Validated initialization of the shared DCE-MRI/PINN data context."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a dataset without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_dataset(
    path: Path,
    *,
    declared_temporal_resolution_minutes: float = 1.0,
    declared_spatial_resolution_mm: float = 0.1,
) -> tuple[dict[str, Any], list[str]]:
    """Inspect the result file and distinguish observed from declared metadata."""
    import numpy as np
    from scipy.io import loadmat

    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = loadmat(source)
    required = {"txyz_smm", "u_mm_s", "v_mm_s", "w_mm_s", "K_m2"}
    missing = sorted(required - raw.keys())
    if missing:
        raise KeyError(f"Missing required MATLAB fields: {', '.join(missing)}")
    coordinates = np.asarray(raw["txyz_smm"], dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 4:
        raise ValueError(f"txyz_smm must have shape (n, >=4), got {coordinates.shape}")
    times = np.unique(coordinates[:, 0])
    observed_spacing = {}
    for column, axis in enumerate(("x", "y", "z"), 1):
        unique = np.unique(coordinates[:, column])
        deltas = np.diff(unique)
        observed_spacing[axis] = float(np.median(deltas)) if deltas.size else None
    spatial_matches = all(
        value is not None and np.isclose(value, declared_spatial_resolution_mm, rtol=0, atol=1e-8)
        for value in observed_spacing.values()
    )
    metadata = {
        "source_file": str(source),
        "source_sha256": file_sha256(source),
        "data_kind": "trained MR-AIV/PINN result derived from DCE-MRI",
        "sample_count": int(coordinates.shape[0]),
        "available_fields": sorted(key for key in raw if not key.startswith("__")),
        "acquisition": {
            "declared_temporal_resolution_minutes": declared_temporal_resolution_minutes,
            "declared_spatial_resolution_mm": declared_spatial_resolution_mm,
        },
        "observed": {
            "time_values": [float(value) for value in times],
            "time_point_count": int(times.size),
            "coordinate_spacing_mm": observed_spacing,
            "spatial_resolution_matches_declaration": bool(spatial_matches),
        },
        "units": {
            "coordinates": "mm", "source_velocity": "mm/s",
            "reported_velocity": "um/s", "source_permeability": "m^2",
            "reported_permeability": "mm^2",
        },
    }
    limitations = [
        "Velocity and permeability are model-inferred fields, not independently measured biological mechanisms.",
        "Atlas labels, orientation, and a validated atlas transformation are not present in this result file.",
    ]
    if times.size < 2:
        limitations.append(
            "The result file contains one unique time value; the declared one-minute DCE-MRI acquisition resolution "
            "cannot be independently verified or used for temporal analysis from this file alone."
        )
    if not spatial_matches:
        limitations.append("Observed coordinate spacing does not match the declared spatial resolution on every axis.")
    return metadata, limitations
