"""Allow-listed specialist NotebookLM routing with a deterministic fallback."""

from __future__ import annotations

import json
from pathlib import Path

from models import QueryNotebookRoute, SearchQuery, SpecialistNotebook


def load_registry(path: Path) -> list[SpecialistNotebook]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    notebooks = [SpecialistNotebook.model_validate(item) for item in payload.get("notebooks", [])]
    if not notebooks:
        raise ValueError("Notebook registry contains no specialist notebooks")
    ids = [item.notebook_id for item in notebooks]
    if len(ids) != len(set(ids)):
        raise ValueError("Notebook registry contains duplicate notebook IDs")
    return notebooks


def deterministic_routes(queries: list[SearchQuery], notebooks: list[SpecialistNotebook]) -> list[QueryNotebookRoute]:
    by_specialty = {item.specialty: item for item in notebooks}
    routes: list[QueryNotebookRoute] = []
    for query in queries:
        text = query.query.lower()
        if any(term in text for term in ("mri", "imaging", "segmentation", "partial volume", "mr-aiv")):
            specialties = ["mri"]
        elif any(term in text for term in ("pinn", "training", "loss", "neural", "model inference")):
            specialties = ["pinns", "applied_mathematics"]
        elif any(term in text for term in ("velocity", "permeability", "flow", "transport", "boundary", "diffusion")):
            specialties = ["fluid_dynamics", "applied_mathematics"]
        elif any(term in text for term in ("brain", "biological", "anatom", "physiolog", "mouse", "ventric")):
            specialties = ["neuroscience"]
        else:
            specialties = ["applied_mathematics"]
        selected = [by_specialty[item].notebook_id for item in specialties if item in by_specialty][:2]
        routes.append(QueryNotebookRoute(query_id=query.query_id, notebook_ids=selected, reason="Deterministic domain-keyword fallback."))
    return routes
