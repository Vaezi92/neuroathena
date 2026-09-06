"""Question aggregation, selection, and duplicate detection."""

from __future__ import annotations

import re
from collections import Counter

from .models import ASPECTS, CandidateQuestion, RunConfig, Score


def _weighted_average(values: dict[str, float], weights: dict[str, float]) -> float:
    active = [(value, weights.get(key, 1.0)) for key, value in values.items() if weights.get(key, 1.0) > 0]
    if not active:
        raise ValueError("No positively weighted values")
    return sum(value * weight for value, weight in active) / sum(weight for _, weight in active)


def aggregate_candidate(candidate: CandidateQuestion, config: RunConfig) -> CandidateQuestion:
    """Average grader scores by aspect, then apply configurable aspect weights."""
    if not candidate.grades:
        raise ValueError(f"Candidate {candidate.candidate_id} has no grades")
    dimensions = {
        aspect: _weighted_average(
            {grader: getattr(score, aspect) for grader, score in candidate.grades.items()},
            config.grader_weights,
        )
        for aspect in ASPECTS
    }
    candidate.aggregate = Score(**dimensions)
    candidate.overall_score = _weighted_average(dimensions, config.aspect_weights)
    return candidate


def select_best(candidates: list[CandidateQuestion]) -> CandidateQuestion:
    """Select one eligible candidate using deterministic score tie-breaks."""
    eligible = [item for item in candidates if item.status == "proposed" and item.aggregate is not None]
    if not eligible:
        raise ValueError("No valid candidates are available")
    return max(
        eligible,
        key=lambda item: (
            item.overall_score,
            item.aggregate.importance,
            item.aggregate.feasibility,
            item.aggregate.novelty,
            item.candidate_id,
        ),
    )


def normalized_tokens(text: str) -> list[str]:
    """Normalize question text for deterministic duplicate comparison."""
    return re.findall(r"[a-z0-9]+", text.lower())


def similarity(first: str, second: str) -> float:
    """Multiset Dice similarity; deterministic and dependency-free."""
    a, b = Counter(normalized_tokens(first)), Counter(normalized_tokens(second))
    if not a or not b:
        return 0.0
    # Similar syntax does not imply duplicate scientific content. Preserve
    # questions that target different primary measured/inferred quantities.
    discriminators = {
        "velocity", "permeability", "concentration", "advection", "mass",
        "metadata", "uncertainty", "slice", "experiment",
    }
    concepts_a, concepts_b = set(a) & discriminators, set(b) & discriminators
    if concepts_a and concepts_b and concepts_a.isdisjoint(concepts_b):
        return 0.0
    overlap = sum((a & b).values())
    return 2 * overlap / (sum(a.values()) + sum(b.values()))


def find_duplicate(text: str, known: dict[str, str], threshold: float) -> str | None:
    """Return the most similar known ID when similarity meets the threshold."""
    matches = [(similarity(text, prior), identifier) for identifier, prior in known.items()]
    if not matches:
        return None
    score, identifier = max(matches)
    return identifier if score >= threshold else None
