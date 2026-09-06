"""Explicit trust-promotion policy for analyzer-authored functions."""

from __future__ import annotations

from .models import AnalysisResult, CandidateFunction


class AcceptedDeterministicPromotionPolicy:
    """Promote only critic-accepted candidates that passed two isolated replays."""

    def permits(self, function: CandidateFunction, result: AnalysisResult) -> bool:
        """Return whether validation and provenance satisfy the trust policy."""
        validation = result.validation
        provenance = result.provenance
        return (
            result.function_name == function.name
            and validation.get("static_validation") is True
            and validation.get("deterministic") is True
            and validation.get("deterministic_replays", 0) >= 2
            and provenance.get("execution_mode") == "isolated_subprocess"
            and bool(provenance.get("function_sha256"))
            and bool(provenance.get("dataset_sha256"))
        )
