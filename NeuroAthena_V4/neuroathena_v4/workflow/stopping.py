"""Explicit stopping policy for the autonomous question loop."""

from __future__ import annotations

from ..models import RunConfig, RunState, SelectedQuestion


class StoppingPolicy:
    """Apply maximum-step and consecutive-low-score termination rules."""

    def __init__(self, config: RunConfig):
        self.config = config

    def evaluate(self, state: RunState, question: SelectedQuestion) -> None:
        """Mutate only the stopping counter and optional termination reason."""
        score = question.aggregate
        low = all(
            getattr(score, aspect) < self.config.low_score_threshold
            for aspect in ("feasibility", "novelty", "importance")
        )
        state.consecutive_low_score_steps = state.consecutive_low_score_steps + 1 if low else 0
        if state.consecutive_low_score_steps >= self.config.consecutive_low_score_limit:
            state.termination_reason = "scientific_saturation"
        elif state.step >= self.config.maximum_steps:
            state.termination_reason = "maximum_steps"
