"""Reusable, bounded scientific procedures for NeuroAthena V3."""

from __future__ import annotations

from models import SkillInvocation, ToolCall


SKILL_VERSION = "2.0.0"


def route_skills(goal: str) -> list[SkillInvocation]:
    """Select transparent recipes using keywords; no free-form tool generation."""
    goal_lower = goal.lower()
    selected: list[SkillInvocation] = []
    if any(word in goal_lower for word in ("velocity", "speed", "flow", "transport", "inspect")):
        selected.append(
            SkillInvocation(
                skill_name="inspect_velocity_field",
                version=SKILL_VERSION,
                reason="The objective requests velocity/speed diagnostics.",
                tool_calls=[
                    ToolCall(tool_name="mraiv.plot_distribution", parameters={"field": "velocity"}),
                    ToolCall(tool_name="mraiv.plot_velocity_slice"),
                ],
                cautions=[
                    "Velocity is model-inferred, not independently measured flow.",
                    "Spatial structure may reflect masks, boundaries, or preprocessing.",
                ],
            )
        )
    if any(word in goal_lower for word in ("permeability", "heterogeneity", "inspect")):
        selected.append(
            SkillInvocation(
                skill_name="inspect_permeability_field",
                version=SKILL_VERSION,
                reason="The objective requests permeability/heterogeneity diagnostics.",
                tool_calls=[
                    ToolCall(tool_name="mraiv.plot_distribution", parameters={"field": "permeability"}),
                    ToolCall(tool_name="mraiv.plot_permeability_slice"),
                ],
                cautions=[
                    "Permeability interpretation must retain units and inference assumptions.",
                    "Extreme values require artifact review before biological interpretation.",
                ],
            )
        )
    if not selected:
        selected = route_skills("inspect velocity and permeability")
    return selected


def artifact_review_skill() -> SkillInvocation:
    return SkillInvocation(
        skill_name="evaluate_possible_artifact",
        version=SKILL_VERSION,
        reason="All apparent biological patterns require an artifact alternative.",
        tool_calls=[ToolCall(tool_name="mraiv.get_metadata")],
        cautions=[
            "Check boundaries, masks, missing values, interpolation, and partial-volume effects.",
            "Do not use anatomical direction terms without orientation metadata.",
        ],
    )
