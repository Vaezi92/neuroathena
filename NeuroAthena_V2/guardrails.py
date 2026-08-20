"""Scientific claim and provenance checks."""

from __future__ import annotations

from models import GuardrailResult


OVERCLAIM_TERMS = (" proves ", " demonstrates causality ", " confirmed mechanism ")
ANATOMICAL_TERMS = ("left", "right", "anterior", "posterior", "superior", "inferior", "ventricular")


def evaluate_guardrails(state: object) -> list[GuardrailResult]:
    results: list[GuardrailResult] = []
    artifact_ids = {item["artifact_id"] for item in state.artifacts}
    claims = state.quantitative_observations + state.visual_observations + state.hypotheses
    bad_refs = [claim["claim_id"] for claim in claims if not set(claim.get("related_artifact_ids", [])).issubset(artifact_ids)]
    results.append(GuardrailResult(guardrail="figure_provenance", passed=not bad_refs, message=f"Invalid claim references: {bad_refs}" if bad_refs else "All figure/data references resolve."))

    text = " ".join(claim["text"].lower() for claim in claims)
    overclaims = [term.strip() for term in OVERCLAIM_TERMS if term in f" {text} "]
    results.append(GuardrailResult(guardrail="biological_claim", passed=not overclaims, message=f"Prohibited overclaims: {overclaims}" if overclaims else "Claim language is calibrated."))

    orientation_known = bool(state.dataset_metadata.get("orientation"))
    anatomy_terms = [term for term in ANATOMICAL_TERMS if term in text]
    results.append(GuardrailResult(guardrail="orientation", passed=orientation_known or not anatomy_terms, message="Orientation metadata supports anatomical terms." if orientation_known else (f"Unsupported anatomical terms: {anatomy_terms}" if anatomy_terms else "No unsupported anatomical-direction language.")))

    query_intents = {item["intent"] for item in state.search_queries}
    needed = {"support", "contradiction", "alternative", "limitation"} if state.hypotheses else set()
    results.append(GuardrailResult(guardrail="confirmation_bias", passed=needed.issubset(query_intents), message="Support, contradiction, alternative, and limitation searches are represented." if needed.issubset(query_intents) else f"Missing search intents: {sorted(needed - query_intents)}"))

    record_ids = {item["record_id"] for item in state.literature_records}
    cited_ids = {record_id for assessment in state.evidence_assessments for key in ("supporting_record_ids", "contradicting_record_ids", "alternative_record_ids", "limitation_record_ids") for record_id in assessment[key]}
    results.append(GuardrailResult(guardrail="citation", passed=cited_ids.issubset(record_ids), message="Every literature reference resolves to a retrieved record." if cited_ids.issubset(record_ids) else "An evidence assessment cites an unknown record."))
    notebook_records = [item for item in state.literature_records if item.get("provider") == "notebooklm"]
    notebook_citations_valid = not state.notebooklm_notebook or bool(notebook_records) and all(
        item.get("source_id") and item.get("title") and item.get("excerpt") for item in notebook_records
    )
    results.append(
        GuardrailResult(
            guardrail="notebooklm_citations",
            passed=notebook_citations_valid,
            message=(
                "NotebookLM evidence has source IDs, titles, and cited excerpts."
                if notebook_citations_valid and state.notebooklm_notebook
                else "NotebookLM was not used."
                if not state.notebooklm_notebook
                else "NotebookLM returned no complete source citations; its answer cannot support a scientific claim."
            ),
        )
    )
    return results
