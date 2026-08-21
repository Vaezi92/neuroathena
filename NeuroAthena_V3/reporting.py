"""Self-contained, escaped HTML report renderer for NeuroAthena V3."""

from __future__ import annotations

import base64
import html
import mimetypes
from pathlib import Path
from typing import Any


def _h(value: Any) -> str:
    return html.escape(str(value))


def _image_data(path: str) -> str:
    figure = Path(path)
    media_type = mimetypes.guess_type(figure.name)[0] or "image/png"
    return f"data:{media_type};base64,{base64.b64encode(figure.read_bytes()).decode('ascii')}"


def _list(items: list[Any], empty: str = "None recorded.") -> str:
    return "<p class='muted'>" + _h(empty) + "</p>" if not items else "<ul>" + "".join(f"<li>{_h(item)}</li>" for item in items) + "</ul>"


def _citation_links(record_ids: list[str], numbering: dict[str, int]) -> str:
    links = [f"<a class='citation' href='#{_h(record_id)}'>[{numbering[record_id]}]</a>" for record_id in record_ids if record_id in numbering]
    return " ".join(links)


def _evidence_group(
    title: str,
    key: str,
    assessments: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    numbering: dict[str, int],
) -> str:
    record_ids = list(dict.fromkeys(record_id for item in assessments for record_id in item.get(key, [])))
    cards = "".join(
        f"<article class='evidence-card'><h3>{_h(records[record_id]['title'])} {_citation_links([record_id], numbering)}</h3>"
        f"<p>{_h(records[record_id].get('excerpt') or 'No excerpt was stored.')}</p>"
        f"<small>{_h(records[record_id].get('journal') or records[record_id].get('provider',''))}</small></article>"
        for record_id in record_ids if record_id in records
    )
    if not cards:
        cards = "<p class='muted'>No retrieved record was assigned to this evidence category.</p>"
    return f"<section><h2>{_h(title)}</h2>{cards}</section>"


def _critic_table(assessments: list[dict[str, Any]]) -> str:
    rows = "".join(
        f"<tr><td>{_h(item['claim_id'])}</td><td><span class='strength {_h(item['strength'])}'>{_h(item['strength'])}</span></td>"
        f"<td>{len(item.get('supporting_record_ids',[]))}</td><td>{len(item.get('contradicting_record_ids',[]))}</td>"
        f"<td>{len(item.get('alternative_record_ids',[]))}</td><td>{len(item.get('limitation_record_ids',[]))}</td>"
        f"<td>{_h(item.get('rationale',''))}</td></tr>" for item in assessments
    ) or "<tr><td colspan='7'>No evidence assessment was stored.</td></tr>"
    return f"<table><thead><tr><th>Claim</th><th>Strength</th><th>Support</th><th>Contradiction</th><th>Alternatives</th><th>Limitations</th><th>Critic rationale</th></tr></thead><tbody>{rows}</tbody></table>"


def render_html_report(state: Any, output: Path) -> Path:
    synthesis = state.final_synthesis or {}
    records = {item["record_id"]: item for item in state.literature_records}
    cited_ids = list(dict.fromkeys(
        synthesis.get("citation_record_ids", [])
        + [record_id for response in state.notebooklm_responses for record_id in response.get("record_ids", [])]
        + [record_id for assessment in state.evidence_assessments for key in ("supporting_record_ids", "contradicting_record_ids", "alternative_record_ids", "limitation_record_ids") for record_id in assessment.get(key, [])]
    ))
    cited = [records[item] for item in cited_ids if item in records]
    numbering = {item["record_id"]: index for index, item in enumerate(cited, 1)}
    figures = "".join(
        f"<figure><img src='{_image_data(item['figure_path'])}' alt='{_h(item['artifact_id'])}'>"
        f"<figcaption><strong>{_h(item['artifact_id'])}</strong> — {_h(item['tool_name'])}</figcaption></figure>"
        for item in state.artifacts if item.get("figure_path") and Path(item["figure_path"]).is_file()
    )
    answers = "".join(
        f"<article><h3>Round {_h(item['round'])}</h3><p>{_h(item['verbatim_text'])}</p>"
        f"<small>Question ID: {_h(item['related_question_id'])}</small></article>"
        for item in state.human_interpretations
    )
    evidence = "".join(
        f"<tr id='{_h(item['record_id'])}'><td>[{index}]</td><td>{_h(item['title'])}</td>"
        f"<td>{_h(item.get('excerpt') or 'No excerpt available')}</td>"
        f"<td><a href='{_h(item['url'])}'>{_h(item.get('source_id') or item.get('doi') or item['record_id'])}</a></td></tr>"
        for index, item in enumerate(cited, 1)
    ) or "<tr><td colspan='4'>No literature source was cited in this run.</td></tr>"
    notebook_findings = "".join(
        f"<article><h3>{_h(item.get('intent','NotebookLM finding'))}</h3><p>{_h(item.get('answer') or 'No answer text returned.')}</p>"
        f"<small>Query: {_h(item.get('query_id',''))} · citations: {_citation_links(item.get('record_ids',[]), numbering)}</small></article>"
        for item in state.notebooklm_responses
    ) or "<p class='muted'>NotebookLM was not used in this run.</p>"
    notebook_titles = {item["notebook_id"]: item["title"] for item in state.notebook_registry}
    routing_rows = "".join(
        f"<tr><td>{_h(item['query_id'])}</td><td>{_h(', '.join(notebook_titles.get(value, value) for value in item['notebook_ids']))}</td><td>{_h(item['reason'])}</td></tr>"
        for item in state.notebook_routes
    ) or "<tr><td colspan='3'>No specialist routing was performed.</td></tr>"
    failure_rows = "".join(
        f"<tr><td>{_h(notebook_titles.get(item.get('notebook_id',''), item.get('notebook_id','')))}</td>"
        f"<td>{_h(item.get('query_id',''))}</td><td>{_h(item.get('intent',''))}</td>"
        f"<td>{_h(item.get('attempts',0))}</td><td>{_h(item.get('error',''))}</td></tr>"
        for item in state.retrieval_failures
    ) or "<tr><td colspan='5'>No retrieval failures were recorded.</td></tr>"
    document = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>NeuroAthena V3 — {_h(state.run_id)}</title><style>
body{{font-family:STIXGeneral,'Times New Roman',serif;background:#f5f3ef;color:#20252b;margin:0;line-height:1.55}}
main{{max-width:1120px;margin:auto;padding:44px 24px}} header{{background:#182936;color:white;padding:32px;border-radius:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}} section{{background:white;margin:22px 0;padding:26px;border-radius:14px;box-shadow:0 5px 22px #18293612}}
figure{{margin:0;background:#f8f8f8;padding:12px;border-radius:10px}} img{{width:100%;height:auto;display:block}} figcaption{{padding-top:9px}}
table{{width:100%;border-collapse:collapse}} th,td{{text-align:left;vertical-align:top;border-bottom:1px solid #ddd;padding:10px}} .muted,small{{color:#65717b}} h1,h2,h3{{line-height:1.2}}
.citation{{display:inline-block;color:#245f75;font-weight:bold;text-decoration:none;margin-left:4px}}
.evidence-card{{border-left:4px solid #4d7c86;padding:4px 18px;margin:14px 0;background:#f8faf9}} .strength{{font-weight:bold;text-transform:uppercase}} .strength.insufficient{{color:#9b3d30}} .strength.weak{{color:#9a681c}} .strength.moderate{{color:#336d48}} .strength.strong{{color:#245f75}}
</style></head><body><main><header><h1>NeuroAthena V3 scientific report</h1><p>Run {_h(state.run_id)} · model-inferred MR-AIV results</p></header>
<section><h2>Analysis objective</h2><p>{_h(state.requested_goal)}</p><p><strong>Dataset SHA-256:</strong> {_h(state.dataset_metadata.get('source_sha256',''))}</p></section>
<section><h2>Key figures</h2><div class='grid'>{figures}</div></section>
<section><h2>Verbatim expert explanations</h2>{answers or '<p>No expert explanation recorded.</p>'}</section>
<section><h2>NotebookLM-retrieved findings</h2>{notebook_findings}</section>
<section><h2>Specialist notebook routing</h2><table><thead><tr><th>Query</th><th>Selected specialist notebook(s)</th><th>Reason</th></tr></thead><tbody>{routing_rows}</tbody></table></section>
<section><h2>Retrieval completeness</h2><table><thead><tr><th>Notebook</th><th>Query</th><th>Intent</th><th>Attempts</th><th>Failure</th></tr></thead><tbody>{failure_rows}</tbody></table></section>
<section><h2>Evidence-grounded synthesis</h2><h3>Executive summary</h3><p>{_h(synthesis.get('executive_summary','No OpenAI synthesis was requested.'))}</p>
<h3>Integrated interpretation</h3><p>{_h(synthesis.get('integrated_interpretation',''))}</p><div class='grid'><article><h3>Fluid dynamics</h3><p>{_h(synthesis.get('fluid_dynamics_interpretation',''))}</p></article>
<article><h3>Biological/anatomical perspective</h3><p>{_h(synthesis.get('biological_anatomical_interpretation',''))}</p></article></div>
<p><strong>Citations used in synthesis:</strong> {_citation_links(synthesis.get('citation_record_ids',[]), numbering) or 'None.'}</p>
<h3>Alternative explanations</h3>{_list(synthesis.get('alternative_explanations',[]))}<h3>Limitations</h3>{_list(synthesis.get('limitations',[]))}
<h3>Recommended next experiments</h3>{_list(synthesis.get('recommended_next_experiments',[]))}</section>
{_evidence_group('Supporting evidence', 'supporting_record_ids', state.evidence_assessments, records, numbering)}
{_evidence_group('Contradicting evidence', 'contradicting_record_ids', state.evidence_assessments, records, numbering)}
{_evidence_group('Alternative explanations from evidence', 'alternative_record_ids', state.evidence_assessments, records, numbering)}
{_evidence_group('Methodological limitations from evidence', 'limitation_record_ids', state.evidence_assessments, records, numbering)}
<section><h2>Cited evidence</h2><table><thead><tr><th>No.</th><th>Source</th><th>Cited excerpt</th><th>Identifier</th></tr></thead><tbody>{evidence}</tbody></table></section>
<section><h2>Evidence critic assessment</h2>{_critic_table(state.evidence_assessments)}</section>
<section><h2>Guardrails</h2>{_list([('PASS' if item['passed'] else 'FAIL') + ' — ' + item['guardrail'] + ': ' + item['message'] for item in state.guardrail_results])}</section>
</main></body></html>"""
    output.write_text(document, encoding="utf-8")
    return output
