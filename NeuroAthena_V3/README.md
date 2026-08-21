# NeuroAthena V3

NeuroAthena V3 evolves V2 into the roadmap's final biological-interpretation system. It combines deterministic MR-AIV analysis, curated NotebookLM retrieval, contextual expert dialogue, citation-constrained OpenAI synthesis, provenance, guardrails, and a self-contained HTML report.

The numerical and plotting functions in `../NeuroAthena_V1/deterministic_tools.py` remain the source of truth. OpenAI models never calculate fields or generate arbitrary plotting code.

## Workflow

```text
Load + Validate
      ↓
Analysis Planner
      ↓
Skill Router
      ↓
MCP Tool Contract
      ↓
Artifact Registry
      ↓
Observation Agent
      ↓
Round 1: three fixed perspectives
      ↓
Claim + NotebookLM evidence refresh
      ↓
Rounds 2–3: contextual OpenAI questions
      ↓
Claim Extractor
      ↓
Literature Search
      ↓
Evidence Critic
      ↓
Guardrails
      ↓
OpenAI Scientific Synthesizer
      ↓
Self-contained HTML Report
```

## Scientific boundaries

- Velocity and permeability are model-inferred fields, not independently measured biological mechanisms.
- The current `.mat` file has no atlas ROI labels or orientation metadata.
- V3 therefore supports reproducible coordinate-bounded ROIs but does not invent anatomical region names.
- Anatomical-direction claims are rejected unless orientation metadata becomes available.
- Literature mode intentionally searches for support, contradiction, alternatives, and limitations.
- Retrieved papers are candidates for expert review; retrieval alone does not validate a claim.

## Setup

```bash
cd /Users/mohammadvaezi/Documents/neuroathena/NeuroAthena_V3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

Set your OpenAI API key in the shell (never in source code), authenticate NotebookLM, and obtain the canonical notebook ID:

```bash
export OPENAI_API_KEY="YOUR_KEY"
notebooklm login --browser chrome
notebooklm list --json
```

Run the complete workflow:

```bash
python workflow.py
```

The default `notebooks.json` registry contains five allow-listed specialists: MRI, applied mathematics, neuroscience, PINNs, and fluid dynamics. For every evidence query, OpenAI selects the smallest sufficient set of one or two notebooks. NeuroAthena validates every returned ID against the registry, records the decision and rationale, and shows the routing table in `report.html`.

Selected notebooks are searched concurrently, with at most three active notebook workers by default. Queries within one notebook remain sequential to reduce throttling risk. Change the bound when needed:

```bash
python workflow.py --notebook-concurrency 2
```

Each failed query is retried once, recorded with its underlying MCP error if it still fails, and skipped. Successful results survive, guardrails mark the report as partial, and HTML generation continues.

To force one notebook for a diagnostic run, override routing explicitly:

```bash
python workflow.py --notebooklm-notebook "CANONICAL-NOTEBOOK-ID"
```

Round 1 asks exactly three questions: your overall scientific explanation, your fluid-dynamics explanation, and your biological/anatomical interpretation. After those answers, NeuroAthena retrieves evidence and uses the OpenAI Responses API with Pydantic structured output to generate contextual questions for rounds 2 and 3. Type `resolved` when enough information has been collected.

Rounds 2 and 3 contain at most two concise questions that must be answerable from current expert knowledge. Requests to compute maps or statistics, run code, retrain PINNs, provide per-voxel records, or perform ablations are rejected as HITL questions and retained as possible recommendations for the final report.

By default, V3 opens the generated figures before the first question. Closing the image viewer does not stop the workflow. For automated or headless runs:

```bash
python workflow.py --no-show-figures
```

At each review prompt, type the expert response verbatim. Include `resolved`, `sufficient`, `finish`, or `stop` when further questions are unnecessary. Otherwise the graph asks progressively narrower questions and stops after round three.

The terminal is concise by default. Full structured HITL payloads can be printed with:

```bash
python workflow.py --verbose
```

Normal runs print concise live progress for validation, each deterministic analysis, notebook routing, each specialist notebook and query, contextual-question synthesis, final synthesis, and HTML rendering. Third-party MCP startup logs are suppressed. Since specialist NotebookLM retrieval can take a minute or more per notebook, elapsed time and cited-passage counts are shown as each notebook finishes.

To perform live Europe PMC searches after the expert review:

```bash
python workflow.py --online-literature
```

### NotebookLM evidence retrieval and OpenAI synthesis

NotebookLM retrieves passages from the sources you selected. NeuroAthena passes only structured artifacts, verbatim expert explanations, retrieved excerpts, and source IDs to OpenAI for synthesis. The model output is schema-validated and may cite only record IDs present in the retrieval registry.

This connector uses the community `notebooklm-py` MCP server because personal NotebookLM does not currently provide an official public MCP API. Treat it as a preview integration. NeuroAthena invokes only read operations and never adds, deletes, or changes notebooks or sources.

Authenticate once. The login is saved under `~/.notebooklm/`, outside this repository and outside Git:

```bash
notebooklm login --browser chrome
notebooklm list --json
```

Confirm that the five IDs in `notebooks.json` match the list output, then run automatic routing:

```bash
python workflow.py
```

For a separate account/profile:

```bash
notebooklm -p work login --browser chrome
notebooklm -p work list --json
python workflow.py --notebooklm-profile work
```

For each extracted claim, NeuroAthena independently asks for support, contradiction, an alternative explanation, and limitations. It requests full NotebookLM references and rejects NotebookLM evidence records that lack a source ID, title, or cited excerpt. Raw answer metadata and accepted citations are preserved in the run report.

For an automated offline smoke run that spends no API credits:

```bash
python workflow.py --no-openai --no-notebooklm --no-show-figures --no-open-report \
  --auto-review "Scientific explanation." \
  --auto-review "Fluid-dynamics explanation." \
  --auto-review "Biological explanation; resolved."
```

Reports and figures are stored under `artifacts/<run-id>/`. At completion, `report.html` opens automatically in your default browser. It contains embedded figures, verbatim user explanations, NotebookLM findings, cited excerpts, the integrated synthesis, alternatives, limitations, next experiments, evidence assessments, and guardrail outcomes. Use `--no-open-report` to suppress the browser. `report.json` preserves machine-readable provenance and model-call metadata.

To rebuild the HTML presentation later from saved JSON without rerunning analysis, NotebookLM, or OpenAI:

```bash
python render_existing_report.py artifacts/<run-id>/report.json
```

This creates `report_rebuilt.html` beside the JSON.

## MCP server

Configure the source and run the stdio server:

```bash
export NEUROATHENA_MAT_FILE="/Users/mohammadvaezi/Documents/neuroathena/Brain/Real/10ROI_perm_FT_thr25to75_concentration_WTM1-Train:0.5_mass:1steady_fields_high.mat"
export NEUROATHENA_RUN_ID="mcp-demo"
python mcp_server.py
```

Exposed tools:

- `mraiv.get_metadata`
- `mraiv.get_roi_statistics`
- `mraiv.plot_velocity_slice`
- `mraiv.plot_permeability_slice`
- `mraiv.plot_distribution`
- `mraiv.compare_rois`
- `mraiv.retrieve_artifact`

## Test

```bash
python -m unittest -v
```
