# NeuroAthena V2

NeuroAthena V2 extends the working V1 LangGraph with a modular MR-AIV analysis interface, reusable scientific skills, bounded expert review, structured claims, literature retrieval, evidence criticism, provenance, and scientific guardrails.

The numerical and plotting functions in `../NeuroAthena_V1/deterministic_tools.py` remain the source of truth. V2 exposes them through a stable MCP contract and does not allow arbitrary Python execution.

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
Question Generator ↔ HITL (maximum 3 rounds)
      ↓
Claim Extractor
      ↓
Literature Search
      ↓
Evidence Critic
      ↓
Guardrails
      ↓
Scientific Summary
```

## Scientific boundaries

- Velocity and permeability are model-inferred fields, not independently measured biological mechanisms.
- The current `.mat` file has no atlas ROI labels or orientation metadata.
- V2 therefore supports reproducible coordinate-bounded ROIs but does not invent anatomical region names.
- Anatomical-direction claims are rejected unless orientation metadata becomes available.
- Literature mode intentionally searches for support, contradiction, alternatives, and limitations.
- Retrieved papers are candidates for expert review; retrieval alone does not validate a claim.

## Setup

```bash
cd /Users/mohammadvaezi/Documents/neuroathena/NeuroAthena_V2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

Offline mode creates analyses, HITL records, claims, provenance, and an evidence table without making unsupported literature claims:

```bash
python workflow.py
```

By default, V2 opens the generated figures in the desktop image viewer before the first expert-review question. The images are already saved, and the viewer is launched separately, so closing it does not stop the workflow. For automated or headless runs, disable this behavior with:

```bash
python workflow.py --no-show-figures
```

At each review prompt, type the expert response verbatim. Include `resolved`, `sufficient`, `finish`, or `stop` when further questions are unnecessary. Otherwise the graph asks progressively narrower questions and stops after round three.

The terminal is concise by default: it shows only the review round and question. Full structured HITL payloads and artifact paths remain in the saved report and can be printed for debugging with:

```bash
python workflow.py --verbose
```

To perform live Europe PMC searches after the expert review:

```bash
python workflow.py --online-literature
```

### NotebookLM evidence retrieval

V2 can use your curated NotebookLM as an optional retrieval layer. NotebookLM retrieves passages from the sources you selected; NeuroAthena keeps the analysis, claim extraction, evidence criticism, citation checks, and final synthesis in the local workflow.

This connector uses the community `notebooklm-py` MCP server because personal NotebookLM does not currently provide an official public MCP API. Treat it as a preview integration. NeuroAthena invokes only read operations and never adds, deletes, or changes notebooks or sources.

Authenticate once. The login is saved under `~/.notebooklm/`, outside this repository and outside Git:

```bash
notebooklm login --browser chrome
notebooklm list --json
```

Copy the canonical notebook ID from the list output, then run:

```bash
python workflow.py --notebooklm-notebook "YOUR-CANONICAL-NOTEBOOK-ID"
```

For a separate account/profile:

```bash
notebooklm -p work login --browser chrome
notebooklm -p work list --json
python workflow.py --notebooklm-notebook "YOUR-CANONICAL-NOTEBOOK-ID" --notebooklm-profile work
```

For each extracted claim, NeuroAthena independently asks for support, contradiction, an alternative explanation, and limitations. It requests full NotebookLM references and rejects NotebookLM evidence records that lack a source ID, title, or cited excerpt. Raw answer metadata and accepted citations are preserved in the run report.

For an automated offline smoke run:

```bash
python workflow.py --auto-review "The review is sufficient and resolved." --no-show-figures
```

Reports and figures are stored under `artifacts/<run-id>/`. The JSON report preserves complete structured provenance; the Markdown report contains the evidence table and guardrail outcomes.

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
