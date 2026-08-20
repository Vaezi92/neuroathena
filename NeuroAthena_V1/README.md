# NeuroAthena V1

NeuroAthena V1 is a deterministic analysis workflow for trained MR-AIV/PINN MATLAB results. It uses LangGraph for controlled orchestration and a single human-in-the-loop review step.

> Trained datasets, generated figures, virtual environments, local configuration, and API keys are intentionally excluded from version control.

## V1 workflow

```text
LoadData
   ↓
ValidateData
   ↓
AnalysisPlanner
   ↓
ToolExecutor
   ↓
ObservationNode
   ↓
HumanReview
   ↓
FinalSummary
```

Each run deterministically generates four artifacts:

- Speed and velocity-component distributions
- Permeability distribution
- Central sagittal, coronal, and axial speed slices
- Central sagittal, coronal, and axial permeability slices

## Expected MATLAB fields

- `txyz_smm`: time and spatial coordinates
- `u_mm_s`, `v_mm_s`, `w_mm_s`: predicted velocity components in mm/s
- `K_m2`: predicted permeability in m²

Velocity is converted to µm/s and permeability is converted to mm² without normalization. Every figure is linked to a structured artifact record containing its tool name, version, parameters, source, units, and numerical summary.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place a compatible trained result in the repository-level `Brain/Real/` directory. The default filename is:

```text
10ROI_perm_FT_thr25to75_concentration_WTM1-Train:0.5_mass:1steady_fields_high.mat
```

## Run

Use the default result:

```bash
python langgraph_workflow.py
```

Or provide another compatible file:

```bash
python langgraph_workflow.py ../Brain/Real/your_result.mat
```

The graph pauses at `HumanReview`, attempts to open the four PNGs, prints every artifact path, accepts your review, and writes a JSON report under `artifacts/<run_id>/`.

For a non-interactive smoke run:

```bash
python langgraph_workflow.py --auto-review "Figures reviewed." --no-open
```

## Optional OpenAI observation node

Offline deterministic observations are the default and require no API key. To enable the schema-constrained OpenAI observation node:

```bash
export OPENAI_API_KEY="your-key"
python langgraph_workflow.py --use-openai
```

The model is not given arbitrary-code execution. Set `OPENAI_MODEL` to override the default model.

## Test

```bash
python -m unittest -v
```

The end-to-end test runs when the local trained `.mat` file is available; otherwise it is skipped.
