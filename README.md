# NeuroAthena

NeuroAthena is a versioned scientific workflow for deterministic analysis and evidence-grounded interpretation of MR-AIV/PINN mouse-brain results.

## Repository layout

```text
neuroathena/
├── Brain/                  # Shared local input data
│   └── Real/
├── NeuroAthena_V1/         # First working LangGraph workflow
├── NeuroAthena_V2/         # Second-version implementation
├── .gitignore
└── README.md
```

Scientific `.mat` inputs, virtual environments, local secrets, and generated artifacts are intentionally excluded from Git. Each version keeps its own source code, dependencies, tests, documentation, and generated-artifact directory.

## Version 1

```bash
cd NeuroAthena_V1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python langgraph_workflow.py
```

See [`NeuroAthena_V1/README.md`](NeuroAthena_V1/README.md) for details.

## Version 2

V2 implements the modular scientific-assistant stage of the Notion roadmap: an MR-AIV MCP server, reusable scientific skills, bounded three-round HITL, structured claims, literature retrieval, evidence criticism, provenance, and stronger scientific guardrails.

```bash
cd NeuroAthena_V2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python workflow.py
```

See [`NeuroAthena_V2/README.md`](NeuroAthena_V2/README.md) for the full workflow and scientific limitations.
