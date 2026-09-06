# NeuroAthena V4

NeuroAthena V4 is an autonomous, evidence-traceable scientific investigation loop for MR-AIV/PINN brain data. Four domain-specific Question Generator (QG) agents repeatedly propose and anonymously cross-grade questions. A coordinator selects exactly one question per step, an analyzer answers it with allow-listed deterministic computations, a critic verifies the method and result and may request bounded re-analysis, and a synthesizer creates the final report only after the loop stops.

V4 does not treat model-inferred velocity or permeability as an independently measured biological mechanism. Anatomical claims require validated orientation and atlas-registration metadata.

For a file-by-file explanation of the implementation, read [`docs/CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md).

## Architecture

```text
Shared Analyzed Data + Shared Q&A + Lessons Learned
                         |
             snapshot for investigation step n
                         |
       +-----------------+-----------------+
       | MRI QG | Fluid QG | Math QG | Neuroscience QG |
       +-----------------+-----------------+
                         |
              3 candidate questions each
                         |
     Coordinator anonymizes, blocks duplicates, and distributes
                         |
              anonymous cross-grading by all QGs
                         |
              weighted score aggregation
                         |
              select exactly one question
                         |
                  Analyzer proposal
                         |
                 pre-execution critic
                         |
              deterministic function execution
                         |
               post-execution critic
                    /          \
          revise (<= limit)    terminal verdict
                    \          /
        update Shared Q&A, Shared AD, and Lessons Learned
                         |
                  evaluate stopping
                    /          \
                 continue     Synthesizer
```

## Question scoring

Every candidate receives a feasibility, novelty, and importance score in `[0, 1]` from each of the four QG agents. Candidate authorship is hidden during grading. Self-grading is supported and is treated equally by default.

For aspect `a`, grader weights `g`, and aspect weights `w`:

```text
aspect_score[a] = sum(grader_weight[g] * grade[g,a]) / sum(grader_weight[g])
overall_score   = sum(aspect_weight[a] * aspect_score[a]) / sum(aspect_weight[a])
```

All weights default to `1`. Ties are resolved deterministically by importance, feasibility, novelty, then candidate ID.

## Duplicate and novelty gate

The coordinator blocks questions that repeat a previously proposed, selected, or answered question, or request a deterministic quantity already present in Shared Analyzed Data. The initial implementation uses normalized token similarity with a configurable threshold. Blocked candidates remain in the audit log with a reference to their match but cannot be selected.

## Analyzer–critic repair loop

The analyzer does not directly mutate the trusted deterministic-function library. It creates an `AnalysisProposal` that either identifies a trusted function or contains a new `CandidateFunction` and parameters. The critic reviews the complete proposal—including candidate source—before execution, then reviews the produced result and written answer.

Candidate source must define only `analyze(data, parameters)`, cannot import modules or use dynamic execution, private attributes, networking, or file-I/O operations, and is restricted to allow-listed NumPy analysis operations. It runs in a credential-free isolated Python subprocess with a timeout and strict JSON result contract. The dataset hash is rechecked before execution, and a candidate must produce identical results in two isolated replays.

Promotion is explicit: only a candidate whose post-execution critic verdict is `accept`, whose static validation passed, whose two replay results are identical, and whose provenance identifies the isolated execution and dataset/function hashes is promoted into the run's trusted registry. Promotion never occurs on a proposal review, failed execution, revision, partial answer, or rejection.

The critic returns one of:

- `accept`
- `revise`
- `partially_answered`
- `unanswerable_from_data`
- `implementation_failed`
- `invalid_question`
- `critic_rejected`

Re-analysis is bounded and defaults to three attempts. Exhausted revisions become a terminal result and the main question loop continues. Only critic-accepted results enter trusted Shared Analyzed Data; rejected attempts remain in the event log and Lessons Learned.

## Shared resources

`SharedQAPool` stores every candidate, anonymous grade, selection rationale, analyzer attempts, critic verdict, and final answer status. It is updated when a question is selected and again after analysis reaches a terminal verdict.

`SharedAnalyzedData` stores fixed dataset metadata and limitations plus critic-approved deterministic quantities and artifacts. Each quantity includes its dataset hash, function/version, parameters, units, provenance, and originating question.

`LessonMemory` stores reusable process knowledge such as failed approaches, unavailable variables, invalid assumptions, and successful analysis patterns. It must not promote tentative biological interpretations into facts.

Every mutation is also appended to a JSONL event log. Step snapshots are stable: agents participating in a step read the same state version.

## Stopping rules

The user can configure all limits. Defaults are:

- Maximum main-loop steps: `10`
- Low-score threshold: `0.25`
- Consecutive low-score selections required: `5`
- Analyzer–critic attempts per selected question: `3`

A step is low-scoring only when the selected question's averaged feasibility, novelty, and importance are all strictly below the threshold. Any step with at least one aspect at or above the threshold resets the consecutive counter. The loop stops at five consecutive low-score steps or the maximum number of steps.

The final report records the exact termination reason: `scientific_saturation`, `maximum_steps`, or `no_valid_candidates`.

## Current implementation boundary

The live workflow supports four OpenAI QG agents grounded in four specialist NotebookLM notebooks, OpenAI analyzer function authoring, OpenAI critic review, isolated deterministic execution, policy-controlled function promotion, and OpenAI final synthesis. `--offline-demo` retains deterministic placeholder adapters for inexpensive orchestration checks; the authored-function vertical slice is covered by offline and mocked integration tests.

The temporary CLI default activates only `fluid_dynamics` and `neuroscience` to reduce NotebookLM latency. Select any configured subset explicitly with `--qg-specialties`; for example, restore all four with `--qg-specialties mri fluid_dynamics applied_mathematics neuroscience`.

## Run

Run with the trained MR-AIV/PINN result:

```bash
python -m neuroathena_v4.cli \
  --mat-file ../Brain/Real/10ROI_perm_FT_thr25to75_concentration_WTM1-Train:0.5_mass:1steady_fields_high.mat \
  --run-dir artifacts/run-01 \
  --maximum-steps 10
python -m unittest discover -s tests -v
```

Authenticate before the first live run:

```bash
export OPENAI_API_KEY="YOUR_KEY"
../NeuroAthena_V3/.venv/bin/notebooklm login --browser chrome
../NeuroAthena_V3/.venv/bin/notebooklm list --json
```

Alternatively, place the key alone in `api.txt` at the V4 project root. This file is ignored by Git. The `OPENAI_API_KEY` environment variable takes precedence when both are present. Credentials are never included in shared state, events, prompts, or reports.

Dataset initialization occurs before investigation step 1. It hashes and inspects the MATLAB file, validates its schema and spatial spacing, and records declared acquisition metadata separately from values observable in the result. The supplied trained-result file contains only one unique time value, so its declared one-minute source DCE-MRI resolution cannot by itself support temporal analysis.

For a no-cost orchestration test, add `--offline-demo`. Live mode never silently falls back when OpenAI or NotebookLM authentication is missing.
