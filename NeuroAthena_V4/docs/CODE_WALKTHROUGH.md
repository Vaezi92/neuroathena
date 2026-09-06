# NeuroAthena V4 code walkthrough

Read the system in this order when deciding what to change.

## 1. Whole workflow

Start with `neuroathena_v4/coordinator.py`. Its loop contains only four ideas:

```text
run one question round
run one analysis round
evaluate stopping rules
save a checkpoint
```

After termination, it asks the synthesizer for the final report. Agents cannot update shared state directly.

## 2. Question round

`neuroathena_v4/workflow/question_round.py` implements one complete QG round:

1. Each active QG receives the same snapshot.
2. Each proposes exactly three questions.
3. Repetitive candidates are marked as duplicates.
4. Authorship is hidden before grading.
5. Every QG grades every valid candidate.
6. Scores are aggregated.
7. Exactly one question is selected.

Change this file when changing proposal count, anonymization, grading completeness, or selection.

## 3. Scoring

`neuroathena_v4/scoring.py` contains score aggregation, deterministic tie-breaking, and duplicate similarity. It has no provider or persistence logic.

`RunConfig.aspect_weights` controls feasibility, novelty, and importance weights. `RunConfig.grader_weights` controls grader weights.

## 4. Analysis round

`neuroathena_v4/workflow/analysis_round.py` implements:

```text
analyzer trusted-function selection or candidate-function authoring
→ critic pre-review
→ candidate validation and isolated deterministic execution
→ critic result review
→ accept and optional policy-controlled promotion, revise, or terminate
```

Only critic-accepted results enter Shared Analyzed Data. All attempts, including candidate source and both critic reviews, remain in the Q&A audit record.

`neuroathena_v4/deterministic_execution.py` statically validates the single-function source contract, verifies the dataset hash, launches the credential-free subprocess, enforces the timeout and strict JSON result, and requires two identical replays. `neuroathena_v4/isolated_runner.py` is the narrow subprocess entry point. `neuroathena_v4/promotion.py` contains the explicit trust-promotion rule.

## 5. Stopping

`neuroathena_v4/workflow/stopping.py` owns the complete stopping rule: maximum steps or the configured number of consecutive questions whose three aggregate scores are all below the threshold.

## 6. Shared state

`neuroathena_v4/models.py` defines the system vocabulary:

- `CandidateQuestion`: proposal, grades, and duplicate status.
- `SelectedQuestion`: one winner for a step.
- `QARecord`: question, attempts, final status, and answer.
- `AnalysisProposal`: requested deterministic function call.
- `AnalysisResult`: value, units, answer, and provenance.
- `CriticReview`: accept, revise, or terminal decision.
- `AnalyzedQuantity`: critic-accepted trusted result.
- `RunState`: Shared Q&A, Shared Analyzed Data, and Lessons Learned.

## 7. Live agents

`neuroathena_v4/live_agents.py` contains the QG, analyzer, critic, and synthesizer implementations. NotebookLM supplies private background to each QG. OpenAI returns schema-validated questions and grades. Questions go to the coordinator, not NotebookLM.

## 8. Providers

- `providers/openai_gateway.py`: only OpenAI SDK boundary.
- `providers/notebooklm_client.py`: provider-facing NotebookLM boundary.
- `notebooklm.py`: read-only MCP transport inherited from V3.
- `credentials.py`: credential loading that never enters shared state.

## 9. Deterministic layer

`registry.py` owns built-in and promoted functions plus execution provenance. It starts with metadata summary and analyzed-quantity count, executes new candidates through the isolated boundary, and promotes only candidates permitted by `AcceptedDeterministicPromotionPolicy` after critic acceptance.

`dataset.py` inspects the MATLAB file before step 1 and separates observed facts from declared acquisition metadata.

## 10. Persistence and CLI

`store.py` writes the append-only `events.jsonl` and atomic `state.json`. `cli.py` selects implementations and connects dependencies; it should contain no scientific rules.

## 11. Testing rule

Tests should read as requirements: authorship is hidden, every agent grades every candidate, one question wins, duplicates are blocked, critic revisions are bounded, only accepted data is trusted, and stopping rules are exact.
