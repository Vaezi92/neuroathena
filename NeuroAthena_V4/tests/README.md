# Reading the tests

`test_v4.py` is organized by behavior:

- `ScoringTests` verifies weighted averaging, deterministic selection, and duplicate detection.
- `DatasetTests` verifies that declared acquisition metadata remains separate from values observed in the MATLAB file.
- `LiveAgentWiringTests` verifies that a QG consults NotebookLM, produces three questions, and grades anonymous candidates without making real network calls.
- `CoordinatorTests` verifies complete loop behavior, persistence, stopping, grading participation, and critic-requested re-analysis.
- `FunctionAuthoringTests` verifies offline and mocked-OpenAI question-to-accepted-result vertical slices, deterministic replay, critic gates, promotion, and rejection of unsafe candidate operations.

The test doubles are intentionally small:

- `FixedQG` always returns three predictable questions and scores.
- `RuleBasedAnalyzer`, `RuleBasedCritic`, and `MarkdownSynthesizer` exercise orchestration without API cost.
- `unittest.mock.patch` replaces only external provider calls.

Read a test as **arrange → act → assert**:

1. Arrange a configuration and deterministic agents.
2. Act by running the coordinator or one policy function.
3. Assert the scientific-workflow requirement, not an internal call sequence.

Live OpenAI and NotebookLM calls are intentionally excluded from unit tests. They require credentials, are slow, and are nondeterministic. Their adapters are checked with schema-valid mocked responses; a separate manual live smoke run validates credentials and provider availability.
