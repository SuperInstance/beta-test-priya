# beta-test-priya

[![Language: Python](https://img.shields.io/badge/language-Python-blue.svg)]()
[![SuperInstance](https://img.shields.io/badge/part%20of-SuperInstance-9cf.svg)](https://github.com/SuperInstance)

CS student usability test of the SuperInstance ternary ecosystem packages. Priya walks through `negative-space-core` and `avoidance-cascade`, documenting what clicked, what didn't, and where the docs fall short.

## What This Is

Priya (junior CS student, some Python experience, basic agent-based modeling knowledge) sat down with the two core packages and built `tutorial.py` — a step-by-step walkthrough that exercises every API. Her feedback in `BETA-REPORT.md` captures the real learning curve.

## Overall Rating: 7/10

| Aspect | Score | Notes |
|--------|-------|-------|
| Ease of learning | 7/10 | Productive in ~1 hour |
| API design | 7/10 | Mostly intuitive, some inconsistencies |
| Documentation | 5/10 | Signatures yes, examples no |
| Concept clarity | 6/10 | Core ideas cool, need more explanation |
| Class project potential | 9/10 | Would definitely use |

## The Tutorial

`tutorial.py` is a complete walkthrough of both packages:

```
Step 1: Create 100 ternary agents (-1, 0, +1)
Step 2: Run agents through environments with AvoidanceTracker
Step 3: Analyze avoidance across population scales (BatchAnalyzer)
Step 4: Verify the Conservation Law
Step 5: Detect cascade events (CascadeDetector)
Step 6: Balanced Learning prevents cascades (BalancedLearner)
Step 7: Inference from gaps (InferenceEngine)
Step 8: Feedback Loop — prediction learning
```

### Running the Tutorial

```bash
pip install -r requirements.txt
python tutorial.py
```

Dependencies: `negative-space-core >= 0.1.0`, `avoidance-cascade >= 0.1.0`, `pytest >= 7.0`.

## API Coverage

### negative-space-core

| Class | Methods Tested |
|-------|---------------|
| `AvoidanceTracker(positions)` | `record(actions)`, `avoid_ratio()`, `choose_ratio()`, `unknown_ratio()`, `avoid_std()`, `generations()` |
| `BatchAnalyzer` | `add_batch(batch)`, `population_summary()`, `conservation_std()` |
| `ConservationLaw(threshold)` | `test_all_scales(data)`, `all_conserved()`, `report()` |
| `FeedbackLoop(decay)` | `run_round(options)`, `accuracy()` |
| `InferenceEngine` | `record_avoidance(pos, count)`, `find_gaps()`, `infer()`, `high_confidence_deductions(min_confidence)` |

### avoidance-cascade

| Class | Methods Tested |
|-------|---------------|
| `CascadeDetector(threshold, window, on_alert)` | `record(avoided)`, `reset()` |
| `BalancedLearner(exploration_rate, decay_factor, explore_interval)` | `decide(agent_id, avoid_bias)`, `record_outcome(agent_id, avoided, reward)`, `get_memory(agent_id)` |
| `CascadeMetrics` | `record(avoidance_ratio, ...)`, `summary()`, `since(step)` |
| `ExplorationScheduler(interval, selector, on_explore)` | `record_avoid(agent_id, avoided)` |

## Tests

```bash
pytest tests/
```

40+ tests covering all 5 laws across both packages:

- **Law 1 (Conservation)**: `TestConservationLaw` — verifies ratio stability across scales
- **Law 2 (Avoidance Tracking)**: `TestAvoidanceTracker` — per-position ratio calculations
- **Law 3 (Cascade Detection)**: `TestCascadeDetector` — threshold triggering, balanced data rejection
- **Law 4 (Balanced Learning)**: `TestBalancedLearner`, `TestExplorationScheduler` — forced exploration prevents runaway
- **Law 5 (Inference from Gaps)**: `TestInferenceEngine` — gap finding and confidence filtering

## Bugs Found

| Bug | Severity | Description |
|-----|----------|-------------|
| CascadeDetector partial window | Low | `window=50` with 30 records still fires alerts |
| FeedbackLoop.accuracy() pre-data | Cosmetic | Returns 0.0 instead of None before `run_round()` |
| InferenceEngine confidence calibration | Low | `min_confidence=0.6` passes more deductions than expected |
| AvoidanceTracker no validation | Low | Values outside {-1,0,1} accepted silently |

## Concepts That Need Better Docs

1. **Conservation Law** — "avoidance ratio is conserved across population scales" needs a one-sentence intuition. What does it *mean*? Answer: the avoid:choose ratio stays stable whether you have 10 agents or 1,000.

2. **Inference from Gaps** — Learning from what agents *avoid* rather than what they *choose*. Needs concrete example: "if agents avoid positions 1–5 and 10–15, positions 6–9 are interesting."

3. **FeedbackLoop decay/margin** — When would you set `decay=0.5` vs `0.99`? What do these control?

4. **BalancedLearner vs ExplorationScheduler** — Both prevent cascades via forced exploration. When to use which?

## Repository Structure

```
beta-test-priya/
├── tutorial.py          # 8-step walkthrough (~250 lines)
├── tests/
│   └── test_tutorial.py # 40+ pytest tests
├── BETA-REPORT.md       # Full usability assessment
├── requirements.txt     # negative-space-core, avoidance-cascade, pytest
├── .benchmarks/         # Performance measurements
└── docs/                # Integration notes
```

## Related Repos

| Repo | Role |
|------|------|
| `negative-space-core` | Core ternary tracking (tested here) |
| `avoidance-cascade` | Cascade detection (tested here) |
| `beta-test-elena` | Mathematical rigor testing |
| `beta-test-marcus` | Investor due-diligence |
| `superinstance-architecture` | Architecture specification |
