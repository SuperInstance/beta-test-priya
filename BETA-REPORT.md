# Beta-Report: SuperInstance Ternary Ecosystem

**Tester:** Priya (CS student, learning agent systems & evolutionary computation)  
**Date:** 2026-06-04  
**Packages tested:** `negative-space-core` v0.1.0, `avoidance-cascade` v0.1.0

---

## Overall Learning Experience: 7/10

The packages were surprisingly approachable. I'm a junior CS student with some Python experience and a vague understanding of agent-based modeling. The core concepts — ternary actions (-1, 0, +1), avoidance tracking, cascade detection — clicked pretty quickly once I started coding. The hardest part was understanding *why* the conservation law matters, not *how* to use it.

## Was the API Intuitive?

**Mostly yes (7/10).**

**What worked well:**
- `AvoidanceTracker.record(actions)` — dead simple. Pass a list of -1/0/+1, get back stats.
- `BatchAnalyzer` — the naming is clear. You add batches, you get summaries.
- `CascadeDetector` with `on_alert` callback — this is a nice pattern, feels very Pythonic.
- `InferenceEngine.find_gaps()` — the metaphor of "gaps in avoidance zones" is evocative and the API maps to it well.

**What was confusing:**
- `ConservationLaw.test_all_scales(data)` expects `dict[int, list[float]]` where keys are population sizes. This wasn't immediately obvious from the signature — I had to experiment. A docstring example would help.
- `BalancedLearner.decide()` returns a `bool` (True = avoid), which felt inconsistent with the ternary theme. Would returning -1/0/+1 be more consistent?
- `FeedbackLoop.predict()` returns an int, but the ternary mapping wasn't documented in the method docstring. Had to check the module docs.
- `CascadeMetrics.record()` accepts `**extra` kwargs but it's unclear what gets stored and what's ignored.

## Which Concepts Were Hardest to Understand?

1. **Conservation Law** — "avoidance ratio is conserved across population scales" is abstract. I had to think about this for a while. What does it *mean* for a ratio to be "conserved"? Eventually I understood it as: the avoid:choose ratio stays stable whether you have 10 agents or 1000. But a one-sentence intuition in the docs would help a lot.

2. **Inference from Gaps** — the idea that you can learn from *what agents avoid* rather than *what they choose* is cool but philosophically tricky. More concrete examples would help (e.g., "if agents avoid positions 1-5 and 10-15, we infer something interesting about positions 6-9").

3. **FeedbackLoop.decay and margin** — the `decay` and `margin` parameters on FeedbackLoop were not intuitive. What do they control exactly? When would I set decay to 0.5 vs 0.99?

4. **BalancedLearner vs ExplorationScheduler** — these seem to solve overlapping problems (preventing cascades via forced exploration). The distinction wasn't clear. When would I use one vs the other?

## What Examples Would Help?

1. **A minimal "hello world"** — just create a tracker, record one generation, print the ratio. Give me a 5-line starting point.
2. **Conservation Law walkthrough** — show data at two scales, compute ratios, demonstrate they match. Connect the math to the code.
3. **Cascade scenario** — start with balanced data, then show what happens when avoidance spikes. Visual (even ASCII) would be great.
4. **End-to-end pipeline** — create agents → track → detect cascades → use inference → feedback loop. Basically what I built in `tutorial.py`, but in the docs.
5. **Comparison example** — show BalancedLearner vs raw agent decisions. Why does forced exploration matter? Show the numbers.

## Bug Reports

### Bug 1: CascadeDetector alert count depends on window in unexpected ways
- **Severity:** Low (UX)
- When `window=50` and I feed 30 all-True records, I expected no alerts (not enough data for the window). Instead I got an alert. The threshold comparison might not account for partial windows correctly, or the behavior should be documented.

### Bug 2: FeedbackLoop.accuracy() returns 0.0 before any predictions
- **Severity:** Cosmetic
- `accuracy()` returns 0.0 before `run_round` is called. This is technically correct (0/0 avoidance), but could return `None` or raise an error to be clearer.

### Bug 3: InferenceEngine.high_confidence_deductions() returns all deductions with min_confidence=0.6
- **Severity:** Low
- Even deductions with what seemed like low confidence passed the 0.6 threshold. The confidence scoring might need calibration, or the docs should explain what confidence values mean in practice.

### Bug 4: AvoidanceTracker.record() doesn't validate action values
- **Severity:** Low
- Passing values like `2` or `-5` doesn't raise an error. Should validate that actions are in {-1, 0, 1}.

## Would You Use This for a Class Project?

**Yes, absolutely.** I'd use this for a project on agent-based modeling or emergent behavior. The packages give you enough structure to run experiments quickly without getting bogged down in implementation. The ternary action model is a nice abstraction — simple enough to explain in a presentation, rich enough to produce interesting dynamics.

Specifically, I'd want to:
- Experiment with different avoidance biases and see how cascades emerge
- Use the InferenceEngine to do something creative (maybe map avoidance zones to a real-world scenario like traffic or disease spread)
- Compare BalancedLearner vs ExplorationScheduler performance
- Build a visualization of cascade events over time

**One request:** Jupyter notebook examples would make this 10x more useful for coursework. Being able to run cells and see results incrementally is how I learn best.

## Summary

| Aspect | Rating | Notes |
|--------|--------|-------|
| Ease of learning | 7/10 | Got productive in ~1 hour |
| API design | 7/10 | Mostly intuitive, some inconsistencies |
| Documentation | 5/10 | Signatures yes, examples no |
| Concept clarity | 6/10 | Core ideas are cool, need more explanation |
| Class project potential | 9/10 | Would definitely use |
| Bug severity | 3/10 | Minor issues only |

**Overall:** Solid foundation, needs better docs and examples. The ideas are genuinely interesting — I found myself thinking about the conservation law and inference-from-gaps concepts even after I finished coding. That's a good sign.
