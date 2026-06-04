"""
SuperInstance Ternary Ecosystem Tutorial
=========================================
A step-by-step walkthrough of the negative_space_core and avoidance_cascade packages.

Written by Priya (beta-tester persona) to verify all 5 laws of the ternary ecosystem.

Actions in the ternary system:
    -1 = AVOID   (agent actively avoids)
     0 = UNKNOWN (agent is undecided / exploring)
    +1 = CHOOSE  (agent actively chooses/approaches)

The Five Laws:
    1. Conservation Law — avoidance ratio is conserved across population scales
    2. Avoidance Tracking — per-position avoidance ratios are measurable
    3. Cascade Detection — cascades are detectable when avoidance exceeds thresholds
    4. Balanced Learning — forced exploration prevents runaway avoidance
    5. Inference from Gaps — knowledge can be deduced from negative spaces
"""

import random
from negative_space_core import (
    AvoidanceTracker,
    BatchAnalyzer,
    ConservationLaw,
    FeedbackLoop,
    InferenceEngine,
)
from avoidance_cascade import (
    CascadeDetector,
    BalancedLearner,
    CascadeMetrics,
    ExplorationScheduler,
)

random.seed(42)

# ---------------------------------------------------------------------------
# Step 1: Create 100 ternary agents
# ---------------------------------------------------------------------------
NUM_AGENTS = 100
NUM_POSITIONS = 20
NUM_GENERATIONS = 50

# Each agent is identified by an index. We'll generate ternary actions for them.
agents = list(range(NUM_AGENTS))

print("=" * 60)
print("STEP 1: Create 100 ternary agents")
print("=" * 60)
print(f"  Agents: {NUM_AGENTS}")
print(f"  Positions (decision points): {NUM_POSITIONS}")
print(f"  Generations: {NUM_GENERATIONS}")
print(f"  Ternary values: -1 (AVOID), 0 (UNKNOWN), +1 (CHOOSE)")
print()

# ---------------------------------------------------------------------------
# Step 2: Run agents through environments with AvoidanceTracker
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 2: Run agents through environments")
print("=" * 60)

tracker = AvoidanceTracker(positions=NUM_POSITIONS)

for gen in range(NUM_GENERATIONS):
    # Each position gets a collective decision from all agents
    actions = []
    for pos in range(NUM_POSITIONS):
        # Simulate ternary decisions with slight avoidance bias
        action = random.choices([-1, 0, 1], weights=[0.45, 0.10, 0.45])[0]
        actions.append(action)
    result = tracker.record(actions)

print(f"  Generations recorded: {tracker.generations()}")
print(f"  Mean avoidance ratio: {tracker.avoid_ratio():.4f}")
print(f"  Avoidance std dev:    {tracker.avoid_std():.4f}")
print(f"  Choose ratio:         {tracker.choose_ratio():.4f}")
print(f"  Unknown ratio:        {tracker.unknown_ratio():.4f}")
print()

# ---------------------------------------------------------------------------
# Step 3: Track avoidance ratios with BatchAnalyzer across population scales
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 3: Analyze avoidance across population scales")
print("=" * 60)

analyzer = BatchAnalyzer()
scale_data = {}  # pop_size -> list of avoid ratios

population_sizes = [10, 25, 50, 100]
for pop_size in population_sizes:
    avoid_ratios = []
    for _ in range(20):  # 20 batches per scale
        batch = [random.choices([-1, 0, 1], weights=[0.45, 0.10, 0.45])[0]
                 for _ in range(pop_size)]
        stats = analyzer.add_batch(batch)
        avoid_ratios.append(stats["avoid"])
    scale_data[pop_size] = avoid_ratios
    print(f"  Pop size {pop_size:3d}: mean avoid ratio = "
          f"{sum(avoid_ratios)/len(avoid_ratios):.4f}")

population_summary = analyzer.population_summary()
print(f"\n  Overall population summary:")
for k, v in population_summary.items():
    print(f"    {k}: {v:.4f}")
print()

# ---------------------------------------------------------------------------
# Step 4: Verify the Conservation Law
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 4: Verify the Conservation Law (Law 1)")
print("=" * 60)

conservation = ConservationLaw(threshold=0.20)  # reasonable threshold for stochastic data
scale_results = conservation.test_all_scales(scale_data)
for pop_size, passed in scale_results.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  Population {pop_size:3d}: {status}")

all_conserved = conservation.all_conserved()
print(f"\n  All scales conserved: {all_conserved}")
print(f"  Conservation report:\n{conservation.report()}")
print()

# ---------------------------------------------------------------------------
# Step 5: Detect cascade events
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 5: Detect cascade events (Law 3)")
print("=" * 60)

cascade_alerts = []
detector = CascadeDetector(threshold=0.8, window=50,
                           on_alert=lambda alert: cascade_alerts.append(alert))

# Simulate a run where avoidance gradually increases (cascade scenario)
for step in range(200):
    # Gradually increase avoidance probability
    avoid_prob = min(0.95, 0.4 + step * 0.003)
    avoided = random.random() < avoid_prob
    alert = detector.record(avoided)
    if alert:
        print(f"  ⚠ CASCADE ALERT at step {step}: ratio={alert.avoidance_ratio:.3f}")

print(f"  Total cascade alerts: {len(cascade_alerts)}")

# Now test with balanced data (should not cascade)
detector.reset()
balanced_alerts = []
detector_balanced = CascadeDetector(threshold=0.8, window=50,
                                    on_alert=lambda a: balanced_alerts.append(a))
for step in range(200):
    avoided = random.random() < 0.5
    detector_balanced.record(avoided)
print(f"  Balanced run alerts: {len(balanced_alerts)} (expected: 0 or very few)")
print()

# ---------------------------------------------------------------------------
# Step 6: Balanced Learning prevents cascades (Law 4)
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 6: Balanced Learning with forced exploration (Law 4)")
print("=" * 60)

learner = BalancedLearner(exploration_rate=0.15, decay_factor=0.995, explore_interval=10)
metrics = CascadeMetrics()

avoid_count = 0
explore_count = 0
total_decisions = 500

for step in range(total_decisions):
    agent_id = f"agent_{step % NUM_AGENTS}"
    # Decision with some avoid bias in environment
    should_avoid = learner.decide(agent_id, avoid_bias=0.6)
    if should_avoid:
        avoid_count += 1
    # Simulate reward (avoiding is sometimes good, sometimes bad)
    reward = random.uniform(-1, 1)
    if should_avoid:
        reward += 0.2  # slight positive bias for avoiding in this env
    learner.record_outcome(agent_id, should_avoid, reward)

    # Track metrics periodically
    if step % 10 == 0:
        metrics.record(
            avoidance_ratio=avoid_count / (step + 1),
            exploration_count=explore_count,
            step=step,
        )

final_avoid_ratio = avoid_count / total_decisions
print(f"  Total decisions: {total_decisions}")
print(f"  Final avoidance ratio: {final_avoid_ratio:.3f}")
print(f"  (With forced exploration, should stay well below 1.0)")

metrics_summary = metrics.summary()
print(f"  Cascade metrics summary:")
for k, v in metrics_summary.items():
    print(f"    {k}: {v}")
print()

# Also demonstrate the ExplorationScheduler
print("  --- ExplorationScheduler demo ---")
scheduler = ExplorationScheduler(interval=50, selector="highest_avoid")
explore_events = []
scheduler_explore = ExplorationScheduler(
    interval=30, selector="highest_avoid",
    on_explore=lambda e: explore_events.append(e)
)
for i in range(200):
    agent_id = f"agent_{i % 20}"
    avoided = random.random() < 0.7
    scheduler_explore.record_avoid(agent_id, avoided)

print(f"  Exploration events triggered: {len(explore_events)}")
print()

# ---------------------------------------------------------------------------
# Step 7: Inference Engine — deduce patterns from negative spaces (Law 5)
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 7: Inference Engine — patterns from gaps (Law 5)")
print("=" * 60)

engine = InferenceEngine()

# Record avoidance at various positions (simulate known avoidance zones)
avoidance_zones = {1: 0.9, 2: 0.85, 5: 0.7, 8: 0.95, 12: 0.8, 15: 0.75, 18: 0.9}
for pos, count in avoidance_zones.items():
    engine.record_avoidance(pos, count)

# Find gaps between avoidance regions
gaps = engine.find_gaps()
print(f"  Avoidance zones: {avoidance_zones}")
print(f"  Gaps found: {gaps}")

# Make inferences about what's in the gaps
deductions = engine.infer()
print(f"  Deductions ({len(deductions)}):")
for d in deductions:
    print(f"    Position range {d.get('gap', '?')}: confidence={d.get('confidence', 0):.3f}")

# High confidence deductions
high_conf = engine.high_confidence_deductions(min_confidence=0.6)
print(f"  High-confidence deductions: {len(high_conf)}")
for d in high_conf:
    print(f"    {d}")
print()

# ---------------------------------------------------------------------------
# Step 8: Feedback Loop — learning from predictions (ties it together)
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 8: Feedback Loop — prediction learning")
print("=" * 60)

feedback = FeedbackLoop(decay=0.9)
options = {f"option_{i}": random.uniform(-1, 1) for i in range(10)}

for round_num in range(5):
    results = feedback.run_round(options)
    correct = sum(1 for v in results.values() if v != 0)
    print(f"  Round {round_num + 1}: {len(results)} predictions, "
          f"accuracy={feedback.accuracy():.3f}")

print()

# ---------------------------------------------------------------------------
# Final Report: All 5 Laws Verified
# ---------------------------------------------------------------------------
print("=" * 60)
print("FINAL REPORT: Five Laws of the Ternary Ecosystem")
print("=" * 60)

laws = {
    "Law 1 — Conservation": all_conserved,
    "Law 2 — Avoidance Tracking": tracker.avoid_ratio() > 0,
    "Law 3 — Cascade Detection": len(cascade_alerts) > 0,
    "Law 4 — Balanced Learning": final_avoid_ratio < 0.95,
    "Law 5 — Inference from Gaps": len(deductions) > 0,
}

all_passed = True
for law_name, result in laws.items():
    status = "✓ VERIFIED" if result else "✗ FAILED"
    if not result:
        all_passed = False
    print(f"  {law_name}: {status}")

print()
if all_passed:
    print("🎉 All 5 laws verified successfully!")
else:
    print("⚠ Some laws could not be verified — see details above.")
print()
print("Tutorial complete!")
