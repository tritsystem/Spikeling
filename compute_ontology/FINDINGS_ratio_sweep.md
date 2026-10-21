# Findings: specialist:LIF ratio sweep — dose-dependence test

**Status:** measured + written up 2026-09-04 (`heterogeneous_memory_ratio_sweep_test.py`).

## Question

The mixed-ensemble rung (`FINDINGS_mixed_ensemble.md`) found that a 16-specialist/16-LIF
mix does NOT rescue the high-pattern-load memory collapse — it tracks pure heterogeneous's
collapse almost exactly. Two live explanations were left open: (a) dose-dependent — 16/16
just wasn't enough dilution, a smaller specialist core would rescue more; (b) hard floor —
any meaningful specialist fraction triggers the collapse regardless of ratio.

## Design

Swept two new ratios (8-specialist = 2/type + 24-LIF; 24-specialist = 6/type + 8-LIF),
same total 32 neurons, same diagnostic P points (2,4,6,8,11), same 8 reps/point, same
real 95% CIs, reusing `run_capacity_test`/`mean_ci95` unchanged. Generalized factory
function validated bit-for-bit identical to the original `make_mixed_pops` at matching
parameters (same seed -> same output) before trusting new ratios. Combined with existing
reference data (0-specialist = pure homogeneous, 16-specialist = the already-measured
mixed condition, 32-specialist = pure heterogeneous) for the full 5-point sweep.

## Result: hard floor confirmed, dose-dependence FALSIFIED

| P | 0-spec | 8-spec | 16-spec | 24-spec | 32-spec |
|---|---|---|---|---|---|
| 2  | 82.1% ± 4.1% | 80.4% ± 6.1% | 83.8% ± 3.6% | 81.7% ± 4.8% | 86.2% ± 6.2% |
| 4  | 55.2% ± 5.6% | 52.5% ± 9.3% | 52.3% ± 8.4% | 51.2% ± 9.0% | 47.3% ± 11.5% |
| 6  | 48.3% ± 6.1% | 43.3% ± 10.5% | 43.8% ± 7.8% | 42.5% ± 10.4% | 44.2% ± 7.2% |
| 8  | **52.5% ± 5.5%** | 33.0% ± 7.8% | 38.5% ± 7.7% | 33.6% ± 4.9% | 34.5% ± 6.0% |
| 11 | **51.7% ± 2.3%** | 34.0% ± 2.3% | 34.0% ± 2.8% | 31.4% ± 2.4% | 33.6% ± 2.6% |

At P=8 and P=11, all four specialist-containing conditions (8/16/24/32) sit in the same
~31-38% band, with no monotonic trend across the ratio axis at all — 8-specialist (25%
of the ensemble, the most-diluted condition tested) collapses just as hard as pure
heterogeneous (32-specialist, 100%). Every specialist-containing condition sits clearly,
non-overlappingly below the 0-specialist homogeneous control at high P. This is a genuine
cliff, not a dose-response curve: presence of ANY specialist fraction triggers the
collapse; the amount doesn't matter across the 25%-100% range tested.

**Bonus finding, also decisive:** the P=2 "heterogeneous advantage" shows no trend
either — 80.4% to 86.2% across all five ratios, all mutually overlapping CIs. Combined
with the mixed-ensemble rung's earlier observation that this edge didn't clearly
replicate in a second independent measurement, this sweep is a third independent
data point showing no separation at P=2 across 5 different neuron-composition conditions.
The original P=2 "heterogeneous beats homogeneous, non-overlapping CIs" result from the
first crossover measurement now looks like it was most likely noise at n=8 reps for a
genuinely small or nonexistent low-P effect, not a robust finding.

## Mechanistic implication

Rules out "dilution by count" as the rescue mechanism (explanation (a) from the mixed-
ensemble rung) — cleanly falsified across a 4x range of specialist count (8 to 32). Leaves
explanation (b) as the live account: the specialists' confident-wrong completions dominate
the ridge-regression readout's learned weights out of proportion to their population share,
so even a 25% specialist minority is enough to corrupt the readout at high pattern load.
Still untested directly (would need to inspect trained weight magnitudes per neuron-type
block, not just behavioral recall accuracy, to confirm this mechanism rather than infer it).

## Bottom line

Two rungs now agree: the low-P heterogeneous advantage looks like noise, not a real effect,
and the high-P collapse is a hard, ratio-independent floor triggered by any specialist
presence. The practical implication for `heterogeneous-neuron-ontology`: composition-based
mitigation (mixing in plain LIF at any tested ratio) does not fix the memory-capacity
tradeoff. A real fix would need to target the readout mechanism itself (e.g. per-type
regularization on the ridge weights, or an ensemble-of-readouts approach), not the neuron
population mixture — untested, a different kind of next rung than this one.
