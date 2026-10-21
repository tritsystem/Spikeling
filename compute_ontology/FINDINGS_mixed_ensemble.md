# Findings: does a mixed specialist+LIF ensemble rescue the high-P memory collapse?

**Status:** measured 2026-08-30 (`heterogeneous_memory_mixed_test.py`, commit `dd53c1a`),
write-up completed 2026-09-04 — the run existed and was correct, but no finding
document or vault entry was ever written for it until now.

## Pre-registration (as stated in the script's own docstring, written before the run)

Direct follow-up to the real, replicated crossover in
[[heterogeneous-memory-capacity-crossover-finding]]: pure heterogeneous
(8 each of LIF/Izhikevich/AdEx/Resonator) beats pure homogeneous (32x LIF) at
dropped-channel recall for low stored-pattern load (P=2), but *loses*,
collapsing to a tight below-chance plateau, at high load (P>=8).

**Working mechanism proposed for the collapse:** each specialist binds
tightly to its own channel — good for detection, but when that channel's
evidence is missing from a corrupted cue, the specialist produces a
confident, systematically *wrong* completion rather than hedging. Plain LIF's
blurrier population code regularizes toward the mean instead, degrading more
gracefully as more patterns compete.

**Prediction under test:** a MIXED ensemble — a smaller specialist core
(4 of each of the 4 types = 16) plus a larger plain-LIF regularizer block
(16), same 32 total neurons as both pure conditions — recovers the low-P
specialist advantage while damping the high-P collapse, because the LIF
sub-population supplies the missing "hedge."

**Falsify:** mixed tracks pure-heterogeneous's high-P collapse (no rescue),
and/or fails to show pure-heterogeneous's low-P advantage over homogeneous.

**Design:** same real methodology as the original crossover rung — 8
independent replications per P (different stored pattern *set* each rep, not
just different noise), real 95% CIs (normal approximation), same ridge-regression
dropped-channel-recall readout, same reused code
(`run_capacity_test`/`mean_ci95` imported unchanged from
`heterogeneous_memory_test.py`). Diagnostic points only (P = 2, 4, 6, 8, 11 —
the pure-condition sweep's low/mid/high anchors), not the full 1..11 sweep,
to keep runtime bounded (~29 min total across 3 conditions x 5 P x 8 reps).

## Result: DISCONFIRMED

| P | Heterogeneous (pure) | Mixed (specialist core + LIF) | Homogeneous (pure, 32x LIF) |
|---|---|---|---|
| 2  | 86.2% ± 6.2% | 83.8% ± 3.6% | 82.1% ± 4.1% |
| 4  | 47.3% ± 11.5% | 52.3% ± 8.4% | 55.2% ± 5.6% |
| 6  | 44.2% ± 7.2% | 43.8% ± 7.8% | 48.3% ± 6.1% |
| 8  | 34.5% ± 6.0% | 38.5% ± 7.7% | **52.5% ± 5.5%** |
| 11 | 33.6% ± 2.6% | 34.0% ± 2.8% | **51.7% ± 2.3%** |

Chance = 50%.

**High-P collapse: NOT rescued.** At P=11, mixed (34.0% ± 2.8%) is
statistically indistinguishable from pure heterogeneous (33.6% ± 2.6%) — the
CIs are near-identical — and both remain clearly, non-overlappingly below
homogeneous (51.7% ± 2.3%). Diluting the specialist population with an equal
number of plain-LIF regularizers did essentially nothing to damp the
collapse. Same story at P=8 (38.5% vs 34.5%, homogeneous still clearly ahead
at 52.5%).

**Low-P advantage: did not clearly replicate either.** At P=2 this run, all
three conditions overlap (86.2%, 83.8%, 82.1%, all within ~4-6 points with
overlapping CIs) — homogeneous alone came within CI of matching pure
heterogeneous here, unlike the original crossover finding's P=2 result
(85.8% ± 4.4% vs 76.7% ± 4.1%, non-overlapping). Both are real, independently-seeded
8-rep measurements of the same design; the discrepancy is most likely
run-to-run variance at a genuinely small effect size (8 reps is not a lot of
statistical power for a ~4-9 point gap), not a bug — but it means the
low-P heterogeneous edge should be held with less confidence than the
original single non-overlapping-CI result implied on its own.

## Honest mechanistic read

The proposed "dilution as regularization" mechanism does not hold at the
tested ratio (50% specialist / 50% plain LIF by neuron count). Two live
possibilities, neither tested here:
1. The specialists' confident-wrong completions dominate the readout's
   learned weights disproportionately to their population share — diluting
   by count isn't the same as diluting by influence, and a 50/50 split may
   still be specialist-dominated in the weights ridge regression actually
   assigns.
2. The mechanism itself (confident-wrong vs. graceful hedging) may be the
   wrong explanation for the collapse — an alternative untested account is
   that specialist channels simply carry less *linearly decodable*
   information about the other three channels' cross-correlations at high P,
   regardless of confidence, and no amount of LIF mixed in fixes a genuine
   information deficit in the specialist sub-population's code.

Distinguishing these needs either a specialist-ratio sweep (e.g. 8/24,
16/16 tested here, 24/8) or inspecting the trained readout's weight
magnitudes per neuron-type block directly — neither done in this rung.

## Bottom line

The mixed-ensemble hypothesis is a clean disconfirm, not a refinement: adding
plain-LIF regularizers in a 1:1 ratio with specialists does not recover
associative-memory capacity at high pattern load. The real, still-standing
result remains the plain crossover from the prior rung — heterogeneous wins
small-working-memory tasks, homogeneous wins larger stored-pattern sets — and
this rung failed to find an easy composition-based fix for that tradeoff.

## Next rungs (not run)

- Sweep the specialist:LIF ratio itself (8/24, 24/8) rather than only the
  one 16/16 split, to check whether the collapse is dose-dependent or a hard
  floor regardless of mixture.
- Inspect trained ridge-regression weight magnitudes per neuron-type block to
  test mechanism (1) vs (2) above directly, rather than only the behavioral
  recall accuracy.
- Re-run the pure P=2 heterogeneous-vs-homogeneous comparison with more than
  8 replications to resolve whether that low-P advantage is real and just
  noisy at n=8, or was itself closer to a false positive than the original
  single measurement suggested.
