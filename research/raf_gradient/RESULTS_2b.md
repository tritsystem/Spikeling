# Rung 2b results — 2026-09-07

Rung 2 redone with the three fixes its addenda demanded. Harness
`raf_gradient_rung2b.py`; VecBank asserted bit-identical to `ResonatorState.step()`
(Δ = 0.0). Primary run `--steps 400 --ticks 2000 --seed 0`; multi-seed check
seeds 0–3. Raw: `results_2b_compare.json`; figure `rung2b_summary.png`; two dated
addenda in `PREREGISTER_2b.md`.

## Verdict: partial pass — the practical claim holds, the mechanistic one does not

| pre-reg | result | reading |
|---|---|---|
| **Gate T** BPTT solves the task > 0.90 | **0.968–0.980** across seeds (train loss ~0.01) | **PASS** — task validated (rung 2's was not) |
| **Gate V** vec == real resonator | Δ = 0.0 | PASS |
| **Gate W** init `r(δ̂,fd) ≥ 0.9`, weighted sign-agree ≥ 0.9 | r **0.992**, sign-agree **1.000** | PASS |
| **S1** RAF-full ≥ 0.90× BPTT acc in ≤ 20× steps | raf_full **0.960–0.985** vs BPTT 0.968–0.980, **every seed** | **PASS** — solid |
| **S2** sign-agree(δ̂, fd) ≥ 0.85 for both w2 signs, signal-present steps | **+w2 0.93–1.00** (pass) · **−w2 0.52 / 0.77 / 0.82 / 1.00** (seeds 0–3) | **FAIL** for −w2 — and *unreliable*, not just low |
| **S3** sign-agree ≥ 0.75 around w2 zero-crossings | `tcross_weight = 0` every seed | **N/A** — never exercised (see below) |
| **S4** tracking rule keeps unreachable < 10% and below the capped rule | tracking **0.00** vs capped **0.13–0.18**, every seed | **PASS** — clean win |
| **R4** RAF-sign ≥ 0.80× RAF-full acc | 0.965 vs 0.774 | PASS |
| **S5** cost | ~3500× forward-evals per step | as pre-registered |

## What the two fixes bought

1. **Task validation (Gate T).** 2-D XOR-of-Gaussians, N=400/400, no distractor dims:
   surrogate BPTT reaches 0.97–0.98 every seed. Accuracy comparisons are now readable.

2. **Threshold-tracking dither rule (S4) — the unambiguous win.**
   `A_j = clip(1.2·quantile(|v_j|, 0.30), 0.02, 3.0)` grows as a unit drifts from the
   flip. Result: **0% unreachable unit-steps** across all seeds, vs 13–18% under
   rung 2's capped `A_j ≤ 0.5` rule. The specific mechanism asked for works.

3. **As a consequence, S1 now passes solidly.** RAF node-perturbation — one real-math
   resonator per hidden unit, `δ̂_j = sign(⟨v_j·ref⟩)·sqrt(energy_j)`, listening to a
   single global loss broadcast — **trains the 2-layer spiking net to BPTT-level test
   accuracy on every seed.** Rung 2 could not show this because its task was broken.

## What still fails, and why it matters

**S2: per-unit hidden-gradient *sign fidelity* for minority-sign (−w2) units is
unreliable** — 0.52 to 1.00 across four seeds (mean ~0.78), while +w2 units sit at
0.93–1.00. The variance is the finding: you cannot count on the RAF readout to get a
−w2 hidden unit's gradient sign right. Contributing factors, in order of evidence:

- **Signal collapse near convergence** (middle panel of the figure). Once train-loss
  drops below ~0.10 the hard-classification loss is a coarse staircase — one sample
  flip changes `L` by `1/800` — so `fd_j` itself is 1–2-sample quantization noise and
  its sign is close to random. This hits node perturbation regardless of the readout;
  BPTT's surrogate is immune because `σ'(v)` is nonzero for every sample. The
  signal-present filter (loss > 0.10) removes most but not all of this.
- **Rectification asymmetry.** `h_j = 1[v_j>0]` is one-sided around the dither; for a
  unit whose samples sit mostly on one side of threshold, the perturbation "sees" an
  unrepresentative subset of samples, and for `w2_j < 0` the marginal loss sign of
  that subset can disagree with the batch-summed `fd` sign.
- **Low power.** Few −w2 units are ever in the gradient-carrying "kept" set, so the
  per-seed −w2 number is an average over a handful of unit-steps.

**S3 was never exercised.** The validated task converges in ~40 steps; after that the
gradient-carrying units have stable `w2` signs, so no *load-bearing* `w2_j` crosses
zero during the informative window on any seed. The sign-*inversion-in-flight* case
that motivated this whole rung still has no measurement.

## The tension, stated plainly

**S1 passes while S2 fails.** RAF node-perturbation trains the net fine even though its
per-unit sign estimate for minority-sign hidden units is unreliable — because those
updates are small (scale-matched to a small `fd`), the +w2 units and the exact
output-layer update carry the load, and the errors are early-averaged-out noise rather
than systematic. **Training success here does not validate the credit assignment; it
tolerates its errors.**

## Consequence for the ladder

- The **substrate-feasibility claim survives**: a resonator bank in the backward path,
  with a threshold-tracking dither, can train a small multi-layer spiking net to
  BPTT-level accuracy at ~3500× the forward-evaluation cost. That is a real, if narrow,
  positive result — worth a short write-up with the honest cost multiplier.
- The **mechanistic claim does not**: the readout is not a faithful per-unit gradient
  for minority-sign units, and near convergence node perturbation on a hard classifier
  has little signal to read at all.
- **S3 remains open.** A rung 2c would need a task that keeps a load-bearing hidden
  unit near threshold with an oscillating outgoing weight long enough to measure sign
  tracking through an actual inversion — and a loss that does not saturate into a
  staircase (e.g. a regression target, or many more samples).
- Still nothing supports the original "each gradient spike its own SNN" recursion.
