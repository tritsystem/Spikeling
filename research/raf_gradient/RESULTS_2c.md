# Rung 2c results — 2026-09-07

Regression target (hard spiking hidden layer, linear readout, MSE) + a mid-training
`β` sign-flip that forces a subset of `w2` through zero. Harness
`raf_gradient_rung2c.py` (reuses rung 2b's dither rule, FD ground truth, resonator
probe; VecBank bit-identical to `ResonatorState`). Primary `--steps 300 --ticks 1500
--seed 0`; multi-seed 0–3. Raw `results_2c_compare.json`; figure `rung2c_summary.png`;
one dated addendum in `PREREGISTER_2c.md`.

## Verdict: the two things 2c was built to test PASS; the tight training/fidelity bars FAIL

| pre-reg | result (seed 0 · multi-seed 0–3) | reading |
|---|---|---|
| **Gate T** BPTT test R² ≥ 0.85 | 0.966 · 0.96–0.97 all seeds | PASS — task validated |
| **Gate V** vec == real resonator | Δ = 0.0 | PASS |
| **Gate W** init sign-agree = 1.00 (≥3/4 seeds) | 1.000 all seeds (r 0.89–1.00 where ≥3 units kept) | PASS (addendum: r-criterion was underpowered, sign-agree is the gate) |
| **Gate F** `max\|fd\|` last-20% ≥ 0.25× first-20% — *does the signal survive convergence?* | **0.40** | **PASS — the regression target fixes rung 2b's staircase collapse** |
| **U1** RAF-full test MSE ≤ 1.25× BPTT | 0.045 vs 0.029 (R² 0.935); multi-seed R² **0.79 / 0.81 / 0.93 / 0.94** | **FAIL** — learns substantially, misses the tight bar, seed-variable |
| **U2** bulk sign-agree ≥ 0.85 both w2 signs, all + late windows | −w2 **0.62/0.73/0.67/0.78**; +w2 also ~0.5–0.7 | **FAIL** — bulk per-step per-unit sign is unreliable for *both* signs |
| **U3** sign-agree ≥ 0.75 in ±3 steps around every `w2_j` zero-crossing — *the measurement rungs 2/2b never got* | **0.987 / 1.000 / 1.000 / 1.000** across 4 seeds, ~24 crossings total | **PASS — clean, replicated** |
| **U4** RAF-sign MSE ≤ 1.5× RAF-full | 0.045 vs 0.067 | PASS |
| **U5** cost | ~3000× forward-evals per step | as pre-registered |

## What 2c establishes that 2b could not

1. **Gate F — the FD signal no longer collapses.** With a regression target the density
   of samples at each hidden threshold stays nonzero at the optimum (residuals small
   but continuously valued, not quantized to `1/2N`). `max|fd|` ends at 0.40× its
   early value vs rung 2b where it fell to the one-sample-flip floor and its sign went
   random. 2c's premise is validated.

2. **U3 — sign-tracking through an outgoing-weight inversion works.** The `β` flip
   forces gradient-carrying hidden units' `w2_j` through zero (5–7 crossings per seed).
   In the ±3-step window around each, `sign(δ̂_j) == sign(fd_j)` at **0.99–1.00** on
   every seed. When a hidden unit's role is actively inverting, the RAF
   velocity-phase readout follows it. `raf_sign` matches; only the digital-lock-in
   ceiling (`nodepert_ideal`) does *worse* at crossings (0.86, low weight).

## Why U1 / U2 still fail — and it is not the resonator

The readout is **sharp exactly at the inversions (U3 ≈ 1.0) and mushy in the bulk
(U2 ≈ 0.6)**. Mechanism: 6 hidden units serving a 2-feature target is heavily
over-parameterised. In the steady state multiple hidden units are near-interchangeable,
so an individual unit's gradient *sign* is genuinely ill-posed — the FD "truth" itself
trades off between redundant units step to step. At a `w2_j` zero-crossing that unit's
role is momentarily well-defined by the residual, so the sign sharpens. So U2's low
number is partly a property of the problem, not only the estimator; U1's shortfall
(R² 0.79–0.94 vs BPTT 0.96) follows from feeding those mushy bulk signs into the
weight update.

**Kill condition is triggered** (U1 fails and U2 fails for `w2 < 0`). Per the method:
stop, no epicycles. But U3 — the reason this rung exists — is a genuine replicated
pass.

## Cross-rung synthesis (rungs 1 → 2c)

| claim | status |
|---|---|
| resonator bank = frequency-multiplexed gradient **readout** | **holds** (rung 1) — magnitude fragile, needs the dither to straddle the threshold |
| **sign** recoverable from `ResonatorState.v` phase | **holds** (rung 1b isolated; rung 2c through inversions) |
| threshold-**tracking dither** keeps units reachable | **holds** (rung 2b: 0% vs 13–18% unreachable) |
| RAF backward-path bank can **train** a multi-layer spiking net | **holds, qualified** — classification acc 0.97 (2b), regression R² 0.79–0.94 (2c), at ~3000–3500× forward-eval cost |
| readout gives a **faithful per-step per-unit** gradient | **does not hold** — bulk sign-agree 0.5–0.8; and in over-parameterised nets the per-unit sign is partly ill-posed |
| tracks hidden gradient sign **through a `w2` inversion** | **holds** (2c U3, 0.99–1.00, 4 seeds) |
| "each gradient spike is its own SNN" recursion | **unsupported** at every rung |

A rung 2d, if pursued, would be about **network sizing** (match `n_hid` to the task so
per-unit signs are well-posed) and a real learning-rate/step-budget sweep for U1 — not
about the resonator readout, which has now been characterised.
