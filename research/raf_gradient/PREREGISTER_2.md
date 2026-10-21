# Rung 2 pre-registration — RAF node-perturbation training + sign-inversion stress

**Written before any rung-2 result is read. Append dated addenda; do not rewrite.**

## Question

Rungs 1 / 1b established, on a *single* threshold unit, that a bank of the real
Spikeling `ResonatorState` oscillators can read a per-parameter gradient by resonance
(`sign(⟨v·ref⟩)·sqrt(energy)`), robust to channel crowding for the sign, fragile for
the magnitude, and **only where the dither straddles the threshold**. Rung 2 asks the
question that actually matters:

> Can that readout **train a real 2-layer spiking network** end-to-end — with one
> resonator per hidden unit listening to a single global error signal — to within
> reach of surrogate-gradient BPTT, **including when hidden-unit gradient signs invert
> during training** because their outgoing weights change sign?

## Network + task (fixed)

- `SpikeNet`: `n_in → n_hid → 1`. Hidden `h_j = 1[v_j > 0]`, `v_j = (W1ᵀx)_j + b1_j`.
  Output `ŷ = 1[w2ᵀh + b2 > 0]`. Loss `L(θ) = mean_n ½(ŷ_n − y_n)²`.
- **BPTT baseline** uses the fast-sigmoid surrogate `σ'(v) ≈ 1/(1+10|v|)²` at both
  layers. This is the reference gradient and the reference accuracy.
- **Stress task** (`--task stress`): binary classification requiring the hidden layer
  (curved boundary, two Gaussian clusters per class), `n_in=4`, `n_hid=6`, `N=48`
  train / `48` test, fixed seed. `w2` is **initialised with alternating signs**
  `[+,−,+,−,+,−]·0.5`; the task + learning rate are chosen so at least two `w2_j`
  cross zero during training (verified from the BPTT run's weight trace).

## Methods compared

1. **RAF-full** — node perturbation with the real resonator readout. Each gradient
   step: dither every hidden bias `b1_j(t) = b1_j0 + A_j sin(ω_j t)` at its own
   `f_j` (log-spaced 300–3000 Hz); feed the DC-removed `Σ_n L_n(t)` as one shared
   drive to a per-unit `ResonatorState`; estimate
   `δ̂_j = sign(⟨v_j·sin ω_j t⟩) · sqrt(energy_ema_j)`, scale-matched to the ideal
   node-perturbation δ by a single global least-squares factor per step.
   Weight update: `ΔW1_ij = −lr·δ̂_j·mean_n x_{n,i}`, `Δb1_j = −lr·δ̂_j`.
   Output layer `w2, b2` updated with its **local exact-cheap** gradient
   `∂L/∂w2_j = mean_n (∂L/∂ŷ_n)·h_{n,j}` (surrogate on the output step) — this scalar
   `∂L/∂ŷ` broadcast is the only top-down signal the hidden resonators ever see.
- **Per-unit dither rule (the thing being tested for "smoothing"):**
  `A_j = clip(1.5 · mean_n|v_{n,j}|, 0.05, 0.5)`, recomputed each step.
2. **RAF-sign** — identical but `δ̂_j = sign(⟨v_j·ref⟩)` only (unit magnitude).
3. **nodepert-ideal** — identical pipeline, but δ_j from an ideal digital lock-in on
   the loss (`2⟨L·sin ω_j t⟩/A_j`). Ceiling for the perturbative family.
4. **BPTT** — full surrogate backprop. Reference.

## Ground truth for the sign diagnostic

Per step, the reference hidden error `δ_j^ref = ∂L/∂b1_j` from the BPTT backward pass
on the *current* weights (also cross-checked once against central FD on `b1_j`).

## Pre-registered predictions

- **R1 (it learns).** RAF-full test accuracy reaches **≥ 0.90 ×** the BPTT baseline's
  final test accuracy within **20×** the gradient steps.
- **R2 (sign inversion is tracked).** Over all training steps,
  `sign(δ̂_j) == sign(δ_j^ref)` at rate **≥ 0.85** for **both** the `w2_j > 0` and the
  `w2_j < 0` hidden units, and **≥ 0.75** in the ±3-step window around any `w2_j`
  zero-crossing. The readout must *follow* the inversion, not assume a fixed sign.
- **R3 (cost, not speed).** Report forward-evaluations-per-gradient-step (RAF ≈
  `ticks`, BPTT = 1) and accuracy-vs-evaluations curves. Expected: RAF within a few %
  of BPTT accuracy at **100–1000×** the forward evaluations. This is a
  substrate-feasibility result, not a wall-clock win.
- **R4 (sign is most of it).** RAF-sign reaches **≥ 0.80 ×** RAF-full final accuracy —
  consistent with rung 1b ("sign is the robust half").

## Instrument checks (gates)

- The vectorised resonator used in the training loops matches the real
  `ResonatorState.step()` to **< 1e-9** max abs state difference over 8000 ticks on a
  fixed drive (`--selftest`).
- BPTT baseline test accuracy on the stress task **> 0.60** (else task/harness broken).
- On a random init, `r(δ̂, δ^ref) > 0.9` (RAF readout wired right before any training).

## Kill condition

If **R1 fails** (RAF-full cannot get within 0.90× baseline in 20× steps) **or R2 fails**
(sign-agreement < 0.85 for the negative-`w2` units) — rung 2 is an **honest negative**:
the frequency-multiplexed RAF backward pass does not train a real multi-layer net.
Record it, stop, do not add epicycles (deeper recursion, per-weight frequencies, etc.).

## Addendum 2026-09-07 (after `--selftest`, before any training run)

Instrument gate #3 as written (`r(δ̂, δ^ref) > 0.9` over all live units, unweighted)
**FAILED at r = 0.62**. Diagnosis: the failure is a metric artifact, not a readout
failure. With a hard staircase loss, hidden units far from every sample's threshold
contribute ≈ 2e-4 to δ^ref — an order of magnitude below the units that carry the
gradient — and their phase is undefined noise on *both* the reference and the estimate.
The flat unweighted correlation gives those numerically-zero units equal weight. On the
units that actually matter the readout is faithful: **|δ^ref|-weighted sign-agreement =
0.977; on units carrying > 10% of max|δ^ref| (the ones that get a real weight update),
sign-agreement = 1.000 and r = +0.98.**

Corrected instrument gate (decided before reading any training result):

- **Gate 3a:** |δ^ref|-weighted sign-agreement(δ̂, δ^ref) **≥ 0.90** on a random init.
- **Gate 3b:** unweighted r(δ̂, δ^ref) **≥ 0.90** restricted to units with
  |δ^ref| > 0.1·max|δ^ref|.
- The unweighted all-units r is still reported, as context, not as a gate.

R2's sign-inversion agreement rates are likewise now **|δ^ref|-weighted within each
w2-sign subset**, so near-zero units cannot dominate the diagnostic.

## Addendum 2026-09-07 #2 (after the first training run)

Two findings, both recorded before the RESULTS write-up:

1. **The instrument (task) is not validated.** Trained to convergence the BPTT
   baseline reaches only **0.688 test acc** (train loss 0.042 — clear overfit) on the
   48-train / 48-test stress task. For an accuracy-based comparison to mean anything
   the reference method must cleanly solve the task (> 0.9). It does not, so **R1 and
   R4 are uninterpretable** — a pass or fail on them here is within run-to-run noise
   (raf_full 0.833 "beating" BPTT is noise, not signal). Only R2 (a sign diagnostic
   vs the per-step reference δ, not an accuracy number) and R3 (a cost count) are
   readable from this run.

2. **R2 fails and it is not cross-talk.** At init the RAF readout is faithful on
   gradient-carrying units of *both* w2 signs (−w2 units: sign-correct on the two
   units with |δ| ≫ 0, wrong only on the one with |δ| ≈ 7e-4). Averaged over 150
   training steps, −w2 sign-agreement collapses to **0.362** (vs +w2 **0.994**).
   Dithering the −w2 units in isolation (no +w2 channels active) reproduces the *same*
   init pattern — so the training-time collapse is **not** inter-channel cross-talk.
   Leading candidate: as weights grow, −w2 hidden units are driven far enough from
   threshold that the capped adaptive dither (A ≤ 0.5) can no longer straddle them,
   so their phase estimate is noise — the rung-1 "no smoothing past the dither
   amplitude" wall, now hitting one w2-sign subset selectively. `n_crossings = 0`:
   the "during a w2 zero-crossing" sub-test was never exercised.

**Kill condition is triggered** (R2 −w2 sign-agreement 0.362 < 0.85). Per the method:
record as an honest negative, stop, add no epicycles. A future rung-2 redo must first
validate the task (BPTT > 0.9) and give the adaptive dither a way to track units that
run away from threshold.

## What a pass licenses

A write-up of "the backward pass as a bank of resonate-and-fire neurons" as a
*substrate-feasible* on-chip learning rule for multi-layer spiking nets — with the
honest cost multiplier from R3 — and only then a look at whether more than one
resonator layer (the original "each gradient spike its own SNN") buys anything.
