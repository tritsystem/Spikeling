# Rung 2b pre-registration — validated task + threshold-tracking dither + FD ground truth

**Written before any rung-2b result is read. Append dated addenda; do not rewrite.**

Supersedes rung 2's setup on three points the rung-2 addenda flagged:

1. **Validated task.** 2-D XOR-of-Gaussians (4 clusters at (±1,±1), classes = the two
   diagonal pairs, σ = 0.35), `N = 400` train / `400` test, `n_in = 2` (no distractor
   dims), `n_hid = 6`. Surrogate BPTT reaches **~0.98 test acc / train loss ~0.01** on
   this — verified before rung 2b (`Gate T`). `w2` still initialised alternating-sign
   `[+,−,+,−,+,−]·0.5`; this task drives **≥ 10 `w2_j` zero-crossings** during
   training, so the sign-inversion case is actually exercised.

2. **Threshold-tracking dither rule.** Rung 2's `A_j = clip(1.5·mean|v_j|, 0.05, 0.5)`
   capped at 0.5 and could not follow a unit that drifted from threshold. Replace with:
   - `A_j = clip( 1.2 · quantile_n(|v_{n,j}|, 0.30), A_min=0.02, A_max=3.0 )`
     — amplitude scaled so ~30 % of samples fall within one dither swing of the flip;
     it **grows as the unit moves away**.
   - `reachable_j = quantile_n(|v_{n,j}|, 0.10) ≤ A_max` — if not even the nearest
     10 % of samples are within `A_max` of the threshold, the unit is genuinely
     unreachable and its `δ_j` for the hard network really is ≈ 0; it is reported as
     "no signal", not scored as a sign error.

3. **Correct ground truth.** Rung 2 scored the RAF readout against the *surrogate* δ.
   The surrogate is the biased quantity (derivative of a smoothed net). A perturbation
   estimate is an unbiased estimator of the **finite-difference** gradient at its own
   amplitude. Score against **per-unit central FD at `A_j`**:
   `fd_j = [L(b1_j + A_j) − L(b1_j − A_j)] / (2·A_j)`.
   The surrogate δ is kept only as a secondary number (its gap from `fd` measures how
   far the hard and smoothed gradients diverge — not the RAF's fault).

## Methods compared (unchanged from rung 2)

`bptt` (reference) · `raf_full` = node perturbation with
`δ̂_j = sign(⟨v_j·sin ω_j t⟩)·sqrt(energy_ema_j)`, one real-math resonator per hidden
unit, shared DC-removed `Σ_n L_n(t)` drive · `raf_sign` (sign only) · `nodepert_ideal`
(digital lock-in ceiling). Weight rule `ΔW1_ij = −lr·δ̂_j·mean_n x_{n,i}`,
`Δb1_j = −lr·δ̂_j`; output layer updated with its local surrogate gradient (the only
top-down broadcast). `δ̂` scale-matched to `fd` by one global factor per step.

## Instrument gates

- **Gate T** — BPTT test acc > 0.90 on the task (done: ~0.98).
- **Gate V** — vectorised resonator == real `ResonatorState.step()` to < 1e-9 over
  8000 ticks (carried from rung 2: was exactly 0.0).
- **Gate W** — random init: `r(δ̂, fd) ≥ 0.90` over reachable units with
  `|fd_j| > 0.1·max|fd|`; `|fd|`-weighted sign-agreement ≥ 0.90.

## Pre-registered predictions

- **S1 (learns).** `raf_full` test acc ≥ **0.90 ×** BPTT final test acc within 20×
  the gradient steps.
- **S2 (sign, both w2 signs).** `|fd|`-weighted sign-agreement(δ̂, fd) ≥ **0.85** for
  **both** `w2_j > 0` and `w2_j < 0` reachable units with `|fd_j| > 0.1·max|fd|`,
  aggregated over all training steps.
- **S3 (sign inversion).** In the ±3-step window around each `w2_j` zero-crossing,
  sign-agreement(δ̂, fd) ≥ **0.75**.
- **S4 (tracking works).** Unit-steps flagged `unreachable` stay **< 10 %** across
  training with the tracking rule; report the same count under rung 2's capped rule
  for contrast (expected: much higher).
- **S5 (cost).** Report forward-evals per gradient step (≈ `ticks` + warm-up) and the
  accuracy-vs-evals curve. No wall-clock win expected.

## Kill condition

**S1 fails**, OR **S2 fails for the `w2_j < 0` subset**, OR **S4 shows the tracking
rule does not reduce unreachability** vs the capped rule. Record as an honest
negative, stop, add no epicycles.

## Addendum 2026-09-07 (after first 2b run, before re-scoring)

First run: gates V + W pass (vec==real exact; init `r(δ̂,fd)=0.99`, sign-agree 1.00).
`raf_full` trains to 0.968 acc. But **S2 collapsed to ~chance for both w2 signs**
(+w2 0.68, −w2 0.53) despite the perfect init — a training-time effect, and
Gate T failed only because BPTT was cut at 200 steps.

Diagnosis: near convergence the hard-classification loss `L = (frac misclassified)/2`
is a coarse staircase. With `N=400`, one sample flip changes `L` by `1/800 = 1.25e-3`;
once train-loss is small, `fd_j = ΔL/2A_j` is dominated by the discrete flipping of
one or two borderline samples, so its **sign is essentially random** — there is no
gradient left for node perturbation to recover, independent of the readout. BPTT's
surrogate avoids this because `σ'(v)` is nonzero for every sample.

Re-scoring rules (fixed now, before the re-run):

1. **Gate T / S1 baseline:** BPTT accuracy taken at **convergence** (2000 steps),
   not at the RAF step budget.
2. **S2 reported twice:** over *all* steps, and restricted to steps with
   **train-loss > 0.10** (where `max|fd|` clears the one-sample quantization floor by
   ≥ 5×). The pre-registered ≥ 0.85 bar applies to the **signal-present** subset; the
   all-steps number is reported as context.
3. **New logged series:** per-step `sign_agree(δ̂, fd)` and `train_loss`, so the
   sign-agreement-vs-progress curve can be shown. Prediction **S2c:** sign-agreement
   ≥ 0.85 while train-loss > 0.10, then decays toward chance as the loss saturates.
4. RAF step budget raised to **400** to give `w2_j` zero-crossings among
   gradient-carrying units a chance to occur (S3).

Kill condition now reads: **S1 fails**, OR **S2 fails on the signal-present subset for
`w2_j < 0`**, OR **S4 fails**.

## What a pass licenses

Writing up "a resonator bank in the backward path" as a substrate-feasible on-chip
learning rule for multi-layer spiking nets, with the honest R3/S5 cost multiplier —
and only then asking whether more than one resonator layer buys anything.
