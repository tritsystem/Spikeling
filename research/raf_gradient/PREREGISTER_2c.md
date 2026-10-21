# Rung 2c pre-registration — regression target + forced sign inversion

**Written before any rung-2c result is read. Append dated addenda; do not rewrite.**

Rung 2b left two things unresolved: (S2) minority-sign sign-fidelity was unreliable
because the hard-classification loss **saturates into a staircase** near convergence,
and (S3) no *load-bearing* `w2_j` crossed zero during the informative window, so
sign-tracking through an inversion was never measured. Rung 2c fixes both by task
design; the rung-2b dither rule and FD ground truth carry over unchanged.

## Task + network (fixed)

- **Regression.** Hidden layer stays hard spiking: `h_j = 1[v_j > 0]`,
  `v_j = (W1ᵀx)_j + b1_j`. Output is **linear**: `ŷ = w2ᵀh + b2` (continuous).
  Loss `L(θ) = mean_n ½(ŷ_n − y_n)²`.
- **Inputs** `x ∈ ℝ²`, `x ~ U(−1.5, 1.5)²`, `N = 400` train / `400` test, fixed seed.
- **Target** `y = g1(x) + β · g2(x)`, `g1(x) = tanh(2 x₀)`, `g2(x) = tanh(2 x₁)`.
  `β = +1` for the first half of training, **`β = −1` for the second half.** Hidden
  units that serve `g2` must invert their outgoing weight at the midpoint; `g1`-servers
  do not — this deliberately drives a *subset* of `w2` through zero while the rest hold
  sign (the minority-sign-crossing stress).
- `w2` initialised alternating-sign `[+,−,+,−,+,−]·0.5`.

Why the loss no longer saturates: samples are spread across every hidden unit's
threshold at all times, so the density of `v_{:,j}` at 0 stays nonzero and
`dL/db1_j` stays informative even at the regression optimum (residuals small but
continuously valued, not quantized).

## Methods (unchanged from rung 2b)

`bptt` (surrogate on hidden, **exact** on the linear output) · `raf_full`
(`δ̂_j = sign(⟨v_j·ref⟩)·sqrt(energy_j)`, one real-math resonator per hidden unit,
shared DC-removed `Σ_n L_n(t)` drive) · `raf_sign` · `nodepert_ideal` (loss lock-in
ceiling). Dither `A_j = clip(1.2·quantile(|v_j|, 0.30), 0.02, 3.0)`,
`reachable_j` per rung 2b. Weight rule `ΔW1_ij = −lr·δ̂_j·mean_n x_{n,i}`,
`Δb1_j = −lr·δ̂_j`, `δ̂` scale-matched to `fd` by one global factor per step. Output
`w2, b2` updated with its exact gradient.

## Ground truth

Per-unit central FD at each unit's own amplitude:
`fd_j = [L(b1_j + A_j) − L(b1_j − A_j)] / (2 A_j)`. Surrogate δ kept as a secondary
number only.

## Instrument gates

- **Gate T** — `bptt` reaches test **R² ≥ 0.85** on each phase's target.
- **Gate V** — vec resonator == real `ResonatorState.step()` < 1e-9 (carried: was 0.0).
- **Gate W** — random init: `r(δ̂, fd) ≥ 0.90` and `|fd|`-weighted sign-agree ≥ 0.90
  over reachable units with `|fd_j| > 0.1·max|fd|`.
- **Gate F (the rung-2c point)** — `max_j |fd_j|` in the last 20 % of training is
  **≥ 0.25 ×** its value in the first 20 % (signal does **not** collapse). If Gate F
  fails, 2c has not fixed 2b's core problem and U2/U3 are not scored.

## Pre-registered predictions

- **U1 (learns).** `raf_full` final test MSE ≤ **1.25 ×** `bptt` final test MSE,
  within 20× the gradient steps.
- **U2 (sign, both w2 signs, no longer collapsing).** `|fd|`-weighted
  sign-agree(δ̂, fd) ≥ **0.85** for **both** `w2_j > 0` and `w2_j < 0` reachable
  meaningful-`|fd|` units, aggregated over steps ≥ 10 — **including the last 20 % of
  training** (reported separately; both windows must clear 0.85).
- **U3 (sign inversion in flight — the measurement 2/2b never got).** Over the
  ±3-step window around every `w2_j` zero-crossing on a gradient-carrying unit,
  sign-agree(δ̂, fd) ≥ **0.75**. Report the crossing count (expected ≫ 0, forced by
  the β flip) and the windowed agreement, split by crossing direction (+→− vs −→+).
- **U4.** `raf_sign` final test MSE ≤ 1.5 × `raf_full` final test MSE.
- **U5.** Forward-evals per gradient step (≈ `ticks` + warm-up); accuracy-vs-evals.

## Kill condition

**Gate F fails** (task didn't fix the collapse), OR **U1 fails**, OR **U2 fails for the
`w2_j < 0` subset in either window**, OR **U3 fails**. Record as an honest negative,
stop, add no epicycles.

## Addendum 2026-09-07 (after `--selftest`, before the training run)

Gate W's `r ≥ 0.90` criterion is underpowered at initialisation: the
`reach & (|fd| > 0.1·max|fd|)` keep-filter leaves only **1–6 units** depending on
seed, and a Pearson r over 1–4 points is meaningless (seed 2: 1 kept unit → r
undefined). Across seeds 0–3 the **sign-agreement is 1.000 on every seed**; r, where
≥ 3 units are kept, is 0.89 / 0.96 / — / 1.00.

Corrected Gate W (decided before the training run): **`|fd|`-weighted sign-agreement =
1.000 on ≥ 3 of 4 seeds** (met), with r reported as context where ≥ 3 units are kept.
This does not relax any sign prediction — U2/U3 are sign-only and that is exactly what
Gate W now certifies.

## What a pass licenses

A full write-up: "a resonator bank in the backward path is a substrate-feasible
learning rule for multi-layer spiking nets that tracks hidden-unit gradient signs
through outgoing-weight inversions" — with the honest cost multiplier — and only then
the question of whether more than one resonator layer buys anything.
