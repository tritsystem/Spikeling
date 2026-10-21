# Rung 2d pre-registration — matched network sizing (over-parameterisation as a controlled variable)

**Written before any rung-2d result is read. Append dated addenda; do not rewrite.**

Rung 2c passed Gate F (FD signal survives) and U3 (sign-tracking through `w2`
inversions) but failed U1 (RAF R² 0.79–0.94 vs BPTT 0.96) and U2 (bulk per-step
per-unit sign-agree ~0.6). RESULTS_2c attributed both to **over-parameterisation** —
6 hidden units on a 2-feature target, so individual hidden-unit gradient *signs* are
ill-posed (redundant, interchangeable units). Rung 2d tests that attribution directly
by making width a swept variable against a task with a **known** required width.

## Task — teacher/student, same architecture class

- **Teacher:** `y = c₀ + Σ_{j=1}^{k} c_j · 1[a_j·x − b_j > 0]`, `k = 4` random
  hyperplanes in ℝ², `a_j ~ N(0,I)` unit-normalised, `b_j ~ U(−1,1)`,
  `c_j = s_j·U(0.6,1.4)` with `s_j ∈ {±1}` (two +, two −), `c₀` set so `mean(y)=0`.
  Piecewise-constant, `k` pieces. A hard-threshold student with `n_hid = k` can
  represent it **exactly** (R² → 1 achievable).
- **Inputs** `x ~ U(−1.5,1.5)²`, `N = 400` train / `400` test, fixed seed.
- **Forced inversion:** at the training midpoint, negate `c_j` for `j ∈ {k/2 … k}`.
  The student units aligned with those teacher planes must invert their outgoing
  weight — carries the U3 test into 2d.
- Student: hard hidden `h_j = 1[v_j>0]`, **linear** readout `ŷ = w2ᵀh + b2`, MSE.
  `w2` init alternating-sign `·0.5`, `b1 ~ N(0,0.3)`.

## Methods, dither, ground truth — unchanged from rung 2b/2c

`bptt` (surrogate hidden, exact linear output) · `raf_full`
(`δ̂_j = sign(⟨v_j·ref⟩)·sqrt(energy_j)`, one real-math resonator per hidden unit) ·
`raf_sign` · `nodepert_ideal`. Dither `A_j = clip(1.2·q30(|v_j|), 0.02, 3.0)`.
Ground truth = per-unit central FD at `A_j`. `δ̂` scale-matched to `fd` by one global
factor/step. Resonator = vectorised `ResonatorState`, asserted bit-identical.

## Swept variable

`n_hid ∈ {4, 6, 8, 12, 16}` = {k, 1.5k, 2k, 3k, 4k}. Everything else fixed.
Primary run seed 0; headline predictions re-checked on seeds 1–3.

## Instrument gates

- **Gate T** — `bptt` test R² ≥ **0.90** at `n_hid = k` (student can fit the teacher).
- **Gate V** — vec == real `ResonatorState` < 1e-9 (carried; was 0.0).
- **Gate W** — init `|fd|`-weighted sign-agree = 1.00 on ≥ 3/4 seeds at `n_hid = k`
  (r reported where ≥ 3 units kept; per the 2c addendum sign-agree is the gate).

## Pre-registered predictions

- **V1 (matched sizing recovers training).** At `n_hid = k`, `raf_full` final test
  R² ≥ **0.90 ×** `bptt` R², on seed 0 **and** ≥ 3/4 seeds.
- **V2 (matched sizing recovers per-unit sign).** At `n_hid = k`, bulk per-step
  `|fd|`-weighted sign-agree(δ̂, fd) ≥ **0.85** for **both** `w2>0` and `w2<0`
  reachable meaningful-`|fd|` units, in **both** the all-steps (≥10) and last-20%
  windows. This is the core test: if the readout is faithful once unit roles are
  well-posed, 2c's U2 failure was the network, not the resonator.
- **V3 (degradation is caused by excess width).** Bulk sign-agree at `n_hid = 4k` is
  ≥ **0.10 lower** (absolute) than at `n_hid = k`, and the sequence over
  {k,1.5k,2k,3k,4k} is monotone non-increasing (one adjacent inversion allowed for
  noise). Without this, 2c's failure is not explained by over-parameterisation and
  the readout is simply unreliable.
- **V4 (U3 carried).** Sign-agree in ±3 steps around each `w2_j` zero-crossing ≥
  **0.75** at every `n_hid`.
- **V5 (lr/step retry for V1).** With `--sweep lr` over `lr ∈ {0.1,0.3,0.6}` ×
  `steps ∈ {200,400}` at `n_hid = k`, the best `raf_full` R² ≥ **0.90 ×** `bptt` R².
  (Redundant if V1 already passes at default `lr = 0.3`.)

## Kill condition

**V1 fails at matched size even after the V5 lr/step sweep** (readout can't train a
well-posed net) **OR V3 fails** (excess width does not explain 2c's degradation →
the readout is unreliable regardless of sizing). Either way: honest negative, stop,
no epicycles.

## What a pass licenses

Rungs 1→2d then read as: a resonator bank in the backward path, with a
threshold-tracking dither, is a substrate-feasible learning rule for **appropriately
sized** multi-layer spiking nets — faithful per-unit gradient signs when unit roles
are well-posed, degrading gracefully with redundancy — tracking sign through outgoing-
weight inversions, at ~3000× the forward-evaluation cost. The recursion idea stays
unsupported; the natural next question is depth > 1 hidden layer, not a second
resonator layer.
