# Rung 1 pre-registration — RAF resonator bank as a gradient probe

**Written before any result is read. Do not edit after the first run; append a dated
addendum instead.**

## Question

Can a bank of the *real* Spikeling `ResonatorState` oscillators (core/runtime/runtime.py),
each tuned to a distinct per-parameter dither frequency `f_i`, recover the local
gradient `∂loss/∂θ_i` of a hard-threshold readout **by resonance alone** — i.e. from
how strongly the scalar loss wiggles at `f_i` — with fidelity comparable to a finite
difference taken at the same perturbation scale?

This is the smallest testable piece of the larger idea ("the backward pass is itself an
SNN of resonate-and-fire neurons"). If the bank cannot read a gradient off a *one-layer*
system, the multi-layer version is not worth building.

## Method (fixed)

- **Target system.** `D` parameters `θ`. Fixed unit-norm random input `x`.
  Pre-activation `v(θ) = Σ_i θ_i x_i − bias`.
  - `--readout soft`: `y = σ(β v)`, `loss = ½(y − target)²` — has a closed-form
    gradient, used to **validate the harness**.
  - `--readout hard`: `y = 1[v > 0]`, `loss = ½(y − target)²` — the true spike; no
    analytic gradient; finite-difference is the only ground truth.
  - `target = 0`. `bias` is set so the operating point sits `margin` logistic-units
    from the flip.
- **Dither.** Every tick (`dt = 1/40000 s`, Spikeling's native rate)
  `θ_i(t) = θ_i0 + A sin(ω_i t)`, `ω_i = 2π f_i`. `f_i` log-spaced across `[f_lo, f_hi]`.
- **Demodulation front-end.** `drive(t) = G · (loss(θ(t)) − ⟨loss⟩_EMA)` — DC-removed
  loss wiggle, one shared scalar. `G` auto-calibrated from a warm-up so the median
  channel amplitude is ~1e-2.
- **Bank.** One `ResonatorState(freq_hz=f_i, damping, coupling=DEFAULT_RESONATOR_BASE_GAIN·ω_i²,
  gate_threshold=0.0)` per parameter, stepped with the shared `drive`
  (`ResonatorState.step(drive, dt)` — the real method, unmodified).
  `gate_threshold=0.0` only to force the ungated energy path; nothing else touched.
- **Estimator.** `grad_est_i = sqrt(resonator_i.energy_ema)` after a settle-and-measure
  window (`T` ticks, first `T//2` discarded). **Magnitude only** — the energy EMA
  squares `x`, so sign is not recoverable from the resonator. An ideal digital lock-in
  (`2·mean(loss·sin ω_i t)/A`, signed) is computed alongside as a reference ceiling.
- **Ground truth.** Central finite difference at step `e = A` (same smoothing scale as
  the dither): `fd_i = (loss(θ+e e_i) − loss(θ−e e_i)) / 2e`. A sharp FD at `e = 1e-4`
  is also recorded.
- **Anchors.** Correlation of a random vector with FD (null, expect ≈ 0); correlation of
  the analytic surrogate with FD (instrument check, soft readout).

## Pre-registered predictions

- **P1 (viability).** In the most favourable cell — `D = 4`, maximum frequency spacing,
  operating point within 1 unit of threshold, `soft` readout — Spearman
  ρ(`grad_est`, `|fd_A|`) **≥ 0.8**.
- **P2 (cross-talk).** Holding `D` fixed, ρ falls **monotonically** as the frequency
  band is compressed ×½ then ×¼ (spectral leakage + intermodulation `f_i ± f_j`).
  Report the ρ-vs-spacing curve.
- **P3 (bandwidth saturation).** Holding the band fixed, ρ falls monotonically as
  `D ∈ {4, 8, 16, 32}` (each channel gets less spectral room). Report the ρ-vs-`D` curve.
- **P4 (measuring the real thing).** Far from threshold (`|v0| ≥ 2`), `grad_est` and the
  analytic surrogate **collapse together** — ρ between them stays ≥ 0.9 — confirming the
  bank tracks the true local slope, not a fixed artefact.
- **Instrument check (gate).** ρ(surrogate, `fd_A`) **> 0.95** on the soft readout, or the
  harness is invalid and no resonator number is trusted.

## Kill condition

If **P1 fails** in that most-favourable cell, the resonator-bank readout is not viable
as a gradient probe. Record it as an honest negative, do not proceed to rung 2
(frequency-multiplexed multi-layer training).

## Addendum 2026-09-07 (before any scored cell — only `--selftest` had been run)

The `--selftest` output showed the harness is valid (ρ surrogate-vs-FD = 1.000) but
also that at `D = 4` a Spearman correlation has almost no statistical power (24 distinct
rank orderings; the random null anchor came back at −0.80 purely from 4-point noise).
Methodological fix, decided before reading any scored result:

- **P1 is judged at `D = 8`** (still small, still the favourable band/margin). The
  `D = 4` cell is still reported for continuity but is not the pass/fail gate.
- All ρ values at `D ≤ 8` are read as directional, not precise; the P2/P3 *curves*
  (shape as spacing/`D` change) carry the weight, not any single cell's third decimal.

## What a pass licenses (rung 2, NOT run here)

2-layer spiking net, each unit dithered at its own `f_i`, a per-unit resonator reading a
**global** error broadcast; compare test accuracy **and** measurements-per-step against
surrogate-gradient BPTT.
