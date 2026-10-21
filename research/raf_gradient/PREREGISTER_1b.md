# Rung 1b pre-registration — phase-preserving RAF readout (sign recovery)

**Written before any 1b result is read. Append dated addenda, do not rewrite.**

## Question

Rung 1 showed the resonator bank recovers gradient **magnitude** ranking but the
`energy_ema` readout squares `x` and loses **sign**. Rung 1b asks: does the sign fall
out of the *phase* of the resonator's response relative to the known dither reference,
using only quantities the shipped `ResonatorState` already exposes?

## Physics being exploited (fixed)

A damped driven oscillator `ẍ + 2ζω ẋ + ω² x = c·F(t)` driven at its own resonance
`F(t) = D₀ sin(ωt)` has steady state:

- `x(t) ∝ −cos(ωt)` — displacement lags the drive by 90°, sign-ambiguous under squaring
- `v(t) = ẋ(t) ∝ +sin(ωt)` — **velocity is in phase with the drive**

So if channel `i`'s contribution to the shared drive is `+g_i·A·sin(ω_i t)` (positive
gradient) then `v_i(t) ∝ +sin(ω_i t)` and `⟨v_i · sin(ω_i t)⟩ > 0`; if the gradient is
negative the drive component flips to `−sin` and the correlation goes negative.

## Estimators under test

- **RAF-v (primary):** `sign_i = sign ⟨v_i(t) · sin(ω_i t)⟩_window`,
  `mag_i = sqrt(energy_ema_i)`, combined `est_i = sign_i · mag_i`.
  Uses `ResonatorState.v` and `.energy_ema` only.
- **RAF-IQ (variant):** full quadrature on the resonator displacement —
  `I = ⟨x_i·sin⟩`, `Q = ⟨x_i·cos⟩`, `est_i = −Q_i` calibrated by `mag_i`
  (at resonance the in-phase-with-drive component of `x` is `−Q`). Tests whether the
  second quadrature buys anything over velocity alone.
- **loss-lock-in (ceiling):** `2⟨loss(t)·sin(ω_i t)⟩ / A` — the signed digital
  demodulator from rung 1, as a reference ceiling.

## Ground truth

Signed central finite difference at `e = A` (`fd_A`, from rung 1's harness). For the
soft readout `sign(fd_A_i) == sign(surrogate_i)` for every `i` by construction — that
is the instrument check.

## Pre-registered predictions

- **Q1 (sign recovery).** Favourable cell (`D = 8`, band ×10, margin 0.3, `soft`):
  sign-agreement rate between RAF-v and `sign(fd_A)` **≥ 0.9**, and Pearson
  r(`est`, `fd_A`) **≥ 0.9**.
- **Q2 (sign is cross-talk-robust).** As the band compresses ×10 → ×5 → ×2.5, RAF-v
  sign-agreement stays **≥ 0.85** and degrades **slower** (in absolute drop) than the
  rung-1 magnitude ρ did over the same compression (1.00 → 0.93). Rationale:
  `⟨v_i·sin(ω_i t)⟩` rejects off-`f_i` energy by sinusoid orthogonality over the
  window, on top of the resonator's own Q.
- **Q3 (hard step, no rescue).** On `--readout hard`: inside the near-threshold band
  (margin ≲ A) RAF-v sign-agreement **≥ 0.8**; outside it, undefined — the phase
  readout does **not** fix the missing-smoothing cliff from rung 1, it only adds sign
  where there is already a signal.
- **Q4 (velocity is enough).** RAF-v sign-agreement is within **0.05** of RAF-IQ in
  every live cell — the second quadrature is redundant at resonance.

## Instrument check (gate)

Soft readout: `sign(fd_A_i) == sign(surrogate_i)` for all `i` (trivially true by
construction) — confirms the signed reference is wired correctly. If it ever fails the
harness is broken.

## Kill condition

If **Q1 fails** in the favourable cell, sign is not recoverable from the shipped
`ResonatorState`'s exposed phase and would need an external lock-in — record as an
honest negative; the "backward pass as a bank of RAF neurons" loses its main appeal
(you'd be carrying a digital lock-in alongside every neuron anyway).

## What a pass licenses

Rung 2 can use `sign(⟨v·ref⟩)·sqrt(energy)` as a **complete** (signed, magnitude-
calibrated) per-unit gradient estimate, and only needs to solve smoothing
(per-unit dither amplitude) and the global error broadcast.
