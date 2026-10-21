# Rung 1b results — 2026-09-07

Harness: `raf_gradient_rung1b.py` (imports rung-1's `Readout`/`make_bank` verbatim).
Real `ResonatorState`, `gate_threshold=0.0` only; additionally reads `.x`/`.v` each
tick. `ticks=20000`, `damping=0.02`, `beta=12`, `seed=0`. Raw: `results_1b_*.json`;
figure `rung1b_summary.png`.

## Verdict: PASS on all four predictions — with one construction caveat

| pre-reg | result | reading |
|---|---|---|
| **Instrument check** `sign(fd)==sign(surrogate)` ∀i | PASS every cell | signed reference wired right |
| **Q1** favourable cell: RAF-v sign-agree ≥ 0.9, r ≥ 0.9 | sign-agree **1.000**, r **+0.998** | PASS |
| **Q2** sign-agree ≥ 0.85 through band ×10→×2.5, degrading slower than rung-1 magnitude | sign-agree **1.000 / 1.000 / 1.000** while rung-1 \|grad\| ρ went **1.00 → 0.98 → 0.93** | PASS — sign is flat where magnitude bends |
| **Q3** hard step: sign-agree ≥ 0.8 near threshold, undefined outside | **1.000** at margin 0.3 (r +0.84); **dead** at margin ≥ 1.0 | PASS — phase readout adds sign to an existing signal, doesn't create one |
| **Q4** RAF-v within 0.05 of RAF-IQ everywhere | identical to 3 dp in **every** cell | PASS — velocity alone carries the sign at resonance; second quadrature redundant |

Extra: **D-sweep** — RAF-v sign-agree stays **1.000 to D=32** while the rung-1
magnitude ρ collapses to 0.80 and RAF-v signed-r eases 1.00 → 0.958. **Sign is the
robust half of the readout; magnitude is the fragile half.**

## What rung 1b establishes

A **complete** per-parameter gradient estimate is available from the shipped
`ResonatorState` with no modification to the class:

    est_i  =  sign( ⟨ v_i(t) · sin(ω_i t) ⟩ )  ·  sqrt( energy_ema_i )
             └─────────── sign ───────────┘     └──── magnitude ────┘

The sign comes from the fact that a resonator driven at its own resonance has
**velocity in phase with the drive** — so correlating `ResonatorState.v` against the
known dither reference recovers whether the loss rose or fell when that parameter went
up. This correlation also rejects off-`f_i` energy by sinusoid orthogonality, which is
why sign-agreement survives channel crowding that the energy-only magnitude readout
does not.

## Construction caveat (disclosed, not in the pre-reg)

In the `soft` readout every channel shares **one positive** gradient-scaling scalar
`C = (y−t)·y·(1−y)·β > 0`, so `sign(∂loss/∂θ_i) = sign(x_i)` by construction and the
drive phase tracks it trivially. Rung 1b therefore proves the **wiring is correct** and
that cross-talk does not *flip* signs up to D=32 / band ×2.5 — it does **not** exercise
a case where nonlinearity could genuinely invert a channel's apparent drive phase
(e.g. a hidden layer, or a loss whose curvature changes sign across the dither swing).
That belongs in rung 2's target network, where the hidden units do have
`θ`-dependent gradient signs. Flagged there.

## Net effect on the ladder

Rung 2 may now treat `sign(⟨v·ref⟩)·sqrt(energy)` as a full signed, magnitude-
calibrated per-unit gradient estimate. The two open problems are unchanged from
rung 1's RESULTS.md: **(a) smoothing** — the estimate is still exactly zero on a true
step unless the per-unit dither straddles threshold; **(b) the global error
broadcast** for multi-layer credit assignment. Rung 2's pre-registration must also
add a **sign-inversion stress cell** (nonlinear hidden layer) per the caveat above.
