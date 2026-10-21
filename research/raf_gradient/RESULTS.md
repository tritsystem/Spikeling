# Rung 1 results — 2026-09-07

Harness: `raf_gradient_rung1.py`, real `core/runtime/runtime.py::ResonatorState`
(unmodified except `gate_threshold=0.0`). `ticks=20000`, `damping=0.02`, `beta=12`,
`seed=0`. Raw numbers in `results_*.json`; figure `rung1_summary.png`.

## Verdict: qualified PASS — proceed to rung 2 with two hard constraints

The kill condition (P1 fails in the favourable cell) was **not** triggered. But two
findings narrow what rung 2 can assume.

| pre-reg | result | reading |
|---|---|---|
| **Instrument check** ρ(surrogate, FD) > 0.95 | **1.000** | harness valid |
| **P1** viability, D=8 soft, ρ(res, \|FD\|) ≥ 0.8 | **1.000** | PASS (see caveat 1) |
| **P1-hard** near-threshold cell | ρ **0.93–1.00** (D=8–4) | PASS — bank ranks \|grad\| when the dither straddles the step |
| **P2** cross-talk, band ×10→×5→×2.5 | ρ **1.00 → 0.98 → 0.93** | PASS — monotone decline as predicted |
| **P3** bandwidth saturation, D 4→8→16→32 | ρ **1.00 → 1.00 → 0.98 → 0.80** | PASS — monotone decline; hits the 0.8 bar exactly at D=32 |
| **P4** far from threshold, estimator+surrogate collapse together ρ≥0.9 | **FAIL as stated** | on a true step the bank goes from ρ≈0.93 (margin 0.3) to **undefined/dead** (margin ≥ 1.0): no graceful decay, an all-or-nothing cliff |

## What rung 1 actually establishes

1. **Frequency-multiplexed demodulation works on the real resonator.** A bank of
   Spikeling `ResonatorState` oscillators, one per parameter at its own dither
   frequency, recovers per-parameter gradient **magnitude rankings** from a single
   shared scalar-loss trace. It degrades predictably with channel crowding (P2) and
   channel count (P3) — the resonators' finite Q is doing real narrowband separation
   and in these cells beats a naive finite-window digital lock-in (grey curve in the
   figure), which does not integer-period-window the non-commensurate frequencies.

2. **The resonator adds nothing to the threshold problem itself.** On a genuine step
   (`--readout hard`) the bank reads a usable gradient **only inside a band roughly one
   dither-amplitude wide around the flip**. Outside it the loss is constant, the drive
   is zero, and the estimate is exactly undefined — the same dead-gradient wall that
   surrogate gradients exist to get around. The resonator bank is a **readout**
   mechanism, not a **smoothing** mechanism. Smoothing still has to come from
   elsewhere: dither amplitude tuned per-unit to straddle threshold (stochastic
   smoothing), or a blended analytic surrogate.

3. **Energy readout loses sign.** `energy_ema` squares `x`, so the bank ranks
   `|∂loss/∂θ_i|` but cannot tell uphill from downhill. Sign only survived in the
   in-phase digital lock-in column (`r` 0.86–0.98 when the cell was alive). A
   resonate-and-fire readout that keeps **spike phase relative to the dither
   reference** — not accumulated energy — is the open question for rung 1b.

## Caveats found by running the harness (not in the original pre-reg)

1. **Small-D rank correlations are low-power.** D=4 has 24 orderings; the random null
   anchor swung to −0.8 on noise alone. P1 was moved to D=8 by the dated addendum;
   even that is 8 points. The P2/P3 *curves* carry the weight, not any single ρ.
2. **Soft-mode P1 is partly trivial.** With `beta·A ≈ 0.012` the logistic never leaves
   its locally-linear regime, and there the gradient is just a rank-preserving scaling
   of the fixed input `x`. Soft mode tests "can the bank demodulate a multiplexed
   near-linear signal" (clearly yes); it does not test threshold behaviour at all.
   That is why P4-soft shows a flat ρ=1.000 — there is no threshold to move away from.
3. **The "naive lock-in" column is a baseline, not a bound.** Relabelled in the code.

## Rung 2 is licensed, but must:

- **supply its own smoothing** — per-unit dither amplitude scaled to each unit's
  distance-to-threshold, or a surrogate/dither hybrid — because the resonator will not
  do it;
- **recover sign** — phase-preserving RAF readout, benchmarked against the in-phase
  lock-in column here;
- carry a **global error broadcast** (rung 1 is single-layer; multi-layer credit
  assignment is untested and not implied by anything above).

Nothing in rung 1 motivates the deeper recursion ("each gradient spike its own SNN") —
one level (a resonator bank in the backward path) is all the evidence supports.
