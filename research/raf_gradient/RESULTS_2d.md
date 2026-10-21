# Rung 2d results — 2026-09-07

Over-parameterisation made a controlled variable: hard-threshold teacher with `k = 4`
hyperplanes (the student can represent it **exactly** at `n_hid = 4`), student width
swept `{4, 6, 8, 12, 16}`. Harness `raf_gradient_rung2d.py` (reuses rung 2b/2c dither,
FD ground truth, resonator probe, `_diagnose`). Raw `results_2d_nhid.json`,
`results_2d_lr.json`; figure `rung2d_summary.png`.

## Verdict: 2c's over-parameterisation explanation is FALSIFIED. Kill condition triggered.

| pre-reg | result | reading |
|---|---|---|
| **Gate T** BPTT R² ≥ 0.90 at `n_hid = k` | **0.954** | PASS — student fits the teacher |
| **Gate V** vec == real resonator | Δ = 0.0 | PASS |
| **Gate W** init sign-agree = 1.00 on ≥ 3/4 seeds at `n_hid = k` | 1.00 / 0.92 / 1.00 / 1.00 | PASS (3/4) |
| **V1** RAF R² ≥ 0.90× BPTT at matched size | 0.856 vs bar 0.859 (default lr); **0.860 vs 0.858** after the V5 sweep | **bare pass / effectively FAIL** — RAF plateaus at R² ~0.86 vs BPTT 0.95 |
| **V2** matched-size bulk sign-agree ≥ 0.85 both w2 signs | +w2 0.97/0.90 · **−w2 0.47/0.06** (all/late) | **FAIL** — minority-sign units are *least* reliable at matched size |
| **V3** sign fidelity degrades with excess width (drop ≥ 0.10, monotone) | **−w2 sign-agree: 0.47 → 0.83 → 0.89 → 0.90 → 0.92** as width goes 4→16 | **FAIL — it RISES with width. The hypothesis is backwards.** |
| **V4** U3 carried: sign-agree ≥ 0.75 at w2 crossings, every width | 1.00 at every width (weight negligible at `n_hid=4`, real at ≥ 6) | PASS |
| **V5** best lr/step raf R² ≥ 0.90× BPTT | best 0.860 (lr 0.6) vs 0.858; RAF ceiling is flat 0.85–0.86 across lr 0.3–0.6 | bare pass — no setting closes the ~0.10 R² gap |

## Retraction

**`RESULTS_2c.md` attributed U1/U2 to over-parameterisation** ("6 hidden units on a
2-feature target … an individual unit's gradient sign is genuinely ill-posed … U2's
low number is partly a property of the problem, not only the estimator"). **Rung 2d
tested that as a pre-registered prediction (V2/V3) and it is false.** With unit roles
maximally well-posed — matched width, an exactly-representable teacher — RAF still
underfits BPTT (R² 0.86 vs 0.95) and minority-sign per-unit sign fidelity is *worse*
(0.47), not better. That explanation is withdrawn.

## What is actually going on

1. **RAF node-perturbation underfits BPTT by a roughly width-independent margin.**
   R² plateaus at ~0.85–0.88 for every width and every learning rate tried, while BPTT
   climbs 0.95 → 0.99. The resonator readout carries enough signal to learn most of the
   function and no setting recovers the last ~10 % of variance. This is a property of
   the *perturbative signal*, not of network sizing.

2. **The width dependence of sign fidelity is a statistical-power effect, not a
   well-posedness effect.** At `n_hid = 4` only two units are `−w2`; the `|fd|`-weighted
   sign-agree over two units is dominated by whichever one the optimiser has parked in
   a flat region (small, sign-random `fd`). Add width → 3, 4, 6, 8 `−w2` units, more of
   them near threshold at any step → the weighted average is taken over more well-posed
   cases → the number climbs. More units *dilutes* the noisy ones; it does not make any
   individual unit's sign better-posed.

3. **U3 survives 2d.** Sign-tracking through `w2` zero-crossings is 1.00 at every width
   (meaningfully weighted at `n_hid ≥ 6`). It remains the one robustly positive result
   across rungs 1b → 2d.

**Kill condition triggered** (V3 fails outright; V1 is a coin-flip). Per the method:
stop, no epicycles, and correct the record (done above).

## Cross-rung synthesis (rungs 1 → 2d) — final

| claim | status |
|---|---|
| resonator bank = frequency-multiplexed gradient **readout** | holds (1) — magnitude fragile |
| **sign** from `ResonatorState.v` phase | holds (1b isolated; 2c/2d through inversions) |
| threshold-**tracking dither** keeps units reachable | holds (2b: 0 % vs 13–18 %) |
| bank **trains** a multi-layer spiking net | **holds but LOSSY** — consistently underfits BPTT (~few % acc / ~0.10 R²), not closable by width, lr, or step budget |
| **faithful per-step per-unit** gradient | **does not hold** — bulk sign-agree 0.5–0.9, unreliable; **not caused by over-parameterisation** (2d) and not fixed by sizing |
| tracks sign **through a `w2` inversion** | **holds** (2c U3, 2d V4 — 1.00 at every width) |
| "each gradient spike is its own SNN" recursion | **unsupported** at every rung |

The honest one-liner: **a resonator bank in the backward path is a working but lossy
training signal for multi-layer spiking nets — it learns, never matches BPTT, and has
a reliable per-unit sign only at outgoing-weight inversions.** Further rungs would be
about *why* the perturbative signal plateaus ~10 % short of BPTT (variance of the
one-shared-scalar drive? the node-perturbation weight approximation
`ΔW1_ij ≈ δ̂_j·mean_n x_{n,i}`?), not about the resonator, which is now characterised.
