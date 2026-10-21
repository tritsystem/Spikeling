# Rung 2 results — 2026-09-07

Harness: `raf_gradient_rung2.py` (`--steps 150 --ticks 2000 --lr 0.4 --seed 0`).
Vectorised resonator asserted **bit-identical** to the shipped `ResonatorState.step()`
(max |Δstate| = 0.0 over 8000 ticks). Raw: `results_2_compare.json`; figure
`rung2_summary.png`. Two dated addenda in `PREREGISTER_2.md` record the gate
correction and the findings below.

## Verdict: honest negative — kill condition triggered (R2)

| pre-reg | result | reading |
|---|---|---|
| **Gate 1** vec == real `ResonatorState` | Δ = 0.0 | PASS |
| **Gate 3a** \|δ\|-weighted sign-agree(δ̂, δ_ref) ≥ 0.9 at init | 0.977 | PASS (after gate correction — see addendum #1) |
| **Gate 3b** r on units > 10% max\|δ_ref\| ≥ 0.9 | 0.959 | PASS |
| **Gate 2** BPTT baseline solves the task > 0.6 | 0.688–0.729, overfits (train loss 0.042) | **task not validated** — see below |
| **R1** RAF-full ≥ 0.90× BPTT accuracy in ≤ 20× steps | raf_full 0.833 vs 0.9×base 0.66 | **uninterpretable** — within run noise; not claimed as a pass |
| **R2** sign-agree ≥ 0.85 for **both** w2-sign subsets | **+w2 0.994, −w2 0.362** | **FAIL — kill condition** |
| **R3** cost, RAF ≈ ticks fwd-evals/step vs 1 | 525k vs 150 total (~3500×/step) | as pre-registered — substrate feasibility, no speed win |
| **R4** RAF-sign ≥ 0.80× RAF-full accuracy | 0.688 vs 0.667 | **uninterpretable** (same noise-floor problem) |

## The two findings

### 1. The task is near the noise floor — accuracy predictions are unreadable

48 train / 48 test, 2 informative + 2 distractor dims, overlapping Gaussians. Trained
to convergence the BPTT baseline reaches only **0.688** test accuracy (train loss
0.042 — plain overfitting). With the reference method unable to cleanly solve the
task, ±0.1 accuracy swings between methods are run-to-run noise. **R1 and R4 are not
scored.** A rung-2 redo needs a task BPTT solves > 0.9 (more data, cleaner margin, or
a standard small benchmark) before any accuracy comparison means anything.

### 2. The phase readout fails for minority-sign hidden units — and it is not cross-talk

At initialisation the RAF readout (`sign(⟨v·ref⟩)·sqrt(energy)`) is faithful on the
hidden units that carry real gradient, **for both w2 signs** — the only miss is a unit
with |δ_ref| ≈ 7e-4 (numerically zero, undefined phase). Averaged over 150 training
steps the picture splits hard by the sign of the outgoing weight:

- `w2_j > 0` units: sign-agreement **0.994**
- `w2_j < 0` units: sign-agreement **0.362** (below chance)

Dithering the −w2 units **in isolation** (no +w2 channels active) reproduces the same
init pattern, so the training-time collapse is **not** inter-channel cross-talk or
intermodulation. Leading mechanism: as the weights grow during training, the −w2
hidden units are pushed far enough from threshold that the capped adaptive dither
(`A_j ≤ 0.5`) can no longer straddle them — so their velocity-phase estimate is noise.
This is the **rung-1 wall** ("the resonator reads nothing unless the dither crosses the
threshold"), now hitting one w2-sign subset selectively because those units happen to
run away from threshold faster under this task + update rule. `n_crossings = 0` — no
`w2_j` actually crossed zero, so the headline "during a sign inversion" case was never
even exercised; the readout fails on *statically* negative `w2` first.

## Consequence for the ladder

Per `PREREGISTER_2.md` and the method: **stop here, add no epicycles** (no deeper
recursion, no per-weight frequency budget, no hand-tuning). The honest state after
three rungs:

- **Rung 1:** the resonator bank is a working frequency-multiplexed gradient *readout*,
  not a *smoother*; magnitude fragile, reads zero off a true step unless the dither
  straddles it.
- **Rung 1b:** sign is recoverable from `ResonatorState.v` phase; robust to crowding on
  an isolated unit.
- **Rung 2:** that readout does **not** deliver a faithful multi-layer gradient during
  training — the "no smoothing past the dither amplitude" limit resurfaces as a
  selective failure on hidden units that drift from threshold, and the adaptive-dither
  rule as written does not track them. A viable rung 2 would need (a) a validated task,
  (b) a dither rule that follows a unit's distance-to-threshold as it changes, and
  (c) then a re-test of the w2 zero-crossing case that this run never reached.

Nothing across the three rungs supports the original "each gradient spike is its own
SNN" recursion. The one-level version (a resonator bank in the backward path) is a
real *readout* mechanism with a hard dependence on per-unit dither amplitude that
rung 2 shows is not yet solved.
