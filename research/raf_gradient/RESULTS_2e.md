# Rung 2e results — 2026-09-07

Diagnostic rung: attribute the ~0.10–0.13 test-R² gap between RAF node-perturbation
(plateau ~0.86) and surrogate BPTT (~0.95) found in rung 2d. Harness
`raf_gradient_rung2e.py`, teacher/student from rung 2d (`k = 4`, `n_hid = 4`,
exactly representable), `seed = 0`. Raw `results_2e.json`; figure `rung2e_summary.png`.

## Verdict: neither pre-registered suspect. The perturbative gradient itself is the cap.

The pre-registration named (a) drive variance and (b) the node-perturbation weight
approximation, and allowed for "something neither prediction anticipated — report
honestly." That is the outcome.

| method (test R²) | | |
|---|---|---|
| `bptt` (surrogate backprop) | **0.954** | reference |
| `fd_nodepert` (exact node error, `·mean_n x` rule) | 0.841–0.846 | |
| `raf_full` (resonator node error, same rule) | 0.856 | = rung 2d |
| `fd_weights` (exact central FD on every `W1_ij`) | 0.797–0.824 | |
| `raf_weights` (one resonator per `W1_ij`) | 0.808 | |

| pre-reg | result | reading |
|---|---|---|
| **W1** `fd_weights` ≥ 0.90× `bptt` | 0.797 vs 0.858 | **FAIL** — exact per-weight FD of the hard net *also* plateaus |
| **W2** approximation is the main cost (`fd_weights` ≥ 0.05 above `fd_nodepert`) | `fd_weights` is 0.05 *below* `fd_nodepert` | **FAIL** — the "proper" weight gradient does no better (slightly worse) |
| **W3** drive/readout noise costs < 0.05 R² | `raf_full` − `fd_nodepert` = **+0.01**; `raf_weights` − `fd_weights` = **−0.01** | **PASS** — the resonator adds essentially nothing; it faithfully delivers the FD gradient |
| **W4** per-weight RAF closes the gap | `raf_weights` 0.808 | **FAIL** |
| **C1** dropped covariance is large | `‖g−ĝ‖/‖g‖` median **0.99**, 100% of states ≥ 0.25; `cos(g,ĝ)` = **0.18** | **PASS (extreme)** |
| **ALT** drive variance is the cause | `raf_full` ≈ `fd_nodepert` | **no** |

## What is actually going on

1. **W3 is the load-bearing result: the resonator is not the bottleneck.** Swapping the
   exact FD readout for the resonator bank changes final R² by ±0.01 in both the
   node-pert and per-weight rows. Rungs 1b/2c already showed the readout tracks sign
   faithfully; 2e shows that at the training level too.

2. **Every method that descends the *true* (finite-difference) gradient of the hard
   SNN plateaus at R² ≈ 0.80–0.86**, and it is a genuine plateau, not undertraining
   (`fd_nodepert`: 0.841 at 600 steps = 0.841 at 2000). Surrogate BPTT reaches 0.954.

3. **It is not surrogate *smoothing*.** BPTT with surrogate `1/(1+α|v|)²` gives R²
   0.954 for α ∈ {2, 10, 50, 200} — even a near-delta surrogate matches a smooth one.
   The surrogate's value is not that it is soft; it is that it is **sign-consistent** —
   always a monotone "downhill on the smoothed loss" direction. The FD gradient of the
   staircase loss is **sign-noisy**: between crossings it is ~0, at crossings it is a
   signed spike, and averaging those over any realisable dither amplitude can point the
   wrong way when the local crossings are label-unrepresentative. The unbiased gradient
   is the worse descent direction.

4. **C1's ≈1.0 rel-error is a second, independent bias** specific to the node-pert
   *weight* rule with **zero-mean inputs**: `ΔW1_ij = δ_j·mean_n x_{n,i}` and
   `mean_n x_{n,i} ≈ 0`, so the rule barely moves `W1` at all — the reconstruction ĝ is
   nearly orthogonal to the true weight gradient (`cos = 0.18`). This is why
   `fd_nodepert`/`raf_full` lean on the bias terms and the exact `w2` update. But
   fixing it (`fd_weights`, `raf_weights` — no node-pert approximation) does **not**
   help, because finding (3) still caps those at ~0.80.

## Attribution, stated plainly

The RAF backward pass plateaus below BPTT because **it faithfully estimates the true
gradient of a hard-threshold SNN, and that gradient is a ~10–15% worse descent
direction than the surrogate** — the surrogate is biased but sign-consistent, the true
gradient is unbiased but sign-noisy on the staircase. Corollary: a resonance-measured
backward pass is inherently capped below "assume a smooth surrogate and backprop,"
because a surrogate gradient is an analytic choice, not a physical measurement. Rung 0
/ 1b's "biased-but-useful vs unbiased-but-worse" tension, now demonstrated at the
network-training level.

## Ladder — closed

| rung | one-line outcome |
|---|---|
| 1 | resonator bank = frequency-multiplexed gradient **readout** (magnitude fragile, needs the dither to straddle the flip) |
| 1b | **sign** recoverable from `ResonatorState.v` phase; cross-talk-robust |
| 2 | honest negative (task/dither/ground-truth all mis-specified); superseded |
| 2b | validated task + tracking dither → RAF **trains** a 2-layer spiking net to BPTT-level *classification* acc; per-unit sign unreliable |
| 2c | regression target keeps FD signal alive; **U3 PASS** — sign-tracking *through* `w2` inversions works (0.99–1.00, replicated) |
| 2d | 2c's over-parameterisation explanation **falsified**; RAF plateaus ~0.86 R² at every width/lr |
| 2e | the plateau is the **FD gradient of a hard SNN being a worse descent direction than the surrogate** — the resonator is faithful and not the bottleneck |

**Net:** a resonator bank in the backward path is a substrate-feasible, sign-faithful,
**inherently ~10–15%-below-BPTT** learning signal for hard SNNs, tracking sign through
outgoing-weight inversions, at ~3000× the forward-evaluation cost. "Each gradient spike
its own SNN" is unsupported at every rung. There is no rung 2f: the cap is not an
engineering defect to fix, it is the price of measuring the true gradient instead of
assuming a surrogate.
