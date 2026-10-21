# Rung 2e pre-registration — why does the perturbative signal plateau ~10% R² short of BPTT?

**Written before any rung-2e result is read. Append dated addenda; do not rewrite.**

Rung 2d established: RAF node-perturbation trains but plateaus at test R² ~0.86 at
every width and every learning rate, while surrogate BPTT reaches 0.95–0.99. Rung 2e
attributes that gap. Two named suspects:

- **(a) drive variance** — the single shared scalar `Σ_n L_n(t)` drive is noisy, so the
  resonator estimate `δ̂_j` has high variance and the optimiser can't settle.
- **(b) the node-perturbation weight approximation** — the dither is one scalar offset
  on `b1_j` applied to every sample, so the readout recovers only the *batch-summed*
  node error `δ_j = Σ_n δ_j^{(n)}`. The weight update then uses
  `ΔW1_ij = −lr·δ̂_j·mean_n x_{n,i}`, which approximates
  `∂L/∂W1_ij = mean_n(δ_j^{(n)} x_{n,i})` by `mean_n(δ_j^{(n)})·mean_n(x_{n,i})` —
  dropping `cov_n(δ_j, x_i)`, the term that says which input direction the hidden unit
  should rotate toward.

## Design — a 2×2 that isolates readout noise from the weight approximation

Same teacher/student as rung 2d (`k = 4` hard-threshold teacher, student `n_hid = 4`,
exactly representable). Six training methods, all else identical (`lr = 0.3`,
`steps = 250`, `seed = 0`, FD/dither/resonator machinery from rungs 2b–2d):

| | **exact readout** (FD) | **resonator readout** (RAF) |
|---|---|---|
| **node-pert weight rule** (`·mean_n x`) | `fd_nodepert` | `raf_full` (the rung-2d method) |
| **per-parameter** (dither/FD each `W1_ij` + `b1_j`) | `fd_weights` | `raf_weights` |

Plus `bptt` (surrogate reference). `raf_weights` dithers each of the 8 `W1_ij` at its
own frequency (`n_in·n_hid` freqs) with `A_ij = clip(A_j / rms_n(x_i), 0.02, 3.0)`
where `A_j` is the rung-2d per-unit amplitude; `fd_weights` is central FD on each
`W1_ij` and `b1_j` directly.

## Direct covariance diagnostic (no training)

At 20 random-init + mid-training states, compute the true weight gradient
`g_ij = mean_n(δ_j^{(n)} x_{n,i})` (from the BPTT backward) and the node-pert
reconstruction `ĝ_ij = mean_n(δ_j^{(n)})·mean_n(x_{n,i})`. Report
`‖g − ĝ‖ / ‖g‖` and `cos(g, ĝ)`. If the dropped covariance is a large fraction of the
true gradient, the approximation is provably lossy independent of any readout.

## Pre-registered predictions

- **W1 (hard-net FD ≈ surrogate BPTT).** `fd_weights` final test R² ≥ **0.90 ×**
  `bptt` R². If this fails, part of the "gap" is simply that the hard net has less
  usable gradient than its smooth surrogate — not a perturbation problem.
- **W2 (the approximation is the main cost).** `fd_nodepert` R² is **≥ 0.05 below**
  `fd_weights` R² (absolute), **and** `raf_full` R² is within **0.03** of
  `fd_nodepert` R². I.e. most of the ~0.10 gap is the `·mean_n x` approximation, and
  resonator noise adds little on top.
- **W3 (drive variance is secondary).** `raf_full` R² ≥ `fd_nodepert` R² − **0.05**
  and `raf_weights` R² ≥ `fd_weights` R² − **0.05**. Resonator/drive noise costs
  < 0.05 R² in both rows.
- **W4 (per-parameter RAF closes the gap).** `raf_weights` final test R² ≥ **0.90 ×**
  `bptt` R² — paying for per-weight frequencies recovers BPTT-level fit, confirming the
  bottleneck was the node-pert approximation, not the resonator readout.
- **C1 (covariance diagnostic).** `‖g − ĝ‖ / ‖g‖` ≥ **0.25** at a majority of the 20
  states — the dropped term is materially large.

## Alternative outcome (also pre-registered)

If **W2 fails** (`fd_nodepert` ≈ `fd_weights`, gap < 0.05) **and** `raf_full` ≪
`fd_nodepert` (> 0.05 below), then **drive variance / readout noise is the cause**, not
the weight approximation. Record that; it points a rung 2f at the drive (longer
integration, per-sample dither, multi-drive) rather than the weight rule.

## Kill / stop condition

This rung is diagnostic, not a go/no-go — it has no kill condition. It ends when the
~0.10 R² gap is attributed to (a), (b), a mix, or something neither prediction
anticipated (report honestly). No epicycles: one 2×2 + one diagnostic, then write it up.

## What a resolved attribution licenses

If **(b)**: the honest framing is "RAF node-perturbation on the *bias* is cheap but
biased; per-weight RAF is unbiased but costs `n_in×` more frequencies" — a stated
cost/accuracy knob, ladder genuinely complete.
If **(a)**: a narrow rung 2f on the drive, then stop regardless.
