# raf_gradient — is the backward pass an SNN of resonate-and-fire neurons?

Origin: a chat thread on surrogate gradients → dither/lock-in gradient estimation →
"what if each gradient spike were its own SNN of RAF neurons". This directory is
**rung 1** of the scoping ladder: the smallest falsifiable piece.

## The rung-1 question

Dither every parameter of a one-layer threshold readout at its own frequency `f_i`;
feed the DC-removed scalar loss as one shared drive into a bank of the **real Spikeling
`ResonatorState`** oscillators, one tuned to each `f_i`. Can the bank read the
per-parameter gradient off resonance alone, vs a finite-difference ground truth?

## Files

| file | what |
|---|---|
| `PREREGISTER.md` | rung 1 predictions P1–P4, instrument check, kill condition — fixed before results, one dated addendum |
| `raf_gradient_rung1.py` | rung 1 harness. `--selftest`, `--sweep {spacing,n,margin}`, `--readout {soft,hard}`, `--A` |
| `plot_rung1.py` | `rung1_summary.png` |
| `RESULTS.md` | rung 1, 2026-09-07: qualified PASS — bank is a gradient *readout*, not a *smoother*; energy readout loses sign |
| `PREREGISTER_1b.md` | rung 1b predictions Q1–Q4 — recover sign from response phase |
| `raf_gradient_rung1b.py` | rung 1b harness. Same flags; reads `ResonatorState.v`/`.x` and correlates vs the dither reference |
| `plot_rung1b.py` | `rung1b_summary.png` |
| `RESULTS_1b.md` | rung 1b, 2026-09-07: PASS Q1–Q4 — `sign(⟨v·ref⟩)·sqrt(energy)` is a full signed estimate; sign is the cross-talk-robust half |
| `PREREGISTER_2.md` | rung 2 predictions R1–R4 + sign-inversion stress; two dated addenda (gate correction, then the negative) |
| `raf_gradient_rung2.py` | rung 2 harness: 2-layer SpikeNet, RAF node-perturbation vs surrogate BPTT, vectorised resonator asserted bit-equal to the real class |
| `plot_rung2.py` | `rung2_summary.png` |
| `RESULTS_2.md` | rung 2, 2026-09-07: **honest negative** — kill condition triggered (R2), and the task itself isn't validated (BPTT overfits to ~0.69) |
| `PREREGISTER_2b.md` | rung 2b: validated task + threshold-tracking dither + FD ground truth; two dated addenda |
| `raf_gradient_rung2b.py` | rung 2b harness (imports VecBank from rung 2) |
| `plot_rung2b.py` | `rung2b_summary.png` |
| `RESULTS_2b.md` | rung 2b, 2026-09-07: **partial pass** — S1/S4 pass solidly (RAF trains to BPTT level; tracking rule fixes reachability), S2 still fails for −w2 units (unreliable, 0.52–1.00 across seeds), S3 never exercised |
| `PREREGISTER_2c.md` | rung 2c: regression target + forced mid-training `w2` sign flip; one dated addendum |
| `raf_gradient_rung2c.py` | rung 2c harness (imports rung 2b's dither/FD/probe) |
| `plot_rung2c.py` | `rung2c_summary.png` |
| `RESULTS_2c.md` | rung 2c, 2026-09-07: **Gate F + U3 pass** (signal survives convergence; sign-tracking *through* `w2` inversions works, 0.99–1.00 × 4 seeds) — U1/U2 fail (blamed on over-parameterisation — **withdrawn in 2d**) |
| `PREREGISTER_2d.md` | rung 2d: over-parameterisation as a swept variable (teacher/student, `n_hid` 4→16) |
| `raf_gradient_rung2d.py` | rung 2d harness (imports rung 2b dither/FD/probe, rung 2c `_diagnose`) |
| `plot_rung2d.py` | `rung2d_summary.png` |
| `RESULTS_2d.md` | rung 2d, 2026-09-07: **2c's over-parameterisation explanation FALSIFIED** — sign fidelity *rises* with width; RAF underfits BPTT (~0.10 R²) at every width/lr; U3 survives. Kill condition triggered; retraction recorded. |
| `PREREGISTER_2e.md` | rung 2e: attribute the ~0.13 R² gap — 2×2 {exact FD, resonator} × {node-pert rule, per-weight} + a covariance diagnostic |
| `raf_gradient_rung2e.py` | rung 2e harness |
| `plot_rung2e.py` | `rung2e_summary.png` |
| `RESULTS_2e.md` | rung 2e, 2026-09-07: **gap attributed** — the FD gradient of a hard SNN is a ~10–15% worse descent direction than the surrogate (sign-noisy vs sign-consistent); the resonator is faithful and **not** the bottleneck (W3 pass, ±0.01). Ladder closed. |
| `results*.json` | raw per-cell / per-run output |

## One-line result

**Rung 1:** the resonator bank is a working frequency-multiplexed gradient *readout*
(degrades predictably with channel crowding and count), but not a *smoothing*
mechanism — on a true step it reads zero unless the dither already straddles the
threshold — and its energy readout loses gradient sign. (`RESULTS.md`)

**Rung 1b:** sign is recovered by correlating `ResonatorState.v` against the dither
reference (velocity is in phase with the drive at resonance).
`sign(⟨v·ref⟩)·sqrt(energy)` is a complete signed estimate; sign-agreement holds at
1.000 through D=32 and band ×2.5 where the magnitude readout has already decayed —
sign is the robust half. (`RESULTS_1b.md`)

**Rung 2 (honest negative):** first attempt at RAF node-perturbation training —
task too small (BPTT overfit to ~0.69), dither rule capped so it couldn't follow units
that drift from threshold, and scored against the biased surrogate. Kill condition
triggered. Superseded by 2b. `RESULTS_2.md`.

**Rung 2b (partial pass):** the three fixes — validated 2-D XOR task (BPTT 0.97–0.98),
threshold-tracking dither `A_j = clip(1.2·q30(|v_j|), 0.02, 3.0)`, per-unit
finite-difference ground truth.
- **S4 PASS** (the headline fix): 0% unreachable unit-steps vs 13–18% under the capped
  rule, every seed.
- **S1 PASS solid**: RAF node-perturbation trains the 2-layer spiking net to BPTT-level
  test accuracy (0.96–0.985) on every seed.
- **S2 FAIL**: per-unit sign fidelity for minority-sign (−w2) hidden units is
  *unreliable* — 0.52–1.00 across seeds (vs +w2 0.93–1.00). Near convergence the hard
  loss saturates into a staircase and node perturbation has little signal to read.
- **S3 still open**: no load-bearing `w2_j` zero-crossing occurred in the informative
  window — the sign-inversion-in-flight case remains unmeasured.

**Tension:** S1 passes while S2 fails — training tolerates the unreliable credit
assignment, it doesn't validate it. The substrate-feasibility claim survives (train a
multi-layer spiking net with a backward-path resonator bank, ~3500× fwd-eval cost);
the "faithful per-unit gradient" claim does not. `RESULTS_2b.md`.

**Rung 2c (Gate F + U3 pass; U1/U2 fail):** regression target (hard spiking hidden,
linear readout, MSE) + a mid-training `β` flip that forces `w2_j` through zero.
- **Gate F PASS** — with a regression loss the FD signal *survives convergence*
  (`max|fd|` late/early = 0.40); rung 2b's staircase collapse is fixed.
- **U3 PASS, replicated** — sign-agree(δ̂, fd) = **0.99–1.00 across 4 seeds** in the
  ±3-step window around ~24 forced `w2` zero-crossings. Sign-tracking *through* an
  outgoing-weight inversion works — the measurement rungs 2/2b never reached.
- **U1/U2 FAIL** — RAF trains to R² 0.79–0.94 (vs BPTT 0.96) and bulk per-step
  per-unit sign fidelity is ~0.6 for *both* w2 signs. The readout is sharp *at* the
  inversion, mushy in the redundant bulk — 6 hidden units on a 2-feature target is
  over-parameterised, so an individual unit's gradient sign is partly ill-posed
  regardless of readout. Kill condition triggered. `RESULTS_2c.md`.

**Rung 2d (falsifies 2c's explanation; kill condition):** over-parameterisation swept
as a variable — teacher/student, student `n_hid` 4→16 against a `k=4` teacher it can
fit exactly.
- **V3 FALSIFIED** — 2c predicted per-unit sign fidelity would *degrade* with excess
  width. It **rises**: −w2 bulk sign-agree 0.47 → 0.83 → 0.89 → 0.90 → 0.92 as
  n_hid 4→16. The width effect is statistical power (more −w2 units dilute the noisy
  ones), not well-posedness. **RESULTS_2c's over-parameterisation claim is withdrawn.**
- **V1 bare pass / effectively fail** — RAF R² plateaus at ~0.86 at *every* width and
  *every* lr (0.3–0.6); BPTT climbs 0.95→0.99. A persistent ~0.10 R² underfitting gap
  that no sizing/lr/step setting closes.
- **V2 FAIL** — matched-size −w2 sign-agree 0.47/0.06 (all/late), *worst* at matched
  size.
- **V4 PASS** — U3 (sign through w2 inversions) = 1.00 at every width.
`RESULTS_2d.md`.

**Rung 2e (gap attributed; ladder closed):** a 2×2 {exact FD readout, resonator
readout} × {node-pert weight rule, per-weight dither} + a covariance diagnostic.
- **W3 PASS — the resonator is not the bottleneck.** Swapping the exact FD readout for
  the resonator bank moves final R² by ±0.01 in both rows.
- **Every method that descends the *true* FD gradient of the hard SNN plateaus at
  R² 0.80–0.86** (genuine plateau, not undertraining); surrogate BPTT reaches 0.95.
- **Not surrogate smoothing** — BPTT with surrogate `1/(1+α|v|)²` gives 0.954 for
  α ∈ {2…200} (even near-delta). The surrogate wins by being **sign-consistent**; the
  FD gradient of the staircase loss is **sign-noisy**.
- **C1** — the node-pert weight rule `ΔW1_ij = δ_j·mean_n x_i` is ~orthogonal to the
  true gradient for zero-mean inputs (`cos = 0.18`); fixing it doesn't help because the
  FD-gradient cap above still binds.
`RESULTS_2e.md`.

**Cross-rung synthesis (final):** a resonator bank in the backward path is a
substrate-feasible, **sign-faithful**, threshold-tracking learning signal for hard
SNNs — it trains multi-layer nets, tracks sign through outgoing-weight inversions
(rung 2c U3), and adds ~zero noise over an exact FD readout (rung 2e W3) — but it is
**inherently ~10–15% R² below surrogate BPTT**, because it measures the *true* gradient
of a hard SNN and that is a worse descent direction than the biased-but-sign-consistent
surrogate. That cap is not an engineering defect; it is the price of measuring instead
of assuming. "Each gradient spike its own SNN" is unsupported at every rung. **No rung
2f.**

## Reproduce

```
cd research/raf_gradient
python raf_gradient_rung1.py --selftest
python raf_gradient_rung1.py --sweep spacing
python raf_gradient_rung1.py --sweep n
python raf_gradient_rung1.py --readout hard --sweep margin
python plot_rung1.py
```
