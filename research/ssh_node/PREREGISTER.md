# Pre-registration — SSHNode vs ResonatorState

Written **before** `benchmark.py` was run. Fill `RESULTS.md` with the
verdict per hypothesis (CONFIRMED / REFUTED / DOWNGRADED) and the evidence
line. Keep the ones that turn out wrong.

## What is being tested

`SSHNode` (`ssh_node.py`) — one neuron whose hidden state is an N=8 SSH
chain of damped driven oscillators, output read from the topological edge
site (site 0), **zero training**. It plugs into the runtime exactly like
`core/runtime/runtime.py`'s `ResonatorState` (`step(drive, dt) -> bool`,
same energy-EMA firing).

## Task

Tone detection at the normalised scale (`omega ≈ 1`, `dt = 0.05`, matching
`research/ssh_resonator_bridge.py`). The drive over a window is:

* **target window**: `A·sin(omega·t)` + `A_d·sin(3·omega·t)` + white noise
* **distractor-only window**: `A_d·sin(3·omega·t)` + white noise

The node should fire during target windows, not during distractor-only
windows. Score = **balanced accuracy** `(TPR + (1 − FPR)) / 2` over many
windows, at a **matched false-alarm rate**: each arm's firing threshold is
calibrated on a held-out run to `FPR ≈ 0.05` before scoring (fair-tuning
requirement — no arm is scored at its own favourable operating point).

## Arms (all read out by an edge-crossing on an RMS-energy level)

| id | what | hidden dim | trained? |
|---|---|---|---|
| `res_single` | one real `ResonatorState` | 1 | no |
| `res_pop_trained` | N `ResonatorState` (same freq) + ridge linear readout over their energies | N | **yes** |
| `ssh_topo` | `SSHNode` g = +0.6 (topological), edge readout | N | no |
| `ssh_trivial` | `SSHNode` g = −0.6 (trivial), edge readout | N | no |
| `ssh_uniform` | `SSHNode` g = 0.0 (uniform chain), edge readout | N | no |

`res_pop_trained` is the bar to beat — this portfolio's record is that a
trained linear decoder over a plain-oscillator population is hard to beat
and topological structure keeps not adding value on top of it.

## Perturbation axis (the actual research question)

Interior sites only (never the readout site), `k ∈ {0, 1, 2, 4}`:

* **chiral-preserving** — `decouple_sites`: the site is cleanly removed
  from the chain (row/col of K zeroed, input coupling zeroed, state frozen).
* **chiral-breaking** — `detune_sites`: `omega_i → omega_i·(1 ± 0.3)`.

Applied to `res_pop_trained` (drop a unit / detune a unit) and the three
`ssh_*` arms. `res_single` runs at `k = 0` only, as the reference floor.

Seeds: **8** (variance-check discipline — 3 was flagged as noise-prone in
`research/fibonacci_resonator_check.py`; report mean and a 95% CI, and if a
CI spans the decision line the verdict is INCONCLUSIVE, not a lean).

## Hypotheses and predictions

**H1 — edge-mode localisation transfers.**
`ssh_topo.edge_localization` (static) ≫ `ssh_trivial` and `ssh_uniform` at
`k = 0`; predict `> 0.5` for topo, `< 0.2` for the other two.
*Prior basis:* held 5/5 in the Spikeling LIF chain and 15/15-ish in
`ssh_resonator_bridge.py`. **Predict CONFIRMED.**

**H2 — clean-condition parity.**
At `k = 0`, `res_pop_trained` balanced accuracy ≥ `ssh_topo` (CI on the
difference includes 0 or favours the trained arm). Topology buys nothing
when the chain is intact.
**Predict CONFIRMED** (trained decoder ≥ fixed edge readout).

**H3 — topological defect-tolerance as a *win*.**
Under the **chiral-preserving** defect, `ssh_topo`'s balanced-accuracy
degradation slope (vs `k`) is shallower than `ssh_trivial` **and shallower
than `res_pop_trained`**, with the win-rate 95% CI lower bound `> 50%`.
*Prior basis:* INCONCLUSIVE in `ssh_resonator_bridge.py` (60%, CI 35–85%);
FALSIFIED on the photonic array (25.7%, CI 15.7–35.7%).
**Predict REFUTED / INCONCLUSIVE** — `ssh_topo` degrades slower than
`ssh_trivial`/`ssh_uniform` (that's H1 doing work) but does **not** beat
`res_pop_trained`.

**H4 — the chiral-symmetry mechanism.**
`ssh_topo` tolerates the chiral-preserving defect *better than* the
chiral-breaking defect (preserving-slope shallower than breaking-slope),
CI on the difference excludes 0 in the predicted direction.
*Prior basis:* failed 0/5 in the LIF chain, 0/5 in the resonator bridge,
both robustly in the wrong direction; photonic reversed even harder.
**Predict REFUTED**, likely wrong-direction.

**H5 — the one that would make it worth shipping.**
Under the chiral-preserving defect, `ssh_topo` (zero training) *matches*
`res_pop_trained` (trained) on degradation slope — CI on the difference
includes 0. If so: a training-free robust readout, synthesisable on the
Verilog backend, "as robust, for free".
**Predict REFUTED** — but this is the hypothesis the rung exists to check;
a CONFIRMED here is the surprise worth a write-up.

## Falsification bars (any one → the overall claim is a negative)

* H1 REFUTED (edge localisation doesn't even transfer) → the whole premise is dead.
* H3 REFUTED **and** H5 REFUTED → SSHNode is strictly dominated by a
  trained oscillator population; no reason to use it.
* `ssh_topo` NaNs / diverges on any seed → the internal SSH neuron is not a
  viable primitive at this scale without more machinery than a drop-in.

## Controls baked in

* `ssh_trivial` (opposite dimerisation, same two coupling values) isolates
  *dimerisation order* from "a coupled chain exists".
* `ssh_uniform` (g = 0) isolates *topology* from "any nearest-neighbour coupling".
* matched-FPR threshold calibration per arm.
* `res_single` as the no-chain floor.
* 8 seeds, CI-gated verdicts.
