# SSHNode — an SSH-chain neuron, benchmarked against the real ResonatorState

**Question.** A neuron whose hidden state is a small Su-Schrieffer-Heeger
chain (staggered internal coupling `t1, t2, t1, …`), read out at its
topological **edge site** with **zero training** — does that internal
structure buy anything (robustness, selectivity) over the real Spikeling
`ResonatorState`, or over a trained linear readout on a population of them?

This leans on the part of the [topological-phononics] work that *did*
transfer across substrates (edge-vs-bulk localisation, 5/5 in the Spikeling
LIF chain) rather than the part that didn't (network defect-tolerance —
inconclusive on the Resonator bridge, falsified on the photonic array).

## Files

| file | what |
|---|---|
| `ssh_node.py` | `SSHNode` — drop-in for `ResonatorState` (`step(drive, dt) -> bool`, same energy-EMA firing), internally an N-site SSH chain. `decouple_sites` (chiral-preserving defect) / `detune_sites` (chiral-breaking). |
| `PREREGISTER.md` | hypotheses H1–H5 and their predicted verdicts, written before any run. |
| `benchmark.py` | tone-detection task, matched-FPR scoring, defect sweep, 8 seeds → `RESULTS.json` + console table + pre-registered verdicts. |
| `RESULTS.md` | filled in after the run: verdict per hypothesis + honest read. |

## Run

```bash
python benchmark.py                 # N=8, 8 seeds
python benchmark.py --n-sites 4     # smaller chain — interior defects bite harder (real test of H5)
python benchmark.py --smoke         # 1 seed, tiny windows — just checks nothing NaNs
```

## Design notes

- **Substrate fidelity.** Each SSH site integrates the *exact*
  `ResonatorState.step()` dynamics (symplectic Euler, `-2 damping omega v`
  term, gated `x^2` energy EMA), extended with an internal coupling matrix
  `K`. `K` is spectrally normalised so `rho(K) < 0.7 * min(omega_i^2)` — an
  unnormalised chain diverges to NaN (the lesson from
  `../ssh_resonator_bridge.py`'s first run).
- **Drive.** `drive_mode="edge"`: the tone is injected only at the readout
  edge (a physical SSH chain is boundary-driven), boosted by `N` so total
  injected power matches a uniform drive. `drive_mode="all"` is also
  available.
- **Fair tuning.** Task difficulty was tuned *before* the scored run so a
  single `ResonatorState` lands ~0.80–0.90 balanced accuracy (room to move
  either way). Every arm's operating point is set to `FPR ≈ 0.05` on a
  calibration split, then scored on a disjoint test split.
- **Controls.** `ssh_trivial` (opposite dimerisation) isolates dimerisation
  *order* from "a chain exists"; `ssh_uniform` (`g=0`) isolates *topology*
  from "any coupling"; `res_single` is the no-chain floor; `res_pop_trained`
  is the bar (trained decoder over an oscillator bank).

## Prior context (why the predictions lean negative)

`topological_phononics_finding` (published), `spikeling_phononic_bridge`,
the SSH LIF-chain chiral experiment, `photonic_ternary_ssh_negative_result`
— three prior measurements, all: edge/bulk distinction transfers,
topological defect-tolerance does **not** transfer as a robust win, the
chiral-symmetry mechanism fails (often wrong-direction). PREREGISTER.md
predicts the same here and treats a CONFIRMED H5 ("as robust as the trained
bank, for free") as the surprise worth writing up.
