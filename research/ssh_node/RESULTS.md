# Results — SSHNode vs ResonatorState

8 seeds each at **N=8** (`RESULTS_N8.json`) and **N=4** (`RESULTS_N4.json`),
tone-detection task, balanced accuracy at matched FPR, interior-defect sweep.
Verdicts are against the predictions in `PREREGISTER.md` (written first).

## Scoreboard (balanced accuracy, mean over 8 seeds)

**N = 8**

| arm | clean | preserving k=1/2/4 | breaking k=1/2/4 |
|---|---|---|---|
| `res_single` | 0.909 [0.882, 0.935] | — | — |
| `res_pop_trained` (the bar) | 0.909 [0.882, 0.935] | 0.909 / 0.909 / 0.909 | 0.909 / 0.909 / 0.909 |
| `ssh_topo` (g=+0.6) | 0.904 [0.873, 0.934] | 0.906 / 0.904 / **0.876** | 0.904 / 0.906 / 0.909 |
| `ssh_trivial` (g=−0.6) | **0.842** [0.815, 0.869] | 0.850 / 0.841 / 0.892 | 0.854 / 0.903 / 0.903 |
| `ssh_uniform` (g=0) | 0.877 [0.850, 0.905] | 0.891 / 0.876 / 0.877 | 0.884 / 0.905 / 0.905 |

**N = 4** (interior = 2 sites, so k=4 not reachable) — same shape:
`ssh_topo` 0.904 clean, `res_pop_trained` 0.909 flat under all defects,
`ssh_trivial` worst at 0.838.

Static edge localisation `|psi_0|^2` of the K≈0 mode:

| | N=8 | N=4 | delocalised floor (1/N) |
|---|---|---|---|
| `ssh_topo` | 0.469 | 0.474 | 0.125 / 0.25 |
| `ssh_uniform` | 0.216 | 0.362 | |
| `ssh_trivial` | 0.092 | 0.281 | |

## Verdicts

| H | prediction | verdict | evidence |
|---|---|---|---|
| **H1** edge localisation transfers | CONFIRMED | **DOWNGRADED** (N=8), **REFUTED** by the pre-set bar (N=4) | Ranking holds both N (topo > uniform > trivial), and at N=8 topo (0.47) is ~5× trivial (0.09). But the pre-registered bars (topo > 0.5, others < 0.2) are missed — the finite chain's two ends hybridise, more so as N shrinks. The distinction is real; the localisation is weak at these N. |
| **H2** trained pop ≥ SSH topo on clean data | CONFIRMED | **CONFIRMED** | 0.909 vs 0.904, both N. The internal SSH structure buys nothing when the chain is intact. |
| **H3** SSH topo wins on defect-tolerance | REFUTED / INCONCLUSIVE | **REFUTED / INCONCLUSIVE** | `ssh_topo` does not beat `res_pop_trained` under either defect; the trained bank is flat (0.909) and `ssh_topo` if anything dips (0.876 at preserving k=4, N=8). |
| **H4** chiral-preserving easier than chiral-breaking | REFUTED | **REFUTED** | No preserving-vs-breaking asymmetry for `ssh_topo`. Fourth substrate, fourth time the chiral-symmetry mechanism fails to appear. |
| **H5** zero-training SSH readout as robust as the trained bank | REFUTED (but the one worth checking) | **REFUTED** | `ssh_topo`'s preserving-defect drop (≈0.028 at k=4, N=8) exceeds `res_pop_trained`'s (≈0). Not "as robust, for free." |

No falsification bar was tripped on viability: `SSHNode` never NaN'd across
16 runs × 8 seeds × 7 conditions — the integrator normalisation holds and
the class is a clean drop-in for `ResonatorState`.

## Honest read

**In this frequency-detection regime an internal SSH chain buys nothing
over a single `ResonatorState`.** Best case it matches (`ssh_topo` ≈
`res_single` ≈ `res_pop_trained` ≈ 0.90); worst case — the wrong
dimerisation order (`ssh_trivial`) — it is ~7 points *worse*, because a
non-topological chain spreads the driven response away from the readout
site. `ssh_uniform` sits between. This is exactly the pattern of
[topological_phononics_finding] / [spikeling_phononic_bridge] /
[photonic_ternary_ssh_negative_result]: the **edge-vs-bulk distinction
transfers** (here: weakly, and it does shape the readout — `ssh_topo`
beats its own trivial/uniform controls), while a **functional topological
advantage does not**.

## Load-bearing caveat — the defect axis never bit

`res_pop_trained` is flat under every defect because a small bank of
identical-frequency resonators is trivially redundant: drop or detune a
couple of eight and the survivors still detect the tone. So H3/H5 were
never a real contest — "match the trained bank's robustness" is a low bar
when the bank has nothing to lose. A fair robustness test needs a task
where units carry **complementary** information (multi-frequency
discrimination, or a NARMA-style memory task like the one
`../ssh_resonator_bridge.py` used) so that losing a unit actually costs
accuracy. That is the natural rung-2.

The **primary** question this rung set out to answer — does wrapping a
detector's hidden state in an SSH chain make the detector itself better —
is answered: **no**, and it is answered cleanly.

## Rung-2, if pursued

- Complementary-information task (multi-tone or NARMA memory) so the
  defect axis can actually separate the arms.
- `drive_mode="all"` vs `"edge"` as an axis — this rung used `"edge"` only.
- Larger N (16, 32) — the edge localisation should sharpen; check whether
  H1 recovers its pre-registered bars and whether that changes anything
  functionally (predict: still no).
- Only worth doing if rung-2's H5 shows real signal; otherwise this line
  is closed the same way the photonic one was.
