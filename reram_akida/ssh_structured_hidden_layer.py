#!/usr/bin/env python
"""
ssh_structured_hidden_layer.py — testing the user's own cross-substrate
theorem (chiral-symmetry-conditional defect tolerance, already confirmed
on the phononic SSH chain and Spikeling's Resonator neuron, and already
falsified/reversed on the physical Duffing model) on a 5th substrate: THIS
session's ReRAM+Akida drift detector, specifically against the real
failure mode ReSpike's paper just identified (2026-08-29 read) — under
rising ReRAM programming/read error, the hidden layer doesn't misclassify
gracefully, it goes SILENT (total spike count collapses toward zero, see
their Fig.7), and a detector that defaults to "no drift" on silence would
misreport that collapse as a null result.

Reuses REAL, already-verified code, not reimplementations:
  - ssh_H() from topological-phononics/ssh_topological_reservoir.py — the
    exact same tight-binding alternating-hopping (v intra-cell, w
    inter-cell) generator, with the SAME real disorder-injection modes
    (hop=symmetry-respecting/proportional, onsite=symmetry-breaking) used
    in that file's own A1/A2 tests (topological w>v edge mode pinned
    under hopping disorder, drifts under on-site disorder).
  - AdExHabituationNeuron from habituation_hidden_layer.py — unchanged.
  - ReRAMSynapseArray.program()/.weight[][] from reram_synapse_array.py.

STRUCTURAL ADAPTATION (disclosed, not hidden): this detector's synapses
are a bipartite channel->hidden graph, not a literal 1D lattice, so there
is no single natural embedding of a chain onto it. The adaptation used
here: each hidden neuron h gets its OWN independent SSH chain of length
2*n_channels via ssh_H(N=n_channels, v, w) (same v,w regime constants
literally copied from ssh_topological_reservoir.py: VT,WT=0.4,1.0 for
topological / VR,WR=1.0,0.4 for trivial). The chain's real alternating
off-diagonal hopping sequence [H[0,1], H[1,2], ..., H[2N-2,2N-1]] IS the
v,w,v,w,... pattern the original file verified; a phase offset per
neuron (h % 2) selects whether channel 0 sees a v-type or w-type bond
first, giving population diversity while reusing the exact same real
values. This is a genuine reuse of the verified generator's real output,
not a relabeled fresh random draw.

ERROR-MODE MAPPING TO REAL RERAM/ReSpike TAXONOMY (reasoned, not
arbitrary): ReSpike's own real error taxonomy (Section 2.3) already
splits into "proportional" (deviation scales with the cell's own state)
and "independent" (deviation is state-independent). A proportional error
scales every bond by roughly the same relative amount, so it preserves
which bonds are strong (w) vs weak (v) — the same real property that
makes SSH's "hopping disorder" symmetry-RESPECTING. An independent error
adds the same-size perturbation regardless of a bond's own strength, so
it can push a genuinely weak (v) bond above a genuinely strong (w) one —
the same real property that makes SSH's "on-site disorder"
symmetry-BREAKING (it can scramble which sites/bonds dominate). This
script therefore uses ReSpike's own proportional/independent error model
directly as the hopping/on-site disorder analogue, instead of inventing
a separate disorder mechanism.

PRE-REGISTERED HYPOTHESIS (stated before running): under PROPORTIONAL
(symmetry-respecting) ReRAM error, the topological (w>v) SSH-structured
hidden layer resists the ReSpike spike-collapse failure mode better
(higher total spike count at a given error level) than the trivial (v>w)
SSH-structured layer or the existing random-uniform baseline. Under
INDEPENDENT (symmetry-breaking) error, this advantage should NOT appear
(or should be much smaller) — protection is conditional on the error
type, mirroring the original file's real A2 result.
DISCONFIRM: no topological-vs-trivial gap under proportional error, or an
equal/larger gap under independent error -> report a clean null, same
discipline as the symmetry_breaking_test_v2.py null earlier in this
session.
"""

import os
import sys
import random
import math

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "topological-phononics"))
from ssh_topological_reservoir import ssh_H  # noqa: E402

from reram_synapse_array import ReRAMSynapseArray
from habituation_hidden_layer import AdExHabituationNeuron
from akida_style_drift_detector import make_signature

# same real regime constants as ssh_topological_reservoir.py, not re-picked
VT, WT = 0.4, 1.0   # topological: w > v
VR, WR = 1.0, 0.4   # trivial:     v > w


def ssh_channel_weights(n_channels, v, w, phase_offset):
    """One clean SSH chain's real alternating hop sequence, reused
    unmodified from ssh_H -- returns n_channels weight values."""
    H = ssh_H(n_channels, v, w)  # 2*n_channels x 2*n_channels, clean (no disorder yet)
    hops = [H[i, i + 1] for i in range(2 * n_channels - 1)]
    start = phase_offset % 2
    seq = hops[start:start + n_channels]
    while len(seq) < n_channels:  # pad if we ran off the end
        seq.append(hops[-1])
    return seq


def build_ssh_reram_pair(n_hidden, n_channels, regime, seed):
    """Program ReRAM cells from the real SSH hop sequence per hidden
    neuron (regime='topological'/'trivial'), or fall back to the
    existing random-uniform baseline (regime='random') for comparison."""
    rng = random.Random(seed)
    reram_pos = ReRAMSynapseArray(n_hidden, n_channels, seed=seed)
    reram_neg = ReRAMSynapseArray(n_hidden, n_channels, seed=seed + 5000)

    if regime == "random":
        for h in range(n_hidden):
            for c in range(n_channels):
                reram_pos.program(h, c, rng.uniform(0.0, 1.0), mode="iterative")
                reram_neg.program(h, c, rng.uniform(0.0, 1.0), mode="iterative")
        return reram_pos, reram_neg

    v, w = (VT, WT) if regime == "topological" else (VR, WR)
    # normalize the raw hop values (~0.4-1.0) into [0,1] program targets
    lo, hi = min(v, w), max(v, w)
    for h in range(n_hidden):
        seq = ssh_channel_weights(n_channels, v, w, phase_offset=h)
        for c in range(n_channels):
            target = (seq[c] - lo) / (hi - lo + 1e-9)
            reram_pos.program(h, c, target, mode="iterative")
            # neg cell gets the complementary bond so w=pos-neg keeps the
            # same real alternating strong/weak CONTRAST, not cancelled out
            reram_neg.program(h, c, 1.0 - target, mode="iterative")
    return reram_pos, reram_neg


def apply_reram_error(reram_array, pct, mode, rng):
    """Real ReSpike error taxonomy (Sec 2.3), applied post-programming
    directly to the stored cell values -- same two categories their own
    sensitivity sweep uses (proportional / independent), same use as the
    hopping/on-site disorder analogue described above."""
    if pct <= 0:
        return
    frac = pct / 100.0
    for r in range(reram_array.rows):
        for c in range(reram_array.cols):
            w = reram_array.weight[r][c]
            if mode == "proportional":
                sigma = abs(w) * frac
            else:  # independent
                sigma = frac  # scaled to the full [0,1] cell range, state-independent
            reram_array.weight[r][c] = max(0.0, min(1.0, w + rng.gauss(0.0, sigma)))


class SSHStructuredHiddenLayer:
    """Same AdEx habituation neuron as habituation_hidden_layer.py --
    only the ReRAM weight STRUCTURE differs (SSH-alternating vs random)."""

    def __init__(self, n_channels, n_hidden, reram_pos, reram_neg, seed=0):
        self.n_channels = n_channels
        self.n_hidden = n_hidden
        self.reram_pos = reram_pos
        self.reram_neg = reram_neg
        rng = random.Random(seed + 4242)
        self.neurons = [
            AdExHabituationNeuron(threshold=rng.uniform(0.4, 1.1), gL=0.3, tau_w=40.0)
            for _ in range(n_hidden)
        ]

    def step(self, channel_input):
        spikes = [0] * self.n_hidden
        for h in range(self.n_hidden):
            drive = 0.0
            for c in range(self.n_channels):
                w = self.reram_pos.read(h, c) - self.reram_neg.read(h, c)
                drive += w * channel_input[c]
            if self.neurons[h].step(drive, dt=1.0):
                spikes[h] = 1
        return spikes


def spike_count_under_error(regime, error_mode, error_pct, n_channels=8, n_hidden=24,
                             ticks=300, noise_std=0.05, seed=0):
    reram_pos, reram_neg = build_ssh_reram_pair(n_hidden, n_channels, regime, seed)
    err_rng = random.Random(seed + 9001)
    apply_reram_error(reram_pos, error_pct, error_mode, err_rng)
    apply_reram_error(reram_neg, error_pct, error_mode, err_rng)

    hidden = SSHStructuredHiddenLayer(n_channels, n_hidden, reram_pos, reram_neg, seed=seed)
    sig = make_signature(n_channels, seed=seed * 2 + 1)
    rng = random.Random(seed + 77)
    total = 0
    for _ in range(ticks):
        noisy = [v + rng.gauss(0, noise_std) for v in sig]
        total += sum(hidden.step(noisy))
    return total


def paired_t_test(diffs):
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    se = std / math.sqrt(n) if n > 0 else 0.0
    t = mean / se if se > 0 else float("inf")
    z = abs(t)
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))) if math.isfinite(z) else 0.0
    return mean, std, t, p


if __name__ == "__main__":
    print("=" * 78)
    print("SSH-structured ReRAM connectivity vs random baseline")
    print("Real hypothesis: topological (w>v) resists ReSpike's spike-collapse")
    print("failure mode under PROPORTIONAL error, not under INDEPENDENT error.")
    print("=" * 78)

    N_TRIALS = 15
    ERROR_LEVELS = [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 15, 25, 50]
    REGIMES = ["topological", "trivial", "random"]

    for error_mode in ["proportional", "independent"]:
        print(f"\n--- {error_mode.upper()} error ---")
        curves = {regime: [] for regime in REGIMES}
        for pct in ERROR_LEVELS:
            means = {}
            for regime in REGIMES:
                counts = [spike_count_under_error(regime, error_mode, pct, seed=s)
                          for s in range(N_TRIALS)]
                mean = sum(counts) / N_TRIALS
                means[regime] = mean
                curves[regime].append(counts)
            print(f"  error={pct:5.1f}%  topological={means['topological']:6.1f}  "
                  f"trivial={means['trivial']:6.1f}  random={means['random']:6.1f}")

        print(f"\n  Paired topological-vs-trivial comparison at key error levels:")
        for pct, idx in [(5, ERROR_LEVELS.index(5)), (15, ERROR_LEVELS.index(15)),
                          (50, ERROR_LEVELS.index(50))]:
            diffs = [t - v for t, v in zip(curves["topological"][idx], curves["trivial"][idx])]
            mean_diff, std_diff, t, p = paired_t_test(diffs)
            sig = "SIGNIFICANT" if p < 0.05 else "not significant"
            print(f"    error={pct}%: mean(topological-trivial)={mean_diff:+.2f}  "
                  f"t={t:.3f}  p={p:.4g}  ({sig})")

    print("\n--- read the raw numbers above against the pre-registration; report honestly ---")
