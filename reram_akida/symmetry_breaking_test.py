#!/usr/bin/env python
"""
symmetry_breaking_test.py — testing the user's own proven theorem against
this 4th substrate (the ReRAM+Akida-style drift detector).

THE THEOREM (from acoustic-vortex-sim/reservoir_computing, proven to 1e-9
on FEM modes, see PROOF_selection_rule.txt): on a D4-symmetric plate,
integral(phi^3) = 0 for every non-A1 mode -- point symmetry makes
even-order/higher-order computation structurally DEAD. Breaking that
symmetry (quasicrystal disorder, low-symmetry perturbation) LIFTS the
dead fraction and activates real even-order computational capacity.
Portable principle, already confirmed on THREE independent physical
substrates: a MEMS quasicrystal plate, a 24-node lattice ring (dihedral
symmetry kills exactly 21/24 modes, 3k=0 mod N), and a coupled Spikeling
resonator bank.

PRE-REGISTERED HYPOTHESIS (stated before running this): a ReRAM hidden-
layer weight population built with explicit CYCLIC symmetry across input
channels (every hidden neuron's weight vector is a cyclic shift of one
shared base pattern -- the closest discrete analog to the plate's point-
symmetry group available in an 8-channel array) will show a SMALLER
differential population response (spike-rate change) to the drift
signature than a symmetry-BROKEN population (independently random
weights per neuron, no shared structure) -- extending the theorem to a
4th, non-physical substrate.

This does NOT reuse or reinterpret the exact D4/dihedral machinery from
the plate work (that's a 2D point-group symmetry on a physical mode
shape; an 8-channel synapse vector has no literal rotation group in the
same sense) -- cyclic-shift symmetry across channels is the honest,
stated discrete analogy being tested, not a claim of mathematical
identity with the proven plate theorem.
"""

import random

from reram_synapse_array import ReRAMSynapseArray
from akida_style_drift_detector import HiddenLayer, make_signature


def build_symmetric_reram_pair(n_hidden, n_channels, seed):
    """Every hidden neuron's (pos, neg) weight vector is a cyclic shift of
    ONE shared base pattern -- explicit symmetry across the population,
    the discrete analog of the plate's point-symmetry group."""
    rng = random.Random(seed)
    base_pos = [rng.uniform(0.0, 1.0) for _ in range(n_channels)]
    base_neg = [rng.uniform(0.0, 1.0) for _ in range(n_channels)]

    reram_pos = ReRAMSynapseArray(n_hidden, n_channels, seed=seed)
    reram_neg = ReRAMSynapseArray(n_hidden, n_channels, seed=seed + 5000)
    for h in range(n_hidden):
        shift = h % n_channels
        for c in range(n_channels):
            reram_pos.program(h, c, base_pos[(c + shift) % n_channels], mode="iterative")
            reram_neg.program(h, c, base_neg[(c + shift) % n_channels], mode="iterative")
    return reram_pos, reram_neg


def build_broken_symmetry_reram_pair(n_hidden, n_channels, seed):
    """Independently random weight per (neuron, channel) cell -- no shared
    structure across the population. The existing baseline construction
    from akida_style_drift_detector.py, used here as the symmetry-BROKEN
    condition."""
    rng = random.Random(seed)
    reram_pos = ReRAMSynapseArray(n_hidden, n_channels, seed=seed)
    reram_neg = ReRAMSynapseArray(n_hidden, n_channels, seed=seed + 5000)
    for h in range(n_hidden):
        for c in range(n_channels):
            reram_pos.program(h, c, rng.uniform(0.0, 1.0), mode="iterative")
            reram_neg.program(h, c, rng.uniform(0.0, 1.0), mode="iterative")
    return reram_pos, reram_neg


def measure_differential_response(reram_pos, reram_neg, n_channels, n_hidden, seed,
                                   noise_std=0.05, settle_ticks=300, measure_ticks=300):
    """Real measurement: population spike-rate under the normal signature
    vs. under the fully-drifted signature, same network, same noise
    process, only the signature differs. Returns (normal_rate,
    drift_rate, relative_increase)."""
    rng = random.Random(seed + 77)
    normal_sig = make_signature(n_channels, seed=seed * 2 + 1)
    drift_sig = make_signature(n_channels, seed=seed * 2 + 2)
    hidden = HiddenLayer(n_channels, n_hidden, reram_pos, reram_neg, seed=seed)

    normal_total = 0
    for _ in range(settle_ticks):
        noisy = [v + rng.gauss(0, noise_std) for v in normal_sig]
        spikes = hidden.step(noisy)
        normal_total += sum(spikes)
    normal_rate = normal_total / settle_ticks

    drift_total = 0
    for _ in range(measure_ticks):
        noisy = [v + rng.gauss(0, noise_std) for v in drift_sig]
        spikes = hidden.step(noisy)
        drift_total += sum(spikes)
    drift_rate = drift_total / measure_ticks

    rel = (drift_rate - normal_rate) / normal_rate if normal_rate > 0 else float("nan")
    return normal_rate, drift_rate, rel


if __name__ == "__main__":
    print("=" * 78)
    print("Symmetry-breaking theorem, applied to a 4th substrate")
    print("Pre-registered BEFORE this run: symmetric weight population shows a")
    print("SMALLER differential response to drift than a symmetry-broken one.")
    print("=" * 78)

    N_TRIALS = 15
    n_channels, n_hidden = 8, 24

    sym_rels, broken_rels = [], []
    for seed in range(N_TRIALS):
        rp, rn = build_symmetric_reram_pair(n_hidden, n_channels, seed)
        nr, dr, rel = measure_differential_response(rp, rn, n_channels, n_hidden, seed)
        sym_rels.append(rel)

        rp2, rn2 = build_broken_symmetry_reram_pair(n_hidden, n_channels, seed)
        nr2, dr2, rel2 = measure_differential_response(rp2, rn2, n_channels, n_hidden, seed)
        broken_rels.append(rel2)

        print(f"  seed={seed:2d}  symmetric: normal={nr:.3f} drift={dr:.3f} rel={rel:+.3f}"
              f"   |   broken: normal={nr2:.3f} drift={dr2:.3f} rel={rel2:+.3f}")

    valid_sym = [r for r in sym_rels if r == r]     # drop NaNs (normal_rate==0 cases)
    valid_broken = [r for r in broken_rels if r == r]

    print(f"\n{N_TRIALS} real trials, same seeds/signatures/noise for both conditions:")
    if valid_sym:
        print(f"  Symmetric population:      mean relative response = {sum(valid_sym)/len(valid_sym):+.3f}"
              f"  ({len(valid_sym)}/{N_TRIALS} valid, rest had zero normal-rate baseline)")
    if valid_broken:
        print(f"  Symmetry-broken population: mean relative response = {sum(valid_broken)/len(valid_broken):+.3f}"
              f"  ({len(valid_broken)}/{N_TRIALS} valid)")

    if valid_sym and valid_broken:
        sym_mean = sum(valid_sym) / len(valid_sym)
        broken_mean = sum(valid_broken) / len(valid_broken)
        held = broken_mean > sym_mean
        print(f"\nHypothesis (broken-symmetry response > symmetric response): "
              f"{'HELD' if held else 'DID NOT HOLD'}")
        print(f"  broken - symmetric = {broken_mean - sym_mean:+.4f}")
