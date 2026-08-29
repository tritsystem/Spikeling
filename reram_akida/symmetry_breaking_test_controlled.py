#!/usr/bin/env python
"""
symmetry_breaking_test_controlled.py — confound-controlled follow-up to
symmetry_breaking_test.py. The first pass found a real ~2x mean-|drive|
magnitude difference between the symmetric (cyclic-shift) and broken-
symmetry (independent random) weight populations, confounding the
differential-response comparison. This version equalizes real measured
drive magnitude between the two conditions BEFORE comparing differential
response to drift, isolating "symmetric structure" as the only remaining
variable -- same discipline the user's own quasicrystal work applied
("peer review forced a FRAMING CORRECTION: the effect is driven by
BROKEN POINT SYMMETRY, not quasicrystallinity").

Method: for each trial, measure each population's real mean |drive|
against that trial's actual normal signature, then scale the SYMMETRIC
condition's channel input by (broken_mean_drive / symmetric_mean_drive)
-- mathematically equivalent to rescaling its weights, without needing
to re-program the ReRAM array. This equalizes realized drive amplitude
while leaving the cyclic-shift correlation structure (the actual thing
being tested) untouched.
"""

import random

from reram_synapse_array import ReRAMSynapseArray
from akida_style_drift_detector import HiddenLayer, make_signature
from symmetry_breaking_test import build_symmetric_reram_pair, build_broken_symmetry_reram_pair


def mean_abs_drive(reram_pos, reram_neg, n_hidden, n_channels, signal):
    total = 0.0
    for h in range(n_hidden):
        d = sum((reram_pos.read(h, c) - reram_neg.read(h, c)) * signal[c] for c in range(n_channels))
        total += abs(d)
    return total / n_hidden


def measure_differential_response_scaled(reram_pos, reram_neg, n_channels, n_hidden, seed,
                                          input_gain=1.0, noise_std=0.05,
                                          settle_ticks=300, measure_ticks=300):
    """Same measurement as the uncontrolled version, but every channel
    input is multiplied by `input_gain` before driving the network --
    equivalent to rescaling the weight matrix, used here to equalize
    realized drive magnitude across conditions."""
    rng = random.Random(seed + 77)
    normal_sig = make_signature(n_channels, seed=seed * 2 + 1)
    drift_sig = make_signature(n_channels, seed=seed * 2 + 2)
    hidden = HiddenLayer(n_channels, n_hidden, reram_pos, reram_neg, seed=seed)

    normal_total = 0
    for _ in range(settle_ticks):
        noisy = [(v + rng.gauss(0, noise_std)) * input_gain for v in normal_sig]
        spikes = hidden.step(noisy)
        normal_total += sum(spikes)
    normal_rate = normal_total / settle_ticks

    drift_total = 0
    for _ in range(measure_ticks):
        noisy = [(v + rng.gauss(0, noise_std)) * input_gain for v in drift_sig]
        spikes = hidden.step(noisy)
        drift_total += sum(spikes)
    drift_rate = drift_total / measure_ticks

    rel = (drift_rate - normal_rate) / normal_rate if normal_rate > 0 else float("nan")
    return normal_rate, drift_rate, rel


if __name__ == "__main__":
    print("=" * 78)
    print("Confound-controlled symmetry-breaking test")
    print("Drive magnitude equalized per-trial BEFORE comparing differential")
    print("response -- isolates weight-structure (symmetric vs broken) as the")
    print("only remaining variable.")
    print("=" * 78)

    N_TRIALS = 15
    n_channels, n_hidden = 8, 24

    sym_rels, broken_rels = [], []
    gains_applied = []

    for seed in range(N_TRIALS):
        normal_sig = make_signature(n_channels, seed=seed * 2 + 1)

        rp_sym, rn_sym = build_symmetric_reram_pair(n_hidden, n_channels, seed)
        rp_brk, rn_brk = build_broken_symmetry_reram_pair(n_hidden, n_channels, seed)

        sym_drive = mean_abs_drive(rp_sym, rn_sym, n_hidden, n_channels, normal_sig)
        brk_drive = mean_abs_drive(rp_brk, rn_brk, n_hidden, n_channels, normal_sig)
        gain = brk_drive / sym_drive if sym_drive > 0 else 1.0
        gains_applied.append(gain)

        nr, dr, rel = measure_differential_response_scaled(
            rp_sym, rn_sym, n_channels, n_hidden, seed, input_gain=gain)
        sym_rels.append(rel)

        nr2, dr2, rel2 = measure_differential_response_scaled(
            rp_brk, rn_brk, n_channels, n_hidden, seed, input_gain=1.0)
        broken_rels.append(rel2)

        print(f"  seed={seed:2d}  gain_applied={gain:5.2f}x  "
              f"symmetric(scaled): normal={nr:.3f} drift={dr:.3f} rel={rel:+.3f}"
              f"   |   broken: normal={nr2:.3f} drift={dr2:.3f} rel={rel2:+.3f}")

    valid_sym = [r for r in sym_rels if r == r]
    valid_broken = [r for r in broken_rels if r == r]

    print(f"\n{N_TRIALS} real trials, drive-magnitude-equalized:")
    print(f"  mean gain applied to symmetric condition: {sum(gains_applied)/len(gains_applied):.2f}x")
    if valid_sym:
        print(f"  Symmetric population (scaled):   mean relative response = "
              f"{sum(valid_sym)/len(valid_sym):+.3f}  ({len(valid_sym)}/{N_TRIALS} valid)")
    if valid_broken:
        print(f"  Symmetry-broken population:       mean relative response = "
              f"{sum(valid_broken)/len(valid_broken):+.3f}  ({len(valid_broken)}/{N_TRIALS} valid)")

    if valid_sym and valid_broken:
        sym_mean = sum(valid_sym) / len(valid_sym)
        broken_mean = sum(valid_broken) / len(valid_broken)
        held = broken_mean > sym_mean
        print(f"\nHypothesis (broken-symmetry response > symmetric response), "
              f"NOW CONFOUND-CONTROLLED: {'HELD' if held else 'DID NOT HOLD'}")
        print(f"  broken - symmetric = {broken_mean - sym_mean:+.4f}")
