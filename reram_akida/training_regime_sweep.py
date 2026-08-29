#!/usr/bin/env python
"""
training_regime_sweep.py — pushing further on the habituation detector's
training regime, per the real Cerebra reference (10 AKD1000 chips, a much
larger population than our 24-neuron toy, and sustained unsupervised
training on an empty room before any real detection was demonstrated).

PRE-REGISTERED HYPOTHESIS (stated before running): increasing hidden
population size and/or training duration will increase the real additive
differential response (drift_rate - normal_rate) and improve detection
reliability (fewer near-zero-response trials), compared to the documented
24-neuron/300-tick baseline (0.44 -> 0.47 population rate, ~7% relative,
too small for reliable threshold detection).

Method: a real parameter sweep, not a single lucky config -- population
size in {24, 48, 96} x training duration in {300, 1500} ticks, N=15 paired
trials per cell, ADDITIVE metric only (the relative/fractional metric
already proved untrustworthy near zero baselines in the symmetry test).
Same signed-weight ReRAM construction and the already-calibrated AdEx
threshold/gL from habituation_hidden_layer.py -- only population size and
training duration change.
"""

import math
import random

from reram_synapse_array import ReRAMSynapseArray
from habituation_hidden_layer import HabituationHiddenLayer
from akida_style_drift_detector import make_signature


def build_reram_pair(n_hidden, n_channels, seed):
    rng = random.Random(seed)
    reram_pos = ReRAMSynapseArray(n_hidden, n_channels, seed=seed)
    reram_neg = ReRAMSynapseArray(n_hidden, n_channels, seed=seed + 5000)
    for h in range(n_hidden):
        for c in range(n_channels):
            reram_pos.program(h, c, rng.uniform(0.0, 1.0), mode="iterative")
            reram_neg.program(h, c, rng.uniform(0.0, 1.0), mode="iterative")
    return reram_pos, reram_neg


def run_one(n_hidden, n_channels, train_ticks, seed, noise_std=0.05, measure_ticks=300):
    rng = random.Random(seed + 77)
    reram_pos, reram_neg = build_reram_pair(n_hidden, n_channels, seed)
    normal_sig = make_signature(n_channels, seed=seed * 2 + 1)
    drift_sig = make_signature(n_channels, seed=seed * 2 + 2)
    hidden = HabituationHiddenLayer(n_channels, n_hidden, reram_pos, reram_neg, seed=seed)

    normal_total = 0
    settle_window = min(100, max(20, train_ticks // 3))  # last third, floor 20/cap 100
    recent = []
    for t in range(train_ticks):
        noisy = [v + rng.gauss(0, noise_std) for v in normal_sig]
        spikes = hidden.step(noisy)
        recent.append(sum(spikes))
    normal_rate = sum(recent[-settle_window:]) / settle_window

    drift_total = 0
    for _ in range(measure_ticks):
        noisy = [v + rng.gauss(0, noise_std) for v in drift_sig]
        spikes = hidden.step(noisy)
        drift_total += sum(spikes)
    drift_rate = drift_total / measure_ticks

    return normal_rate, drift_rate


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
    print("Training-regime sweep: population size x training duration")
    print("Additive metric only. N=15 paired trials per cell.")
    print("=" * 78)

    N_TRIALS = 15
    n_channels = 8
    pop_sizes = [24, 48, 96]
    durations = [300, 1500]

    baseline_diffs = None  # (24, 300) cell, for the paired comparison vs. every other cell

    results = {}
    for n_hidden in pop_sizes:
        for train_ticks in durations:
            additives = []
            zero_count = 0
            for seed in range(N_TRIALS):
                nr, dr = run_one(n_hidden, n_channels, train_ticks, seed)
                add = dr - nr
                additives.append(add)
                if nr == 0.0 and dr == 0.0:
                    zero_count += 1
            mean_add = sum(additives) / N_TRIALS
            results[(n_hidden, train_ticks)] = additives
            print(f"  n_hidden={n_hidden:3d}  train_ticks={train_ticks:5d}  "
                  f"mean additive change={mean_add:+.4f}  "
                  f"({zero_count}/{N_TRIALS} trials silent in both conditions)")

    baseline = results[(24, 300)]
    print(f"\nPaired comparisons against the documented baseline (n=24, train=300):")
    for (n_hidden, train_ticks), additives in results.items():
        if (n_hidden, train_ticks) == (24, 300):
            continue
        diffs = [a - b for a, b in zip(additives, baseline)]
        mean_diff, std_diff, t, p = paired_t_test(diffs)
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        direction = "larger" if mean_diff > 0 else "smaller"
        print(f"  n_hidden={n_hidden:3d} train={train_ticks:5d} vs baseline: "
              f"mean paired diff={mean_diff:+.4f}, t={t:.3f}, p={p:.4g} "
              f"({sig}, {direction} response than baseline)")
