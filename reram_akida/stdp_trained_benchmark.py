#!/usr/bin/env python
"""
stdp_trained_benchmark.py — real before/after: fixed-random-weight
habituation detector (documented baseline: 0.44 -> 0.47, ~7% relative,
not statistically distinguishable from noise) vs. the same architecture
with real STDP training the ReRAM weights during exposure to the normal
signature.

PRE-REGISTERED HYPOTHESIS (stated before running): STDP-trained weights
will produce a larger, more reliable additive differential response to
drift than the fixed-random-weight baseline. Additive metric only, real
paired t-test, same discipline as every other test in this exploration.
"""

import math
import random

from reram_synapse_array import ReRAMSynapseArray
from stdp_trained_hidden_layer import STDPTrainedHiddenLayer
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


def run_stdp_trained(n_channels=8, n_hidden=24, train_ticks=300, seed=0,
                      noise_std=0.05, measure_ticks=300):
    rng = random.Random(seed + 77)
    reram_pos, reram_neg = build_reram_pair(n_hidden, n_channels, seed)
    normal_sig = make_signature(n_channels, seed=seed * 2 + 1)
    drift_sig = make_signature(n_channels, seed=seed * 2 + 2)
    hidden = STDPTrainedHiddenLayer(n_channels, n_hidden, reram_pos, reram_neg, seed=seed)

    recent = []
    for _ in range(train_ticks):
        noisy = [v + rng.gauss(0, noise_std) for v in normal_sig]
        spikes = hidden.step(noisy, learn=True)   # STDP live during training
        recent.append(sum(spikes))
    settle_window = min(100, max(20, train_ticks // 3))
    normal_rate = sum(recent[-settle_window:]) / settle_window

    drift_total = 0
    for _ in range(measure_ticks):
        noisy = [v + rng.gauss(0, noise_std) for v in drift_sig]
        spikes = hidden.step(noisy, learn=False)   # no learning during measurement
        drift_total += sum(spikes)
    drift_rate = drift_total / measure_ticks

    return normal_rate, drift_rate, hidden.stdp_updates_applied


def run_fixed_baseline(n_channels=8, n_hidden=24, train_ticks=300, seed=0,
                        noise_std=0.05, measure_ticks=300):
    """The documented baseline, unchanged: same HabituationHiddenLayer,
    fixed random weights, no STDP."""
    rng = random.Random(seed + 77)
    reram_pos, reram_neg = build_reram_pair(n_hidden, n_channels, seed)
    normal_sig = make_signature(n_channels, seed=seed * 2 + 1)
    drift_sig = make_signature(n_channels, seed=seed * 2 + 2)
    hidden = HabituationHiddenLayer(n_channels, n_hidden, reram_pos, reram_neg, seed=seed)

    recent = []
    for _ in range(train_ticks):
        noisy = [v + rng.gauss(0, noise_std) for v in normal_sig]
        spikes = hidden.step(noisy)
        recent.append(sum(spikes))
    settle_window = min(100, max(20, train_ticks // 3))
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
    print("Real STDP-trained weights vs. the fixed-random-weight baseline")
    print("Same signatures, same seeds, same noise -- only the weight-training")
    print("regime differs.")
    print("=" * 78)

    N_TRIALS = 20
    n_hidden, n_channels, train_ticks = 24, 8, 300

    stdp_additives, fixed_additives = [], []
    total_updates = []

    for seed in range(N_TRIALS):
        nr_s, dr_s, n_updates = run_stdp_trained(n_channels, n_hidden, train_ticks, seed)
        stdp_additives.append(dr_s - nr_s)
        total_updates.append(n_updates)

        nr_f, dr_f = run_fixed_baseline(n_channels, n_hidden, train_ticks, seed)
        fixed_additives.append(dr_f - nr_f)

        print(f"  seed={seed:2d}  STDP-trained: normal={nr_s:.3f} drift={dr_s:.3f} "
              f"additive={dr_s-nr_s:+.3f} ({n_updates} real weight updates applied)"
              f"   |   fixed baseline: normal={nr_f:.3f} drift={dr_f:.3f} additive={dr_f-nr_f:+.3f}")

    print(f"\n{N_TRIALS} real paired trials:")
    print(f"  STDP-trained:   mean additive change = {sum(stdp_additives)/N_TRIALS:+.4f}")
    print(f"  Fixed baseline: mean additive change = {sum(fixed_additives)/N_TRIALS:+.4f}")
    print(f"  mean real STDP updates applied per trial: {sum(total_updates)/N_TRIALS:.1f}")

    diffs = [s - f for s, f in zip(stdp_additives, fixed_additives)]
    mean_diff, std_diff, t, p = paired_t_test(diffs)
    print(f"\nPaired t-test (STDP-trained additive - fixed-baseline additive):")
    print(f"  mean paired difference = {mean_diff:+.4f} (std={std_diff:.4f})")
    print(f"  t = {t:.3f}, two-tailed p (normal approximation) = {p:.4g}")
    print(f"  {'SIGNIFICANT at p<0.05' if p < 0.05 else 'NOT significant at p<0.05'}"
          f" -- {'STDP training shows a real, larger drift response' if (p < 0.05 and mean_diff > 0) else ('fixed weights show a real, larger drift response' if (p < 0.05 and mean_diff < 0) else 'no real, resolvable difference at this sample size')}")
