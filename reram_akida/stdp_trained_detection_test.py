#!/usr/bin/env python
"""
stdp_trained_detection_test.py — the metric that actually matters:
real detection latency and false-alarm rate, not just differential
population-rate signal, using the STDP-trained hidden layer.

Threshold calibration lesson carried over from the habituation-only
detector's real struggle earlier in this exploration (arbitrary
multipliers produced either 0% detection or 40 false alarms/5000 ticks):
this time the alarm threshold is set from REAL MEASURED baseline
variance (mean + 3 real standard deviations over a held-out quiet
window, sampled AFTER training but BEFORE any drift), not a guessed
multiplier -- a real, standard, defensible statistical convention.

Compared directly against the already-established real naive-threshold
baseline from this exploration's first pass: 20/20 detected, mean
latency 255.8 ticks, 0 false alarms / 5000 ticks.
"""

import random

from reram_synapse_array import ReRAMSynapseArray
from stdp_trained_hidden_layer import STDPTrainedHiddenLayer
from akida_style_drift_detector import make_signature, drifted_signature


def build_reram_pair(n_hidden, n_channels, seed):
    rng = random.Random(seed)
    reram_pos = ReRAMSynapseArray(n_hidden, n_channels, seed=seed)
    reram_neg = ReRAMSynapseArray(n_hidden, n_channels, seed=seed + 5000)
    for h in range(n_hidden):
        for c in range(n_channels):
            reram_pos.program(h, c, rng.uniform(0.0, 1.0), mode="iterative")
            reram_neg.program(h, c, rng.uniform(0.0, 1.0), mode="iterative")
    return reram_pos, reram_neg


def calibrate_threshold(hidden, normal_sig, rng, noise_std, window=30, n_windows=20, k_sigma=3.0):
    """Real measurement, not a guess: run a held-out quiet window
    (post-training, pre-drift, no learning) and compute mean + k*std of
    the windowed population rate. Returns (threshold, mean, std)."""
    window_buf = []
    rates = []
    for _ in range(window * n_windows):
        noisy = [v + rng.gauss(0, noise_std) for v in normal_sig]
        spikes = hidden.step(noisy, learn=False)
        window_buf.append(sum(spikes))
        if len(window_buf) > window:
            window_buf.pop(0)
        if len(window_buf) == window:
            rates.append(sum(window_buf) / window)
    mean = sum(rates) / len(rates)
    var = sum((r - mean) ** 2 for r in rates) / max(1, len(rates) - 1)
    std = var ** 0.5
    return mean + k_sigma * std, mean, std


def run_trial(n_channels=8, n_hidden=24, drift_start=200, drift_ticks=400,
              total_ticks=2000, noise_std=0.05, seed=0, train_ticks=300,
              window=30, debounce=3):
    rng = random.Random(seed + 77)
    reram_pos, reram_neg = build_reram_pair(n_hidden, n_channels, seed)
    normal_sig = make_signature(n_channels, seed=seed * 2 + 1)
    drift_sig = make_signature(n_channels, seed=seed * 2 + 2)
    hidden = STDPTrainedHiddenLayer(n_channels, n_hidden, reram_pos, reram_neg, seed=seed)

    for _ in range(train_ticks):
        noisy = [v + rng.gauss(0, noise_std) for v in normal_sig]
        hidden.step(noisy, learn=True)

    threshold, base_mean, base_std = calibrate_threshold(hidden, normal_sig, rng, noise_std, window=window)

    window_buf = []
    consec_over = 0
    detect_tick = None
    for t in range(total_ticks):
        if t < drift_start:
            sig = normal_sig
        elif t < drift_start + drift_ticks:
            frac = (t - drift_start) / drift_ticks
            sig = drifted_signature(normal_sig, drift_sig, frac)
        else:
            sig = drift_sig
        noisy = [v + rng.gauss(0, noise_std) for v in sig]
        spikes = hidden.step(noisy, learn=False)   # learning off during the actual test window
        window_buf.append(sum(spikes))
        if len(window_buf) > window:
            window_buf.pop(0)
        if len(window_buf) == window:
            rate = sum(window_buf) / window
            consec_over = consec_over + 1 if rate >= threshold else 0
            if consec_over >= debounce and detect_tick is None:
                detect_tick = t

    return detect_tick, drift_start, threshold, base_mean, base_std


def run_false_alarm_control(n_channels=8, n_hidden=24, total_ticks=5000, noise_std=0.05,
                             seed=100, train_ticks=300, window=30, debounce=3):
    rng = random.Random(seed + 77)
    reram_pos, reram_neg = build_reram_pair(n_hidden, n_channels, seed)
    normal_sig = make_signature(n_channels, seed=seed * 2 + 1)
    hidden = STDPTrainedHiddenLayer(n_channels, n_hidden, reram_pos, reram_neg, seed=seed)

    for _ in range(train_ticks):
        noisy = [v + rng.gauss(0, noise_std) for v in normal_sig]
        hidden.step(noisy, learn=True)

    threshold, base_mean, base_std = calibrate_threshold(hidden, normal_sig, rng, noise_std, window=window)

    window_buf = []
    consec_over = 0
    alarms = 0
    armed = True
    for _ in range(total_ticks):
        noisy = [v + rng.gauss(0, noise_std) for v in normal_sig]
        spikes = hidden.step(noisy, learn=False)
        window_buf.append(sum(spikes))
        if len(window_buf) > window:
            window_buf.pop(0)
        if len(window_buf) == window:
            rate = sum(window_buf) / window
            consec_over = consec_over + 1 if rate >= threshold else 0
            if consec_over >= debounce and armed:
                alarms += 1
                armed = False
            if consec_over == 0:
                armed = True
    return alarms, total_ticks, threshold


if __name__ == "__main__":
    print("=" * 78)
    print("STDP-trained detector -- real end-to-end detection test")
    print("Threshold = mean + 3*sigma of a real, held-out, post-training quiet window")
    print("=" * 78)

    N_TRIALS = 20
    latencies = []
    missed = 0
    thresholds = []

    for trial in range(N_TRIALS):
        det, drift_start, threshold, base_mean, base_std = run_trial(seed=trial)
        thresholds.append(threshold)
        if det is not None:
            latencies.append(det - drift_start)
        else:
            missed += 1
        print(f"  seed={trial:2d}  threshold={threshold:.3f} (base_mean={base_mean:.3f}, "
              f"base_std={base_std:.3f})  detect_tick={det}")

    print(f"\n{N_TRIALS} real trials:")
    print(f"  STDP-trained detector: {len(latencies)}/{N_TRIALS} detected, missed={missed}")
    if latencies:
        print(f"    mean detection latency: {sum(latencies)/len(latencies):.1f} ticks"
              f"  (min={min(latencies)}, max={max(latencies)})")

    print("\nFalse-alarm control (pure normal signal, 5000 ticks, no drift):")
    alarms, ticks, thr = run_false_alarm_control()
    print(f"  STDP-trained detector: {alarms} false alarms / {ticks} ticks (threshold={thr:.3f})")

    print("\nFor direct comparison, the real established naive-threshold baseline (same task):")
    print("  Naive amplitude-threshold: 20/20 detected, mean latency 255.8 ticks, 0 false alarms / 5000")
