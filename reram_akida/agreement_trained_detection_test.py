#!/usr/bin/env python
"""
agreement_trained_detection_test.py — direct, apples-to-apples comparison
of AgreementTrainedHiddenLayer against §9's real STDP-trained results,
using the EXACT same protocol: same N=50 combined seed pool (0-19 tuning
set + 100-129 held-out), same mean+3sigma threshold calibration, same
real/spurious/missed classification (a "detection" before drift_start
counts as spurious, not real, matching the correction made in §9 after
the original headline number was found to hide this).

Real, established §9 STDP numbers this compares against:
  train_ticks=300:  real_det=38/50 (mean_lat=365.7)  spurious=1/50 (2%)   missed=11/50
  train_ticks=2400: real_det=41/50 (mean_lat=152.2)  spurious=9/50 (18%)  missed=0/50
"""

import random

from stdp_trained_detection_test import build_reram_pair, calibrate_threshold
from agreement_trained_hidden_layer import AgreementTrainedHiddenLayer
from akida_style_drift_detector import make_signature, drifted_signature


def run_trial(seed, train_ticks, n_channels=8, n_hidden=24, drift_start=200,
              drift_ticks=400, total_ticks=2000, noise_std=0.05, window=30, debounce=3):
    rng = random.Random(seed + 77)
    reram_pos, reram_neg = build_reram_pair(n_hidden, n_channels, seed)
    normal_sig = make_signature(n_channels, seed=seed * 2 + 1)
    drift_sig = make_signature(n_channels, seed=seed * 2 + 2)
    hidden = AgreementTrainedHiddenLayer(n_channels, n_hidden, reram_pos, reram_neg, seed=seed)

    for _ in range(train_ticks):
        noisy = [v + rng.gauss(0, noise_std) for v in normal_sig]
        hidden.step(noisy, learn=True)

    threshold, base_mean, base_std = calibrate_threshold(hidden, normal_sig, rng, noise_std, window=window)

    window_buf, consec_over, detect_tick = [], 0, None
    for t in range(total_ticks):
        if t < drift_start:
            sig = normal_sig
        elif t < drift_start + drift_ticks:
            frac = (t - drift_start) / drift_ticks
            sig = drifted_signature(normal_sig, drift_sig, frac)
        else:
            sig = drift_sig
        noisy = [v + rng.gauss(0, noise_std) for v in sig]
        spikes = hidden.step(noisy, learn=False)
        window_buf.append(sum(spikes))
        if len(window_buf) > window:
            window_buf.pop(0)
        if len(window_buf) == window:
            rate = sum(window_buf) / window
            consec_over = consec_over + 1 if rate >= threshold else 0
            if consec_over >= debounce and detect_tick is None:
                detect_tick = t
    return detect_tick, drift_start


def run_false_alarm_control(seed, train_ticks, n_channels=8, n_hidden=24,
                             total_ticks=5000, noise_std=0.05, window=30, debounce=3):
    rng = random.Random(seed + 77)
    reram_pos, reram_neg = build_reram_pair(n_hidden, n_channels, seed)
    normal_sig = make_signature(n_channels, seed=seed * 2 + 1)
    hidden = AgreementTrainedHiddenLayer(n_channels, n_hidden, reram_pos, reram_neg, seed=seed)
    for _ in range(train_ticks):
        noisy = [v + rng.gauss(0, noise_std) for v in normal_sig]
        hidden.step(noisy, learn=True)
    threshold, base_mean, base_std = calibrate_threshold(hidden, normal_sig, rng, noise_std, window=window)

    window_buf, consec_over, alarms, armed = [], 0, 0, True
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
    return alarms, total_ticks


def classify(train_ticks, seeds):
    real_det, spurious, missed = [], [], 0
    for seed in seeds:
        det, drift_start = run_trial(seed, train_ticks)
        if det is None:
            missed += 1
        else:
            lat = det - drift_start
            (real_det if lat >= 0 else spurious).append(lat)
    alarms, ticks = run_false_alarm_control(seeds[0], train_ticks)
    return real_det, spurious, missed, alarms, ticks


if __name__ == "__main__":
    SEEDS = list(range(0, 20)) + list(range(100, 130))  # same N=50 pool as §9

    print("=" * 90)
    print("Agreement-trained (SADP-inspired) hidden layer vs. §9's real STDP results")
    print("Same N=50 seed pool, same threshold calibration, same real/spurious/missed split")
    print("=" * 90)

    for train_ticks in [300, 2400]:
        rd, sp, missed, alarms, ticks = classify(train_ticks, SEEDS)
        mean_lat = sum(rd) / len(rd) if rd else float("nan")
        print(f"\n  train_ticks={train_ticks}:")
        print(f"    real_det={len(rd)}/50 (mean_lat={mean_lat:6.1f})  "
              f"spurious={len(sp)}/50 ({100*len(sp)/50:4.0f}%)  missed={missed}/50  "
              f"false_alarms={alarms}/{ticks}")

    print("\n  Real, established §9 STDP numbers for direct comparison:")
    print("    train_ticks=300:  real_det=38/50 (mean_lat=365.7)  spurious=1/50 (2%)   missed=11/50")
    print("    train_ticks=2400: real_det=41/50 (mean_lat=152.2)  spurious=9/50 (18%)  missed=0/50")
    print("\n--- read the raw numbers above against the pre-registration; report honestly ---")
