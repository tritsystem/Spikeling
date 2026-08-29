#!/usr/bin/env python
"""
calibration_window_fix.py — chasing §9's newly-found regression directly:
at the winning STDP config (train_ticks=2400, window_ticks=3, tau=3.0),
18% of trials (9/50, held-out N=50 pool) trigger BEFORE drift_start even
begins -- a real cost hidden by the tuning-set-only headline number.

PRE-REGISTERED HYPOTHESIS (stated before running): the mean+3σ threshold
calibration window (currently window=30 ticks x n_windows=20 samples =
600 ticks total) underestimates a heavily-trained network's true
baseline variance -- more STDP exposure plausibly makes firing more
correlated/bursty rather than steadily noisy, so a short calibration
sample undersamples the tail. Widening n_windows (more independent
rolling-window samples, same per-window length) should lower the
spurious-pre-drift-trigger rate by producing a better-estimated,
appropriately-higher threshold, without materially hurting real
detection latency.
DISCONFIRM: spurious rate doesn't drop with more calibration samples ->
the mechanism isn't calibration-sample-count; something else (e.g. a
real drift in what "quiet" baseline activity even means after 2400
ticks of STDP) is responsible instead -- report that honestly, don't
force a fix that doesn't work.

Same real N=50 combined seed pool as §9 (0-19 tuning-set + 100-129
held-out) for a fair, non-cherry-picked before/after comparison. Reuses
build_reram_pair, calibrate_threshold, STDPTrainedHiddenLayer,
make_signature/drifted_signature directly -- only n_windows and k_sigma
are newly exposed as sweepable (calibrate_threshold already accepted
them as parameters; they just weren't varied yet).
"""

import random

from stdp_trained_hidden_layer import STDPTrainedHiddenLayer
from akida_style_drift_detector import make_signature, drifted_signature
from stdp_trained_detection_test import build_reram_pair, calibrate_threshold

TRAIN_TICKS, STDP_WINDOW_TICKS, STDP_TAU = 2400, 3, 3.0  # winning config from §9, held fixed


def run_trial(seed, n_windows, k_sigma, n_channels=8, n_hidden=24, drift_start=200,
              drift_ticks=400, total_ticks=2000, noise_std=0.05, window=30, debounce=3):
    rng = random.Random(seed + 77)
    reram_pos, reram_neg = build_reram_pair(n_hidden, n_channels, seed)
    normal_sig = make_signature(n_channels, seed=seed * 2 + 1)
    drift_sig = make_signature(n_channels, seed=seed * 2 + 2)
    hidden = STDPTrainedHiddenLayer(n_channels, n_hidden, reram_pos, reram_neg, seed=seed,
                                     window_ticks=STDP_WINDOW_TICKS, stdp_tau=STDP_TAU)

    for _ in range(TRAIN_TICKS):
        noisy = [v + rng.gauss(0, noise_std) for v in normal_sig]
        hidden.step(noisy, learn=True)

    threshold, base_mean, base_std = calibrate_threshold(
        hidden, normal_sig, rng, noise_std, window=window, n_windows=n_windows, k_sigma=k_sigma)

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


def run_false_alarm_control(seed, n_windows, k_sigma, n_channels=8, n_hidden=24,
                             total_ticks=5000, noise_std=0.05, window=30, debounce=3):
    rng = random.Random(seed + 77)
    reram_pos, reram_neg = build_reram_pair(n_hidden, n_channels, seed)
    normal_sig = make_signature(n_channels, seed=seed * 2 + 1)
    hidden = STDPTrainedHiddenLayer(n_channels, n_hidden, reram_pos, reram_neg, seed=seed,
                                     window_ticks=STDP_WINDOW_TICKS, stdp_tau=STDP_TAU)
    for _ in range(TRAIN_TICKS):
        noisy = [v + rng.gauss(0, noise_std) for v in normal_sig]
        hidden.step(noisy, learn=True)
    threshold, base_mean, base_std = calibrate_threshold(
        hidden, normal_sig, rng, noise_std, window=window, n_windows=n_windows, k_sigma=k_sigma)

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


def evaluate(n_windows, k_sigma, seeds, label=""):
    real_det, spurious, missed = [], [], 0
    for seed in seeds:
        det, drift_start = run_trial(seed, n_windows, k_sigma)
        if det is None:
            missed += 1
        else:
            lat = det - drift_start
            (real_det if lat >= 0 else spurious).append(lat)
    alarms, ticks = run_false_alarm_control(seeds[0], n_windows, k_sigma)
    mean_lat = sum(real_det) / len(real_det) if real_det else float("nan")
    print(f"  {label:<32} real_det={len(real_det)}/{len(seeds)} (mean_lat={mean_lat:6.1f})  "
          f"spurious={len(spurious)}/{len(seeds)} ({100*len(spurious)/len(seeds):4.0f}%)  "
          f"missed={missed}/{len(seeds)}  false_alarms={alarms}/{ticks}")
    return len(real_det), mean_lat, len(spurious), missed


if __name__ == "__main__":
    SEEDS = list(range(0, 20)) + list(range(100, 130))  # same N=50 pool as §9

    print("=" * 90)
    print("Stage A: widen n_windows (more calibration samples), k_sigma=3.0 fixed")
    print(f"Baseline from §9 (n_windows=20): real_det=41/50 (mean_lat=152.2), spurious=9/50 (18%), missed=0/50")
    print("=" * 90)
    for nw in [20, 40, 80, 160]:
        evaluate(nw, 3.0, SEEDS, label=f"n_windows={nw}")

    print("\n" + "=" * 90)
    print("Stage B: raise k_sigma instead (more conservative threshold), n_windows=20 fixed")
    print("=" * 90)
    for ks in [3.0, 3.5, 4.0, 5.0]:
        evaluate(20, ks, SEEDS, label=f"k_sigma={ks}")

    print("\n--- read the raw numbers above against the pre-registration; report honestly ---")
