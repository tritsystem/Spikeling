#!/usr/bin/env python
"""
stdp_lever_sweep.py — closing the real, honest gap left open in the
vault ledger: the STDP-trained detector (16/20 detected, mean latency
457.0, 1 false alarm/5000) still doesn't beat the naive amplitude
threshold (20/20, 255.8, 0). Three real, untried levers, tested as a
disciplined sequential sweep (not a full grid, to keep this runnable):
train_ticks, then STDP_WINDOW_TICKS, then stdp_tau -- each stage fixes
the winner of the previous stage rather than re-testing all combinations.

PRE-REGISTERED HYPOTHESIS (stated before running): more STDP training
ticks will let the weights converge further from their random
initialization, directly improving detection completeness/latency --
UNLIKE the earlier training_regime_sweep.py negative result, which
scaled POPULATION SIZE with FIXED (non-learning) weights and correctly
found no benefit (more copies of the same unstructured randomness).
This is a different lever: the weights themselves are now learning, so
more exposure time is a real, different mechanism, not the same one
already falsified in §7 of the ledger.
DISCONFIRM: no monotonic improvement with train_ticks, or no config
beats the current 16/20 baseline -> report the honest ceiling instead of
tuning until something looks better by chance (single run per config
would risk exactly that; N=20 seeds per config, same as the original
detection test, guards against it).

Reuses real, already-verified pieces directly: build_reram_pair,
calibrate_threshold from stdp_trained_detection_test.py;
STDPTrainedHiddenLayer from stdp_trained_hidden_layer.py (now with
window_ticks promoted to a real per-instance parameter, see that file's
2026-08-29 edit).
"""

import random

from reram_synapse_array import ReRAMSynapseArray
from stdp_trained_hidden_layer import STDPTrainedHiddenLayer
from akida_style_drift_detector import make_signature, drifted_signature
from stdp_trained_detection_test import build_reram_pair, calibrate_threshold


def run_trial(n_channels=8, n_hidden=24, drift_start=200, drift_ticks=400,
              total_ticks=2000, noise_std=0.05, seed=0, train_ticks=300,
              window=30, debounce=3, stdp_window_ticks=5, stdp_tau=3.0):
    rng = random.Random(seed + 77)
    reram_pos, reram_neg = build_reram_pair(n_hidden, n_channels, seed)
    normal_sig = make_signature(n_channels, seed=seed * 2 + 1)
    drift_sig = make_signature(n_channels, seed=seed * 2 + 2)
    hidden = STDPTrainedHiddenLayer(n_channels, n_hidden, reram_pos, reram_neg,
                                     seed=seed, window_ticks=stdp_window_ticks,
                                     stdp_tau=stdp_tau)

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


def run_false_alarm_control(n_channels=8, n_hidden=24, total_ticks=3000, noise_std=0.05,
                             seed=100, train_ticks=300, window=30, debounce=3,
                             stdp_window_ticks=5, stdp_tau=3.0):
    rng = random.Random(seed + 77)
    reram_pos, reram_neg = build_reram_pair(n_hidden, n_channels, seed)
    normal_sig = make_signature(n_channels, seed=seed * 2 + 1)
    hidden = STDPTrainedHiddenLayer(n_channels, n_hidden, reram_pos, reram_neg,
                                     seed=seed, window_ticks=stdp_window_ticks,
                                     stdp_tau=stdp_tau)

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
    return alarms, total_ticks


def _rank_key(result):
    """Maximize detected count first, then minimize mean latency among
    ties -- explicit tie-break, not an accident of dict ordering."""
    detected, mean_lat, _ = result
    return (detected, -mean_lat if mean_lat == mean_lat else float("-inf"))  # NaN-safe


def evaluate_config(train_ticks, stdp_window_ticks, stdp_tau, n_trials=20, label=""):
    latencies = []
    missed = 0
    for trial in range(n_trials):
        det, drift_start = run_trial(seed=trial, train_ticks=train_ticks,
                                      stdp_window_ticks=stdp_window_ticks, stdp_tau=stdp_tau)
        if det is not None:
            latencies.append(det - drift_start)
        else:
            missed += 1
    alarms, ticks = run_false_alarm_control(train_ticks=train_ticks,
                                             stdp_window_ticks=stdp_window_ticks, stdp_tau=stdp_tau)
    mean_lat = sum(latencies) / len(latencies) if latencies else float("nan")
    print(f"  {label:<38} detected={len(latencies)}/{n_trials}  "
          f"mean_latency={mean_lat:6.1f}  false_alarms={alarms}/{ticks}")
    return len(latencies), mean_lat, alarms


if __name__ == "__main__":
    print("=" * 78)
    print("Stage 1: train_ticks sweep (window_ticks=5, stdp_tau=3.0 fixed -- current defaults)")
    print("=" * 78)
    TRAIN_TICKS_OPTIONS = [300, 600, 1200, 2400]
    stage1 = {}
    for tt in TRAIN_TICKS_OPTIONS:
        stage1[tt] = evaluate_config(tt, 5, 3.0, label=f"train_ticks={tt}")
    best_train_ticks = max(stage1, key=lambda k: _rank_key(stage1[k]))
    print(f"  -> best train_ticks so far: {best_train_ticks} "
          f"({stage1[best_train_ticks][0]}/20 detected, mean latency {stage1[best_train_ticks][1]:.1f})")

    print("\n" + "=" * 78)
    print(f"Stage 2: STDP_WINDOW_TICKS sweep (train_ticks={best_train_ticks}, stdp_tau=3.0 fixed)")
    print("=" * 78)
    WINDOW_OPTIONS = [3, 5, 8, 12]
    stage2 = {}
    for wt in WINDOW_OPTIONS:
        stage2[wt] = evaluate_config(best_train_ticks, wt, 3.0, label=f"window_ticks={wt}")
    best_window = max(stage2, key=lambda k: _rank_key(stage2[k]))
    print(f"  -> best window_ticks so far: {best_window} "
          f"({stage2[best_window][0]}/20 detected, mean latency {stage2[best_window][1]:.1f})")

    print("\n" + "=" * 78)
    print(f"Stage 3: stdp_tau sweep (train_ticks={best_train_ticks}, window_ticks={best_window} fixed)")
    print("=" * 78)
    TAU_OPTIONS = [1.5, 3.0, 5.0, 8.0]
    stage3 = {}
    for tau in TAU_OPTIONS:
        stage3[tau] = evaluate_config(best_train_ticks, best_window, tau, label=f"stdp_tau={tau}")
    best_tau = max(stage3, key=lambda k: _rank_key(stage3[k]))

    print("\n" + "=" * 78)
    print("Best config found vs. the real established baselines:")
    print("=" * 78)
    best_detected, best_latency, best_alarms = stage3[best_tau]
    print(f"  Best STDP-trained config (train_ticks={best_train_ticks}, "
          f"window_ticks={best_window}, stdp_tau={best_tau}): "
          f"{best_detected}/20 detected, mean latency {best_latency:.1f}, "
          f"{best_alarms}/3000 false alarms")
    print(f"  Original STDP-trained defaults (300, 5, 3.0):      "
          f"{stage1[300][0]}/20 detected, mean latency {stage1[300][1]:.1f}")
    print(f"  Naive amplitude threshold (real established):       20/20 detected, mean latency 255.8, 0 false alarms/5000")
    print("\n--- read the raw numbers above against the pre-registration; report honestly ---")
