#!/usr/bin/env python
"""
sync_mesh_repeated_trials.py — pushes on the real noise-floor problem
found in sync_mesh_test_two_devices.py: two runs after fixing the gain
mismatch gave gap=+0.124 then gap=-0.104 -- a sign flip, meaning a single
4s trial isn't a reliable measurement, not that the effect necessarily
isn't there.

TWO REAL CHANGES FROM THE PRIOR SCRIPT, both aimed at the noise floor
itself rather than re-running the same measurement and hoping:

1. LONGER per-trial recordings (10s, not 4s) -- less single-trial PLV
   variance from finite averaging.
2. EXPLICIT PER-TRIAL LAG CORRECTION: the earlier cross-correlation
   diagnostic searched for the best alignment lag but never actually
   APPLIED it before computing PLV -- it only used the (weak, 0.13) peak
   correlation to argue against a large fixed misalignment. This script
   finds the best lag via cross-correlation on each trial's own real
   data, then computes PLV both WITHOUT and WITH that lag correction --
   isolating how much of the trial-to-trial noise is a fixable alignment
   problem vs. genuine variability in the underlying signal.

N_TRIALS independent trials, reporting mean/std of the gap (real PLV
minus shuffled-control PLV) across trials for both aligned and unaligned
conditions, plus how many trials land positive vs negative -- the actual
statistical question two one-off runs can't answer.

PRE-REGISTERED PREDICTION: if the two-device effect is real but small,
lag-corrected PLV gaps should be MORE consistently positive across
trials than unaligned gaps (mean further from 0, fewer sign flips). If
lag correction doesn't change the sign-flipping pattern, the noise isn't
primarily an alignment problem -- it's either genuine measurement noise
around a true near-zero effect, or the effect isn't there at this
distance/hardware.

    python sync_mesh_repeated_trials.py [freq_hz] [n_trials]
"""
import math
import sys
import threading
import time

import numpy as np
import sounddevice as sd

sys.path.insert(0, "core")
from runtime.runtime import ResonatorState

DEVICE_C922 = 1
DEVICE_QUADCAST = 3
SR = 44100
DURATION_S = 10.0
FREQ_HZ = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
DAMPING = 0.02
COUPLING = 1.0
MAX_LAG_MS = 50.0
LAG_STEP_SAMPLES = 22


def _capture_stream(device, duration_s, samplerate):
    buf = []

    def callback(indata, frames, time_info, status):
        buf.append(indata[:, 0].copy())

    stream = sd.InputStream(device=device, samplerate=samplerate, channels=1,
                             dtype="float32", callback=callback)
    stream.start()
    time.sleep(duration_s)
    stream.stop()
    stream.close()
    return np.concatenate(buf) if buf else np.array([])


def capture_both(duration_s, samplerate):
    results = {}

    def worker(key, device):
        results[key] = _capture_stream(device, duration_s, samplerate)

    t1 = threading.Thread(target=worker, args=("c922", DEVICE_C922))
    t2 = threading.Thread(target=worker, args=("quadcast", DEVICE_QUADCAST))
    t1.start(); t2.start(); t1.join(); t2.join()
    a, b = results["c922"], results["quadcast"]
    n = min(len(a), len(b))
    return a[:n].astype(np.float64), b[:n].astype(np.float64)


def best_lag(a, b, samplerate):
    """Real normalized cross-correlation search over +-MAX_LAG_MS.
    Returns (lag_samples, abs_corr) at the best-aligning lag found."""
    max_lag = int(MAX_LAG_MS / 1000.0 * samplerate)
    a0, b0 = a - a.mean(), b - b.mean()
    best_c, best_l = -1.0, 0
    for lag in range(-max_lag, max_lag + 1, LAG_STEP_SAMPLES):
        if lag >= 0:
            x, y = a0[lag:], b0[:len(b0) - lag] if lag > 0 else b0
        else:
            x, y = a0[:len(a0) + lag], b0[-lag:]
        m = min(len(x), len(y))
        if m < 1000:
            continue
        x, y = x[:m], y[:m]
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        c = float(np.dot(x, y) / denom) if denom > 0 else 0.0
        if abs(c) > best_c:
            best_c, best_l = abs(c), lag
    return best_l, best_c


def apply_lag(a, b, lag):
    if lag >= 0:
        x, y = a[lag:], (b[:len(b) - lag] if lag > 0 else b)
    else:
        x, y = a[:len(a) + lag], b[-lag:]
    m = min(len(x), len(y))
    return x[:m], y[:m]


def run_resonator_pair(ch_a, ch_b, dt):
    ra = ResonatorState(name="A", freq_hz=FREQ_HZ, damping=DAMPING, coupling=COUPLING)
    rb = ResonatorState(name="B", freq_hz=FREQ_HZ, damping=DAMPING, coupling=COUPLING)
    omega = ra.omega
    thetas_a, thetas_b = [], []
    for a, b in zip(ch_a, ch_b):
        ra.step(drive=float(a), dt=dt)
        rb.step(drive=float(b), dt=dt)
        thetas_a.append(math.atan2(-ra.v / omega, ra.x))
        thetas_b.append(math.atan2(-rb.v / omega, rb.x))
    return np.array(thetas_a), np.array(thetas_b)


def plv(theta_a, theta_b):
    diff = theta_a - theta_b
    return float(np.abs(np.mean(np.exp(1j * diff))))


def main():
    print("=" * 70)
    print(f"  SYNC-MESH NOISE-FLOOR PUSH -- {N_TRIALS} trials, {DURATION_S}s each, "
          f"lag-corrected vs not")
    print("=" * 70)
    print(__doc__)

    dt = 1.0 / SR
    rows = []

    for trial in range(1, N_TRIALS + 1):
        print(f"\n--- trial {trial}/{N_TRIALS} ---")
        ch_a, ch_b = capture_both(DURATION_S, SR)
        lag, corr = best_lag(ch_a, ch_b, SR)
        print(f"  best lag: {lag} samples ({lag/SR*1000:.2f}ms), |corr| at that lag: {corr:.4f}")

        theta_a_raw, theta_b_raw = run_resonator_pair(ch_a, ch_b, dt)
        plv_unaligned = plv(theta_a_raw, theta_b_raw)

        a_al, b_al = apply_lag(ch_a, ch_b, lag)
        theta_a_al, theta_b_al = run_resonator_pair(a_al, b_al, dt)
        plv_aligned = plv(theta_a_al, theta_b_al)

        ch_b_shuffled = ch_b[::-1].copy()
        theta_a_ctrl, theta_b_ctrl = run_resonator_pair(ch_a, ch_b_shuffled, dt)
        plv_control = plv(theta_a_ctrl, theta_b_ctrl)

        gap_unaligned = plv_unaligned - plv_control
        gap_aligned = plv_aligned - plv_control
        print(f"  PLV unaligned={plv_unaligned:.4f}  aligned={plv_aligned:.4f}  "
              f"control={plv_control:.4f}")
        print(f"  gap unaligned={gap_unaligned:+.4f}  gap aligned={gap_aligned:+.4f}")

        rows.append({"trial": trial, "lag_ms": lag / SR * 1000, "corr": corr,
                     "plv_unaligned": plv_unaligned, "plv_aligned": plv_aligned,
                     "plv_control": plv_control, "gap_unaligned": gap_unaligned,
                     "gap_aligned": gap_aligned})

    gaps_u = np.array([r["gap_unaligned"] for r in rows])
    gaps_a = np.array([r["gap_aligned"] for r in rows])
    corrs = np.array([r["corr"] for r in rows])

    print("\n" + "=" * 70)
    print(f"VERDICT across {N_TRIALS} trials (measured, not assumed)")
    print("=" * 70)
    print(f"  unaligned gap: mean={gaps_u.mean():+.4f} std={gaps_u.std():.4f}  "
          f"positive={int((gaps_u>0).sum())}/{N_TRIALS}")
    print(f"  aligned gap:   mean={gaps_a.mean():+.4f} std={gaps_a.std():.4f}  "
          f"positive={int((gaps_a>0).sum())}/{N_TRIALS}")
    print(f"  mean best-lag |corr|: {corrs.mean():.4f}")

    if gaps_a.mean() > gaps_u.mean() + 0.02 and (gaps_a > 0).sum() > (gaps_u > 0).sum():
        print("\n  -> Lag correction meaningfully improves both the mean gap and sign "
              "consistency -- a real chunk of the noise WAS an alignment problem, fixable.")
    elif abs(gaps_u.mean()) < 0.05 and abs(gaps_a.mean()) < 0.05:
        print("\n  -> Both aligned and unaligned means sit close to zero across trials -- "
              "this is genuine noise around a near-zero effect, not an alignment artifact. "
              "Honest conclusion: no reliable sync-mesh signal detected at this real "
              "distance/hardware with this method.")
    else:
        print("\n  -> Mixed pattern -- report the numbers as measured, not the clean story "
              "either direction.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
