#!/usr/bin/env python
"""
tdoa_localization_test.py — tests whether the sync-mesh's two real,
independent mics (C922 + HyperX QuadCast) can COMPUTE something useful:
rough sound-source position, via time-difference-of-arrival (TDOA) --
the same cross-correlation lag-finding machinery built for
sync_mesh_repeated_trials.py, repurposed for localization instead of
phase-locking.

SIGN CONVENTION (derived from best_lag's own alignment logic, stated
explicitly so a result is interpretable, not just a number):
  lag > 0  ->  QuadCast's signal arrives EARLIER  -> sound is closer to QuadCast
  lag < 0  ->  C922's signal arrives EARLIER      -> sound is closer to C922

Converts |lag| to a real physical path-length difference via the speed
of sound (343 m/s) -- a genuine computed quantity, not just a raw sample
count.

THE ACTUAL TEST: run this once while making a sound near the C922, once
near the QuadCast. If TDOA is doing real computation (not noise), the
sign of the reported lag should FLIP between the two runs, matching
where the sound actually was.

    python tdoa_localization_test.py [duration_s] [label]
"""
import sys
import threading
import time

import numpy as np
import sounddevice as sd

DEVICE_C922 = 1
DEVICE_QUADCAST = 3
SR = 44100
DURATION_S = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
LABEL = sys.argv[2] if len(sys.argv) > 2 else "trial"
SPEED_OF_SOUND_MPS = 343.0
MAX_LAG_MS = 10.0   # real physical bound: mics a few feet apart -> max real TDOA is
                     # distance/speed_of_sound ~= 1.8m/343m/s ~= 5.3ms; 10ms gives margin
                     # without a search window so wide it locks onto noise peaks instead
                     # of the real lag (a 48ms "best lag" -- physically impossible for this
                     # geometry -- is exactly what a too-wide window produced before this fix)
LAG_STEP_SAMPLES = 4
COUNTDOWN_S = 3


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
    """Full-window cross-correlation -- kept for reference/comparison, but
    NOT what localizes a single transient event: a 4s window is dominated
    by continuous ambient background (same effect that drove the
    sync-mesh PLV results), which can easily swamp one brief clap. This is
    why trial 2 failed to flip sign -- it was likely measuring ambient
    correlation, not the deliberate sound at all."""
    max_lag = int(MAX_LAG_MS / 1000.0 * samplerate)
    a0, b0 = a - a.mean(), b - b.mean()
    best_c, best_l = -1.0, 0
    for lag in range(-max_lag, max_lag + 1, LAG_STEP_SAMPLES):
        if lag >= 0:
            x, y = a0[lag:], (b0[:len(b0) - lag] if lag > 0 else b0)
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


def transient_event_lag(a, b, samplerate, window_ms=2.0, max_lag_ms=MAX_LAG_MS):
    """Finds the actual loud transient (a clap/word) via a short-term
    energy envelope -- but NOT independently in each channel (a first
    attempt at that found a 1.76-SECOND, 603-meter "lag", physically
    impossible for this geometry: the two channels' independent peaks
    were almost certainly two DIFFERENT real acoustic events, not the
    same clap seen twice). Fixed: anchor on whichever channel has the
    stronger overall peak (the louder capture, plausibly the closer
    mic), then search the OTHER channel's envelope ONLY within
    +-max_lag_ms of that same real moment -- guaranteeing both peaks
    refer to the same physical event instead of two unrelated ones."""
    win = max(1, int(window_ms / 1000.0 * samplerate))

    def envelope(x):
        n_win = len(x) // win
        trimmed = x[:n_win * win].reshape(n_win, win)
        return np.sum(trimmed ** 2, axis=1)

    env_a, env_b = envelope(a), envelope(b)
    a_is_anchor = env_a.max() >= env_b.max()
    anchor_env, anchor_raw = (env_a, a) if a_is_anchor else (env_b, b)
    other_raw = b if a_is_anchor else a

    anchor_peak_win = int(np.argmax(anchor_env))
    anchor_peak_sample = anchor_peak_win * win
    anchor_energy = float(anchor_env[anchor_peak_win])

    max_lag_samples = int(max_lag_ms / 1000.0 * samplerate)
    lo = max(0, anchor_peak_sample - max_lag_samples)
    hi = min(len(other_raw), anchor_peak_sample + max_lag_samples)
    segment = other_raw[lo:hi]
    if len(segment) < win:
        return 0, anchor_energy, 0.0

    n_seg_win = len(segment) // win
    seg_energy = np.sum(segment[:n_seg_win * win].reshape(n_seg_win, win) ** 2, axis=1)
    other_peak_local_win = int(np.argmax(seg_energy))
    other_peak_sample = lo + other_peak_local_win * win
    other_energy = float(seg_energy[other_peak_local_win])

    if a_is_anchor:
        lag = anchor_peak_sample - other_peak_sample     # positive: A later -> B led
        return lag, anchor_energy, other_energy
    else:
        lag = other_peak_sample - anchor_peak_sample     # positive: A later -> B led
        return lag, other_energy, anchor_energy


def main():
    print("=" * 70)
    print(f"  TDOA LOCALIZATION TEST -- {LABEL}")
    print("=" * 70)
    print(f"Recording starts in {COUNTDOWN_S}s. Make your sound (clap / say a word) "
          f"as close to the intended mic as you can, right when recording starts.")
    for i in range(COUNTDOWN_S, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1.0)
    print(f"  GO -- recording {DURATION_S}s now!", flush=True)

    ch_a, ch_b = capture_both(DURATION_S, SR)
    lag, peak_a_e, peak_b_e = transient_event_lag(ch_a, ch_b, SR)
    lag_ms = lag / SR * 1000.0
    dist_diff_cm = SPEED_OF_SOUND_MPS * abs(lag) / SR * 100.0

    print("\n" + "=" * 70)
    print("RESULT (measured, not assumed -- transient-event peak, not full-window correlation)")
    print("=" * 70)
    print(f"  best lag: {lag} samples ({lag_ms:+.2f}ms)  "
          f"peak energy: C922={peak_a_e:.6f} QuadCast={peak_b_e:.6f}")
    if lag > 0:
        print(f"  -> sound arrived at QuadCast ~{lag_ms:.2f}ms EARLIER than C922")
        print(f"  -> implied: source is closer to QuadCast by ~{dist_diff_cm:.1f}cm of path length")
    elif lag < 0:
        print(f"  -> sound arrived at C922 ~{abs(lag_ms):.2f}ms EARLIER than QuadCast")
        print(f"  -> implied: source is closer to C922 by ~{dist_diff_cm:.1f}cm of path length")
    else:
        print("  -> no measurable lag found (arrived at both simultaneously, or too quiet/no event)")
    print(f"  (label: {LABEL})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
