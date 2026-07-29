#!/usr/bin/env python
"""
sync_mesh_test_two_devices.py — the real test sync_mesh_test.py couldn't
do: two GENUINELY SEPARATE physical microphones (C922 webcam mic +
HyperX QuadCast S, a completely different USB device, confirmed
producing real non-zero signal before this script was written) instead
of two channels on the same device 2cm apart.

WHY THIS IS HARDER, HONESTLY: the two mics have INDEPENDENT clocks --
no shared word-clock between two separate USB audio devices. sounddevice's
convenience sd.rec()/sd.wait() manage a single global stream and can't
run two concurrent recordings, so this uses two explicit InputStream
objects in separate threads, started as close together as threading
allows. Real clock drift between two independent consumer audio devices
is typically tens-to-hundreds of ppm; over an 8s window that's a
sub-millisecond-to-a-few-ms accumulated offset -- small relative to a
100Hz resonator's ~10ms period, but not zero, and unlike the single-
device test this is a REAL confound that could suppress measured PLV
even if genuine acoustic-driven synchronization exists. Kept the
recording SHORT (4s, not 8s) specifically to limit how much drift can
accumulate.

PRE-REGISTERED PREDICTION: PLV(real, both mics live) > PLV(shuffled
control) by a meaningful margin, same as the single-device result (two
independent replications: 0.66 vs 0.20, then 0.75 vs 0.05) -- but a
SMALLER gap than the single-device case would be an honest, expected
outcome here (real spatial distance + no shared clock), not evidence the
mechanism failed.

    python sync_mesh_test_two_devices.py [freq_hz]
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
DURATION_S = 4.0
FREQ_HZ = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
DAMPING = 0.02
COUPLING = 1.0


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
    """Genuinely concurrent capture from two independent devices, via two
    threads started together -- as close to simultaneous as Python
    threading allows, which is the real, honest limit here."""
    results = {}

    def worker(key, device):
        results[key] = _capture_stream(device, duration_s, samplerate)

    t1 = threading.Thread(target=worker, args=("c922", DEVICE_C922))
    t2 = threading.Thread(target=worker, args=("quadcast", DEVICE_QUADCAST))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    return results["c922"], results["quadcast"]


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
    print("  SYNC-MESH TEST -- TWO SEPARATE PHYSICAL MICS (C922 + HyperX QuadCast)")
    print("=" * 70)
    print(__doc__)
    print(f"Resonator frequency: {FREQ_HZ}Hz, damping={DAMPING}")

    print(f"\nRecording {DURATION_S}s concurrently from BOTH real, independent mics...")
    ch_a, ch_b = capture_both(DURATION_S, SR)
    n = min(len(ch_a), len(ch_b))
    print(f"  c922 samples: {len(ch_a)}  quadcast samples: {len(ch_b)}  using common length: {n}")
    ch_a, ch_b = ch_a[:n], ch_b[:n]
    dt = 1.0 / SR

    print("Running two INDEPENDENT resonators (no coupling) on the REAL two-mic recording...")
    theta_a_real, theta_b_real = run_resonator_pair(ch_a, ch_b, dt)
    plv_real = plv(theta_a_real, theta_b_real)

    print("Running the SAME two resonators on a SHUFFLED control (mic B time-reversed)...")
    ch_b_shuffled = ch_b[::-1].copy()
    theta_a_ctrl, theta_b_ctrl = run_resonator_pair(ch_a, ch_b_shuffled, dt)
    plv_control = plv(theta_a_ctrl, theta_b_ctrl)

    print("\n" + "=" * 70)
    print("VERDICT (measured, not assumed)")
    print("=" * 70)
    print(f"  PLV, real two-mic recording:          {plv_real:.4f}")
    print(f"  PLV, shuffled (decorrelated) control: {plv_control:.4f}")
    gap = plv_real - plv_control
    print(f"  gap: {gap:+.4f}")
    print(f"\n  (single-device replications for comparison: 0.66 vs 0.20 [gap +0.46], "
          f"then 0.75 vs 0.05 [gap +0.71])")
    if gap > 0.15:
        print("  -> SYNC MECHANISM HOLDS ACROSS TWO SEPARATE PHYSICAL DEVICES, not just two "
              "channels on one device -- meaningful evidence the effect isn't an artifact of "
              "sharing a clock/hardware.")
    elif gap > 0.03:
        print("  -> WEAK but real signal across two separate devices -- smaller than the "
              "single-device case, plausibly consistent with real distance/decorrelation "
              "and clock drift, not necessarily a failure of the mechanism.")
    else:
        print("  -> NOT CONFIRMED across two separate devices at this frequency/distance -- "
              "a real, honest negative result worth reporting as such, not reframed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
