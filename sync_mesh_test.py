#!/usr/bin/env python
"""
sync_mesh_test.py — first physical test of the "sync-mesh" invention: can
independent nodes detect anomalies via emergent physical synchronization
(desync as the signal) instead of comparing readings to a calibrated
baseline? That's the OPPOSITE primitive from everything else in this
hardware layer (BaselineDeviation compares a channel to ITS OWN history;
this compares a node to a PEER, with zero direct communication between
them -- only shared exposure to the same real physical environment).

ORIGIN: Tribe's NPC brains phase-locked (Kuramoto r 0.48->0.94) purely
from shared drum-audio feedback, with no direct communication between
brains. Never tested on real physical hardware before this script.

WHAT'S ACTUALLY TESTABLE TONIGHT, HONESTLY: no second physical location
exists yet (the only other mic input on this machine, the Realtek jack,
is confirmed dead -- nothing plugged in). So this tests the CORE
mechanism, not yet the full "different rooms" scenario: two independent
Spikeling Resonator neurons (core/runtime/runtime.py's ResonatorState,
already validated 99.2% for tone detection -- reused here, not
reinvented), driven by the C922's two real mic channels, with NO synapse
or coupling between them at all. If they show measurable phase
correlation purely from listening to the same real room, that's the
"consensus via shared environment, not data transmission" mechanism this
whole invention rests on.

METRIC: phase-locking value (PLV) -- the standard two-signal
synchronization measure (same mathematical family as Kuramoto's order
parameter, just the N=2 form): PLV = |mean_t(exp(i*(theta_A(t) -
theta_B(t))))|, 0 = no phase relationship, 1 = perfectly locked.

CONTROL, not just a raw number: the SAME two real channels, but with one
time-reversed relative to the other -- destroys their real temporal
correlation while preserving each channel's own individual amplitude
statistics. If real PLV isn't measurably higher than this control's PLV,
any apparent "sync" is just resonator math artifact, not a real physical
effect.

PRE-REGISTERED PREDICTION: PLV(real synchronized audio) > PLV(shuffled
control) by a meaningful margin. No claim yet about the exact magnitude
-- this is the first physical measurement of whether the mechanism
exists at all.

    python sync_mesh_test.py [freq_hz]
"""
import math
import sys

import numpy as np
import sounddevice as sd

sys.path.insert(0, "core")
from runtime.runtime import ResonatorState

MIC_DEVICE = 1
SR = 44100
FREQ_HZ = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
DAMPING = 0.02
COUPLING = 1.0


def run_resonator_pair(ch_a, ch_b, dt):
    """Two INDEPENDENT resonators (freq/damping identical, but no synapse,
    no shared state, no coupling between them at all) -- the only thing
    they share is being driven by ch_a / ch_b respectively."""
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
    print("  SYNC-MESH TEST -- real 2-node phase-locking via shared physical exposure")
    print("=" * 70)
    print(__doc__)
    print(f"Resonator frequency: {FREQ_HZ}Hz, damping={DAMPING}")

    print("\nRecording 8s of REAL 2-channel ambient audio...")
    audio = sd.rec(int(8.0 * SR), samplerate=SR, channels=2, device=MIC_DEVICE, dtype="float64")
    sd.wait()
    ch_a_real, ch_b_real = audio[:, 0], audio[:, 1]
    dt = 1.0 / SR

    print("Running two INDEPENDENT resonators (no coupling) on the REAL synchronized channels...")
    theta_a_real, theta_b_real = run_resonator_pair(ch_a_real, ch_b_real, dt)
    plv_real = plv(theta_a_real, theta_b_real)

    print("Running the SAME two resonators on a SHUFFLED control (ch_b time-reversed -- "
          "breaks real temporal correlation, keeps each channel's own amplitude statistics)...")
    ch_b_shuffled = ch_b_real[::-1].copy()
    theta_a_ctrl, theta_b_ctrl = run_resonator_pair(ch_a_real, ch_b_shuffled, dt)
    plv_control = plv(theta_a_ctrl, theta_b_ctrl)

    print("\n" + "=" * 70)
    print("VERDICT (measured, not assumed)")
    print("=" * 70)
    print(f"  PLV, real synchronized audio:      {plv_real:.4f}")
    print(f"  PLV, shuffled (decorrelated) control: {plv_control:.4f}")
    gap = plv_real - plv_control
    print(f"  gap: {gap:+.4f}")
    if gap > 0.15:
        print("  -> REAL SYNC MECHANISM CONFIRMED: two independent, non-coupled resonators "
              "show meaningfully higher phase correlation when driven by the real shared "
              "environment than by a decorrelated control. The core sync-mesh premise holds "
              "at the mechanism level.")
    elif gap > 0.03:
        print("  -> WEAK but real signal: some measurable difference, not dramatic. Worth "
              "more repeats before trusting the exact number.")
    else:
        print("  -> NOT CONFIRMED: no meaningful separation between real and shuffled "
              "conditions at this frequency -- either this frequency isn't where the real "
              "shared signal lives, or the mechanism doesn't produce measurable sync here.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
