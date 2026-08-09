#!/usr/bin/env python
"""
pyspike_reservoir_attention_acoustic_groundtruth.py -- closes the honest
gap flagged after the live dual-sensor test: that test had no
ground-truth target (no defined "correct answer" for a real clap), so
it could only check the confidence gate, not a real accuracy number.
This encodes a KNOWN sequence into real audio, plays/records it through
actual audio hardware, decodes it back, and compares the model's
prediction to the true target -- a real, checkable NMSE, not a smoke
test.

DIAGNOSED FIRST (2026-07-31): tried the true acoustic path (speaker ->
open air -> microphone) first. Confirmed via a Stereo-Mix loopback
control that the playback/recording/RMS-decoding CODE is correct (RMS
scales cleanly and monotonically with output amplitude through Stereo
Mix). But through open air, at safe/reasonable speaker volume, the
laptop's microphone picked up no measurable signal above the noise
floor (measured RMS flat at ~0.0017-0.0023 across output amplitudes
0.05-0.90) -- a real, diagnosed hardware/environment limitation (mic
gain/position/AGC on this specific machine), not a code bug. Using the
Stereo Mix loopback instead: still real, non-synthetic audio hardware
(actual Windows audio stack, DAC output stage, ADC quantization,
device-driver timing) even though it doesn't cross open air. Honest
label: "real audio hardware, not full acoustic" -- a genuine step up
from np.random, short of a true through-air sensor test.

SECOND BUG, DIAGNOSED AND FIXED (2026-07-31): the first full run decoded
garbage (mean |decode error| ~0.7-0.8 on a ~1.0-scale signal, real
hardware NMSE 6.99 -- far worse than guessing the mean). Measured with a
direct cross-correlation test (play a single click, record, correlate):
sd.playrec has a real ~222.6ms (3562-sample) output-to-input latency on
this device/driver combination -- almost two full TICK_DUR=0.12s ticks.
decode_rms_per_tick was slicing the recording assuming zero latency, so
every tick segment contained the WRONG audio. Fixed by measuring this
lag once per run (cross-correlating a calibration click) and shifting
the recording by that many samples before segmenting -- a systematic,
diagnosable timing bug, not a hardware or model failure.

Task: same shape as make_task in the hybrid script, but amplitude-only
(u in [0,1], not [-1,1]) because loudness/RMS is physically
non-negative -- a real constraint of encoding a signed synthetic task
onto a physical audio channel, not an arbitrary choice. One marker
(amplitude 2.5, clearly outside [0,1]) per T=40-tick sequence; target =
value K_LAG ticks after the marker.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sounddevice as sd

from pyspike_reservoir_attention_hybrid import (
    DEVICE, T, K_LAG, M_RESERVOIR, ReservoirBank, ReservoirAttentionReadout, run_stage,
)

SR = 16000
FREQ = 1000.0
TICK_DUR = 0.12          # seconds of tone per tick
REC_DEVICE = 2           # Stereo Mix (Realtek) -- real audio hardware loopback
PLAY_DEVICE = 4          # Speakers (Realtek)


def out_amp(u):
    """Map task amplitude u (in [0,1], marker=2.5) to a safe playback amplitude."""
    return 0.05 + 0.15 * u


def make_task_nonneg(rng, n_samples):
    u = rng.uniform(0.0, 1.0, size=(n_samples, T)).astype(np.float32)
    marker_pos = rng.integers(5, T - K_LAG - 1, size=n_samples)
    for i in range(n_samples):
        u[i, marker_pos[i]] = 2.5
    target = np.array([u[i, marker_pos[i] + K_LAG] for i in range(n_samples)], dtype=np.float32)
    return u, target, marker_pos


def make_tone_sequence(u_seq):
    segs = []
    for u in u_seq:
        n = int(TICK_DUR * SR)
        t = np.arange(n) / SR
        segs.append((out_amp(float(u)) * np.sin(2 * np.pi * FREQ * t)).astype(np.float32))
    return np.concatenate(segs)


def decode_rms_per_tick(rec):
    n = int(TICK_DUR * SR)
    rms = []
    for i in range(T):
        seg = rec[i * n:(i + 1) * n]
        rms.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))))
    return np.array(rms)


def measure_latency():
    """Cross-correlate a played click against what's recorded to find the
    real output-to-input sample delay -- confirmed non-negligible (~3562
    samples / 222.6ms) on this device/driver, large enough to badly
    misalign naive fixed-offset tick segmentation."""
    n_silence = int(0.3 * SR)
    n_click = int(0.05 * SR)
    t = np.arange(n_click) / SR
    click = (0.3 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
    audio = np.concatenate([np.zeros(n_silence, dtype=np.float32), click, np.zeros(n_silence, dtype=np.float32)])
    rec = sd.playrec(audio.reshape(-1, 1), samplerate=SR, channels=1, dtype="float32", device=(REC_DEVICE, PLAY_DEVICE))
    sd.wait()
    rec = rec.flatten()
    corr = np.correlate(rec, audio, mode="full")
    lag = int(np.argmax(np.abs(corr)) - (len(audio) - 1))
    return max(lag, 0)


def play_and_record(audio, lag_samples):
    """Pads the tail so the lag-shifted recording still covers the full
    played sequence, then shifts the recording back into alignment using
    the measured latency."""
    pad = np.zeros(lag_samples + SR // 2, dtype=np.float32)
    padded = np.concatenate([audio, pad])
    rec = sd.playrec(padded.reshape(-1, 1), samplerate=SR, channels=1, dtype="float32", device=(REC_DEVICE, PLAY_DEVICE))
    sd.wait()
    rec = rec.flatten()
    return rec[lag_samples:lag_samples + len(audio)]


if __name__ == "__main__":
    print("=" * 78)
    print("  GROUND-TRUTH TEST: real audio hardware, known encoded targets")
    print("=" * 78)

    rng = np.random.default_rng(0)
    train_u, train_y, _ = make_task_nonneg(rng, 800)
    test_u, test_y, _ = make_task_nonneg(rng, 200)
    reservoir = ReservoirBank(M_RESERVOIR).to(DEVICE)
    FEAT = 2 * M_RESERVOIR

    print("\nTraining on the amplitude-only [0,1] task variant (real constraint: RMS can't be negative)...")
    torch.manual_seed(0)
    model = ReservoirAttentionReadout(FEAT, use_ternary=False, use_spiking=False).to(DEVICE)
    run_stage("synthetic reference (nonneg task)", model, reservoir, train_u, train_y, test_u, test_y)
    model.eval()

    print("\nMeasuring play/record latency (cross-correlation on a known click)...")
    lag_samples = measure_latency()
    print(f"  measured latency: {lag_samples} samples = {lag_samples/SR*1000:.1f} ms")

    print("\nCalibrating: playing known amplitudes through Stereo Mix loopback...")
    cal_amps = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.5])
    measured = []
    for a in cal_amps:
        n = int(TICK_DUR * SR * 3)
        t = np.arange(n) / SR
        audio = (out_amp(a) * np.sin(2 * np.pi * FREQ * t)).astype(np.float32)
        rec = play_and_record(audio, lag_samples)
        measured.append(float(np.sqrt(np.mean(rec.astype(np.float64) ** 2))))
    measured = np.array(measured)
    A = np.vstack([cal_amps, np.ones_like(cal_amps)]).T
    slope, intercept = np.linalg.lstsq(A, measured, rcond=None)[0]
    print(f"  calibration fit: measured_rms = {slope:.5f}*true_amp + {intercept:.5f}")

    def decode_amp(rms):
        return (rms - intercept) / slope

    print("\nRunning real acoustic-hardware trials with KNOWN targets...")
    N_TRIALS = 15
    trial_rng = np.random.default_rng(42)
    u_trials, y_trials, mp_trials = make_task_nonneg(trial_rng, N_TRIALS)

    decoded_u_all = []
    for i in range(N_TRIALS):
        audio = make_tone_sequence(u_trials[i])
        rec = play_and_record(audio, lag_samples)
        rms = decode_rms_per_tick(rec)
        decoded_u = decode_amp(rms)
        decoded_u_all.append(decoded_u)
        print(f"  trial {i:2d}: true marker_pos={mp_trials[i]:2d}  true target={y_trials[i]:+.3f}  "
              f"decode err (mean |decoded-true|)={np.mean(np.abs(decoded_u - u_trials[i])):.4f}")

    decoded_u_all = np.stack(decoded_u_all)
    with torch.no_grad():
        states = reservoir(torch.tensor(decoded_u_all, device=DEVICE, dtype=torch.float32))
        pred, attn1 = model(states)
        found_pos = attn1.argmax(dim=-1).cpu().numpy()
        loc_acc = (found_pos == mp_trials).mean() * 100
        nmse = (F.mse_loss(pred, torch.tensor(y_trials, device=DEVICE)) / torch.tensor(y_trials, device=DEVICE).var()).item()

    print(f"\n{'='*78}")
    print(f"REAL, GROUND-TRUTH-VERIFIED RESULT (real audio hardware, known targets)")
    print(f"{'='*78}")
    print(f"  marker localization accuracy: {loc_acc:.1f}%  ({N_TRIALS} trials)")
    print(f"  NMSE (real hardware): {nmse:.4f}")
