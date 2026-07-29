#!/usr/bin/env python
"""
noise_cancelling_frontend.py — a real physical test of an unused finding
from the topological-phononics reservoir-computing work: a linear readout
can PERFECTLY null a low-rank/correlated interferer at the observation
stage (any amplitude, per noise_feature.py's simulated result), while an
uncorrelated signal in a different subspace survives. That result has
only ever existed in simulation. This is the first attempt to build it as
a real, physical signal-conditioning front-end.

HARDWARE: the Logitech C922 webcam mic reports as a real 2-channel
(stereo) device -- two physical capsules ~2cm apart. That's the minimum
needed for the mechanism at all: a linear combination of >=2 channels has
a null space to project an unwanted correlated component into. A single
mono mic could never test this.

PRE-REGISTERED PREDICTION (stated before running):
  1. A projection direction learned from real ambient-only calibration
     (the eigenvector of the SMALLER eigenvalue of the 2-channel
     calibration covariance) will show LOWER low-frequency ambient energy
     than either raw channel alone, when applied to a fresh ambient
     segment not used for calibration.
  2. A real 1kHz test tone, played through the speakers and captured via
     the SAME mic (so it's genuinely airborne, not injected in software),
     will still be clearly detectable in the projected signal even while
     ambient energy is suppressed -- demonstrating SELECTIVE cancellation,
     not generic attenuation.

HONEST CAVEAT, stated up front: the two mic capsules are only ~2cm apart.
For a source at normal speaker distance (~0.5-1m), the inter-capsule
level/phase difference is real but small (wavelength >> spacing for most
audible frequencies), so this measures WHETHER real selective
cancellation happens, not how dramatic it will be. A weak effect is an
honest possible outcome, not a failure of the script.

    python noise_cancelling_frontend.py
"""
import sys
import time

import numpy as np
import sounddevice as sd

MIC_DEVICE = 1        # C922 Pro Stream Webcam -- real 2-channel capsule pair
SPEAKER_DEVICE = 4     # Speakers (Realtek(R) Audio)
SR = 44100
TONE_HZ = float(sys.argv[1]) if len(sys.argv) > 1 else 1000.0
# Real measured fact (mic_coherence_diagnostic.py + a direct spectral check):
# calibration ambient power is 84% below 200Hz, 97% below 1kHz, ~0.00% above
# 8kHz. The original 1000Hz run tested selectivity using a tone that overlaps
# almost entirely with where the interferer's own power lives -- a real
# confound in the experiment, not just the 2cm spacing. Passing a frequency
# far outside that band (e.g. 5000Hz) tests selectivity without that confound.


def record(seconds, label):
    print(f"  recording {seconds:.1f}s of REAL audio ({label})...", flush=True)
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=2, device=MIC_DEVICE, dtype="float64")
    sd.wait()
    return audio   # shape (N, 2)


def band_energy(x, sr, lo_hz, hi_hz):
    """Real FFT magnitude energy in [lo_hz, hi_hz) -- a real, computed
    quantity from the actual captured waveform, not estimated."""
    n = len(x)
    if n == 0:
        return 0.0
    mags = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= lo_hz) & (freqs < hi_hz)
    return float(np.mean(mags[mask])) if mask.any() else 0.0


def main():
    print("=" * 70)
    print("  NOISE-CANCELLING FRONT-END -- real 2-mic linear null-space test")
    print("=" * 70)
    print(__doc__)

    print("\nSTEP 1: calibrating on REAL ambient audio (5s) -- stay as quiet as you "
          "normally would, real room noise (fans, hum, etc.) IS the interferer.")
    calib = record(5.0, "calibration / ambient")

    cov = np.cov(calib.T)   # real 2x2 covariance from actual captured samples
    eigvals, eigvecs = np.linalg.eigh(cov)   # ascending eigenvalues
    signal_dir = eigvecs[:, 0]    # smaller eigenvalue -- the null-projection direction
    noise_dir = eigvecs[:, 1]     # larger eigenvalue -- the dominant correlated component
    print(f"\n  real calibration covariance eigenvalues: {eigvals[0]:.6e} (null dir), "
          f"{eigvals[1]:.6e} (noise dir)")
    print(f"  rank concentration (noise/(noise+signal) eigenvalue share): "
          f"{eigvals[1] / eigvals.sum():.1%}")
    print(f"  signal (null) direction: {signal_dir}")
    print(f"  noise direction:         {noise_dir}")

    print("\nSTEP 2: fresh ambient segment (3s, NOT used for calibration) -- "
          "measuring real suppression on unseen data.")
    fresh_ambient = record(3.0, "fresh ambient, held out from calibration")
    raw0_amb = band_energy(fresh_ambient[:, 0], SR, 20, 500)
    raw1_amb = band_energy(fresh_ambient[:, 1], SR, 20, 500)
    projected_amb = fresh_ambient @ signal_dir
    proj_amb_energy = band_energy(projected_amb, SR, 20, 500)
    print(f"  low-band (20-500Hz) energy: raw ch0={raw0_amb:.5f}  raw ch1={raw1_amb:.5f}  "
          f"projected={proj_amb_energy:.5f}")

    print(f"\nSTEP 3: playing a REAL {TONE_HZ:.0f}Hz tone through the speakers while "
          f"recording via the same mic (genuinely airborne, not injected in software).")
    t = np.linspace(0, 2.5, int(2.5 * SR), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * TONE_HZ * t)
    sd.play(tone, samplerate=SR, device=SPEAKER_DEVICE)
    time.sleep(0.3)   # let playback actually start before recording begins
    tone_capture = record(2.0, "tone + real ambient, concurrent")
    sd.wait()

    raw0_tone = band_energy(tone_capture[:, 0], SR, TONE_HZ - 50, TONE_HZ + 50)
    raw1_tone = band_energy(tone_capture[:, 1], SR, TONE_HZ - 50, TONE_HZ + 50)
    projected_tone_signal = tone_capture @ signal_dir
    proj_tone_energy = band_energy(projected_tone_signal, SR, TONE_HZ - 50, TONE_HZ + 50)
    proj_tone_lowband = band_energy(projected_tone_signal, SR, 20, 500)

    print(f"  {TONE_HZ:.0f}Hz-band energy during real tone: raw ch0={raw0_tone:.5f}  "
          f"raw ch1={raw1_tone:.5f}  projected={proj_tone_energy:.5f}")
    print(f"  projected low-band (20-500Hz) energy DURING tone: {proj_tone_lowband:.5f} "
          f"(compare to ambient-only projected: {proj_amb_energy:.5f})")

    print("\n" + "=" * 70)
    print("VERDICT (measured, not assumed -- FIXED baseline = channel 0 throughout,")
    print("not whichever channel makes the number look better)")
    print("=" * 70)
    amb_suppression_ch0 = 1.0 - (proj_amb_energy / max(raw0_amb, 1e-12))
    tone_suppression_ch0 = 1.0 - (proj_tone_energy / max(raw0_tone, 1e-12))
    print(f"  ambient low-band suppression vs raw ch0: {amb_suppression_ch0:+.1%}")
    print(f"  tone-band suppression vs raw ch0:        {tone_suppression_ch0:+.1%}")
    gap = amb_suppression_ch0 - tone_suppression_ch0
    print(f"  SELECTIVITY GAP (ambient suppression minus tone suppression): {gap:+.1%}")
    if gap > 0.15:
        print("  -> SELECTIVE: the calibrated interferer is suppressed meaningfully more "
              "than the novel tone -- real evidence the projection targets the calibrated "
              "noise specifically, not just anything correlated.")
    elif gap < -0.05:
        print("  -> INVERTED: the tone was suppressed MORE than the calibrated ambient noise "
              "-- real, unexpected, worth investigating further, not the predicted direction.")
    else:
        print("  -> NOT SELECTIVE: ambient and tone were suppressed by nearly the same amount "
              "-- this is generic attenuation of anything correlated across the pair, not "
              "selective cancellation of the calibrated interferer specifically.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
