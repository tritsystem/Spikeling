#!/usr/bin/env python
"""
mic_coherence_diagnostic.py — diagnoses WHY noise_cancelling_frontend.py's
selectivity was weak, via a real measurement instead of a guess.

The projection can only tell "calibrated noise direction" apart from "a
different source's direction" at frequencies where the two mic channels
carry genuinely different information. Magnitude-squared coherence
Cxy(f) = |Pxy(f)|^2 / (Pxx(f)*Pyy(f)) measures exactly that, per
frequency, from real captured audio: 1.0 = channel 1 is fully predictable
from channel 0 at that frequency (redundant, wavelength >> mic spacing --
nulling can't be selective there), lower values = real independent
information (spatial diversity a projection could actually exploit).

PRE-REGISTERED PREDICTION: if the "2cm spacing << most audible
wavelengths" explanation for the weak selectivity is correct, coherence
should be very close to 1.0 across most of the audible band and only
start dropping meaningfully at HIGH frequencies (where wavelength
approaches the ~2cm capsule spacing, roughly >8kHz). If coherence is
uniformly high everywhere (never drops), spacing isn't the limiting
factor -- something else (mic self-noise floor, gain mismatch) is. If
coherence is low/noisy everywhere including low frequencies, that
contradicts the spacing story entirely.

    python mic_coherence_diagnostic.py
"""
import numpy as np
import sounddevice as sd
from scipy.signal import coherence

MIC_DEVICE = 1
SR = 44100


def main():
    print("=" * 70)
    print("  REAL MIC-PAIR COHERENCE DIAGNOSTIC")
    print("=" * 70)
    print(__doc__)

    print("\nRecording 8s of real ambient audio for a stable coherence estimate...")
    audio = sd.rec(int(8.0 * SR), samplerate=SR, channels=2, device=MIC_DEVICE, dtype="float64")
    sd.wait()
    ch0, ch1 = audio[:, 0], audio[:, 1]

    f, Cxy = coherence(ch0, ch1, fs=SR, nperseg=4096)

    bands = [(20, 200), (200, 1000), (1000, 3000), (3000, 8000), (8000, 16000), (16000, 20000)]
    print("\nReal measured coherence by band (1.0 = fully redundant, lower = real spatial diversity):")
    band_results = []
    for lo, hi in bands:
        mask = (f >= lo) & (f < hi)
        if not mask.any():
            continue
        mean_c = float(np.mean(Cxy[mask]))
        band_results.append((lo, hi, mean_c))
        bar = "#" * int(mean_c * 40)
        print(f"  {lo:6d}-{hi:6d} Hz: coherence={mean_c:.4f}  {bar}")

    print("\n" + "=" * 70)
    print("VERDICT (measured, not assumed)")
    print("=" * 70)
    low_band = [c for lo, hi, c in band_results if hi <= 1000]
    high_band = [c for lo, hi, c in band_results if lo >= 8000]
    low_avg = float(np.mean(low_band)) if low_band else float("nan")
    high_avg = float(np.mean(high_band)) if high_band else float("nan")
    print(f"  low-band (<1kHz) avg coherence:  {low_avg:.4f}")
    print(f"  high-band (>=8kHz) avg coherence: {high_avg:.4f}")

    if low_avg > 0.95 and high_avg < low_avg - 0.1:
        print("  -> CONFIRMS the spacing hypothesis: low frequencies are nearly fully "
              "redundant across the 2 capsules, high frequencies show real decorrelation. "
              "A HIGH-frequency test tone should be more separable from calibrated ambient "
              "noise than the 1kHz tone was.")
    elif low_avg > 0.95 and high_avg > 0.95:
        print("  -> REFUTES the spacing hypothesis as the (sole) explanation: even high "
              "frequencies stay nearly fully redundant. The 2cm spacing may simply be too "
              "small across the ENTIRE audible band for this mic pair specifically, or "
              "there's a shared electrical/gain coupling between the two capsules.")
    else:
        print("  -> Mixed/inconclusive pattern -- coherence doesn't cleanly separate by "
              "band the way the spacing hypothesis predicts. Real result, not the clean "
              "story either direction.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
