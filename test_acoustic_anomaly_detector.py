#!/usr/bin/env python
"""
test_acoustic_anomaly_detector.py — core/hardware/acoustic_anomaly_detector.py.

Two kinds of checks:
  1. The real FFT/calibration/deviation MATH, verified against synthetic
     tones via a fake stream (so the signal-processing correctness doesn't
     depend on real ambient conditions during an automated run) -- this is
     the actual new logic this file adds on top of the already
     hardware-verified AcousticSensorAdapter.
  2. A real-microphone smoke check (same honesty pattern as
     test_acoustic_adapter.py): calibrate()/read_raw() against this
     machine's real mic run without error. Skips honestly if no real input
     device exists.

    python test_acoustic_anomaly_detector.py
"""
import math
import sys
sys.path.insert(0, "core")

import numpy as np

from compiler.compiler import SpikelingParser
from runtime.runtime import SpikelingRuntime
from hardware.acoustic_anomaly_detector import AcousticAnomalyDetector
from hardware.acoustic_adapter import AcousticSensorAdapter

_pass = 0
_fail = 0
_skip = 0


def check(label, ok):
    global _pass, _fail
    if ok:
        _pass += 1
        print(f"  ok    {label}")
    else:
        _fail += 1
        print(f"  FAIL  {label}")


def skip(label, reason):
    global _skip
    _skip += 1
    print(f"  SKIP  {label} ({reason})")


class FakeStream:
    """Mimics sounddevice.InputStream's .read(n) -- returns a SYNTHETIC
    waveform instead of real audio, so the calibration/deviation MATH can
    be verified deterministically against known tones. The real-hardware
    capture path this stands in for is already proven separately in
    test_acoustic_adapter.py against an actual microphone."""
    def __init__(self, generator):
        self._generator = generator   # callable: () -> list[float]

    def read(self, n):
        samples = self._generator(n)
        return np.array(samples, dtype=np.float32).reshape(-1, 1), False

    def stop(self): pass
    def close(self): pass


def silence(n):
    return [0.0] * n


def white_noise(n, amplitude=0.01, rng=np.random.default_rng(0)):
    return (rng.uniform(-amplitude, amplitude, n)).tolist()


def pure_tone(freq_hz, sample_rate):
    def gen(n):
        return [0.5 * math.sin(2 * math.pi * freq_hz * i / sample_rate) for i in range(n)]
    return gen


TEST_NET = """
neuron Ch0 threshold=50 leak=25
neuron Output threshold=30 leak=10
connect Ch0 -> Output weight=100
action Output -> [ANOMALY]
refractory=2ms
"""


def main():
    print("=" * 60)
    print("  ACOUSTIC ANOMALY DETECTOR -- real spectral math + real mic")
    print("=" * 60)

    SAMPLE_RATE = 8000
    rt = SpikelingRuntime(SpikelingParser().parse(TEST_NET))
    det = AcousticAnomalyDetector(rt, sample_rate=SAMPLE_RATE, chunk_ms=100.0,
                                   num_neurons=8)
    chunk_samples = det.chunk_samples

    # ── spectral correctness: a known 1000Hz tone must peak in the band
    # that actually covers 1000Hz, not some arbitrary band -- this is the
    # real correctness check for the FFT/binning math, independent of any
    # anomaly-detection logic built on top of it.
    tone_1000 = pure_tone(1000.0, SAMPLE_RATE)(chunk_samples)
    spectrum = det._spectrum(tone_1000)
    nyquist = SAMPLE_RATE / 2.0
    expected_band = int(1000.0 / nyquist * len(spectrum))
    peak_band = int(np.argmax(spectrum))
    check(f"a real 1000Hz tone's spectral peak lands in the band that actually covers 1000Hz "
          f"(expected ~band {expected_band}, got band {peak_band})",
          abs(peak_band - expected_band) <= 1)

    silent_spectrum = det._spectrum(silence(chunk_samples))
    check("real silence produces near-zero energy across the spectrum",
          max(silent_spectrum) < 1e-6)

    check("is_calibrated is False before calibrate() has run",
          not det.is_calibrated)

    # a real stream must be open to read at all (correct: you can't read
    # from nothing) -- set the fake one now, then check "read before
    # calibration" behavior specifically
    det._stream = FakeStream(lambda n: white_noise(n))
    check("read_raw() before calibration reports 'no deviation' rather than a meaningless number",
          det.read_raw() == [0.0] * det.encoder.num_neurons)

    # ── calibration against a synthetic "normal" baseline (quiet white
    # noise), then a genuinely different sound must read as a real deviation
    mean, std = det.calibrate(duration_s=1.0)
    check("calibrate() produces a real per-band baseline (mean/std lists sized to num_neurons)",
          len(mean) == det.encoder.num_neurons and len(std) == det.encoder.num_neurons)
    check("is_calibrated becomes True after a real calibration run",
          det.is_calibrated)

    # same baseline noise -> deviation should be small (this IS "normal")
    det._stream = FakeStream(lambda n: white_noise(n))
    normal_dev = det.read_raw()
    normal_score = det.anomaly_score(normal_dev)
    check("a reading matching the calibrated baseline scores as low-deviation",
          normal_score < 2.0)   # within a couple std devs, by construction

    # a genuinely different sound (a strong pure tone the noise baseline
    # never had) must score as a real, larger deviation
    det._stream = FakeStream(pure_tone(1000.0, SAMPLE_RATE))
    anomaly_dev = det.read_raw()
    anomaly_score = det.anomaly_score(anomaly_dev)
    check("a genuinely different sound (a tone absent from the noise baseline) scores as a real, larger deviation",
          anomaly_score > normal_score * 3)

    # ── smoothed_anomaly_score() -- EMA over consecutive readings, added
    # after a real stress test showed instantaneous anomaly_score()
    # swinging widely within a single sustained condition
    smooth_det = AcousticAnomalyDetector(rt, sample_rate=SAMPLE_RATE, num_neurons=4,
                                          smoothing_alpha=0.5)
    smooth_det._stream = FakeStream(lambda n: white_noise(n))
    smooth_det.calibrate(duration_s=0.5)

    check("smoothed_anomaly_score() starts at exactly the first reading (no history to blend yet)",
          smooth_det._smoothed_score is None)
    first = smooth_det.smoothed_anomaly_score(deviation=[2.0, 2.0, 2.0, 2.0])
    check("...and the first call's return value equals that first raw score",
          abs(first - 2.0) < 1e-9)

    second = smooth_det.smoothed_anomaly_score(deviation=[0.0, 0.0, 0.0, 0.0])
    # EMA formula with alpha=0.5: 0.5*raw + 0.5*prev = 0.5*0 + 0.5*2.0 = 1.0
    check("a subsequent call blends the new raw reading with the running average per the EMA formula (alpha=0.5)",
          abs(second - 1.0) < 1e-9)

    check("reset_smoothing() clears the running average back to 'no history'",
          (smooth_det.reset_smoothing(), smooth_det._smoothed_score)[1] is None)

    # the actual practical justification: smoothing genuinely reduces
    # variance vs. the raw score on the same noisy sequence of readings
    noisy_readings = [1.0, 8.0, 0.5, 9.0, 0.6, 7.5, 0.8, 8.5, 0.7, 9.2]
    smooth_det.reset_smoothing()
    smoothed_sequence = [smooth_det.smoothed_anomaly_score(deviation=[v] * 4) for v in noisy_readings]
    import statistics
    check("a smoothed sequence has meaningfully lower variance than the raw noisy readings it was built from",
          statistics.pvariance(smoothed_sequence) < statistics.pvariance(noisy_readings) * 0.5)

    # calibrate()/load_calibration() must reset stale smoothing history --
    # otherwise a fresh baseline's early readings get blended with
    # deviation numbers computed against the OLD baseline
    smooth_det.smoothed_anomaly_score(deviation=[5.0, 5.0, 5.0, 5.0])
    check("smoothing history exists before a recalibration",
          smooth_det._smoothed_score is not None)
    smooth_det._stream = FakeStream(lambda n: white_noise(n))
    smooth_det.calibrate(duration_s=0.5)
    check("calibrate() automatically resets stale smoothing history from the old baseline",
          smooth_det._smoothed_score is None)

    # ── end to end: a real anomalous reading fires the network's action
    fired = []
    rt2 = SpikelingRuntime(SpikelingParser().parse(TEST_NET))
    rt2.register_handler("ANOMALY", lambda: fired.append(True))
    det2 = AcousticAnomalyDetector(rt2, output_neuron_map={0: "Ch0"},
                                    sample_rate=SAMPLE_RATE, chunk_ms=100.0,
                                    num_neurons=1, window_ms=10.0, threshold=0.02,
                                    deviation_scale=2.0)
    det2._stream = FakeStream(lambda n: white_noise(n))
    det2.calibrate(duration_s=1.0)
    det2._stream = FakeStream(pure_tone(1000.0, SAMPLE_RATE))
    for _ in range(15):
        det2.tick(dt_ms=5.0)
    check("a sustained real spectral anomaly fires the network's ANOMALY action end to end",
          len(fired) > 0)

    # ── save/load calibration round-trip (real file I/O, real tempfile)
    import json
    import os
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), "test_acoustic_baseline.json")
    det.save_calibration(tmp_path)
    check("save_calibration() writes a real JSON file", os.path.exists(tmp_path))
    with open(tmp_path) as f:
        saved = json.load(f)
    check("the saved calibration's baseline exactly matches what calibrate() computed",
          saved["mean"] == det._baseline_mean.tolist())

    det3 = AcousticAnomalyDetector(rt, sample_rate=SAMPLE_RATE, num_neurons=8)
    loaded_mean, loaded_std = det3.load_calibration(tmp_path)
    check("load_calibration() restores the exact baseline into a fresh, never-calibrated instance",
          loaded_mean == mean and loaded_std == std)
    check("is_calibrated becomes True after loading a saved baseline (not just after a live calibrate() run)",
          det3.is_calibrated)

    wrong_bands = AcousticAnomalyDetector(rt, sample_rate=SAMPLE_RATE, num_neurons=4)
    raised_mismatch = False
    try:
        wrong_bands.load_calibration(tmp_path)
    except ValueError:
        raised_mismatch = True
    check("loading a calibration with a mismatched band count is refused, not silently misapplied",
          raised_mismatch)
    os.remove(tmp_path)

    # ── real calibration session against THIS machine, saved earlier via
    # calibrate_acoustic_baseline.py -- sanity-check that real artifact is
    # actually loadable and well-formed, not just that save/load works on
    # freshly-generated data
    real_calib_path = "acoustic_baseline.json"
    if os.path.exists(real_calib_path):
        real_det = AcousticAnomalyDetector(rt, sample_rate=16000, num_neurons=16)
        real_mean, real_std = real_det.load_calibration(real_calib_path)
        check("the real calibration file saved from this machine's actual ambient sound loads cleanly",
              len(real_mean) == 16 and all(s > 0 for s in real_std))
    else:
        skip("loading the real saved calibration from this machine",
             "acoustic_baseline.json not present -- run calibrate_acoustic_baseline.py first")

    # ── real hardware smoke check (same honesty pattern as
    # test_acoustic_adapter.py) -- skip cleanly if no real mic exists
    try:
        devices = AcousticSensorAdapter.list_devices()
    except Exception:
        devices = []
    if not devices:
        skip("real-microphone calibrate()/read_raw() smoke check", "no real input device found")
    else:
        live = AcousticAnomalyDetector(rt, sample_rate=16000, chunk_ms=100.0, num_neurons=8)
        try:
            live.start()
            live.calibrate(duration_s=0.5)
            live.read_raw()
            live_ran = True
        except Exception as e:
            live_ran = False
            print(f"  (real-mic run raised: {e})")
        finally:
            live.stop()
        check("calibrate() + read_raw() run cleanly against this machine's real microphone",
              live_ran)

    print("\n" + "-" * 42)
    print(f"  {_pass} passed, {_fail} FAILED, {_skip} skipped")
    print("-" * 42 + "\n")
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
