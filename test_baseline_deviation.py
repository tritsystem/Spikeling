#!/usr/bin/env python
"""
test_baseline_deviation.py — core/hardware/baseline_deviation.py, the
shared "verify by relationship to a calibrated baseline" primitive behind
every anomaly-style device (acoustic, EMG, environmental).

    python test_baseline_deviation.py
"""
import os
import statistics
import sys
import tempfile

sys.path.insert(0, "core")

from hardware.baseline_deviation import BaselineDeviation

_pass = 0
_fail = 0


def check(label, ok):
    global _pass, _fail
    if ok:
        _pass += 1
        print(f"  ok    {label}")
    else:
        _fail += 1
        print(f"  FAIL  {label}")


def main():
    print("=" * 60)
    print("  BASELINE DEVIATION -- shared relational-verification primitive")
    print("=" * 60)

    bd = BaselineDeviation(smoothing_alpha=0.5)
    check("not calibrated before calibrate() runs", not bd.is_calibrated)
    check("deviation() before calibration returns all-zeros, not an error",
          bd.deviation([5.0, 5.0]) == [0.0, 0.0])

    # a genuine multi-channel calibration -- works identically whether the
    # channel represents an acoustic frequency band, an EMG reading, or an
    # environmental sensor, since it's just list[float] either way
    samples = [[10.0, 100.0], [10.5, 98.0], [9.5, 102.0], [10.2, 99.0], [9.8, 101.0]]
    mean, std = bd.calibrate(samples)
    check("calibrate() computes a real per-channel mean", abs(mean[0] - 10.0) < 0.1)
    check("is_calibrated becomes True after calibrate()", bd.is_calibrated)

    # a reading identical to the mean deviates ~0
    check("a reading matching the baseline mean deviates ~0",
          all(abs(d) < 1e-6 for d in bd.deviation(mean)))

    # a reading several std-devs off scores as a real, larger deviation
    far_reading = [mean[0] + 10 * std[0], mean[1] + 10 * std[1]]
    near_score = bd.score(mean)
    far_score = bd.score(far_reading)
    check("a reading far from baseline (10 std devs off) scores much higher than one matching it",
          far_score > near_score + 5.0)

    # calibrate() rejects a degenerate/empty call rather than silently
    # producing a baseline of NaN
    bd2 = BaselineDeviation()
    raised = False
    try:
        bd2.calibrate([])
    except ValueError:
        raised = True
    check("calibrate() with no samples is refused, not silently accepted as a degenerate baseline",
          raised)

    # smoothing: reduces variance, matches the EMA formula exactly, resets
    # correctly
    bd3 = BaselineDeviation(smoothing_alpha=0.5)
    bd3.calibrate([[0.0], [0.0], [0.0]])
    first = bd3.smoothed_score(deviation=[2.0])
    check("smoothed_score()'s first call returns exactly the first raw score",
          abs(first - 2.0) < 1e-9)
    second = bd3.smoothed_score(deviation=[0.0])
    check("subsequent calls blend per the EMA formula (alpha=0.5: 0.5*0 + 0.5*2.0 = 1.0)",
          abs(second - 1.0) < 1e-9)
    bd3.reset_smoothing()
    check("reset_smoothing() clears the running average", bd3._smoothed_score is None)

    noisy = [1.0, 8.0, 0.5, 9.0, 0.6, 7.5, 0.8, 8.5]
    bd3.reset_smoothing()
    smoothed_seq = [bd3.smoothed_score(deviation=[v]) for v in noisy]
    check("a smoothed sequence has meaningfully lower variance than the raw noisy sequence",
          statistics.pvariance(smoothed_seq) < statistics.pvariance(noisy) * 0.5)

    # save/load round-trip + channel-count mismatch refusal
    tmp_path = os.path.join(tempfile.gettempdir(), "test_baseline_deviation.json")
    bd.save(tmp_path)
    check("save() writes a real JSON file", os.path.exists(tmp_path))

    bd_loaded = BaselineDeviation()
    loaded_mean, loaded_std = bd_loaded.load(tmp_path)
    check("load() restores the exact saved baseline into a fresh instance",
          loaded_mean == mean and loaded_std == std)
    check("is_calibrated is True after load(), not just after a live calibrate()",
          bd_loaded.is_calibrated)

    bd_mismatch = BaselineDeviation()
    raised_mismatch = False
    try:
        bd_mismatch.load(tmp_path, expected_channels=5)
    except ValueError:
        raised_mismatch = True
    check("load() with expected_channels refuses a channel-count mismatch instead of silently misapplying it",
          raised_mismatch)
    os.remove(tmp_path)

    # calibrate() resets stale smoothing history from a prior baseline
    bd.smoothed_score(deviation=[5.0, 5.0])
    check("smoothing history exists before a recalibration", bd._smoothed_score is not None)
    bd.calibrate(samples)
    check("calibrate() resets stale smoothing history automatically",
          bd._smoothed_score is None)

    print("\n" + "-" * 42)
    print(f"  {_pass} passed, {_fail} FAILED")
    print("-" * 42 + "\n")
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
