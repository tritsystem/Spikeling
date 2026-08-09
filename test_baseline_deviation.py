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

    # ── adapt() -- gated homeostatic drift-tracking ──────────────────────
    bd4 = BaselineDeviation()
    check("adapt() before calibration is a no-op, returns False (nothing to adapt toward)",
          bd4.adapt([5.0]) is False)

    bd4.calibrate([[10.0], [10.0], [10.0], [10.0], [10.0]])   # dead-flat baseline (std floored to 1e-6)
    mean_before = bd4._mean[0]
    adapted = bd4.adapt([10.000001], rate=0.5, gate_threshold=2.0)
    check("adapt() on a reading that already matches baseline returns True (gate passes)",
          adapted is True)
    check("a matching reading nudges the mean toward it (real movement, not a no-op)",
          bd4._mean[0] != mean_before)

    bd5 = BaselineDeviation()
    bd5.calibrate([[v] for v in [9.8, 10.2, 9.9, 10.1, 10.0]])
    mean_before5 = bd5._mean[0]
    real_deviation_reading = [mean_before5 + 50 * max(bd5._std[0], 1e-6)]   # an unmistakable real spike
    gated_out = bd5.adapt(real_deviation_reading, rate=0.9, gate_threshold=2.0)
    check("adapt() on a reading far past gate_threshold returns False (real deviation, not absorbed)",
          gated_out is False)
    check("a gated-out reading leaves the baseline mean completely unchanged",
          bd5._mean[0] == mean_before5)

    # closed-loop drift-tracking over many ticks: a baseline that has
    # genuinely shifted (e.g. a new background service raising idle CPU%)
    # should be tracked by repeated adapt() calls at matching readings,
    # the way a one-time calibrate() never would be
    bd6 = BaselineDeviation()
    bd6.calibrate([[5.0], [5.1], [4.9], [5.0], [5.05]])   # old normal ~5.0
    new_normal = 5.0
    drift_target = 8.0   # the machine's real normal has shifted to ~8.0
    for step in range(400):
        new_normal += (drift_target - new_normal) * 0.02   # the real world drifts gradually, not instantly
        bd6.adapt([new_normal], rate=0.05, gate_threshold=3.0)
    check("repeated adapt() calls track a genuine gradual baseline drift toward the new real normal",
          abs(bd6._mean[0] - drift_target) < 0.5)

    # ── max_channel_z -- regression test for a REAL failure caught live in
    # system_telemetry_adapter.py: a channel perfectly flat during
    # calibration (std floored to 1e-6) produced a billions-scale score
    # from one genuine real reading.
    bd7 = BaselineDeviation(max_channel_z=None)   # default: unclipped, matches every prior test above
    bd7.calibrate([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]])   # channel 1 is perfectly flat (like idle disk I/O)
    runaway_reading = [10.0, 500.0]   # a real, modest jump on the flat channel
    check("without max_channel_z, a flat-calibrated channel's score explodes (reproduces the real bug)",
          bd7.score(runaway_reading) > 1e6)
    check("deviation() itself is enormous on that channel too (this is the raw, honest z, not the bug)",
          abs(bd7.deviation(runaway_reading)[1]) > 1e6)

    bd8 = BaselineDeviation(max_channel_z=20.0)
    bd8.calibrate([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]])
    check("with max_channel_z set, the SAME runaway reading's score stays bounded and interpretable",
          bd8.score(runaway_reading) < 50.0)
    check("max_channel_z does NOT touch deviation() -- the raw per-channel detail is still the true huge number",
          abs(bd8.deviation(runaway_reading)[1]) > 1e6)
    check("a channel that's genuinely NOT degenerate is unaffected by max_channel_z",
          abs(bd8.score([10.0 + 0.0, 0.0]) - bd7.score([10.0, 0.0])) < 1e-6)

    print("\n" + "-" * 42)
    print(f"  {_pass} passed, {_fail} FAILED")
    print("-" * 42 + "\n")
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
