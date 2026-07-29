#!/usr/bin/env python
"""
calibrate_acoustic_baseline.py — runs a REAL calibration session for
AcousticAnomalyDetector against whatever machine/room this is run on, saves
the result, and reports real observed post-calibration deviation scores so
a sensible alert threshold can be picked from actual numbers instead of a
guessed default.

    python calibrate_acoustic_baseline.py [--seconds N] [--out PATH]
"""
import argparse
import sys
import time

sys.path.insert(0, "core")

from compiler.compiler import SpikelingParser
from runtime.runtime import SpikelingRuntime
from hardware.acoustic_anomaly_detector import AcousticAnomalyDetector

# a minimal real network -- this script only needs the detector's own
# read_raw()/anomaly_score(), not a full product network, but
# AcousticSensorAdapter requires a real runtime to construct against.
_NET = """
neuron Ch0 threshold=50 leak=25
neuron Output threshold=30 leak=10
connect Ch0 -> Output weight=100
action Output -> [ANOMALY]
refractory=2ms
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=12.0,
                     help="how long to record as the 'normal' baseline (default 12s)")
    ap.add_argument("--out", default="acoustic_baseline.json",
                     help="where to save the calibration (default acoustic_baseline.json)")
    ap.add_argument("--num-neurons", type=int, default=16,
                     help="frequency bands to calibrate (default 16)")
    ap.add_argument("--post-checks", type=int, default=10,
                     help="how many post-calibration real readings to score, for threshold tuning")
    args = ap.parse_args()

    rt = SpikelingRuntime(SpikelingParser().parse(_NET))
    det = AcousticAnomalyDetector(rt, sample_rate=16000, chunk_ms=100.0,
                                   num_neurons=args.num_neurons)

    devices = AcousticAnomalyDetector.list_devices()
    if not devices:
        print("No real input device found on this machine -- cannot calibrate against nothing.")
        return 1
    print(f"Using input device(s): {', '.join(d['name'] for d in devices[:1])}")

    det.start()
    try:
        print(f"Recording {args.seconds:.0f}s of ambient sound as the 'normal' baseline "
              f"-- don't do anything unusual near the mic right now.")
        mean, std = det.calibrate(duration_s=args.seconds)
        print(f"Calibrated. Per-band baseline (mean +- std), {len(mean)} bands:")
        for i, (m, s) in enumerate(zip(mean, std)):
            print(f"  band {i:2d}: {m:8.4f} +- {s:7.4f}")

        print(f"\nTaking {args.post_checks} real post-calibration readings "
              f"(still normal conditions) to see what a genuine 'normal' score looks like...")
        scores = []
        for i in range(args.post_checks):
            score = det.anomaly_score()
            scores.append(score)
            print(f"  reading {i+1:2d}/{args.post_checks}: anomaly_score = {score:.3f}")
            time.sleep(det.chunk_samples / det.sample_rate)

        import statistics
        s_mean = statistics.mean(scores)
        s_max = max(scores)
        s_stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        # a real, measured recommendation: comfortably above the observed
        # normal-operation ceiling, not an arbitrary constant
        recommended = s_max + 3 * s_stdev if s_stdev > 0 else s_max * 2.0

        print(f"\nObserved normal-operation anomaly_score: mean={s_mean:.3f}, "
              f"max={s_max:.3f}, stdev={s_stdev:.3f}")
        print(f"Recommended alert threshold (max observed + 3 stdev): {recommended:.3f}")
        print("This is a starting point from THIS session's real ambient sound, not a "
              "universal constant -- re-run if the install location/ambient noise changes.")

        det.save_calibration(args.out)
        print(f"\nSaved calibration to {args.out}")
    finally:
        det.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
