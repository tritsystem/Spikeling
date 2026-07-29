#!/usr/bin/env python
"""
stress_test_fan_ramp.py — real validation of the calibrated acoustic
anomaly detector: burns CPU across every core to make the machine's real
cooling fan audibly ramp up, and watches whether anomaly_score() actually
rises above the recommended threshold in response -- an honest end-to-end
check against a real (if mundane) change in ambient sound, not another
synthetic tone.

Requires acoustic_baseline.json (run calibrate_acoustic_baseline.py first).

    python stress_test_fan_ramp.py
"""
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, "core")

from compiler.compiler import SpikelingParser
from runtime.runtime import SpikelingRuntime
from hardware.acoustic_anomaly_detector import AcousticAnomalyDetector

THRESHOLD = 5.674   # from the real calibration session (max observed + 3 stdev)

_NET = """
neuron Ch0 threshold=50 leak=25
neuron Output threshold=30 leak=10
connect Ch0 -> Output weight=100
action Output -> [ANOMALY]
refractory=2ms
"""


def _burn_cpu(stop_event):
    x = 0
    while not stop_event.is_set():
        for i in range(200000):
            x += i * i


def _sample(det, label, n, delay_s):
    over = 0
    for i in range(n):
        score = det.anomaly_score()
        flag = "  <-- OVER THRESHOLD" if score > THRESHOLD else ""
        if flag:
            over += 1
        print(f"  {label} reading {i+1:2d}/{n}: score={score:.3f}{flag}")
        time.sleep(delay_s)
    return over


def main():
    if not os.path.exists("acoustic_baseline.json"):
        print("No acoustic_baseline.json found -- run calibrate_acoustic_baseline.py first.")
        return 1

    rt = SpikelingRuntime(SpikelingParser().parse(_NET))
    det = AcousticAnomalyDetector(rt, sample_rate=16000, chunk_ms=100.0, num_neurons=16)
    det.load_calibration("acoustic_baseline.json")

    det.start()
    try:
        print("=" * 60)
        print(f"  FAN RAMP STRESS TEST -- alert threshold = {THRESHOLD}")
        print("=" * 60)

        print("\nBaseline check (idle, before stress):")
        idle_over = _sample(det, "idle  ", 5, 0.3)

        n_workers = os.cpu_count() or 4
        stop_event = mp.Event()
        procs = [mp.Process(target=_burn_cpu, args=(stop_event,)) for _ in range(n_workers)]
        print(f"\nStarting {n_workers}-core CPU stress to ramp the fan...")
        for p in procs:
            p.start()

        print("Waiting 6s for the fan to actually spin up...")
        time.sleep(6)

        print("\nSampling DURING stress (fan should be audibly ramped by now):")
        stress_over = _sample(det, "stress", 15, 0.5)

        stop_event.set()
        for p in procs:
            p.join(timeout=3)
        print("\nStopped CPU stress. Letting the fan spin back down (6s)...")
        time.sleep(6)

        print("\nSampling AFTER stress (fan should be winding down):")
        after_over = _sample(det, "after ", 8, 0.4)

        print("\n" + "-" * 42)
        print(f"  idle:   {idle_over}/5 readings over threshold (expect 0 -- this is the calibrated baseline)")
        print(f"  stress: {stress_over}/15 readings over threshold (this is the real signal being tested)")
        print(f"  after:  {after_over}/8 readings over threshold (expect trending back toward 0)")
        print("-" * 42 + "\n")
    finally:
        det.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
