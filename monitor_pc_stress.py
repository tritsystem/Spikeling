#!/usr/bin/env python
"""
monitor_pc_stress.py — live PC-stress monitor built on
core/hardware/system_telemetry_adapter.py: real CPU/RAM/disk/GPU
telemetry, calibrated against THIS machine's own real idle state, with
per-channel attribution (not one opaque score) and real top-process
identification when something is actually elevated.

WHY THIS IS THE HONEST VERSION OF "ultimate custom tuning": there is no
universal "stressed" threshold -- what counts as unusual depends entirely
on this specific machine's own real baseline, which is exactly what
calibrate() records. The homeostatic adapt() call each tick lets that
baseline track genuine long-run drift (a new background service, etc.)
without absorbing a real ongoing stress event as the new normal -- see
core/hardware/baseline_deviation.py's adapt() docstring for the gating
rationale.

    python monitor_pc_stress.py [--calib-seconds 15] [--threshold 3.0] [--interval 1.0]
"""
import argparse
import sys
import time

sys.path.insert(0, "core")

from hardware.system_telemetry_adapter import SystemTelemetryAdapter, SystemTelemetrySource, top_processes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib-seconds", type=float, default=15.0,
                     help="how long to record this machine's real idle baseline before monitoring starts")
    ap.add_argument("--threshold", type=float, default=3.0,
                     help="smoothed_stress_score() above this triggers a real-process attribution report")
    ap.add_argument("--interval", type=float, default=1.0,
                     help="seconds between real telemetry reads once monitoring starts")
    ap.add_argument("--adapt-rate", type=float, default=0.01,
                     help="homeostatic drift-tracking rate; 0 disables adapt() entirely")
    args = ap.parse_args()

    print("=" * 70)
    print("  PC STRESS MONITOR -- real telemetry, this machine's own calibrated baseline")
    print("=" * 70)

    adapter = SystemTelemetryAdapter(source=SystemTelemetrySource())
    n_samples = max(3, int(args.calib_seconds / 0.5))
    print(f"\nRecording {n_samples} real idle-state samples over ~{args.calib_seconds:.0f}s "
          f"as this machine's calibrated normal -- let it sit idle now.")
    mean, std = adapter.calibrate(n_samples=n_samples, delay_s=0.5)
    for ch, m, s in zip(adapter._active_channels, mean, std):
        print(f"  {ch:16s} normal = {m:8.2f} +- {s:6.3f}")

    print(f"\nMonitoring every {args.interval:.1f}s. Ctrl+C to stop. "
          f"Attribution reports when smoothed score > {args.threshold:.1f}.")
    print(f"Homeostatic drift-tracking: {'ON (rate=' + str(args.adapt_rate) + ')' if args.adapt_rate > 0 else 'OFF'}\n")

    try:
        while True:
            result = adapter.sample(top_n=3, adapt=args.adapt_rate > 0,
                                     adapt_rate=args.adapt_rate, adapt_gate=2.0)
            score, diag = result["score"], result["diagnosis"]
            diag_str = ", ".join(f"{name}={z:+.2f}sigma" for name, z in diag)
            print(f"  score={score:6.2f}  [{diag_str}]")

            if score > args.threshold:
                print("  " + "-" * 60)
                print(f"  ELEVATED (score {score:.2f} > threshold {args.threshold:.1f}) -- real top processes:")
                for name, pid, val in top_processes(n=3, by="cpu"):
                    print(f"    cpu   {name:28s} pid={pid:<8d} {val:6.1f}%")
                for name, pid, val in top_processes(n=3, by="memory"):
                    print(f"    mem   {name:28s} pid={pid:<8d} {val:6.1f}%")
                print("  " + "-" * 60)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
