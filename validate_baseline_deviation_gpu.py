#!/usr/bin/env python
"""
validate_baseline_deviation_gpu.py — validates the SHARED BaselineDeviation
mechanism (core/hardware/baseline_deviation.py) against real, independently
verifiable data: this machine's actual GPU telemetry (temperature,
utilization, power draw) via nvidia-smi, with real CUDA load via PyTorch.

WHY THIS, NOT "tuning EMG/environmental against real data" DIRECTLY: no
real EMG hardware exists (needs skin-contact electrodes -- no substitute
possible) and no real environmental hardware exists (needs a BME280/SCD40-
style breakout -- also no substitute on this machine). This does NOT tune
either device for its actual intended purpose. What it DOES do, honestly:
proves the shared calibration/deviation/scoring mechanism those two devices
now both use actually detects a REAL physical change -- and unlike the
earlier fan-ramp test, the "did something real actually happen" question is
answered by nvidia-smi's own readings, not assumed. That gap is the whole
point of running this before touching EMG/environmental again with
unverified claims.

    python validate_baseline_deviation_gpu.py
"""
import subprocess
import sys
import threading
import time

sys.path.insert(0, "core")

from hardware.baseline_deviation import BaselineDeviation

CHANNELS = ["gpu_temp_c", "gpu_util_pct", "gpu_power_w"]


def read_gpu_telemetry() -> dict:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    ).stdout.strip()
    temp, util, power = (float(x) for x in out.split(","))
    return {"gpu_temp_c": temp, "gpu_util_pct": util, "gpu_power_w": power}


def _reading_vector(r: dict) -> list:
    return [r[c] for c in CHANNELS]


def _gpu_burn(stop_event):
    import torch
    device = torch.device("cuda")
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    while not stop_event.is_set():
        c = a @ b
        torch.cuda.synchronize()


def _sample(bd, label, n, delay_s):
    scores = []
    for i in range(n):
        r = read_gpu_telemetry()
        score = bd.score(_reading_vector(r))
        scores.append(score)
        print(f"  {label} {i+1:2d}/{n}: temp={r['gpu_temp_c']:.0f}C util={r['gpu_util_pct']:.0f}% "
              f"power={r['gpu_power_w']:.1f}W  ->  deviation_score={score:.2f}")
        time.sleep(delay_s)
    return scores


def main():
    print("=" * 70)
    print("  BASELINE DEVIATION -- validated against REAL, VERIFIABLE GPU telemetry")
    print("=" * 70)

    bd = BaselineDeviation(smoothing_alpha=0.3)

    print("\nRecording real idle GPU telemetry as the calibrated baseline (8 samples)...")
    idle_samples = []
    for i in range(8):
        r = read_gpu_telemetry()
        idle_samples.append(_reading_vector(r))
        print(f"  calib {i+1}/8: temp={r['gpu_temp_c']:.0f}C util={r['gpu_util_pct']:.0f}% power={r['gpu_power_w']:.1f}W")
        time.sleep(0.5)
    mean, std = bd.calibrate(idle_samples)
    print(f"\nCalibrated baseline: temp={mean[0]:.1f}+-{std[0]:.2f}C, "
          f"util={mean[1]:.1f}+-{std[1]:.2f}%, power={mean[2]:.1f}+-{std[2]:.2f}W")

    print("\nIdle check (before load):")
    idle_scores = _sample(bd, "idle ", 5, 0.5)

    stop_event = threading.Event()
    print("\nStarting REAL CUDA matrix-multiply load on the GPU...")
    t = threading.Thread(target=_gpu_burn, args=(stop_event,), daemon=True)
    t.start()
    time.sleep(3)   # let it actually ramp

    r_during = read_gpu_telemetry()
    print(f"\nGROUND TRUTH CHECK (nvidia-smi, not assumed): "
          f"temp={r_during['gpu_temp_c']:.0f}C util={r_during['gpu_util_pct']:.0f}% power={r_during['gpu_power_w']:.1f}W")
    real_load_confirmed = r_during["gpu_util_pct"] > 50.0
    print(f"Real GPU load confirmed by nvidia-smi itself: {real_load_confirmed}")

    print("\nSampling DURING confirmed real load:")
    load_scores = _sample(bd, "load ", 8, 0.5)

    stop_event.set()
    t.join(timeout=5)
    print("\nStopped GPU load. Cooling down (5s)...")
    time.sleep(5)

    print("\nSampling AFTER load:")
    after_scores = _sample(bd, "after", 5, 0.5)

    print("\n" + "-" * 50)
    print(f"  idle mean score:  {sum(idle_scores)/len(idle_scores):.2f}")
    print(f"  load mean score:  {sum(load_scores)/len(load_scores):.2f}  "
          f"(real load confirmed by nvidia-smi: {real_load_confirmed})")
    print(f"  after mean score: {sum(after_scores)/len(after_scores):.2f}")
    print("-" * 50 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
