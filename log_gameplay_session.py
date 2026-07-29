#!/usr/bin/env python
"""
log_gameplay_session.py — real-time telemetry log for one live gameplay
session, built on core/hardware/system_telemetry_adapter.py's real
SystemTelemetrySource.

NOT BaselineDeviation-scored: there's no valid "idle normal" to compare
against once the game is already running -- calibrating now would treat
active gameplay as the baseline, which defeats the whole comparison (the
same discipline behind every calibrate()-before-use design in this
hardware layer). Instead this logs the RAW real trend and produces a
profile summary answering the actual gaming questions: GPU-bound or
CPU-bound (gpu_util_pct near 100% = GPU-bound, low despite high CPU =
something else is the limiter), thermal headroom (sustained gpu_temp_c),
memory pressure, and -- directly verifying the graphics-settings change
from earlier -- whether the game process actually ran on the RTX 5060
throughout, via nvidia-smi's own compute-apps list (ground truth, not
assumed).

    python log_gameplay_session.py --minutes 20 --interval 2 --process RustClient.exe
"""
import argparse
import json
import statistics
import subprocess
import sys
import time

sys.path.insert(0, "core")

import psutil

from hardware.system_telemetry_adapter import SystemTelemetrySource


def _gpu_compute_apps():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        apps = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                apps.append(parts[1])
        return apps
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def _process_cpu_mem(name):
    for p in psutil.process_iter(["pid", "name"]):
        if p.info["name"] and p.info["name"].lower() == name.lower():
            try:
                return p.cpu_percent(None), p.memory_info().rss / (1024 ** 2)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return None, None
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--process", type=str, default="RustClient.exe")
    ap.add_argument("--out", type=str, default="gameplay_session_log.json")
    args = ap.parse_args()

    source = SystemTelemetrySource()
    _process_cpu_mem(args.process)   # prime psutil's per-process cpu_percent timer

    rows = []
    on_dgpu_count = 0
    total_checks = 0
    end = time.time() + args.minutes * 60.0

    print(f"Logging real telemetry every {args.interval}s for {args.minutes} min, watching {args.process}...",
          flush=True)
    while time.time() < end:
        r = source.read()
        apps = _gpu_compute_apps()
        on_dgpu = any(args.process.lower() in a.lower() for a in apps)
        total_checks += 1
        on_dgpu_count += int(on_dgpu)
        cpu_pct, mem_mb = _process_cpu_mem(args.process)

        row = dict(r)
        row["t_s"] = round(time.time() - (end - args.minutes * 60.0), 1)
        row["on_dgpu"] = on_dgpu
        row["proc_cpu_pct"] = cpu_pct
        row["proc_mem_mb"] = mem_mb
        rows.append(row)

        gt = r.get("gpu_temp_c")
        gu = r.get("gpu_util_pct")
        gp = r.get("gpu_power_w")
        print(f"  t+{row['t_s']:6.0f}s  cpu={r['cpu_pct']:5.1f}%  max_core={r['cpu_max_core_pct']:5.1f}%  "
              f"mem={r['mem_pct']:5.1f}%  gpu_temp={gt if gt is None else f'{gt:5.1f}C'}  "
              f"gpu_util={gu if gu is None else f'{gu:5.1f}%'}  gpu_pwr={gp if gp is None else f'{gp:6.1f}W'}  "
              f"on_dGPU={on_dgpu}  proc_cpu={cpu_pct}", flush=True)

        time.sleep(max(0.0, args.interval - 0.3))   # process/gpu queries above already cost real time

    def col(name):
        return [r[name] for r in rows if r.get(name) is not None]

    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    for ch in ["cpu_pct", "cpu_max_core_pct", "mem_pct", "gpu_temp_c", "gpu_util_pct", "gpu_power_w"]:
        vals = col(ch)
        if vals:
            print(f"  {ch:18s} avg={statistics.mean(vals):7.2f}  min={min(vals):7.2f}  max={max(vals):7.2f}")
    pct_on_dgpu = 100.0 * on_dgpu_count / max(1, total_checks)
    print(f"  {args.process} on RTX dGPU: {on_dgpu_count}/{total_checks} checks ({pct_on_dgpu:.0f}%)")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nFull raw log saved to {args.out}")


if __name__ == "__main__":
    main()
