#!/usr/bin/env python
"""
test_mcp_server_tools.py — calls mcp_server.py's tool functions directly
(same underlying callables an MCP client would invoke through the
protocol) against ALL 5 registered real/simulated sources, end to end:
start_sandbox -> calibrate -> read -> diagnose -> run_experiment ->
close_sandbox. Confirms the general framework actually works uniformly
across every source, not just the ones it was designed around.

    python test_mcp_server_tools.py
"""
import sys

import mcp_server as srv

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


def run_source(source, n_calibration=8, n_readings=3, interval_s=0.3):
    print(f"\n--- {source} ---")
    info = srv.start_sandbox(source)
    check(f"[{source}] start_sandbox returns a real sandbox_id", "sandbox_id" in info)
    sid = info["sandbox_id"]

    cal = srv.calibrate(sid, n_samples=n_calibration)
    check(f"[{source}] calibrate() returns real mean/std", "mean" in cal and "std" in cal)

    r = srv.read(sid)
    check(f"[{source}] read() returns a score and raw vector", "score" in r and "raw" in r)
    print(f"  [{source}] read(): score={r['score']}  raw_len={len(r['raw'])}")

    diag = srv.diagnose(sid, top_n=2)
    check(f"[{source}] diagnose() returns up to 2 (name, z) pairs", 0 < len(diag) <= 2)
    print(f"  [{source}] diagnose(): {diag}")

    exp = srv.run_experiment(sid, n_calibration=n_calibration, n_readings=n_readings, interval_s=interval_s)
    check(f"[{source}] run_experiment() returns {n_readings} real scores", len(exp["scores"]) == n_readings)
    check(f"[{source}] run_experiment() saves a real results file", "saved_to" in exp)
    print(f"  [{source}] run_experiment(): avg={exp['avg']:.3f} min={exp['min']:.3f} max={exp['max']:.3f}")

    closed = srv.close_sandbox(sid)
    check(f"[{source}] close_sandbox() reports closed=True", closed["closed"] is True)


def main():
    print("=" * 60)
    print("  MCP SERVER TOOLS -- end to end across all 5 sources")
    print("=" * 60)

    sources = srv.list_sources()
    check("list_sources() returns all 5 registered sources", len(sources) == 5)
    print(f"  {sources}")

    for source in ["system_telemetry", "video", "emg", "environmental", "acoustic"]:
        try:
            run_source(source)
        except Exception as e:
            check(f"[{source}] ran without raising", False)
            print(f"  [{source}] EXCEPTION: {e}")

    print("\n" + "-" * 42)
    print(f"  {_pass} passed, {_fail} FAILED")
    print("-" * 42 + "\n")
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
