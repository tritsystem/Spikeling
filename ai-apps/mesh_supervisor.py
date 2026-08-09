"""Process supervision for mesh_rag_server.py -- the one piece in this whole
stack that's had zero automatic recovery all night (every restart tonight
was done by hand). Adapted from server-guard's real, already-proven
supervisor.py pattern:

  - Crash recovery: launches the target as a subprocess, restarts it if it
    exits, with exponential backoff and crash-loop protection (a real bug
    that makes it die immediately shouldn't retry into the ground forever
    -- a human should see that).
  - Hang detection: server-guard's own heartbeat_watchdog.py checks
    staleness of a DB timestamp guard.py writes every tick -- that's
    specific to a data-collector's tick loop. mesh_rag_server.py is an
    HTTP server, so the correct generalization here is polling its real
    /health endpoint instead: if it stops responding for several
    consecutive checks, the process is hung-but-alive (exactly the gap
    plain crash recovery misses), and gets killed + restarted.

Usage:
    python mesh_supervisor.py
"""
import os
import subprocess
import sys
import time

import requests

MAX_RAPID_FAILURES = 5
RAPID_FAILURE_WINDOW_S = 60.0
BASE_BACKOFF_S = 2.0
MAX_BACKOFF_S = 120.0

HEALTH_URL = "http://127.0.0.1:5055/health"
HEALTH_CHECK_INTERVAL_S = 10.0
HEALTH_TIMEOUT_S = 8.0
HEALTH_CONSECUTIVE_FAILURES_BEFORE_RESTART = 4  # ~40s of real unresponsiveness
STARTUP_GRACE_S = 20.0  # real cold-start loads an embedding model -- don't false-trigger on that

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_SCRIPT = os.path.join(BASE_DIR, "mesh_rag_server.py")
LOG_PATH = os.path.join(BASE_DIR, "mesh_supervisor.log")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def health_ok():
    try:
        r = requests.get(HEALTH_URL, timeout=HEALTH_TIMEOUT_S)
        return r.status_code == 200
    except requests.RequestException:
        return False


def run_supervised():
    failure_times = []

    while True:
        log(f"starting {TARGET_SCRIPT}")
        proc = subprocess.Popen(
            [sys.executable, TARGET_SCRIPT], cwd=BASE_DIR,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        start_time = time.time()
        consecutive_health_failures = 0
        killed_for_hang = False

        while True:
            time.sleep(HEALTH_CHECK_INTERVAL_S)
            if proc.poll() is not None:
                break  # process exited on its own -- fall through to crash-recovery logic below

            if time.time() - start_time < STARTUP_GRACE_S:
                continue  # still inside real cold-start window, don't health-check yet

            if health_ok():
                consecutive_health_failures = 0
                continue

            consecutive_health_failures += 1
            log(f"health check failed ({consecutive_health_failures}/"
                f"{HEALTH_CONSECUTIVE_FAILURES_BEFORE_RESTART})")
            if consecutive_health_failures >= HEALTH_CONSECUTIVE_FAILURES_BEFORE_RESTART:
                log("hung (alive but unresponsive) -- killing and restarting")
                proc.kill()
                proc.wait()
                killed_for_hang = True
                break

        exit_code = proc.returncode
        reason = "hang" if killed_for_hang else f"exit code {exit_code}"
        log(f"process ended ({reason})")

        now = time.time()
        failure_times = [t for t in failure_times if now - t < RAPID_FAILURE_WINDOW_S]
        failure_times.append(now)

        if len(failure_times) >= MAX_RAPID_FAILURES:
            log(f"{MAX_RAPID_FAILURES} failures within {RAPID_FAILURE_WINDOW_S:.0f}s -- "
                "this looks like a real bug, not a transient blip. Stopping supervision "
                "so a human looks at it, instead of looping forever.")
            sys.exit(1)

        backoff = min(MAX_BACKOFF_S, BASE_BACKOFF_S * (2 ** (len(failure_times) - 1)))
        log(f"restarting in {backoff:.0f}s (backoff)")
        time.sleep(backoff)


if __name__ == "__main__":
    run_supervised()
