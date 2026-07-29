#!/usr/bin/env python
"""
mcp_server.py — exposes this portfolio's real sensor/experiment layer
(core/hardware/*) as MCP tools, so any future Claude Code session can run
parametric real-hardware experiments directly (calibrate/read/diagnose/
run_experiment) instead of writing one-off scripts each time. Every
sandbox gets its own timestamped results directory under experiments/,
so concurrent or repeated runs never clobber shared production state
(e.g. acoustic_baseline.json in the project root).

SOURCES REGISTERED (honest status per source, matching each adapter's
own docstring -- this is the single place that status is asserted, so an
agent calling list_sources() gets the truth, not an assumption):
  acoustic          REAL (hardware-verified microphone)
  system_telemetry  REAL (psutil + nvidia-smi)
  video             REAL (hardware-verified webcam)
  emg               SIMULATED (no real EMG hardware connected)
  environmental     SIMULATED (no real environmental hardware connected)

RUNS INTERACTIVELY, NOT AUTONOMOUSLY: this server only does work when a
tool is actually called by a connected client. Nothing here schedules or
loops on its own -- calibrate/read/diagnose/run_experiment each do one
bounded unit of work and return.

    python mcp_server.py
"""
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "core"))

from mcp.server.fastmcp import FastMCP

from hardware.acoustic_anomaly_detector import AcousticAnomalyDetector
from hardware.system_telemetry_adapter import SystemTelemetryAdapter, SystemTelemetrySource
from hardware.video_adapter import VideoSensorAdapter, VideoMotionSource
from hardware.emg_adapter import EMGSensorAdapter, SimulatedEMGSource
from hardware.environmental_adapter import EnvironmentalSensorAdapter, SimulatedEnvironmentalSource

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)

SOURCE_STATUS = {
    "acoustic": "REAL (hardware-verified microphone)",
    "system_telemetry": "REAL (psutil + nvidia-smi)",
    "video": "REAL (hardware-verified webcam)",
    "emg": "SIMULATED (no real EMG hardware connected)",
    "environmental": "SIMULATED (no real environmental hardware connected)",
}

mcp = FastMCP("spikeling-experiments")

_active = {}   # sandbox_id -> {"source": str, "adapter": obj, "created": float}


def _sandbox_dir(sandbox_id: str) -> Path:
    d = EXPERIMENTS_DIR / sandbox_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_adapter(source: str):
    if source == "acoustic":
        a = AcousticAnomalyDetector(None)   # runtime=None: safe, only tick()/encode() ever touch it
        a.start()
        return a
    if source == "system_telemetry":
        return SystemTelemetryAdapter(source=SystemTelemetrySource())
    if source == "video":
        return VideoSensorAdapter(source=VideoMotionSource())
    if source == "emg":
        return EMGSensorAdapter(None, source=SimulatedEMGSource())
    if source == "environmental":
        return EnvironmentalSensorAdapter(None, source=SimulatedEnvironmentalSource())
    raise ValueError(f"unknown source '{source}', must be one of {list(SOURCE_STATUS)}")


def _calibrate(adapter, source: str, n_samples: int):
    """Calls the adapter's own calibrate() for its real side effect
    (populating its internal BaselineDeviation), but reads the resulting
    mean/std back from that internal instance directly rather than
    trusting each adapter's own calibrate() return value -- those vary
    (acoustic/system_telemetry/video/environmental return a (mean, std)
    tuple; EMGSensorAdapter's own documented convention returns a bare
    float baseline instead, a real integration mismatch this caught).
    Every one of these classes stores a BaselineDeviation as either
    `_baseline` (AcousticAnomalyDetector's own naming) or `_deviation`
    (every other adapter here) -- dispatched explicitly by source below,
    NOT by duck-typing attribute existence: EMGSensorAdapter also happens
    to have its own, unrelated `_baseline` attribute (a plain float
    resting-level offset, not a BaselineDeviation instance) -- a real
    name collision that broke a first attempt at `getattr(...) or
    getattr(...)` here, since that picked EMG's float by name before
    ever reaching its actual `_deviation` instance."""
    if source == "acoustic":
        adapter.calibrate(duration_s=max(3.0, n_samples * 0.15))
        bd = adapter._baseline
    else:
        adapter.calibrate(n_samples=n_samples)
        bd = adapter._deviation
    return bd._mean.tolist(), bd._std.tolist()


def _score(adapter):
    for attr in ("anomaly_score", "stress_score", "motion_score", "contraction_score"):
        if hasattr(adapter, attr):
            return getattr(adapter, attr)()
    return None


def _smoothed_score(adapter):
    for attr in ("smoothed_anomaly_score", "smoothed_stress_score",
                 "smoothed_motion_score", "smoothed_contraction_score"):
        if hasattr(adapter, attr):
            return getattr(adapter, attr)()
    return None


def _diagnose(adapter, top_n: int):
    """Uses the adapter's own diagnose() if it has one (system_telemetry,
    video); otherwise builds an equivalent generic version from read_raw()
    (already per-channel deviation on every one of these classes) so
    every source gets consistent per-channel attribution through this
    server even where the underlying class doesn't define it itself."""
    if hasattr(adapter, "diagnose"):
        return adapter.diagnose(top_n=top_n)
    raw = adapter.read_raw()
    names = getattr(adapter, "_active_channels", None) or [f"ch_{i}" for i in range(len(raw))]
    pairs = sorted(zip(names, raw), key=lambda x: -abs(x[1]))
    return pairs[:top_n]


def _release(adapter):
    if hasattr(adapter, "release"):
        adapter.release()
    elif hasattr(adapter, "stop"):
        adapter.stop()


@mcp.tool()
def list_sources() -> dict:
    """List available sensor sources and their honest real/simulated hardware status."""
    return SOURCE_STATUS


@mcp.tool()
def start_sandbox(source: str) -> dict:
    """Open a new isolated experiment session for `source` (acoustic,
    system_telemetry, video, emg, or environmental). Returns a sandbox_id
    to pass to calibrate/read/diagnose/run_experiment/close_sandbox --
    keeps this session's live device and results isolated from any other
    concurrent experiment and from shared production calibration files."""
    if source not in SOURCE_STATUS:
        raise ValueError(f"unknown source '{source}', must be one of {list(SOURCE_STATUS)}")
    sandbox_id = f"{source}_{uuid.uuid4().hex[:8]}"
    _active[sandbox_id] = {"source": source, "adapter": _make_adapter(source), "created": time.time()}
    _sandbox_dir(sandbox_id)
    return {"sandbox_id": sandbox_id, "source": source, "status": SOURCE_STATUS[source]}


@mcp.tool()
def calibrate(sandbox_id: str, n_samples: int = 20) -> dict:
    """Run a real calibration against the sandbox's live device: n_samples
    readings become this session's baseline for scoring/diagnosis."""
    entry = _active[sandbox_id]
    mean, std = _calibrate(entry["adapter"], entry["source"], n_samples)
    result = {"mean": mean, "std": std, "n_samples": n_samples}
    (_sandbox_dir(sandbox_id) / "calibration.json").write_text(json.dumps(result, indent=2))
    return result


@mcp.tool()
def read(sandbox_id: str) -> dict:
    """One real reading from the sandbox's device: current deviation score
    (raw and smoothed) plus the raw per-channel deviation vector."""
    entry = _active[sandbox_id]
    adapter = entry["adapter"]
    return {
        "score": _score(adapter),
        "smoothed_score": _smoothed_score(adapter),
        "raw": adapter.read_raw(),
    }


@mcp.tool()
def diagnose(sandbox_id: str, top_n: int = 3) -> list:
    """Per-channel deviation attribution from the sandbox's device --
    which channel(s) are actually driving the current score."""
    entry = _active[sandbox_id]
    return _diagnose(entry["adapter"], top_n)


@mcp.tool()
def run_experiment(sandbox_id: str, n_calibration: int = 20,
                    n_readings: int = 10, interval_s: float = 1.0) -> dict:
    """Parametric experiment: (re)calibrate, then take n_readings real
    samples interval_s apart, saving everything to this sandbox's own
    results file. Returns real summary stats (avg/min/max score) -- the
    building block an agent calls repeatedly with different parameters as
    it develops and tests a hypothesis, without writing a new script each time."""
    entry = _active[sandbox_id]
    adapter, source = entry["adapter"], entry["source"]

    _calibrate(adapter, source, n_calibration)
    scores = []
    for _ in range(n_readings):
        scores.append(_score(adapter))
        time.sleep(interval_s)

    result = {
        "sandbox_id": sandbox_id,
        "source": source,
        "params": {"n_calibration": n_calibration, "n_readings": n_readings, "interval_s": interval_s},
        "scores": scores,
        "avg": sum(scores) / len(scores) if scores else None,
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
    }
    out_path = _sandbox_dir(sandbox_id) / f"experiment_{int(time.time())}.json"
    out_path.write_text(json.dumps(result, indent=2))
    result["saved_to"] = str(out_path)
    return result


@mcp.tool()
def close_sandbox(sandbox_id: str) -> dict:
    """Release the sandbox's real device (mic stream / webcam) and drop it."""
    entry = _active.pop(sandbox_id, None)
    if entry is None:
        return {"closed": False, "reason": "unknown sandbox_id"}
    _release(entry["adapter"])
    return {"closed": True, "sandbox_id": sandbox_id}


if __name__ == "__main__":
    mcp.run()
