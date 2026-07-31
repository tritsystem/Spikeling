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

ALSO EXPOSED (electronics-assistant / voice-loop / wireless-glasses tools,
added 2026-07-29 -- honest status per tool below, not just per source):
  capture_photo, log_part_note, capture_dismantling_step, log_dismantling
      REAL -- capture_part_photo()/log_electronics_note()/
      capture_step_photo()/log_dismantling_session() from
      electronics_assistant.py, hardware-verified tonight (real C922
      captures, real vault notes written and read back).
  record_voice
      REAL -- record_voice_clip() from voice_loop.py, hardware-verified
      (real mic capture confirmed). Deliberately does NOT wrap
      voicebox_transcribe/voicebox_speak -- those are voicebox's own MCP
      tools, kept separate rather than re-wrapped here.
  capture_wireless_photo, display_on_glasses
      UNTESTED -- capture_wireless_frame()/send_to_glasses_display() from
      wireless_glasses.py. No ESP32S3/ESP32C3 hardware exists yet to
      verify these against; correct shape for when it arrives, same
      discipline as core/hardware/emg_adapter.py's SerialEMGSource stub.
  connect_obd, obd_dtc_codes, obd_live_data, disconnect_obd
      UNTESTED -- automotive_obd.py, real python-OBD library underneath
      (API confirmed against the installed package), no physical ELM327
      adapter exists yet. Deliberately INDEPENDENT of the electronics-
      assistant/photo-logging tools above -- zero shared code, a missing
      OBD-II adapter never affects capture_photo/log_part_note/etc. and
      vice versa. capture_photo/log_part_note/capture_dismantling_step/
      log_dismantling all now also accept a `kind` tag so the SAME
      mechanism logs automotive work too (e.g. kind="automotive-repair"),
      not a separate reimplementation.
  search_vault, read_project, import_data
      REAL -- project_assistant.py, hardware-verified tonight (real
      search of the actual vault notes, real README read from the
      scrapyard-sensor-project directory, real 620-row JSON import).
      Domain-agnostic project-context tools, INDEPENDENT of every other
      module here -- works for any project, not just electronics/
      automotive, and needs none of the camera/OBD/voice hardware.
  run_event_scan
      REAL -- event_scanner.py, hardware-verified tonight. Runs
      motion_scan_trigger.spk (a real Spikeling network: 9 grid-cell
      neurons -> one MotionTrigger neuron) against the live webcam via
      VideoSensorAdapter -- genuine leaky-integrate spike "memory" needs
      SUSTAINED motion across several real ticks to fire, not one noisy
      blip. A REAL BUG was caught and fixed while building this: initial
      neuron thresholds (40/60) were lower than SensorAdapter's own
      default per-spike `drive` (50), so a single stray encoder spike
      instantly fired a neuron with no real accumulation ever happening
      -- thresholds must sit well above a single spike's drive for
      genuine multi-tick "memory" behavior to exist at all. Fixed
      (150/250) but even after the fix, fires fairly often against a
      person actively at their desk (typing/shifting reads as real,
      if modest, sustained motion) -- correct behavior for a desk scene,
      not validated against a genuinely still baseline.

RUNS INTERACTIVELY, NOT AUTONOMOUSLY: this server only does work when a
tool is actually called by a connected client. Nothing here schedules or
loops on its own -- every tool does one bounded unit of work and returns.

    python mcp_server.py
"""
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "core"))

from mcp.server.fastmcp import FastMCP

from electronics_assistant import (
    capture_part_photo,
    log_electronics_note,
    capture_step_photo,
    log_dismantling_session,
)
from voice_loop import record_voice_clip
from wireless_glasses import capture_wireless_frame, send_to_glasses_display
from automotive_obd import (
    connect_obd as _connect_obd,
    read_dtc_codes as _read_dtc_codes,
    read_live_data as _read_live_data,
    disconnect_obd as _disconnect_obd,
)
from project_assistant import search_vault_notes, read_project_directory, import_structured_data
import event_scanner

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
def capture_photo(label: str) -> dict:
    """Real webcam capture (C922) of a salvaged electronics part, for
    Claude to Read and identify/research. Returns the saved photo path --
    Claude still does the actual identification by looking at the file,
    this tool only handles the real capture."""
    path = capture_part_photo(label)
    return {"photo_path": path}


@mcp.tool()
def log_part_note(title: str, photo_paths: list, body_markdown: str,
                   status: str = "identified", kind: str = "electronics-salvage") -> dict:
    """Writes a real vault note (vault/Project Work) for an identified/
    researched part -- same convention as every other note in this vault
    (frontmatter, embedded photos via ![[name]]). `kind` is the domain
    tag in the note's frontmatter -- pass e.g. "automotive-repair" for
    car parts/diagnosis rather than leaving it defaulted to electronics;
    the underlying capture/identify/log mechanism is identical either
    way, only the label changes."""
    path = log_electronics_note(title, photo_paths, body_markdown, status, kind)
    return {"note_path": path}


@mcp.tool()
def capture_dismantling_step(session_label: str, step_num: int) -> dict:
    """Real webcam capture for one step of a dismantling/construction
    session -- named by session + step number so a multi-step session's
    photos stay in order."""
    path = capture_step_photo(session_label, step_num)
    return {"photo_path": path}


@mcp.tool()
def log_dismantling(title: str, steps: list, status: str = "in-progress",
                     kind: str = "electronics-dismantling") -> dict:
    """Writes ONE consolidated vault note for a whole dismantling/
    construction/repair session. `steps`: list of {"photo": path,
    "guidance": text} dicts, one per real step Claude actually looked at
    and gave guidance for. `kind` is the domain tag -- pass e.g.
    "automotive-repair" for a car repair session; the step-by-step
    capture/guidance/logging mechanism is identical for any domain."""
    path = log_dismantling_session(title, steps, status, kind)
    return {"note_path": path}


@mcp.tool()
def record_voice(duration_s: float = 5.0, device_index: int = -1) -> dict:
    """Records real audio from a mic (device_index=-1 uses the system
    default input) for voicebox_transcribe to process afterward. Deliberately
    doesn't call transcribe/speak itself -- those stay as voicebox's own
    separate MCP tools, this only handles capture."""
    device = None if device_index < 0 else device_index
    path = record_voice_clip(duration_s=duration_s, device=device)
    return {"audio_path": path}


@mcp.tool()
def capture_wireless_photo(esp32_cam_ip: str) -> dict:
    """UNTESTED -- fetches a real JPEG snapshot from a wireless ESP32S3
    Sense camera over WiFi. No ESP32S3 hardware exists yet to verify this
    against; will raise a real connection error if the IP is unreachable,
    not silently return a fake frame."""
    path = capture_wireless_frame(esp32_cam_ip)
    return {"photo_path": path}


@mcp.tool()
def display_on_glasses(esp32_display_ip: str, text: str) -> dict:
    """UNTESTED -- sends text to a wireless ESP32C3's OLED display over
    WiFi. No ESP32C3 hardware or matching firmware exists yet to verify
    this against."""
    ok = send_to_glasses_display(esp32_display_ip, text)
    return {"sent": ok}


@mcp.tool()
def connect_obd(port: str = "") -> dict:
    """UNTESTED -- connects to a real ELM327 OBD-II adapter. port=""
    lets python-OBD auto-detect (USB/Bluetooth-as-COM-port adapters);
    pass an explicit port (e.g. "COM5") or a WiFi adapter's
    "socket://<ip>:<port>" string otherwise. No physical adapter exists
    yet to verify this against -- raises a real error if none is found,
    never fakes a successful connection. Fully independent of the
    electronics-assistant/photo-logging tools -- neither affects the other."""
    return _connect_obd(port=port or None)


@mcp.tool()
def obd_dtc_codes() -> list:
    """UNTESTED -- reads real check-engine/diagnostic trouble codes.
    Requires connect_obd() to have succeeded first; raises if not connected."""
    return _read_dtc_codes()


@mcp.tool()
def obd_live_data(pids: list = None) -> dict:
    """UNTESTED -- reads real live sensor values (pids: e.g. ["RPM",
    "COOLANT_TEMP", "SPEED"], defaults to a small common set). Requires
    connect_obd() first; a pid the car doesn't support comes back None."""
    return _read_live_data(pids=pids)


@mcp.tool()
def disconnect_obd() -> dict:
    """Closes the real OBD-II connection, if one is open."""
    return _disconnect_obd()


@mcp.tool()
def search_vault(query: str, limit: int = 10) -> list:
    """Real text search over vault/Project Work + vault/Research's actual
    notes (filename or content match) -- finds what's already been logged
    about a topic before starting fresh on it. Domain-agnostic: works for
    any project, not just electronics/automotive."""
    return search_vault_notes(query, limit=limit)


@mcp.tool()
def read_project(path: str, max_entries: int = 30) -> dict:
    """Real, read-only snapshot of an existing project directory: its
    README (if present) and top-level file/folder listing (not a full
    recursive walk -- stays fast regardless of project size). Raises if
    the path isn't a real existing directory."""
    return read_project_directory(path, max_entries=max_entries)


@mcp.tool()
def import_data(path: str) -> dict:
    """Real CSV/JSON import of a parts list, spec sheet, or any other
    structured data file someone already compiled elsewhere. Raises on
    an unsupported extension or a real parse error."""
    return import_structured_data(path)


@mcp.tool()
def run_event_scan(duration_s: float = 30.0) -> dict:
    """Runs real event-triggered video scanning for duration_s seconds:
    calibrates on the current webcam view, then a real Spikeling network
    (motion_scan_trigger.spk) watches for SUSTAINED motion (genuine
    leaky-integrate spike memory, not a bare threshold) and saves a frame
    each time it fires. Blocks for the full duration_s. Returns the real
    list of saved trigger-frame paths for Claude to Read and describe."""
    frames = event_scanner.main(duration_s=duration_s)
    return {"triggered_frames": frames, "count": len(frames)}


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
