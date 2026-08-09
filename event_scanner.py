#!/usr/bin/env python
"""
event_scanner.py — real event-triggered "live video scanning": instead
of Claude polling frames on a timer (not actually possible -- Claude only
acts within a turn) or a bare numeric threshold check, real per-cell
motion (VideoSensorAdapter, hardware-verified) drives a real Spikeling
network (motion_scan_trigger.spk) with genuine spike-based memory --
leaky integrate-and-fire neurons that need SUSTAINED motion across
several real ticks to actually fire, not one noisy instantaneous blip.
The action-firing/handler-dispatch plumbing (SensorAdapter.tick() ->
SpikelingRuntime.stimulate() -> _fire() auto-invoking the registered
handler) is the SAME already-tested mechanism used throughout this
hardware layer -- nothing new at the runtime level, just a new network
wired to it.

When MotionTrigger fires, the EXACT frame that caused it (cached via
VideoMotionSource.last_frame, not a fresh/different capture) gets saved
for Claude to Read and describe afterward.

    python event_scanner.py [duration_s]
"""
import sys
import time

import cv2

sys.path.insert(0, "core")

from compiler.compiler import SpikelingParser
from runtime.runtime import SpikelingRuntime
from hardware.video_adapter import VideoSensorAdapter, VideoMotionSource

DURATION_S = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
GRID = (3, 3)
CELL_NAMES = [f"Cell{i}" for i in range(GRID[0] * GRID[1])]

triggered_frames = []


def main(duration_s: float = None):
    duration_s = DURATION_S if duration_s is None else duration_s
    triggered_frames.clear()   # fresh list each call -- a prior call's results shouldn't linger

    def _on_scan_trigger():
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = f"electronics_captures/scan_trigger_{ts}.jpg"
        cv2.imwrite(out_path, source.last_frame)
        triggered_frames.append(out_path)
        print(f"  ** SCAN_TRIGGER fired -- real sustained motion, saved {out_path}", flush=True)

    with open("motion_scan_trigger.spk") as f:
        ast = SpikelingParser().parse(f.read())
    runtime = SpikelingRuntime(ast)
    runtime.register_handler("SCAN_TRIGGER", _on_scan_trigger)

    global source
    source = VideoMotionSource(grid=GRID)
    adapter = VideoSensorAdapter(
        runtime=runtime,
        output_neuron_map={i: name for i, name in enumerate(CELL_NAMES)},
        num_neurons=len(CELL_NAMES),
        source=source,
    )

    print("Calibrating on the current scene (stay still for this part)...")
    adapter.calibrate(n_samples=20)

    print(f"Scanning for {duration_s:.0f}s -- real sustained motion will trigger a "
          f"frame save via the actual Spikeling network, not a bare threshold check.")
    end = time.time() + duration_s
    while time.time() < end:
        adapter.tick(dt_ms=200.0)
        time.sleep(0.2)

    print(f"\nDone. {len(triggered_frames)} real trigger(s): {triggered_frames}")
    return triggered_frames


if __name__ == "__main__":
    main()
