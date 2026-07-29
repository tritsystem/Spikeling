#!/usr/bin/env python
"""
test_video_adapter.py — core/hardware/video_adapter.py. Real webcam
hardware, confirmed working at device index 0 before this test was
written (640x480 frames). No simulated fallback -- like acoustic, this
hardware genuinely exists.

    python test_video_adapter.py
"""
import sys
import time
sys.path.insert(0, "core")

from hardware.video_adapter import VideoSensorAdapter, VideoMotionSource

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


def main():
    print("=" * 60)
    print("  VIDEO ADAPTER -- real webcam, real motion signal")
    print("=" * 60)

    source = VideoMotionSource(grid=(3, 3))
    check("channel_names returns 9 real grid cells for a 3x3 grid",
          len(source.channel_names) == 9)

    first = source.read()
    check("first real read() returns all zeros (no prior frame to diff against yet)",
          all(v == 0.0 for v in first))

    time.sleep(0.1)
    second = source.read()
    check("second real read() returns one value per grid cell",
          len(second) == 9)
    check("real frame-diff values are non-negative (absolute pixel difference)",
          all(v >= 0.0 for v in second))

    adapter = VideoSensorAdapter(source=source)
    check("is_calibrated is False before calibrate() has run",
          not adapter.is_calibrated)

    print("\nCalibrating on REAL current camera view (20 frames)...")
    adapter.calibrate(n_samples=20)
    check("is_calibrated becomes True after a real calibration",
          adapter.is_calibrated)

    score_still = adapter.motion_score()
    print(f"  motion_score() right after calibrating (scene should be similar): {score_still:.2f}")
    check("motion_score() right after calibrating against the current view stays low",
          score_still < 5.0)

    diag = adapter.diagnose(top_n=3)
    check("diagnose() returns up to 3 (channel_name, z_score) pairs",
          0 < len(diag) <= 3)
    check("diagnose() is sorted by |z-score| descending",
          all(abs(diag[i][1]) >= abs(diag[i + 1][1]) for i in range(len(diag) - 1)))
    check("diagnose()'s channel names are real, known grid cells",
          all(name in adapter._active_channels for name, _ in diag))
    print(f"  real diagnose() output: {diag}")

    # save/load round-trip
    import os
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), "test_video_baseline.json")
    adapter.save_calibration(tmp_path)
    fresh_source = VideoMotionSource(grid=(3, 3))
    fresh = VideoSensorAdapter(source=fresh_source)
    fresh.load_calibration(tmp_path)
    check("save_calibration()/load_calibration() round-trip a real video baseline",
          fresh.is_calibrated)
    os.remove(tmp_path)

    source.release()
    fresh_source.release()

    print("\n" + "-" * 42)
    print(f"  {_pass} passed, {_fail} FAILED")
    print("-" * 42 + "\n")
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
