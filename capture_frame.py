#!/usr/bin/env python
"""
capture_frame.py — grabs ONE real frame from the C922 (device index 0,
confirmed working earlier when video_adapter.py was built) and saves it
as a JPEG. The "video glasses" live-feedback loop: capture a frame here,
then Claude reads the saved image directly (the Read tool supports
images) to actually see and describe it -- real visual understanding,
not the frame-differencing motion numbers video_adapter.py computes.

    python capture_frame.py [output_path]
"""
import sys

import cv2

OUTPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "live_frame.jpg"
DEVICE_INDEX = 0


def main():
    cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"FAILED: could not open webcam at device index {DEVICE_INDEX}")
        return 1
    # a couple warm-up reads -- the first frame after opening is sometimes
    # a stale/black buffer on some webcam drivers
    for _ in range(3):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("FAILED: webcam opened but frame read failed")
        return 1
    cv2.imwrite(OUTPUT_PATH, frame)
    print(f"OK: saved real frame ({frame.shape[1]}x{frame.shape[0]}) to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
