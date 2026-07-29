"""
core/hardware/video_adapter.py
=================================
VideoSensorAdapter — a real webcam-based motion sensor, following the
same SensorAdapter + BaselineDeviation pattern as every other device in
this hardware layer.

REAL, HARDWARE-VERIFIED (like acoustic_adapter.py, unlike EMG/
environmental): captures from an actual webcam via cv2.VideoCapture --
confirmed working at device index 0, real 640x480 frames read
successfully before this file was written.

SIGNAL: per-cell motion magnitude via simple grayscale frame differencing
(mean absolute pixel difference between consecutive frames, per grid
cell). Deliberately the simplest, most honestly verifiable real
computer-vision signal -- this does not claim to "detect" or "recognize"
any specific content, only "how much did the pixels in this region of
the frame change since the last frame." Multiple channels = spatial grid
cells (the same multi-channel design as environmental_adapter.py's
temperature/humidity/CO2 channels), so calibrate()/diagnose() can
attribute WHICH region of the frame is moving, not just one aggregate
"something moved" score.
"""

import cv2
import numpy as np

from .sensor_adapter import SensorAdapter
from .baseline_deviation import BaselineDeviation


class VideoMotionSource:
    """Real webcam frame-differencing motion source. `grid` (rows, cols)
    splits each frame into cells; each cell's channel value is the mean
    absolute pixel difference from the previous frame in that cell
    (0-255 raw grayscale scale)."""

    def __init__(self, device_index: int = 0, grid: tuple = (3, 3), backend=cv2.CAP_DSHOW):
        self.cap = cv2.VideoCapture(device_index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at device index {device_index} -- "
                                f"real hardware/driver check failed, not assumed to work.")
        self.grid = grid
        self._prev_gray = None

    @property
    def channel_names(self) -> list:
        rows, cols = self.grid
        return [f"cell_r{r}c{c}" for r in range(rows) for c in range(cols)]

    def _capture_gray(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Webcam read failed mid-session -- a real hardware/driver "
                                "issue, not silently papered over.")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def read(self) -> list:
        """Returns one motion-magnitude value per grid cell. The very
        first call after construction (or after a resolution change) has
        no previous frame to diff against, so it honestly returns all
        zeros rather than a fabricated first-frame guess."""
        gray = self._capture_gray()
        rows, cols = self.grid
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return [0.0] * (rows * cols)
        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray
        h, w = diff.shape
        cell_h, cell_w = max(1, h // rows), max(1, w // cols)
        out = []
        for r in range(rows):
            for c in range(cols):
                cell = diff[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
                out.append(float(cell.mean()) if cell.size else 0.0)
        return out

    def release(self) -> None:
        self.cap.release()


class VideoSensorAdapter(SensorAdapter):
    """See module docstring. `source` defaults to a real VideoMotionSource
    -- no simulated fallback, this hardware genuinely exists and was
    confirmed working before this class was written."""

    def __init__(self, runtime=None, *args, source=None, smoothing_alpha: float = 0.3,
                 max_channel_z: float = 20.0, **kwargs):
        """max_channel_z defaults to 20.0 (not None) for the same reason
        as system_telemetry_adapter.py: a grid cell that's perfectly
        static during calibration (e.g. a wall, a dark corner) can have
        near-zero variance, and any real subsequent motion there would
        otherwise produce a runaway aggregate score."""
        super().__init__(runtime, *args, **kwargs)
        self.source = source or VideoMotionSource()
        self._deviation = BaselineDeviation(smoothing_alpha=smoothing_alpha, max_channel_z=max_channel_z)
        self._active_channels = list(self.source.channel_names)

    @property
    def value_range(self):
        return (-4.0, 4.0)   # deviation-scored, same convention as the other detectors

    def calibrate(self, n_samples: int = 30) -> tuple:
        """Records n_samples REAL consecutive frame-diff readings as this
        specific camera/scene's calibrated normal (e.g. an empty room's
        ordinary flicker/compression noise, not literal zero)."""
        samples = [self.source.read() for _ in range(max(2, n_samples))]
        return self._deviation.calibrate(samples)

    @property
    def is_calibrated(self) -> bool:
        return self._deviation.is_calibrated

    def read_raw(self) -> list:
        return self._deviation.deviation(self.source.read())

    def motion_score(self) -> float:
        return self._deviation.score(deviation=self.read_raw())

    def smoothed_motion_score(self) -> float:
        return self._deviation.smoothed_score(deviation=self.read_raw())

    def reset_smoothing(self) -> None:
        self._deviation.reset_smoothing()

    def diagnose(self, top_n: int = 3) -> list:
        """Which grid cell(s) are driving the current motion score --
        the same per-channel-attribution pattern as system_telemetry_
        adapter.py's diagnose()."""
        dev = self.read_raw()
        pairs = sorted(zip(self._active_channels, dev), key=lambda x: -abs(x[1]))
        return pairs[:top_n]

    def save_calibration(self, path: str) -> None:
        self._deviation.save(path)

    def load_calibration(self, path: str) -> tuple:
        return self._deviation.load(path, expected_channels=len(self._active_channels))

    def release(self) -> None:
        self.source.release()
