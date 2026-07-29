"""
core/hardware/emg_adapter.py
=============================
EMGSensorAdapter — a reference SensorAdapter for a bio-signal (muscle
activation / EMG) device.

HONEST STATUS: no real EMG hardware is connected or owned yet. This is
built against a documented SIMULATED source (SimulatedEMGSource) so the
adapter's actual logic -- baseline calibration, buffering, normalization,
encoding -- can be verified now, not left as an unverified sketch. A real
hardware source (SerialEMGSource, for an Arduino/ESP32 streaming ADC counts
from a MyoWare-style sensor over serial) is included in the shape it would
need, but is UNTESTED -- there's no real hardware to test it against yet.
Swap `source=` for a real one once hardware exists; nothing else in
EMGSensorAdapter needs to change.

RELATIONAL SCORING: alongside the existing baseline-subtracted read_raw()
(which drives spike encoding directly, unchanged), calibrate() also builds
a BaselineDeviation (core/hardware/baseline_deviation.py) from the same
resting samples -- the same "verify by relationship to a calibrated
baseline" mechanism acoustic_anomaly_detector.py uses, applied here via
contraction_score()/smoothed_contraction_score(). "Resting" and "elevated"
only mean anything relative to THIS device's own calibrated resting
variance -- electrode placement varies enough that a fixed cutoff can't
work across installs, same reasoning acoustic's per-install baseline uses.
"""

import random

from .sensor_adapter import SensorAdapter
from .baseline_deviation import BaselineDeviation


class SimulatedEMGSource:
    """Documented fake, not real muscle data: baseline ADC noise around a
    resting value, with occasional simulated contraction bursts (a real EMG
    trace at rest is noisy-but-flat, and spikes hard during a contraction --
    this mimics that SHAPE for testing, nothing more)."""

    def __init__(self, resting: float = 512.0, noise: float = 8.0,
                 burst_amplitude: float = 300.0, burst_probability: float = 0.05,
                 rng=None):
        self._rng = rng or random.Random()
        self.resting = resting
        self.noise = noise
        self.burst_amplitude = burst_amplitude
        self.burst_probability = burst_probability
        self._burst_ticks_left = 0

    @property
    def value_range(self):
        return (0.0, 1023.0)   # a typical 10-bit ADC range

    def read(self) -> float:
        if self._burst_ticks_left > 0:
            self._burst_ticks_left -= 1
        elif self._rng.random() < self.burst_probability:
            self._burst_ticks_left = self._rng.randint(3, 10)
        amplitude = self.burst_amplitude if self._burst_ticks_left > 0 else 0.0
        raw = self.resting + self._rng.uniform(-self.noise, self.noise) + amplitude
        return max(0.0, min(1023.0, raw))


class SerialEMGSource:
    """Intended shape for a REAL EMG sensor streamed over serial from an
    Arduino/ESP32 doing analogRead() on a MyoWare-style sensor and printing
    one integer per line. UNTESTED -- no real EMG hardware is connected to
    verify this against. Requires pyserial (`pip install pyserial`), which
    is imported lazily here so its absence doesn't break importing this
    module for the simulated path."""

    def __init__(self, port: str, baud: int = 115200, timeout_s: float = 1.0):
        import serial   # pyserial
        self._ser = serial.Serial(port, baud, timeout=timeout_s)

    @property
    def value_range(self):
        return (0.0, 1023.0)

    def read(self) -> float:
        line = self._ser.readline().decode("utf-8", errors="ignore").strip()
        try:
            return float(line)
        except ValueError:
            return 0.0


class EMGSensorAdapter(SensorAdapter):
    """See module docstring for the real-vs-simulated status. `source` is
    any object with `.read() -> float` and a `.value_range` property --
    SimulatedEMGSource by default."""

    def __init__(self, *args, source=None, buffer_len: int = 32,
                 smoothing_alpha: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = source or SimulatedEMGSource()
        self.buffer_len = buffer_len
        self._baseline = 0.0
        self._deviation = BaselineDeviation(smoothing_alpha=smoothing_alpha)

    def calibrate(self, n_samples: int = 50) -> float:
        """Rest-state baseline calibration: average N readings while the
        muscle is at rest, subtracted from future readings. A REAL, well-
        known EMG requirement -- electrode placement and skin conductivity
        vary enough that raw ADC counts at rest are never exactly 0, so
        every real EMG pipeline calibrates a baseline before use. Also
        calibrates the shared BaselineDeviation from the SAME resting
        samples, so contraction_score() has a genuine per-install notion of
        normal resting VARIANCE, not just an offset. Returns the computed
        baseline (the simple mean, same return value as before)."""
        samples = [self.source.read() for _ in range(max(1, n_samples))]
        self._baseline = sum(samples) / len(samples)
        self._deviation.calibrate([[s] for s in samples])
        return self._baseline

    def read_raw(self) -> list:
        return [self.source.read() - self._baseline for _ in range(self.buffer_len)]

    @property
    def value_range(self):
        lo, hi = self.source.value_range
        # baseline-subtracted, so the range shifts by -baseline too
        return (lo - self._baseline, hi - self._baseline)

    @property
    def is_deviation_calibrated(self) -> bool:
        return self._deviation.is_calibrated

    def contraction_score(self) -> float:
        """Reads ONE fresh sample and scores it against the calibrated
        resting baseline in std-devs of normal resting variance, via the
        shared BaselineDeviation primitive -- the same relational-
        verification mechanism acoustic_anomaly_detector.py uses. ~0 means
        "at rest"; growing values mean growing contraction. Requires
        calibrate() to have run first (a fresh reading against no baseline
        would be meaningless, same reasoning as acoustic's own
        anomaly_score() before calibration)."""
        return self._deviation.score([self.source.read()])

    def smoothed_contraction_score(self) -> float:
        """EMA-smoothed contraction_score() -- same motivation as
        acoustic_anomaly_detector.py's smoothed_anomaly_score() (a single
        instantaneous reading is noisy; smoothing consecutive readings is
        what makes a threshold meaningful for continuous monitoring)."""
        return self._deviation.smoothed_score([self.source.read()])

    def reset_smoothing(self) -> None:
        self._deviation.reset_smoothing()

    def save_calibration(self, path: str) -> None:
        self._deviation.save(path)

    def load_calibration(self, path: str) -> tuple:
        return self._deviation.load(path, expected_channels=1)
