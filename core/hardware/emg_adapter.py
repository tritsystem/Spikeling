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
"""

import random

from .sensor_adapter import SensorAdapter


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

    def __init__(self, *args, source=None, buffer_len: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = source or SimulatedEMGSource()
        self.buffer_len = buffer_len
        self._baseline = 0.0

    def calibrate(self, n_samples: int = 50) -> float:
        """Rest-state baseline calibration: average N readings while the
        muscle is at rest, subtracted from future readings. A REAL, well-
        known EMG requirement -- electrode placement and skin conductivity
        vary enough that raw ADC counts at rest are never exactly 0, so
        every real EMG pipeline calibrates a baseline before use. Returns
        the computed baseline."""
        samples = [self.source.read() for _ in range(max(1, n_samples))]
        self._baseline = sum(samples) / len(samples)
        return self._baseline

    def read_raw(self) -> list:
        return [self.source.read() - self._baseline for _ in range(self.buffer_len)]

    @property
    def value_range(self):
        lo, hi = self.source.value_range
        # baseline-subtracted, so the range shifts by -baseline too
        return (lo - self._baseline, hi - self._baseline)
