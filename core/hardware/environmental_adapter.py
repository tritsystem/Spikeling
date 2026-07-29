"""
core/hardware/environmental_adapter.py
========================================
EnvironmentalSensorAdapter — a reference SensorAdapter for a multi-channel
environmental device (temperature / humidity / CO2).

HONEST STATUS: no real environmental hardware (e.g. a BME280/SCD40-style
breakout) is connected or owned yet. Built against a documented SIMULATED
source (SimulatedEnvironmentalSource) so the adapter's actual logic --
per-channel normalization across wildly different natural ranges, encoding
-- can be verified now. A real hardware source shape (I2CEnvironmentalSource)
is included but UNTESTED -- there's no real hardware to test it against yet.

MULTI-CHANNEL NORMALIZATION: temperature (-40..85 C), humidity (0..100%),
and CO2 (400..5000 ppm) have nothing in common as raw numbers. read_raw()
normalizes each channel to a shared 0..1 scale itself (per channel's own
known range) BEFORE handing off, so the base class's single-value_range
normalize() still applies correctly on top.
"""

import random

from .sensor_adapter import SensorAdapter


class SimulatedEnvironmentalSource:
    """Documented fake, not real sensor data: slow-drifting temperature/
    humidity/CO2 around plausible indoor values. Exists only to exercise
    the adapter's multi-channel plumbing deterministically."""

    def __init__(self, rng=None):
        self._rng = rng or random.Random()
        self._t = 21.0      # deg C
        self._h = 45.0      # % relative humidity
        self._co2 = 600.0   # ppm

    def read(self) -> dict:
        self._t += self._rng.uniform(-0.05, 0.05)
        self._h = max(0.0, min(100.0, self._h + self._rng.uniform(-0.3, 0.3)))
        self._co2 = max(400.0, self._co2 + self._rng.uniform(-15.0, 15.0))
        return {"temperature_c": self._t, "humidity_pct": self._h, "co2_ppm": self._co2}


class I2CEnvironmentalSource:
    """Intended shape for a REAL I2C environmental breakout (e.g. BME280
    for temp/humidity, or an SCD40 for CO2). UNTESTED -- no real hardware
    is connected to verify register reads against, so read() deliberately
    raises rather than pretend to implement an unverified I2C protocol."""

    def __init__(self, bus: int = 1, address: int = 0x76):
        self.bus = bus
        self.address = address

    def read(self) -> dict:
        raise NotImplementedError(
            "Wire up the real chip's register reads here once real "
            "environmental hardware exists to test against (e.g. via "
            "smbus2 on a Raspberry Pi, or the ESP32's own I2C API).")


class EnvironmentalSensorAdapter(SensorAdapter):
    """See module docstring for the real-vs-simulated status and the
    per-channel normalization design. `source` is any object with a
    `.read() -> dict[str, float]` returning any subset of CHANNEL_RANGES'
    keys; SimulatedEnvironmentalSource by default."""

    # channel name -> (lo, hi) its RAW reading is expected to span
    CHANNEL_RANGES = {
        "temperature_c": (-40.0, 85.0),
        "humidity_pct":  (0.0, 100.0),
        "co2_ppm":       (400.0, 5000.0),
    }

    def __init__(self, *args, source=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = source or SimulatedEnvironmentalSource()

    @property
    def value_range(self):
        # read_raw() already normalizes every channel to 0..1 itself
        return (0.0, 1.0)

    def read_raw(self) -> list:
        readings = self.source.read()
        out = []
        for name, (lo, hi) in self.CHANNEL_RANGES.items():
            if name not in readings:
                continue
            span = hi - lo
            v = float(readings[name])
            out.append(max(0.0, min(1.0, (v - lo) / span)) if span > 0 else 0.0)
        return out
