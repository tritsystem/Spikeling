"""
core/hardware/sensor_adapter.py
================================
SensorAdapter — the missing link between a physical (or simulated) sensor
and Spikeling's existing SignalEncoder + SpikelingRuntime.

WHY THIS EXISTS: encoder.SignalEncoder already accepts any list[float] in
[-1, 1] and rate-codes it to spikes -- that part was never the gap. The gap
was that every device would otherwise reinvent, from scratch, the plumbing
of "read real sensor values -> get them into that shape -> stimulate the
right input neurons -> let the runtime's own actions fire." SensorAdapter is
that plumbing, written once. A concrete device driver only has to implement
TWO things:

  1. read_raw()     -- pull one buffer of raw values from the actual sensor,
                        in whatever native units/range it reports
  2. value_range     -- the (lo, hi) that raw buffer is expected to span, so
                        normalize() can map it into SignalEncoder's [-1, 1]

Everything else (normalization, encoding, stimulating the runtime, keeping
a running clock) is shared here. See acoustic_adapter.py for a real,
hardware-backed reference implementation, and emg_adapter.py /
environmental_adapter.py for adapters built against a documented simulated
source (no EMG/environmental hardware exists to test against yet -- see
those files' own docstrings for exactly what's real vs. simulated here;
being honest about that distinction is the whole point, not a footnote).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from encoder.encoder import SignalEncoder


class SensorAdapter:
    """Base class. Subclass and implement read_raw() (and override
    value_range if your sensor's native range isn't already [-1, 1])."""

    def __init__(self, runtime, output_neuron_map=None,
                 num_neurons: int = 16, window_ms: float = 100.0,
                 threshold: float = 0.1, drive: float = 50.0):
        """
        runtime            : a SpikelingRuntime (already built from a
                              compiled .spk network) to stimulate.
        output_neuron_map  : dict of encoder neuron_idx -> runtime neuron
                              name. A spike on encoder channel N only
                              stimulates the runtime if N is a key here --
                              lets a network use fewer input neurons than
                              num_neurons, or route specific channels
                              deliberately. Defaults to an empty map (no-op)
                              so a bare SensorAdapter never crashes on a
                              runtime with different neuron names; a real
                              device subclass is expected to pass a real map.
        num_neurons/window_ms/threshold : passed straight through to
                              SignalEncoder -- see encoder.py for what
                              each one means.
        drive               : the current injected per stimulate() call
                              (SpikelingRuntime.stimulate()'s own default).
        """
        self.runtime = runtime
        self.output_neuron_map = dict(output_neuron_map or {})
        self.encoder = SignalEncoder(num_neurons=num_neurons,
                                      window_ms=window_ms, threshold=threshold)
        self.drive = drive
        self._t_ms = 0.0

    # ── subclass hooks ──────────────────────────────────────────────────

    def read_raw(self) -> list[float]:
        """Return one buffer of raw sensor values. MUST be overridden --
        the base class has no sensor of its own."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement read_raw() -- "
            f"SensorAdapter has no sensor of its own to read from.")

    @property
    def value_range(self) -> tuple[float, float]:
        """(lo, hi) the raw buffer is expected to span. Override if your
        sensor's native units aren't already [-1, 1] (e.g. a temperature
        sensor reporting Celsius, or a raw ADC reading 0-4095)."""
        return (-1.0, 1.0)

    # ── shared plumbing ──────────────────────────────────────────────────

    def normalize(self, raw: list[float]) -> list[float]:
        """Maps `raw` from value_range into SignalEncoder's expected
        [-1, 1], clamping anything that overshoots (sensor noise/spikes
        shouldn't silently break encoding)."""
        lo, hi = self.value_range
        span = hi - lo
        if span <= 0:
            return [0.0 for _ in raw]
        return [max(-1.0, min(1.0, 2.0 * (x - lo) / span - 1.0)) for x in raw]

    def tick(self, dt_ms: float = 100.0) -> list[tuple[int, float]]:
        """One full cycle: read -> normalize -> encode -> stimulate the
        runtime's mapped neurons. Returns the raw (neuron_idx, delay_ms)
        spikes from the encoder, for inspection/testing -- callers don't
        need to touch this to use the adapter, but it's what a test
        verifies against, not the runtime's internal state, so a real
        sensor-classification bug is caught here rather than downstream."""
        raw = self.read_raw()
        if not raw:
            return []
        spikes = self.encoder.encode(self.normalize(raw))
        for neuron_idx, delay_ms in spikes:
            neuron_name = self.output_neuron_map.get(neuron_idx)
            if neuron_name is not None:
                self.runtime.stimulate(neuron_name, self._t_ms + delay_ms, drive=self.drive)
        self._t_ms += dt_ms
        return spikes
