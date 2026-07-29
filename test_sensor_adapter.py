#!/usr/bin/env python
"""
test_sensor_adapter.py — core/hardware/sensor_adapter.py, the shared base
class every device driver (acoustic, EMG, environmental, ...) builds on.

    python test_sensor_adapter.py
"""
import sys
sys.path.insert(0, "core")

from compiler.compiler import SpikelingParser
from runtime.runtime import SpikelingRuntime
from hardware.sensor_adapter import SensorAdapter

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


# A tiny real network: two input neurons feeding one output neuron that
# fires the FIRED action. Real DSL, real compiler, real runtime -- not a
# mock of any of it.
TEST_NET = """
neuron In0 threshold=50 leak=25
neuron In1 threshold=50 leak=25
neuron Output threshold=40 leak=10
connect In0 -> Output weight=100
connect In1 -> Output weight=100
action Output -> [FIRED]
refractory=2ms
"""


class ConstantSensor(SensorAdapter):
    """Test double: read_raw() always returns a fixed buffer -- lets the
    plumbing (normalize -> encode -> stimulate -> action fires) be verified
    deterministically, independent of any real device."""
    def __init__(self, *args, fixed_value=1.0, buffer_len=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_value = fixed_value
        self.buffer_len = buffer_len

    def read_raw(self):
        return [self.fixed_value] * self.buffer_len


def main():
    print("=" * 60)
    print("  SENSOR ADAPTER -- shared base for every device driver")
    print("=" * 60)

    ast = SpikelingParser().parse(TEST_NET)
    rt = SpikelingRuntime(ast)
    fired = []
    rt.register_handler("FIRED", lambda: fired.append(True))

    # a strong, above-threshold constant signal on both mapped channels
    # should reliably fire the output within a handful of ticks
    sensor = ConstantSensor(rt, output_neuron_map={0: "In0", 1: "In1"},
                             num_neurons=2, window_ms=10.0, threshold=0.05,
                             fixed_value=1.0, buffer_len=8)
    for _ in range(20):
        sensor.tick(dt_ms=5.0)
    check("a strong constant signal on both mapped channels fires the network's action",
          len(fired) > 0)

    # base class refuses to be used directly -- it has no sensor of its own
    bare = SensorAdapter(rt)
    raised = False
    try:
        bare.read_raw()
    except NotImplementedError:
        raised = True
    check("the base class's read_raw() refuses to pretend it has a sensor",
          raised)

    # normalize() maps an arbitrary value_range into [-1, 1] correctly
    class RangedSensor(SensorAdapter):
        @property
        def value_range(self):
            return (0.0, 100.0)   # e.g. a 0-100 percent sensor

        def read_raw(self):
            return [0.0, 50.0, 100.0]

    ranged = RangedSensor(rt)
    normalized = ranged.normalize(ranged.read_raw())
    check("normalize() maps the bottom of value_range to -1.0",
          abs(normalized[0] - (-1.0)) < 1e-9)
    check("normalize() maps the middle of value_range to ~0.0",
          abs(normalized[1] - 0.0) < 1e-9)
    check("normalize() maps the top of value_range to 1.0",
          abs(normalized[2] - 1.0) < 1e-9)

    # an out-of-range reading (sensor noise/spike) clamps instead of
    # silently producing an out-of-bounds value the encoder never expects
    over_range = ranged.normalize([150.0, -50.0])
    check("normalize() clamps values that overshoot value_range instead of letting them escape [-1,1]",
          over_range[0] == 1.0 and over_range[1] == -1.0)

    # an unmapped encoder channel must NOT stimulate anything -- silently
    # dropping it is correct (a device can use fewer input neurons than
    # num_neurons), crashing or stimulating the wrong neuron would not be
    empty_map_sensor = ConstantSensor(rt, output_neuron_map={}, num_neurons=2,
                                       fixed_value=1.0, buffer_len=4)
    try:
        empty_map_sensor.tick()
        no_crash = True
    except Exception:
        no_crash = False
    check("an adapter with no output_neuron_map ticks safely (no-op) instead of crashing",
          no_crash)

    # a silent/empty sensor reading is a real, valid case (e.g. a mic buffer
    # under threshold) -- must return cleanly, not raise
    class EmptySensor(SensorAdapter):
        def read_raw(self):
            return []
    empty = EmptySensor(rt)
    check("an empty raw reading returns no spikes instead of erroring",
          empty.tick() == [])

    print("\n" + "-" * 42)
    print(f"  {_pass} passed, {_fail} FAILED")
    print("-" * 42 + "\n")
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
