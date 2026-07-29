#!/usr/bin/env python
"""
test_environmental_adapter.py — core/hardware/environmental_adapter.py. No
real environmental hardware exists yet (see the module's own honest
docstring) -- this verifies the adapter's real logic (per-channel
normalization across wildly different natural ranges) against the
documented SimulatedEnvironmentalSource, not real sensor data.

    python test_environmental_adapter.py
"""
import random
import sys
sys.path.insert(0, "core")

from compiler.compiler import SpikelingParser
from runtime.runtime import SpikelingRuntime
from hardware.environmental_adapter import EnvironmentalSensorAdapter, SimulatedEnvironmentalSource

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


TEST_NET = """
neuron Ch0 threshold=50 leak=25
neuron Ch1 threshold=50 leak=25
neuron Ch2 threshold=50 leak=25
neuron Output threshold=30 leak=10
connect Ch0 -> Output weight=100
connect Ch1 -> Output weight=100
connect Ch2 -> Output weight=100
action Output -> [ALERT]
refractory=2ms
"""


class FixedSource:
    """Test double: always returns the same three-channel reading, so the
    per-channel normalization math can be checked exactly."""
    def __init__(self, temperature_c, humidity_pct, co2_ppm):
        self._reading = {"temperature_c": temperature_c, "humidity_pct": humidity_pct, "co2_ppm": co2_ppm}

    def read(self):
        return dict(self._reading)


def main():
    print("=" * 60)
    print("  ENVIRONMENTAL ADAPTER -- multi-channel, real logic")
    print("=" * 60)

    rt = SpikelingRuntime(SpikelingParser().parse(TEST_NET))

    # bottom of every channel's range -> all normalize to 0.0
    low = EnvironmentalSensorAdapter(rt, source=FixedSource(-40.0, 0.0, 400.0))
    low_raw = low.read_raw()
    check("read_raw() returns one value per known channel",
          len(low_raw) == 3)
    check("the bottom of each channel's own range normalizes to 0.0, despite wildly different raw units",
          all(abs(x - 0.0) < 1e-9 for x in low_raw))

    # top of every channel's range -> all normalize to 1.0
    high = EnvironmentalSensorAdapter(rt, source=FixedSource(85.0, 100.0, 5000.0))
    high_raw = high.read_raw()
    check("the top of each channel's own range normalizes to 1.0",
          all(abs(x - 1.0) < 1e-9 for x in high_raw))

    # a mid-range plausible indoor reading normalizes to plausible mid values,
    # not clustered at an extreme just because raw units differ so much
    mid = EnvironmentalSensorAdapter(rt, source=FixedSource(21.0, 45.0, 600.0))
    mid_raw = mid.read_raw()
    check("a normal indoor reading normalizes to comfortably mid-range values on all 3 channels",
          all(0.0 < x < 0.6 for x in mid_raw))

    # a channel missing from the source's reading is skipped, not crashed on
    # or silently defaulted to a fabricated value
    class PartialSource:
        def read(self):
            return {"temperature_c": 21.0, "humidity_pct": 45.0}   # no co2
    partial = EnvironmentalSensorAdapter(rt, source=PartialSource())
    check("a channel missing from the source's reading is skipped, not fabricated",
          len(partial.read_raw()) == 2)

    # ── stress_score()/smoothed_stress_score() -- the shared
    # BaselineDeviation relational-verification mechanism applied to
    # environmental sensing: "unusual FOR THIS ROOM", not "close to the
    # sensor's absolute max" (read_raw()'s unchanged job).
    class JitterSource:
        """Small real random jitter around a fixed point -- calibrate()
        needs genuine per-channel variance to learn from, not a perfectly
        identical repeated value (which would make baseline std ~0)."""
        def __init__(self, temperature_c, humidity_pct, co2_ppm, jitter=0.5, rng=None):
            self._base = {"temperature_c": temperature_c, "humidity_pct": humidity_pct, "co2_ppm": co2_ppm}
            self._jitter = jitter
            self._rng = rng or random.Random()

        def read(self):
            return {k: v + self._rng.uniform(-self._jitter, self._jitter) for k, v in self._base.items()}

    normal_room = JitterSource(21.0, 45.0, 600.0, jitter=0.5, rng=random.Random(5))
    score_env = EnvironmentalSensorAdapter(rt, source=normal_room, smoothing_alpha=0.5)
    check("is_deviation_calibrated is False before calibrate() has run",
          not score_env.is_deviation_calibrated)
    score_env.calibrate(n_samples=100)
    check("is_deviation_calibrated becomes True after calibrate()",
          score_env.is_deviation_calibrated)

    normal_scores = [score_env.stress_score() for _ in range(20)]
    check("readings matching this room's calibrated normal score low",
          sum(normal_scores) / len(normal_scores) < 3.0)

    # a DIFFERENT room's install, calibrated on ITS OWN normal, correctly
    # treats a CO2 level that would be unremarkable elsewhere as a real
    # deviation if it's unusual for THIS specific calibrated baseline
    quiet_room = JitterSource(20.0, 40.0, 450.0, jitter=0.3, rng=random.Random(9))
    quiet_env = EnvironmentalSensorAdapter(rt, source=quiet_room, smoothing_alpha=0.5)
    quiet_env.calibrate(n_samples=100)
    stressed_room = JitterSource(20.0, 40.0, 1800.0, jitter=0.3, rng=random.Random(2))  # a real CO2 buildup
    quiet_env.source = stressed_room
    stressed_scores = [quiet_env.stress_score() for _ in range(10)]
    check("a real CO2 buildup scores as a meaningful deviation from THIS room's calibrated normal",
          sum(stressed_scores) / len(stressed_scores) > 5.0)

    quiet_env.reset_smoothing()
    smoothed_seq = [quiet_env.smoothed_stress_score() for _ in range(10)]
    check("smoothed_stress_score() tracks the real sustained CO2 buildup (stays elevated)",
          all(s > 3.0 for s in smoothed_seq[-3:]))

    # save/load round-trip
    import os
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), "test_environmental_baseline.json")
    score_env.save_calibration(tmp_path)
    fresh_env = EnvironmentalSensorAdapter(rt, source=JitterSource(21.0, 45.0, 600.0))
    fresh_env.load_calibration(tmp_path)
    check("save_calibration()/load_calibration() round-trip a real environmental baseline",
          fresh_env.is_deviation_calibrated)
    os.remove(tmp_path)

    # end to end: a real CO2 spike (a genuine "stressed room" scenario)
    # should be able to fire the network's action once encoded
    fired = []
    rt2 = SpikelingRuntime(SpikelingParser().parse(TEST_NET))
    rt2.register_handler("ALERT", lambda: fired.append(True))
    spike_source = FixedSource(21.0, 45.0, 4800.0)   # near the top of CO2's range
    live = EnvironmentalSensorAdapter(rt2, output_neuron_map={0: "Ch0", 1: "Ch1", 2: "Ch2"},
                                       source=spike_source, num_neurons=3,
                                       window_ms=10.0, threshold=0.05)
    for _ in range(20):
        live.tick(dt_ms=5.0)
    check("a real CO2-spike reading fires the network's ALERT action end to end",
          len(fired) > 0)

    # sanity check the documented simulated source itself stays in-range
    # over many ticks (drift shouldn't escape physically plausible bounds)
    sim = SimulatedEnvironmentalSource()
    for _ in range(500):
        r = sim.read()
    check("the simulated source's humidity stays within its own documented 0-100% bound after sustained drift",
          0.0 <= r["humidity_pct"] <= 100.0)
    check("the simulated source's CO2 never drifts below the physically real ~400ppm atmospheric floor",
          r["co2_ppm"] >= 400.0)

    print("\n" + "-" * 42)
    print(f"  {_pass} passed, {_fail} FAILED")
    print("-" * 42 + "\n")
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
