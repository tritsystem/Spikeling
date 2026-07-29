#!/usr/bin/env python
"""
test_emg_adapter.py — core/hardware/emg_adapter.py. No real EMG hardware
exists yet (see the module's own honest docstring) -- this verifies the
adapter's real logic (baseline calibration, buffering, normalization,
encoding) against the documented SimulatedEMGSource, not real muscle data.

    python test_emg_adapter.py
"""
import random
import sys
sys.path.insert(0, "core")

from compiler.compiler import SpikelingParser
from runtime.runtime import SpikelingRuntime
from hardware.emg_adapter import EMGSensorAdapter, SimulatedEMGSource

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
neuron Output threshold=30 leak=10
connect Ch0 -> Output weight=100
action Output -> [CONTRACTION]
refractory=2ms
"""


def main():
    print("=" * 60)
    print("  EMG ADAPTER -- real logic, documented simulated source")
    print("=" * 60)

    rng = random.Random(42)   # deterministic -- this test checks real logic, not RNG luck
    source = SimulatedEMGSource(resting=512.0, noise=5.0, burst_amplitude=300.0,
                                 burst_probability=0.0, rng=rng)   # bursts off for the calibration check
    ast = SpikelingParser().parse(TEST_NET)
    rt = SpikelingRuntime(ast)
    adapter = EMGSensorAdapter(rt, output_neuron_map={0: "Ch0"}, source=source,
                                num_neurons=1, buffer_len=8)

    check("before calibration, baseline starts at 0.0 (no assumption about resting level)",
          adapter._baseline == 0.0)

    baseline = adapter.calibrate(n_samples=200)
    check("calibrate() computes a baseline close to the source's real resting value",
          abs(baseline - 512.0) < 5.0)   # 200 samples of +-5.0 noise should average tight

    raw = adapter.read_raw()
    check("read_raw() returns buffer_len samples",
          len(raw) == adapter.buffer_len)
    check("after calibration, resting-state readings sit close to 0 (baseline correctly subtracted)",
          all(abs(x) < 15.0 for x in raw))

    # burst detection: with bursts forced ON, a reading during a burst
    # should clearly exceed the calibrated resting-noise band
    burst_source = SimulatedEMGSource(resting=512.0, noise=5.0, burst_amplitude=300.0,
                                       burst_probability=1.0, rng=random.Random(7))
    burst_adapter = EMGSensorAdapter(rt, output_neuron_map={0: "Ch0"}, source=burst_source, buffer_len=4)
    burst_adapter.calibrate(n_samples=1)   # burst_probability=1.0 means even this sample is mid-burst-ish; just zero the ballpark
    burst_adapter._baseline = 512.0        # pin the baseline explicitly so the burst check is deterministic
    burst_raw = burst_adapter.read_raw()
    check("a simulated muscle contraction burst reads well above the resting baseline",
          max(abs(x) for x in burst_raw) > 100.0)

    # end-to-end: a real contraction should fire the network's action
    fired = []
    rt2 = SpikelingRuntime(SpikelingParser().parse(TEST_NET))
    rt2.register_handler("CONTRACTION", lambda: fired.append(True))
    live_source = SimulatedEMGSource(resting=512.0, noise=5.0, burst_amplitude=400.0,
                                      burst_probability=0.0, rng=random.Random(1))
    live = EMGSensorAdapter(rt2, output_neuron_map={0: "Ch0"}, source=live_source,
                             num_neurons=1, window_ms=10.0, threshold=0.02, buffer_len=8)
    live.calibrate(n_samples=100)
    live_source.burst_probability = 1.0   # force a contraction for the rest of the run
    live_source._burst_ticks_left = 20
    for _ in range(15):
        live.tick(dt_ms=5.0)
    check("a sustained simulated contraction fires the network's CONTRACTION action end to end",
          len(fired) > 0)

    print("\n" + "-" * 42)
    print(f"  {_pass} passed, {_fail} FAILED")
    print("-" * 42 + "\n")
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
