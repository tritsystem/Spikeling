"""
pyspike_bricks.py -- reusable, parameterized circuit-generator helpers
for pyspike.Net, fused in from Sandia National Labs' Fugu framework's
own "bricks" pattern (see Documents/neuromorphic-survey/fugu_vs_spikeling.md
for the real, code-level comparison this was drawn from): a brick is an
ALGORITHM that adds a real, working sub-circuit to a Net, not a fixed
network -- reusable across projects with different parameters, the way
a function is reusable, rather than copy-pasted per-experiment .spk
wiring.

Deliberately NOT a port of Fugu's own heavyweight backend-abstraction
architecture (PyTorch/Lava/Loihi targeting) -- that would fight against
Spikeling's own real strength (a small, dependency-light, embeddable
runtime, the thing that actually makes it usable in the kernel-
scheduler and Arduino-sensor-grid work elsewhere in this portfolio).
Just the composable-circuit-generator IDEA, kept native.

First brick: a real coincidence detector, chosen specifically because
it's the classic use case for the OTHER thing fused in alongside this
(synaptic delay, see runtime.py's Synapse.delay_ms) -- a delay-based
circuit is close to meaningless to demonstrate without a real delay
primitive to build it on, so these two additions are deliberately
paired, not independent.

    python pyspike_bricks.py    # self-test: proves the brick's own documented
                                 # delivery math for real, then proves delay
                                 # compensation -- not just eventual accumulation
                                 # -- is what makes two time-skewed sources
                                 # coincide, via a real negative control.
"""
from pyspike import Net, NeuronRef


def coincidence_detector(net: Net, in_a: NeuronRef, in_b: NeuronRef, name: str,
                          threshold: float = 50.0, weight_per_input: float = 0.6,
                          leak: float = 20.0, delay_a_ms: float = 0.0,
                          delay_b_ms: float = 0.0) -> NeuronRef:
    """Real coincidence detector: fires ONLY when both in_a and in_b
    deliver a spike close enough together in time that neither
    contribution has fully leaked away before the other arrives --
    neither input ALONE is enough to cross threshold.

    The math (with the real defaults above, propagation delivers
    weight*50.0 per spike -- see runtime.py's _deliver_spike()):
      - one input alone: 0.6 * 50.0 = 30.0 potential, threshold=50.0 ->
        does NOT fire alone (30 < 50), by design.
      - two inputs on the SAME tick: 30.0 + 30.0 = 60.0 >= 50.0 -> fires.
      - two inputs one leak-tick apart: 30.0 leaks to 30.0-leak=10.0,
        then +30.0 = 40.0 < 50.0 -> does NOT fire (real, working
        coincidence window, not just eventual accumulation -- leak is
        what makes "coincidence" mean something different from "sum of
        everything that ever arrived").

    delay_a_ms/delay_b_ms: real per-input transmission delay (see
    runtime.py's Synapse.delay_ms) -- set these to compensate for a
    KNOWN real-world propagation-time skew between two physical sources
    (e.g. two sensors at different distances from an event) so
    genuinely simultaneous real-world events register as simultaneous
    spikes at the detector, the classic real use of delay lines in
    neuromorphic coincidence detection.

    Caller is responsible for choosing threshold/weight_per_input/leak
    together for their own real tick rate -- leak in this runtime is
    denominated in potential-per-stimulate()-or-tick()-call, not a
    fixed real-time unit (same as everywhere else leak is used in this
    runtime; this brick doesn't change that, just documents it clearly
    at the one call site that most depends on getting it right)."""
    out = net.neuron(name, threshold=threshold, leak=leak)
    in_a.to(out, weight=weight_per_input, delay_ms=delay_a_ms)
    in_b.to(out, weight=weight_per_input, delay_ms=delay_b_ms)
    return out


# ─────────────────────────────────────────────────────────────────────────────
def _selftest_neither_input_alone_fires() -> bool:
    net = Net(refractory_ms=0)
    rt = net.build_live()  # must precede neuron()/connect() calls -- see build_live()'s docstring
    A = net.neuron("A", threshold=50, leak=0)
    B = net.neuron("B", threshold=50, leak=0)
    coincidence_detector(net, A, B, "OUT")
    rt.stimulate("A", 1.0, 60.0)
    fired = rt.neurons["OUT"].fire_count
    ok = fired == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] one input alone (0.6*50=30 potential < "
          f"threshold 50) does NOT fire OUT (fire_count={fired})")
    return ok


def _selftest_docstring_math_one_leak_tick_apart() -> bool:
    """Exact reproduction of the brick's own documented math: two inputs one
    leak-tick apart -> 30 leaks to 30-20=10, +30=40 < 50 -> no fire."""
    net = Net(refractory_ms=0)
    rt = net.build_live()
    A = net.neuron("A", threshold=50, leak=0)
    B = net.neuron("B", threshold=50, leak=0)
    coincidence_detector(net, A, B, "OUT")  # defaults: weight=0.6, leak=20
    rt.stimulate("A", 1.0, 60.0)
    rt.tick(2.0)                    # one leak tick: OUT potential 30 -> 10
    rt.stimulate("B", 3.0, 60.0)    # +30 = 40 < 50
    fired = rt.neurons["OUT"].fire_count
    ok = fired == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] inputs one leak-tick apart (30 decays to "
          f"10, +30=40 < threshold 50) does NOT fire OUT (fire_count={fired}) -- "
          f"matches the brick's own documented math exactly")
    return ok


def _selftest_delay_compensated_skewed_sources_coincide() -> bool:
    """The real point of pairing this brick with Synapse.delay_ms: two
    physical sources at different distances from a detector (e.g. two
    sensors) whose signals genuinely arrive 20ms apart in real time can
    still be made to COINCIDE at the detector, if each source's own
    conduction delay is tuned to compensate for the arrival-time skew --
    NEAR has a short physical path but a long modeled delay (30ms), FAR
    fired 20ms later but has a short remaining delay (10ms); both land on
    OUT at real time t=31 (NEAR: 1+30=31, FAR: 21+10=31)."""
    net = Net(refractory_ms=0)
    rt = net.build_live()
    NEAR = net.neuron("NEAR", threshold=50, leak=0)
    FAR = net.neuron("FAR", threshold=50, leak=0)
    coincidence_detector(net, NEAR, FAR, "OUT", leak=20.0,
                          delay_a_ms=30.0, delay_b_ms=10.0)
    rt.stimulate("NEAR", 1.0, 60.0)          # real event at t=1
    for t in range(2, 21):
        rt.tick(float(t))
    rt.stimulate("FAR", 21.0, 60.0)          # real event at t=21, 20ms after NEAR
    for t in range(22, 32):
        rt.tick(float(t))                    # t=31 flushes both scheduled deliveries
    fired = rt.neurons["OUT"].fire_count
    ok = fired >= 1
    print(f"  [{'PASS' if ok else 'FAIL'}] delay-compensated coincidence: NEAR and FAR "
          f"fired 20ms apart in real time, but tuned delays (30ms/10ms) land both "
          f"on OUT at the same real time (t=31) -- fires (fire_count={fired}) even "
          f"though neither source alone could")
    return ok


def _selftest_without_delay_compensation_same_skew_does_not_fire() -> bool:
    """Negative control for the test above: SAME 20ms real skew between NEAR
    and FAR, but delay_a_ms/delay_b_ms left at their 0.0 default (no
    compensation). If this fired too, the previous test's result would just
    be eventual accumulation wearing a coincidence-detector costume. It
    doesn't: leak=20 decays NEAR's lone 30-potential to 0 (in 2 ticks) well
    before FAR's uncompensated, instantly-delivered spike arrives 20 ticks
    later -- proving delay compensation, not mere summation, is what made
    the previous test's detector fire."""
    net = Net(refractory_ms=0)
    rt = net.build_live()
    NEAR = net.neuron("NEAR", threshold=50, leak=0)
    FAR = net.neuron("FAR", threshold=50, leak=0)
    coincidence_detector(net, NEAR, FAR, "OUT", leak=20.0)  # delay_a_ms/b_ms default 0.0
    rt.stimulate("NEAR", 1.0, 60.0)          # delivers to OUT INSTANTLY (no delay)
    for t in range(2, 21):
        rt.tick(float(t))                    # 19 leak ticks: 30 -> 0 well before t=21
    rt.stimulate("FAR", 21.0, 60.0)          # delivers to OUT instantly, +30 alone
    fired = rt.neurons["OUT"].fire_count
    ok = fired == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] control, same 20ms skew but NO delay "
          f"compensation: NEAR's charge fully leaked away before FAR arrives, OUT "
          f"does NOT fire (fire_count={fired}) -- confirms the previous test's fire "
          f"was caused by delay tuning, not by eventual accumulation")
    return ok


if __name__ == "__main__":
    print("=" * 78)
    print("  PYSPIKE BRICKS -- coincidence_detector() self-test (native Synapse.delay_ms path)")
    print("=" * 78)
    results = [
        _selftest_neither_input_alone_fires(),
        _selftest_docstring_math_one_leak_tick_apart(),
        _selftest_delay_compensated_skewed_sources_coincide(),
        _selftest_without_delay_compensation_same_skew_does_not_fire(),
    ]
    print("=" * 78)
    print(f"  {sum(results)}/{len(results)} passed")
    if not all(results):
        raise SystemExit(1)
