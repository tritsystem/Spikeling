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
