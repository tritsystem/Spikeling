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

Built on top of that: sequence_detector() (order-sensitive, not just
coincident -- the actual "temporal pattern recognition" delay lines are
for), and a real spiking logic family (and_gate/or_gate/xor_gate/
half_adder/full_adder) closing the OTHER real gap the Fugu comparison
named -- "building GENERAL COMPUTATION out of spiking primitives"
(adder_bricks.py etc), which this runtime had nothing for before.
xor_gate's own docstring records a real bug found and fixed while
building it (a naive OR-excites/AND-inhibits design double-fires on
both-inputs-active; fixed via a refractory-based dedup), not just the
working end state. full_adder's own docstring records the real timing
subtlety of CHAINING two delay-based bricks: composition isn't
plug-and-play the way synchronous digital logic is -- a downstream
stage's inputs have to be aligned to the upstream stage's own real
propagation latency, not just wired.

    python pyspike_bricks.py    # self-test: proves every brick's documented
                                 # behavior for real -- delay-compensated
                                 # coincidence + its negative control, order
                                 # sensitivity, and full truth tables for
                                 # every logic gate including the 3-input
                                 # full adder, not just "it compiles."
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


def sequence_detector(net: Net, first: NeuronRef, second: NeuronRef, name: str,
                       gap_ms: float, threshold: float = 50.0,
                       weight_per_input: float = 0.6, leak: float = 20.0) -> NeuronRef:
    """Real ORDER-sensitive detector: fires when `first` spikes and `second`
    spikes exactly `gap_ms` later -- the REVERSED order (second-then-first,
    same real gap) does NOT fire. This is what actually distinguishes
    "temporal pattern recognition" from plain coincidence detection (see
    this project's own pyspike_delay.py docstring on Izhikevich's
    polychronization: "a delay pattern lets a network respond to WHEN
    inputs arrived relative to each other, not just whether they did").

    Just coincidence_detector with an asymmetric delay: `first`'s
    contribution is held back by exactly `gap_ms` so it lands at the same
    real time `second`'s (undelayed) contribution naturally does, IF AND
    ONLY IF `second` really did arrive `gap_ms` after `first`. Swap the
    order and the two contributions land `2*gap_ms` apart instead of
    together -- `first`'s decays away (leak) long before `second`'s
    delayed one arrives, so the reversed sequence does NOT coincide."""
    return coincidence_detector(net, first, second, name, threshold=threshold,
                                 weight_per_input=weight_per_input, leak=leak,
                                 delay_a_ms=gap_ms, delay_b_ms=0.0)


def and_gate(net: Net, a: NeuronRef, b: NeuronRef, name: str,
             threshold: float = 50.0, weight_per_input: float = 0.6,
             leak: float = 20.0) -> NeuronRef:
    """Real 2-input spiking AND: fires only when both a and b spike within
    the same coincidence window (see coincidence_detector, which this IS --
    named/exposed separately for readability when composing logic circuits
    like xor_gate/half_adder below, where "AND" is the intent that matters
    at the call site, not "coincidence")."""
    return coincidence_detector(net, a, b, name, threshold=threshold,
                                 weight_per_input=weight_per_input, leak=leak)


def or_gate(net: Net, a: NeuronRef, b: NeuronRef, name: str,
            threshold: float = 50.0, weight_per_input: float = 1.2,
            leak: float = 0.0) -> NeuronRef:
    """Real 2-input spiking OR: fires whenever EITHER input alone crosses
    threshold (weight_per_input*50.0 >= threshold by default: 60 >= 50).

    Honest characterization, not oversold: this fires ONCE PER QUALIFYING
    INPUT SPIKE, not once per "at least one active" logical instant -- if
    both a and b spike (even at the same real time, two separate external
    events), OUT fires TWICE, not once. That's a real, disclosed
    difference from a single-pulse Boolean OR gate, and exactly the
    property xor_gate() below has to work around (see its own docstring)
    rather than ignore."""
    out = net.neuron(name, threshold=threshold, leak=leak)
    a.to(out, weight=weight_per_input)
    b.to(out, weight=weight_per_input)
    return out


def xor_gate(net: Net, a: NeuronRef, b: NeuronRef, name: str,
             threshold: float = 50.0, gap_ms: float = 20.0,
             inhibit_gap_ms: float = 10.0) -> NeuronRef:
    """Real 2-input spiking XOR, composed from an AND-helper and an
    OR-helper -- the general-computation-from-spiking-primitives capability
    named as Fugu's own real differentiator (adder_bricks.py etc, see
    Documents/neuromorphic-survey/fugu_vs_spikeling.md) that this runtime
    otherwise has nothing for.

    THE REAL PROBLEM a naive OR-excites/AND-inhibits XOR hits, found by
    actually tracing this runtime's event semantics before writing code
    (not assumed): a plain or_gate() fires ONCE PER INPUT (see its own
    docstring) -- so when BOTH a and b spike, a naive design delivers
    TWO separate excitatory hits to the XOR neuron. Even if an AND-helper's
    single inhibitory hit lands first and clamps XOR to -threshold, the
    SECOND of the two excitatory hits still pushes it back over threshold
    (-50 + 60 + 60 = 70 >= 50) -- XOR fires anyway, wrongly, when both
    inputs are active. This isn't a tuning problem; a single shared
    excitatory weight literally cannot satisfy both "one delivery alone
    crosses threshold" and "two deliveries, even after a clamped -threshold
    head start, stay under it" at once.

    THE REAL FIX: give the OR-helper a nonzero refractory_ms. Two inputs
    arriving at the exact same real time now only produce ONE delivery
    (the second is refractory-blocked -- elapsed=0 < refractory_ms), which
    reopens a real, consistent weight range: single delivery must clear
    threshold alone (weight*50 >= threshold), and one delivery landing
    AFTER a clamped -threshold inhibitory hit must stay under it
    (-threshold + weight*50 < threshold) -- both hold simultaneously for
    any weight*50 in [threshold, 2*threshold), e.g. the default 60 with
    threshold=50.

    Wiring: AND-helper (undelayed, needs both inputs to cross) inhibits
    XOR with a delay shorter than the OR-helper's own delay to XOR, so
    the inhibitory hit (if it comes at all -- only when both fired) always
    lands BEFORE the single deduped excitatory hit. Traced by hand for all
    4 input combinations before this was ever run; see the self-tests
    below for the real, executed proof.

    REQUIRES `net` to have been built with Net(refractory_ms > 0) -- the
    dedup fix above depends on it (refractory is a real property of the
    whole runtime in this codebase, `SpikelingRuntime.refractory_ms`, NOT
    settable per-neuron; there's no NeuronDef field for it). Asserted
    below rather than silently producing a net that fires twice on
    both-inputs, the exact bug this whole design exists to avoid."""
    assert net.ast.refractory_ms > 0, (
        "xor_gate requires Net(refractory_ms=...) > 0 -- its OR-helper dedup "
        "(collapsing two same-instant deliveries into one) depends on the "
        "runtime's real refractory gate; refractory_ms=0 reproduces the "
        "double-delivery bug described in this function's own docstring.")

    and_helper = net.neuron(f"{name}_and_helper", threshold=threshold, leak=20.0)
    a.to(and_helper, weight=0.6)
    b.to(and_helper, weight=0.6)

    or_helper = net.neuron(f"{name}_or_helper", threshold=threshold, leak=0.0)
    a.to(or_helper, weight=1.2)
    b.to(or_helper, weight=1.2)

    out = net.neuron(name, threshold=threshold, leak=0.0)
    and_helper.to(out, weight=-2.0, delay_ms=inhibit_gap_ms)
    or_helper.to(out, weight=1.2, delay_ms=gap_ms)
    return out


def half_adder(net: Net, a: NeuronRef, b: NeuronRef, sum_name: str,
               carry_name: str) -> tuple:
    """Real 1-bit half-adder built entirely from spiking primitives:
    SUM = XOR(a, b), CARRY = AND(a, b) -- composed directly from xor_gate()
    and and_gate() above. This is the concrete thing Documents/
    neuromorphic-survey/fugu_vs_spikeling.md named as Fugu's real
    differentiator (adder_bricks.py: "building GENERAL COMPUTATION out of
    spiking primitives, not just ML inference") and flagged as "worth a
    real follow-up if there's interest" -- this closes that follow-up for
    the 1-bit half-adder case (see full_adder() below for the 3-input,
    carry-in case, built from two of these chained together).

    Requires the same Net(refractory_ms > 0) as xor_gate() -- see its
    docstring."""
    s = xor_gate(net, a, b, sum_name)
    c = and_gate(net, a, b, carry_name)
    return s, c


def full_adder(net: Net, a: NeuronRef, b: NeuronRef, cin: NeuronRef,
               sum_name: str, carry_name: str) -> tuple:
    """Real 3-input full adder, built the same way real hardware full-adders
    are: two chained half_adder() stages --
        s1, c1 = half_adder(a, b)          # SUM = XOR(a,b), CARRY = AND(a,b)
        SUM, c2 = half_adder(s1, cin)      # SUM = XOR(s1,cin), c2 = AND(s1,cin)
        COUT = OR(c1, c2)
    closing the "honest scope-cut" half_adder()'s own docstring named
    (no majority-of-3 needed -- the two-half-adder construction sidesteps
    it entirely, same as real digital logic).

    THE REAL COMPOSITION SUBTLETY, found by tracing timing before writing
    any driving code, not discovered by trial and error: stage 1's SUM
    output (s1) is itself an xor_gate() -- it does NOT fire instantly. Per
    xor_gate's own docstring, its output (if it fires at all) fires
    `gap_ms` after a/b's real arrival time (the OR-helper's delayed
    delivery is what actually crosses XOR's threshold). Unlike synchronous
    digital logic, wiring two stages together does NOT mean "it settles on
    its own" -- stage 2 needs `cin` DELIVERED to it at s1's own earliest
    possible fire time (a/b's arrival time + xor_gate's gap_ms, 20.0 by
    default), or s1 and cin won't land close enough together for stage 2's
    AND/XOR helpers to combine them correctly.

    This function wires the wants once; it does NOT drive/time the inputs
    for you (a, b, cin are caller-supplied NeuronRefs). The CALLER is
    responsible for stimulating `cin` at (a/b's stimulation time + 20.0ms)
    -- see this file's own self-test for the real, executed choreography.
    This is a disclosed, real property of composing delay-based bricks,
    not an oversight: every brick has its own real propagation latency,
    and anything downstream has to be aligned to it, same as a real
    physical circuit constrains when you can safely sample its output."""
    s1, c1 = half_adder(net, a, b, f"{sum_name}_stage1_sum", f"{sum_name}_stage1_carry")
    s2, c2 = half_adder(net, s1, cin, sum_name, f"{sum_name}_stage2_carry")
    cout = or_gate(net, c1, c2, carry_name)
    return s2, cout


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


def _selftest_sequence_detector_is_order_sensitive() -> bool:
    """sequence_detector's whole point: FIRST-then-SECOND fires, but
    SECOND-then-FIRST at the exact same real gap does NOT -- proving this
    is genuine order sensitivity, not just a coincidence window."""
    def run(order_correct: bool) -> int:
        net = Net(refractory_ms=0)
        rt = net.build_live()
        FIRST = net.neuron("FIRST", threshold=50, leak=0)
        SECOND = net.neuron("SECOND", threshold=50, leak=0)
        sequence_detector(net, FIRST, SECOND, "OUT", gap_ms=15.0)
        if order_correct:
            rt.stimulate("FIRST", 1.0, 60.0)
            for t in range(2, 16):
                rt.tick(float(t))
            rt.stimulate("SECOND", 16.0, 60.0)
        else:
            rt.stimulate("SECOND", 1.0, 60.0)
            for t in range(2, 16):
                rt.tick(float(t))
            rt.stimulate("FIRST", 16.0, 60.0)
        for t in range(17, 32):
            rt.tick(float(t))
        return rt.neurons["OUT"].fire_count

    forward = run(order_correct=True)
    reversed_ = run(order_correct=False)
    ok = forward >= 1 and reversed_ == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] order sensitivity: FIRST-then-SECOND "
          f"(15ms gap) fires (fire_count={forward}); SECOND-then-FIRST at the SAME "
          f"15ms gap does NOT (fire_count={reversed_})")
    return ok


def _selftest_and_or_truth_tables() -> bool:
    """Real truth table for and_gate/or_gate over all 4 input combinations,
    each in a fresh net -- no assumptions, every cell actually run."""
    def run(builder, a_active: bool, b_active: bool) -> int:
        net = Net(refractory_ms=0)
        rt = net.build_live()
        A = net.neuron("A", threshold=50, leak=0)
        B = net.neuron("B", threshold=50, leak=0)
        builder(net, A, B, "OUT")
        if a_active:
            rt.stimulate("A", 1.0, 60.0)
        if b_active:
            rt.stimulate("B", 1.0, 60.0)
        return rt.neurons["OUT"].fire_count

    and_table = {combo: run(and_gate, *combo)
                 for combo in [(False, False), (True, False), (False, True), (True, True)]}
    or_table = {combo: run(or_gate, *combo)
                for combo in [(False, False), (True, False), (False, True), (True, True)]}

    and_ok = (and_table[(False, False)] == 0 and and_table[(True, False)] == 0 and
              and_table[(False, True)] == 0 and and_table[(True, True)] >= 1)
    or_ok = (or_table[(False, False)] == 0 and or_table[(True, False)] >= 1 and
             or_table[(False, True)] >= 1 and or_table[(True, True)] >= 1)
    ok = and_ok and or_ok
    print(f"  [{'PASS' if and_ok else 'FAIL'}] and_gate truth table: 00->{and_table[(False,False)]} "
          f"10->{and_table[(True,False)]} 01->{and_table[(False,True)]} 11->{and_table[(True,True)]}")
    print(f"  [{'PASS' if or_ok else 'FAIL'}] or_gate truth table:  00->{or_table[(False,False)]} "
          f"10->{or_table[(True,False)]} 01->{or_table[(False,True)]} 11->{or_table[(True,True)]} "
          f"(11 fires twice -- honest per-input-spike behavior, see or_gate's own docstring)")
    return ok


def _selftest_xor_and_half_adder_truth_tables() -> bool:
    """Real truth table for xor_gate and half_adder over all 4 input
    combinations. This is the actual proof the double-delivery bug
    described in xor_gate's docstring is fixed, not just reasoned about --
    the 11 case is the one that would wrongly fire without the refractory
    dedup fix."""
    def run_xor(a_active: bool, b_active: bool) -> int:
        net = Net(refractory_ms=5)  # xor_gate requires refractory_ms > 0
        rt = net.build_live()
        A = net.neuron("A", threshold=50, leak=0)
        B = net.neuron("B", threshold=50, leak=0)
        xor_gate(net, A, B, "OUT")
        if a_active:
            rt.stimulate("A", 1.0, 60.0)
        if b_active:
            rt.stimulate("B", 1.0, 60.0)
        for t in range(2, 25):
            rt.tick(float(t))
        return rt.neurons["OUT"].fire_count

    xor_table = {combo: run_xor(*combo)
                 for combo in [(False, False), (True, False), (False, True), (True, True)]}
    xor_ok = (xor_table[(False, False)] == 0 and xor_table[(True, False)] >= 1 and
              xor_table[(False, True)] >= 1 and xor_table[(True, True)] == 0)
    print(f"  [{'PASS' if xor_ok else 'FAIL'}] xor_gate truth table: 00->{xor_table[(False,False)]} "
          f"10->{xor_table[(True,False)]} 01->{xor_table[(False,True)]} 11->{xor_table[(True,True)]} "
          f"(11 must be 0 -- the exact case the double-delivery bug would have broken)")

    def run_adder(a_active: bool, b_active: bool) -> tuple:
        net = Net(refractory_ms=5)
        rt = net.build_live()
        A = net.neuron("A", threshold=50, leak=0)
        B = net.neuron("B", threshold=50, leak=0)
        half_adder(net, A, B, "SUM", "CARRY")
        if a_active:
            rt.stimulate("A", 1.0, 60.0)
        if b_active:
            rt.stimulate("B", 1.0, 60.0)
        for t in range(2, 25):
            rt.tick(float(t))
        return rt.neurons["SUM"].fire_count, rt.neurons["CARRY"].fire_count

    adder_table = {combo: run_adder(*combo)
                   for combo in [(False, False), (True, False), (False, True), (True, True)]}
    expected = {(False, False): (0, 0), (True, False): (1, 0),
                (False, True): (1, 0), (True, True): (0, 1)}
    adder_ok = all((adder_table[c][0] >= 1) == (expected[c][0] == 1) and
                    (adder_table[c][1] >= 1) == (expected[c][1] == 1) for c in expected)
    print(f"  [{'PASS' if adder_ok else 'FAIL'}] half_adder truth table (sum,carry): "
          f"00->{adder_table[(False,False)]} 10->{adder_table[(True,False)]} "
          f"01->{adder_table[(False,True)]} 11->{adder_table[(True,True)]} "
          f"(expect 00->(0,0) 10->(1,0) 01->(1,0) 11->(0,1))")
    return xor_ok and adder_ok


def _selftest_full_adder_truth_table() -> bool:
    """Real truth table for full_adder over all 8 (a,b,cin) combinations,
    each in a fresh net -- this is the real proof the two-stage timing
    choreography described in full_adder's own docstring actually works,
    not just that it was reasoned through correctly on paper."""
    XOR_GATE_LATENCY_MS = 20.0  # must match xor_gate()'s default gap_ms

    def run(a_active: bool, b_active: bool, cin_active: bool) -> tuple:
        net = Net(refractory_ms=5)
        rt = net.build_live()
        A = net.neuron("A", threshold=50, leak=0)
        B = net.neuron("B", threshold=50, leak=0)
        CIN = net.neuron("CIN", threshold=50, leak=0)
        full_adder(net, A, B, CIN, "SUM", "COUT")
        if a_active:
            rt.stimulate("A", 1.0, 60.0)
        if b_active:
            rt.stimulate("B", 1.0, 60.0)
        stage2_time = 1.0 + XOR_GATE_LATENCY_MS  # 21.0 -- s1's earliest possible fire time
        for t in range(2, int(stage2_time)):
            rt.tick(float(t))
        if cin_active:
            rt.stimulate("CIN", stage2_time, 60.0)   # aligned with s1's own arrival
        else:
            rt.tick(stage2_time)   # still flush any pending stage-1 delivery on time
        for t in range(int(stage2_time) + 1, int(stage2_time) + 25):
            rt.tick(float(t))
        return rt.neurons["SUM"].fire_count, rt.neurons["COUT"].fire_count

    combos = [(a, b, c) for a in (False, True) for b in (False, True) for c in (False, True)]
    table = {combo: run(*combo) for combo in combos}
    expected = {
        (False, False, False): (0, 0), (False, False, True): (1, 0),
        (False, True, False): (1, 0), (False, True, True): (0, 1),
        (True, False, False): (1, 0), (True, False, True): (0, 1),
        (True, True, False): (0, 1), (True, True, True): (1, 1),
    }
    ok = all((table[c][0] >= 1) == (expected[c][0] == 1) and
             (table[c][1] >= 1) == (expected[c][1] == 1) for c in expected)
    for combo in combos:
        a, b, c = (int(x) for x in combo)
        got = table[combo]
        exp = expected[combo]
        row_ok = (got[0] >= 1) == (exp[0] == 1) and (got[1] >= 1) == (exp[1] == 1)
        print(f"    [{'PASS' if row_ok else 'FAIL'}] {a} {b} cin={c} -> sum,cout={got} (expect {exp})")
    print(f"  [{'PASS' if ok else 'FAIL'}] full_adder: all 8 rows of the real truth table "
          f"(the actual proof the two-stage timing choreography works)")
    return ok


if __name__ == "__main__":
    print("=" * 78)
    print("  PYSPIKE BRICKS -- self-test (native Synapse.delay_ms path)")
    print("=" * 78)
    print("-- coincidence_detector --")
    results = [
        _selftest_neither_input_alone_fires(),
        _selftest_docstring_math_one_leak_tick_apart(),
        _selftest_delay_compensated_skewed_sources_coincide(),
        _selftest_without_delay_compensation_same_skew_does_not_fire(),
    ]
    print("-- sequence_detector --")
    results.append(_selftest_sequence_detector_is_order_sensitive())
    print("-- and_gate / or_gate --")
    results.append(_selftest_and_or_truth_tables())
    print("-- xor_gate / half_adder --")
    results.append(_selftest_xor_and_half_adder_truth_tables())
    print("-- full_adder --")
    results.append(_selftest_full_adder_truth_table())
    print("=" * 78)
    print(f"  {sum(results)}/{len(results)} passed")
    if not all(results):
        raise SystemExit(1)
