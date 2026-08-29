#!/usr/bin/env python
"""
test_synaptic_delay.py -- real, dedicated test for the synaptic-delay
primitive in core/runtime/runtime.py + the .spk DSL's delay= grammar
addition in core/compiler/compiler.py (2026-08-29).

PRIOR STATE, confirmed via git log/grep before writing anything here (not
assumed): the runtime-level delay primitive (Synapse.delay_ms, the
_pending_deliveries queue, _deliver_spike()/_flush_pending_deliveries())
was NOT dead code and NOT missing -- it was already built, wired, and
covered by real tests as of commit 108ab13 (2026-08-16) and 7d207af
(same day), verified here by running pyspike_bricks.py's own self-test
(8/8 pass) before touching anything. The ONLY real gap found was one
layer up: core/compiler/compiler.py's CONNECT_RE never captured a
`delay=` token, so a .spk TEXT file had no syntax to express a delayed
connection at all -- only the Python builder API (pyspike.py's
NeuronRef.to(dst, delay_ms=...)) could reach the existing runtime feature.
This test suite covers BOTH layers: the runtime queue mechanics
(pre-existing, re-verified here at the unit level rather than only via
the higher-level coincidence_detector brick) and the new DSL grammar
addition (delay= parses, defaults to 0.0, and round-trips into a real
Synapse.delay_ms on the runtime built from parsed .spk text).

Predictions, pre-registered before running anything below:
  1. A delayed synapse's effect does NOT land on the source's own fire
     tick -- membrane potential downstream is unchanged immediately
     after the source fires, and only changes once current_time_ms
     reaches the scheduled delivery time.
  2. Two synapses from the same source with different delay_ms values
     land at two DIFFERENT times, matching each one's own delay exactly.
  3. delay_ms=0.0 (explicit, and the "no delay=" DSL default) delivers
     INSTANTLY inside the source's own _fire() call, identical to the
     pre-delay-feature behavior -- no regression for the default path.
  4. Two different delayed synapses independently scheduled to land on
     the SAME future tick BOTH deliver when that tick is reached -- the
     pending-delivery queue does not drop or overwrite either one.
  5. `connect A -> B weight=<f> delay=<f>` parses into
     ConnectionDef.delay_ms; `connect A -> B weight=<f>` (no delay=)
     still parses and defaults to delay_ms=0.0 (backward compatible with
     every .spk file written before this grammar addition).
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "core"))
from compiler.compiler import SpikelingParser, SpikelingAST, NeuronDef, ConnectionDef, ActionDef  # noqa: E402
from runtime.runtime import SpikelingRuntime, Synapse  # noqa: E402

PASS = 0
FAIL = 0


def check(label, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"PASS  {label}")
    else:
        FAIL += 1
        print(f"FAIL  {label}")


def make_ast(connections, refractory_ms=0):
    """Two neurons, A and B, threshold=50 leak=0 (no decay -- isolates
    delay timing from leak timing so each test measures exactly one
    thing), wired by the caller-supplied connection list."""
    ast = SpikelingAST(
        neurons=[
            NeuronDef(name="A", threshold=50, leak=0),
            NeuronDef(name="B", threshold=1_000_000, leak=0),  # never fires on its own -- pure probe
            NeuronDef(name="C", threshold=50, leak=0),
        ],
        connections=connections,
        refractory_ms=refractory_ms,
    )
    return ast


# ─────────────────────────────────────────────────────────────────────────
# 1. Delayed synapse effect does NOT land on the source's own fire tick
# ─────────────────────────────────────────────────────────────────────────
print("-- prediction 1: delayed effect lands LATER, not on the same tick --")
ast1 = make_ast([ConnectionDef(src="A", dst="B", weight=1.0, delay_ms=10.0)])
rt1 = SpikelingRuntime(ast1)
rt1.stimulate("A", current_time_ms=0.0, drive=60.0)   # A fires at t=0 (60 >= threshold 50)
check("A actually fired at t=0 (precondition for this test)",
      rt1.neurons["A"].fire_count == 1)
check("B's potential is UNCHANGED immediately after A fires (delivery not yet due)",
      rt1.neurons["B"].membrane_potential == 0.0)
rt1.tick(5.0)   # before the scheduled delivery (t=10)
check("B's potential is still unchanged at t=5 (delivery due at t=10, not yet)",
      rt1.neurons["B"].membrane_potential == 0.0)
rt1.tick(10.0)  # delivery due now
check("B's potential updates at t=10, exactly when delay_ms=10 says it should "
      f"(got {rt1.neurons['B'].membrane_potential})",
      rt1.neurons["B"].membrane_potential == 1.0 * 50.0)


# ─────────────────────────────────────────────────────────────────────────
# 2. Two different delays from the same source land at different times
# ─────────────────────────────────────────────────────────────────────────
print("-- prediction 2: different delay_ms values land at different times --")
ast2 = SpikelingAST(
    neurons=[
        NeuronDef(name="A", threshold=50, leak=0),
        NeuronDef(name="FAST", threshold=1_000_000, leak=0),
        NeuronDef(name="SLOW", threshold=1_000_000, leak=0),
    ],
    connections=[
        ConnectionDef(src="A", dst="FAST", weight=1.0, delay_ms=5.0),
        ConnectionDef(src="A", dst="SLOW", weight=1.0, delay_ms=20.0),
    ],
)
rt2 = SpikelingRuntime(ast2)
rt2.stimulate("A", current_time_ms=0.0, drive=60.0)
rt2.tick(5.0)
check("FAST (delay=5) has already received its spike at t=5",
      rt2.neurons["FAST"].membrane_potential == 50.0)
check("SLOW (delay=20) has NOT received its spike yet at t=5",
      rt2.neurons["SLOW"].membrane_potential == 0.0)
rt2.tick(20.0)
check("SLOW (delay=20) has received its spike by t=20",
      rt2.neurons["SLOW"].membrane_potential == 50.0)
check("FAST is untouched by SLOW's later delivery (still exactly one delivery's worth)",
      rt2.neurons["FAST"].membrane_potential == 50.0)


# ─────────────────────────────────────────────────────────────────────────
# 3. delay_ms=0.0 delivers instantly -- identical to pre-delay-feature path
# ─────────────────────────────────────────────────────────────────────────
print("-- prediction 3: delay_ms=0.0 behaves identically to the original instant-delivery path (no regression) --")
ast3 = make_ast([ConnectionDef(src="A", dst="B", weight=1.0, delay_ms=0.0)])
rt3 = SpikelingRuntime(ast3)
rt3.stimulate("A", current_time_ms=0.0, drive=60.0)
check("B's potential updates IMMEDIATELY (same call, no tick() needed) when delay_ms=0.0",
      rt3.neurons["B"].membrane_potential == 50.0)

# Same check again, but going through the Synapse dataclass default directly
# (no delay_ms argument at all -- the literal old call signature).
ast3b = SpikelingAST(
    neurons=[NeuronDef(name="A", threshold=50, leak=0), NeuronDef(name="B", threshold=1_000_000, leak=0)],
    connections=[ConnectionDef(src="A", dst="B", weight=1.0)],  # delay_ms omitted entirely
)
rt3b = SpikelingRuntime(ast3b)
rt3b.stimulate("A", current_time_ms=0.0, drive=60.0)
check("ConnectionDef with delay_ms omitted entirely also delivers instantly (dataclass default holds)",
      rt3b.neurons["B"].membrane_potential == 50.0)


# ─────────────────────────────────────────────────────────────────────────
# 4. Two different delayed synapses scheduled for the SAME future tick both land
# ─────────────────────────────────────────────────────────────────────────
print("-- prediction 4: two independently-scheduled deliveries landing on the SAME tick both arrive (no queue data loss) --")
ast4 = SpikelingAST(
    neurons=[
        NeuronDef(name="A", threshold=50, leak=0),
        NeuronDef(name="C", threshold=50, leak=0),
        NeuronDef(name="D", threshold=1_000_000, leak=0),
    ],
    connections=[
        ConnectionDef(src="A", dst="D", weight=1.0, delay_ms=10.0),  # A fires t=0 -> due at t=10
        ConnectionDef(src="C", dst="D", weight=1.0, delay_ms=5.0),   # C fires t=5 -> due at t=10
    ],
)
rt4 = SpikelingRuntime(ast4)
rt4.stimulate("A", current_time_ms=0.0, drive=60.0)   # scheduled delivery #1 @ t=10
rt4.tick(5.0)
rt4.stimulate("C", current_time_ms=5.0, drive=60.0)   # scheduled delivery #2 @ t=10 -- same tick as #1
check("D's potential still 0 at t=5 (neither delivery due yet)",
      rt4.neurons["D"].membrane_potential == 0.0)
rt4.tick(10.0)   # both deliveries due on this exact tick
check("D received BOTH deliveries at t=10 -- two separate 50.0 contributions, not one lost "
      f"(got {rt4.neurons['D'].membrane_potential}, expected 100.0)",
      rt4.neurons["D"].membrane_potential == 100.0)
check("the pending-delivery queue is empty after both same-tick deliveries flush "
      f"(len={len(rt4._pending_deliveries)})",
      len(rt4._pending_deliveries) == 0)


# ─────────────────────────────────────────────────────────────────────────
# 5. DSL grammar: delay= parses; omitted delay= still defaults to 0.0
# ─────────────────────────────────────────────────────────────────────────
print("-- prediction 5: .spk text DSL delay= grammar addition --")
parser = SpikelingParser()

src_with_delay = """
neuron A threshold=50 leak=0
neuron B threshold=50 leak=0
connect A -> B weight=0.75 delay=12.5
action B -> [FIRE_B]
"""
ast_delay = parser.parse(src_with_delay)
check("connect ... delay=12.5 parses into ConnectionDef.delay_ms == 12.5",
      ast_delay.connections[0].delay_ms == 12.5)
check("weight is still parsed correctly alongside delay=",
      ast_delay.connections[0].weight == 0.75)

src_no_delay = """
neuron A threshold=50 leak=0
neuron B threshold=50 leak=0
connect A -> B weight=0.75
action B -> [FIRE_B]
"""
ast_no_delay = parser.parse(src_no_delay)
check("connect line WITHOUT delay= (every pre-existing .spk file) still parses "
      "and defaults delay_ms to 0.0 -- backward compatible",
      ast_no_delay.connections[0].delay_ms == 0.0)

# End-to-end: parsed .spk text -> SpikelingRuntime -> real delayed delivery,
# proving the grammar addition actually reaches the runtime, not just the AST.
rt5 = SpikelingRuntime(ast_delay)
rt5.stimulate("A", current_time_ms=0.0, drive=60.0)
check("end-to-end: parsed-from-text delay=12.5 does NOT deliver on A's own fire tick",
      rt5.neurons["B"].membrane_potential == 0.0)
rt5.tick(12.5)
check("end-to-end: parsed-from-text delay=12.5 delivers exactly at t=12.5 "
      f"(got potential={rt5.neurons['B'].membrane_potential})",
      rt5.neurons["B"].membrane_potential == 0.75 * 50.0)

# Malformed connect lines must still be rejected the same way they always were.
try:
    parser.parse("neuron A threshold=50 leak=0\nneuron B threshold=50 leak=0\nconnect A -> B notweight=1\n")
    check("malformed connect line (no weight=) is still rejected with a SyntaxError", False)
except SyntaxError:
    check("malformed connect line (no weight=) is still rejected with a SyntaxError", True)


print(f"\n=== RESULT: {PASS} passed, {FAIL} failed, {PASS + FAIL} total ===")
print("OVERALL:", "PASS" if FAIL == 0 else "FAIL")
sys.exit(0 if FAIL == 0 else 1)
