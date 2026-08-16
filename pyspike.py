#!/usr/bin/env python
"""
pyspike.py — Python fused with the Spikeling DSL.

The problem this fixes is real and already visible in this project:
spiking_scheduler.py, test_soft_conflicts.py, and test_incremental_scheduling.py
all build networks by f-string-formatting ".spk" TEXT and then re-parsing it
with SpikelingParser -- every single wave, every single arrival. That's a
string-generate-then-regex-parse round trip standing in for what should just
be Python: a loop building neurons and synapses directly.

pyspike.py is a thin builder that constructs the exact same SpikelingAST the
.spk parser produces (compiler.SpikelingAST / NeuronDef / ConnectionDef /
ActionDef) -- so anything built with it is a real Spikeling network, gets the
existing Python runtime for free, AND gets the existing C code generator for
free (CCodeGenerator only needs an AST, it doesn't care how the AST was
built). Nothing about the runtime or hardware backend changes; this only
replaces the TEXT FORMAT with real Python for programmatic construction.

    net = Net(refractory_ms=40)
    S_Work      = net.neuron("S_Work", threshold=50, leak=0)
    Implementer = net.neuron("Implementer", threshold=50, leak=0)

    S_Work >> Implementer                    # excitatory, weight=1.0 (sugar)
    S_Ambiguous.to(Implementer, weight=-2.0) # inhibitory, explicit weight

    @net.action(Implementer)
    def cmd_implement():
        print("implementing!")

    rt = net.build()          # -> a real SpikelingRuntime, handlers already wired
    rt.stimulate("S_Work", 1.0, 60.0)

Programmatic construction (the actual point -- real Python control flow
instead of string-templating a .spk file) reads like this, no regex, no
re-parsing per wave:

    net = Net()
    agents = {name: net.neuron(name, threshold=50, leak=0) for name in names}
    for a, b in conflicting_pairs:
        agents[a].to(agents[b], weight=-3.0)
        agents[b].to(agents[a], weight=-3.0)
    rt = net.build()

    python pyspike.py    # self-test: proves this matches .spk-parsed behavior
                          # bit-for-bit on a real network, then benchmarks the
                          # string-round-trip cost this replaces
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
from compiler.compiler import SpikelingAST, NeuronDef, ConnectionDef, ActionDef  # noqa: E402
from runtime.runtime import SpikelingRuntime, NeuronState, Synapse               # noqa: E402


class NeuronRef:
    """A handle to a neuron already added to a Net. Exists so network
    construction reads as real Python (operators, method chaining) instead
    of passing bare name strings around."""

    def __init__(self, net: "Net", name: str):
        self.net = net
        self.name = name

    def to(self, dst: "NeuronRef", weight: float = 1.0, delay_ms: float = 0.0) -> "NeuronRef":
        """Explicit weighted connect -- the one you reach for when weight
        isn't the default 1.0 (inhibitory synapses, graded severities,
        anything computed rather than constant).

        delay_ms: real per-synapse transmission delay, fused in from a
        direct comparison against Sandia's Fugu framework (see
        Documents/neuromorphic-survey/fugu_vs_spikeling.md) -- this
        runtime had no delay primitive before. Defaults to 0.0 (instant,
        exactly the pre-existing behavior); set it to build delay-based
        circuits like coincidence detectors (see brick helpers in
        pyspike_bricks.py)."""
        self.net.connect(self, dst, weight=weight, delay_ms=delay_ms)
        return dst

    def inhibits(self, dst: "NeuronRef", weight: float = -2.0, delay_ms: float = 0.0) -> "NeuronRef":
        """Same as .to() but the negative default and the name make
        inhibitory wiring self-documenting at the call site."""
        assert weight < 0, "inhibits() expects a negative weight -- use .to() for excitatory"
        return self.to(dst, weight=weight, delay_ms=delay_ms)

    def __rshift__(self, dst: "NeuronRef") -> "NeuronRef":
        """`a >> b` is sugar for the common case: excitatory, weight=1.0.
        Returns dst so `a >> b >> c` chains (a->b, b->c) the way you'd read
        it. Use .to()/.inhibits() when the weight isn't the default."""
        return self.to(dst, weight=1.0)

    def __repr__(self) -> str:
        return f"NeuronRef({self.name!r})"


class Net:
    """Builds a SpikelingAST via real Python instead of a .spk text file.
    .build() hands you a live SpikelingRuntime with handlers already wired --
    same runtime class the .spk parser feeds, so nothing downstream changes."""

    def __init__(self, refractory_ms: int = 0):
        self.ast = SpikelingAST(refractory_ms=refractory_ms)
        self._names: set = set()
        self._handlers: dict = {}
        self._action_counter = 0
        self._live_rt: SpikelingRuntime = None   # set by build_live(); see its docstring

    def neuron(self, name: str, threshold: int = 50, leak: int = 0,
              type: str = "LIF", freq_hz: float = None, damping: float = None,
              coupling: float = None) -> NeuronRef:
        if name in self._names:
            raise ValueError(f"duplicate neuron name '{name}'")
        self._names.add(name)
        self.ast.neurons.append(NeuronDef(
            name=name, threshold=int(threshold), leak=int(leak),
            neuron_type=type, freq_hz=freq_hz, damping=damping, coupling=coupling,
        ))
        if self._live_rt is not None:
            # LIVE mode: this neuron must exist in the runtime immediately,
            # not just in the AST -- see build_live()'s docstring for why.
            self._live_rt.neurons[name] = NeuronState(
                name=name, threshold=float(threshold), leak=float(leak))
        return NeuronRef(self, name)

    def connect(self, src, dst, weight: float = 1.0, delay_ms: float = 0.0) -> None:
        src_name = src.name if isinstance(src, NeuronRef) else src
        dst_name = dst.name if isinstance(dst, NeuronRef) else dst
        self.ast.connections.append(ConnectionDef(
            src=src_name, dst=dst_name, weight=float(weight), delay_ms=float(delay_ms)))
        if self._live_rt is not None:
            self._live_rt.synapses.append(Synapse(
                src=src_name, dst=dst_name, weight=float(weight), delay_ms=float(delay_ms)))

    def action(self, neuron: NeuronRef):
        """Decorator: attach a real Python function directly to a neuron's
        firing event. No command-string indirection to manage by hand -- an
        internal command id is generated and wired to the handler for you,
        collapsing the .spk model's three-step (neuron -> command string ->
        handler dict) into one decorator.

        BUG FOUND AND FIXED (2026-07-17): in build_live() mode this used to
        only update the AST + self._handlers, never the live runtime's own
        .actions/.handlers dicts -- so an action attached after build_live()
        (the normal order: build the empty live runtime, THEN grow it) never
        actually fired, silently. .neuron()/.connect() already mutated the
        live runtime immediately; .action() has to do the same for the same
        reason -- there is no batch step in live mode to pick this up later."""
        def decorator(fn):
            self._action_counter += 1
            cmd = f"__pyspike_cmd_{self._action_counter}_{neuron.name}__"
            self.ast.actions.append(ActionDef(neuron=neuron.name, command=cmd))
            self._handlers[cmd] = fn
            if self._live_rt is not None:
                self._live_rt.actions[neuron.name] = cmd
                self._live_rt.handlers[cmd] = fn
            return fn
        return decorator

    def build(self) -> SpikelingRuntime:
        """Validate + construct the runtime, same as compiler.compile_file()
        does for a .spk file -- reuses the parser's own validator so
        dangling connect/action references are caught the same way."""
        self._validate()
        rt = SpikelingRuntime(self.ast)
        for cmd, fn in self._handlers.items():
            rt.register_handler(cmd, fn)
        return rt

    def build_live(self) -> SpikelingRuntime:
        """Like build(), but returns a runtime you can keep GROWING after the
        fact: every .neuron()/.connect() call made on this Net AFTER calling
        build_live() also mutates the returned runtime immediately, in place
        -- no reparse, no rebuild, no batch step. This is for genuinely
        incremental/streaming use (agents/neurons arriving over time), as
        opposed to build()'s batch model (define the whole network, then run
        it once).

        IMPORTANT -- inhibition in this runtime is EVENT-DRIVEN (a synapse's
        effect is only applied the instant its source neuron FIRES, in
        runtime.py's _fire() propagation loop). So a synapse you .connect()
        AFTER its source already fired will NOT retroactively apply that past
        firing -- the event already happened, propagation doesn't replay. If
        your use case needs a late-arriving inhibitory edge to a neuron that
        may have already fired to still take effect, call
        reconcile_late_edge() right after connecting it (see its docstring).
        This isn't a pyspike limitation to work around -- it's what
        'incremental' has to mean for an event-driven substrate, and it's
        pyspike's job to make that reconciliation easy to do correctly, not
        to hide that it's needed."""
        self._live_rt = SpikelingRuntime.__new__(SpikelingRuntime)
        self._live_rt.neurons = {}
        self._live_rt.resonators = {}
        self._live_rt.synapses = []
        self._live_rt.actions = {}
        self._live_rt.refractory_ms = float(self.ast.refractory_ms)
        self._live_rt.learner = None
        self._live_rt.handlers = dict(self._handlers)
        self._live_rt._spike_log = []
        # Real bug caught by this project's own delay-feature regression
        # test: build_live() constructs the runtime via __new__(), never
        # calling __init__(), so any attribute __init__() sets (like the
        # real synaptic-delay pending-delivery queue added alongside
        # this) has to be set here too, by hand, or live-mode runtimes
        # crash the instant delay code touches it.
        self._live_rt._pending_deliveries = []
        return self._live_rt

    def reconcile_late_edge(self, src: NeuronRef, dst: NeuronRef, weight: float) -> None:
        """After connecting src -> dst on a build_live() runtime, call this to
        apply src's ALREADY-PAST firing to dst, if src already fired before
        this edge existed. Mirrors exactly what _fire()'s own propagation
        step does for edges that existed at firing time -- see build_live()'s
        docstring for why this can't happen automatically."""
        if self._live_rt is None:
            raise RuntimeError("reconcile_late_edge() requires build_live() first")
        src_n = self._live_rt.neurons[src.name]
        dst_n = self._live_rt.neurons[dst.name]
        if src_n.fire_count > 0:
            dst_n.membrane_potential = max(-dst_n.threshold, dst_n.membrane_potential + weight * 50.0)

    def _validate(self) -> None:
        names = self._names
        for c in self.ast.connections:
            if c.src not in names:
                raise NameError(f"connect: unknown neuron '{c.src}'")
            if c.dst not in names:
                raise NameError(f"connect: unknown neuron '{c.dst}'")
        for a in self.ast.actions:
            if a.neuron not in names:
                raise NameError(f"action: unknown neuron '{a.neuron}'")


# ─────────────────────────────────────────────────────────────────────────────
def _selftest_matches_spk_text() -> None:
    """Prove pyspike builds a network that behaves IDENTICALLY to the
    equivalent hand-written .spk text -- not just 'looks plausible', an
    actual side-by-side spike-for-spike comparison."""
    from compiler.compiler import SpikelingParser

    spk_text = """
neuron A threshold=50 leak=0 type=LIF
neuron B threshold=50 leak=0 type=LIF
neuron C threshold=50 leak=0 type=LIF
connect A -> B weight=1.0
connect B -> C weight=-2.0
refractory=0ms
""".strip()
    rt_text = SpikelingRuntime(SpikelingParser().parse(spk_text))

    net = Net(refractory_ms=0)
    A = net.neuron("A", threshold=50, leak=0)
    B = net.neuron("B", threshold=50, leak=0)
    C = net.neuron("C", threshold=50, leak=0)
    A >> B
    B.inhibits(C, weight=-2.0)
    rt_py = net.build()

    for t, (name, drive) in enumerate([("A", 60.0), ("A", 60.0), ("B", 60.0), ("C", 60.0)], start=1):
        cmd_text = rt_text.stimulate(name, float(t), drive)
        cmd_py = rt_py.stimulate(name, float(t), drive)
        assert cmd_text == cmd_py, f"handler dispatch diverged at t={t}: {cmd_text!r} vs {cmd_py!r}"

    for name in ("A", "B", "C"):
        ft, fp = rt_text.neurons[name].fire_count, rt_py.neurons[name].fire_count
        assert ft == fp, f"fire_count diverged for {name}: text={ft} py={fp}"
    print("  [PASS] pyspike-built network matches .spk-text-parsed network spike-for-spike")


def _benchmark_vs_string_roundtrip(n_agents: int = 30, n_trials: int = 200) -> None:
    """Measure the actual cost pyspike removes: f-string generation + regex
    re-parsing of a .spk network, vs building the same network directly."""
    from compiler.compiler import SpikelingParser
    import random
    rng = random.Random(0)
    names = [f"agent_{i}" for i in range(n_agents)]
    edges = [(names[i], names[j], -3.0) for i in range(n_agents) for j in range(i + 1, n_agents)
             if rng.random() < 0.15]

    def build_via_text():
        lines = [f"neuron {n} threshold=50 leak=0 type=LIF" for n in names]
        for a, b, w in edges:
            lines.append(f"connect {a} -> {b} weight={w}")
            lines.append(f"connect {b} -> {a} weight={w}")
        return SpikelingRuntime(SpikelingParser().parse("\n".join(lines)))

    def build_via_pyspike():
        net = Net()
        refs = {n: net.neuron(n, threshold=50, leak=0) for n in names}
        for a, b, w in edges:
            refs[a].to(refs[b], weight=w)
            refs[b].to(refs[a], weight=w)
        return net.build()

    t0 = time.perf_counter()
    for _ in range(n_trials):
        build_via_text()
    text_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_trials):
        build_via_pyspike()
    pyspike_s = time.perf_counter() - t0

    speedup = text_s / pyspike_s if pyspike_s else float("inf")
    print(f"  {n_agents} agents, {len(edges)*2} synapses, {n_trials} builds:")
    print(f"    string round-trip (f-string + regex parse): {text_s*1000:.1f}ms total "
          f"({text_s/n_trials*1000:.3f}ms/build)")
    print(f"    pyspike direct construction:                {pyspike_s*1000:.1f}ms total "
          f"({pyspike_s/n_trials*1000:.3f}ms/build)")
    print(f"    pyspike is {speedup:.1f}x faster to CONSTRUCT the network "
          f"(this is build cost only, not stimulate/spike-propagation cost)")


def _selftest_live_incremental() -> None:
    """Prove build_live() + reconcile_late_edge() behaves correctly: a late
    inhibitory edge to an ALREADY-FIRED neighbor must still veto the
    newcomer, and the raw-NeuronState/Synapse equivalent (what
    test_incremental_scheduling.py did by hand before this existed) must
    match it exactly."""
    net = Net()
    rt = net.build_live()                         # empty live runtime, grows from here
    A = net.neuron("A", threshold=50, leak=0)
    rt.stimulate("A", 1.0, 60.0)
    assert rt.neurons["A"].fire_count == 1, "A should have fired before B ever existed"

    B = net.neuron("B", threshold=50, leak=0)   # arrives AFTER A already fired
    A.inhibits(B, weight=-3.0)                   # late edge -- A's firing is in the past
    net.reconcile_late_edge(A, B, weight=-3.0)    # must be applied manually
    rt.stimulate("B", 2.0, 60.0)
    assert rt.neurons["B"].fire_count == 0, (
        "B should have been vetoed by A's past firing after reconciliation")

    # control: WITHOUT reconcile_late_edge, B should fire (proves the
    # reconciliation call is doing real work, not a no-op)
    net2 = Net()
    rt2 = net2.build_live()
    A2 = net2.neuron("A", threshold=50, leak=0)
    rt2.stimulate("A", 1.0, 60.0)
    B2 = net2.neuron("B", threshold=50, leak=0)
    A2.inhibits(B2, weight=-3.0)
    rt2.stimulate("B", 2.0, 60.0)
    assert rt2.neurons["B"].fire_count == 1, (
        "control failed: B should fire when the late edge is NOT reconciled")

    print("  [PASS] build_live() + reconcile_late_edge() correctly vetoes a late "
          "arrival against an already-fired neighbor (and the un-reconciled control fires)")


def _selftest_live_actions() -> None:
    """Prove @net.action() actually fires in build_live() mode -- found broken
    while building the self-growing-network experiment: action() only ever
    updated the AST + self._handlers, never the live runtime's own
    .actions/.handlers dicts, so an action attached in the normal live-mode
    order (build_live() first, then grow) silently never fired. This also
    proves a handler can spawn a new neuron+synapse that fires in the SAME
    cascade (handlers run synchronously before propagation -- see runtime.py
    _fire()), which is the mechanism the self-growing-network experiment
    depends on."""
    net = Net()
    rt = net.build_live()
    events = []

    seed = net.neuron("seed", threshold=50, leak=0)

    @net.action(seed)
    def _on_seed():
        events.append("seed")
        child = net.neuron("child", threshold=50, leak=0)
        seed.to(child, weight=1.2)   # 1.2*50=60 >= threshold=50

        @net.action(child)
        def _on_child():
            events.append("child")

    rt.stimulate("seed", 1.0, 60.0)
    assert events == ["seed", "child"], f"expected cascade [seed, child], got {events}"
    assert rt.neurons["child"].fire_count == 1, "child should have fired in the same cascade"
    print("  [PASS] @net.action() fires correctly in build_live() mode, INCLUDING a handler-"
          "spawned child neuron that fires in the same cascade (one stimulate() call)")

    print("  [PASS] build_live() + reconcile_late_edge() correctly vetoes a late "
          "arrival against an already-fired neighbor (and the un-reconciled control fires)")


if __name__ == "__main__":
    print("=" * 78)
    print("  PYSPIKE SELF-TEST")
    print("=" * 78)
    _selftest_matches_spk_text()
    _selftest_live_incremental()
    _selftest_live_actions()
    print()
    _benchmark_vs_string_roundtrip()
