"""
spikeling/runtime/runtime.py
============================
Spikeling Python Runtime

Executes a SpikelingAST produced by the compiler.
Handles:
  - Leaky Integrate-and-Fire (LIF) neuron dynamics
  - STDP (Spike-Timing Dependent Plasticity) learning
  - Spike propagation through weighted synapses
  - Action dispatch (maps spike events to named commands)
  - Cross-platform keyboard input (Windows + Unix)
  - Real-time benchmarking vs. traditional processing

Usage:
  from compiler.compiler import compile_file
  from runtime.runtime import SpikelingRuntime

  ast = compile_file("examples/sound_localizer.spk")
  rt  = SpikelingRuntime(ast)
  rt.run_interactive()
"""

import time
import math
import sys
import os
from typing import Optional, Callable
from dataclasses import dataclass, field


def _leak_toward_zero(potential: float, leak: float) -> float:
    """Move a membrane potential toward rest (0) by `leak`, from EITHER side.
    Positive potentials decay down exactly as the original code did; negative
    ones (created by INHIBITORY synapses, weight < 0) recover UP toward 0 rather
    than being snapped straight to 0 -- so hyperpolarization fades gradually and
    an inhibitory veto lasts a few ticks instead of one."""
    if potential > 0.0:
        return max(0.0, potential - leak)
    return min(0.0, potential + leak)


# ─────────────────────────────────────────────
#  Cross-platform keyboard input
# ─────────────────────────────────────────────

if sys.platform == "win32":
    import msvcrt

    def _kbhit() -> bool:
        return msvcrt.kbhit()

    def _getch() -> bytes:
        return msvcrt.getch()

else:
    import tty
    import termios
    import select

    def _kbhit() -> bool:
        return bool(select.select([sys.stdin], [], [], 0)[0])

    def _getch() -> bytes:
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            return ch.encode()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ─────────────────────────────────────────────
#  Neuron state (runtime, separate from AST)
# ─────────────────────────────────────────────

@dataclass
class NeuronState:
    name:               str
    threshold:          float
    leak:               float
    membrane_potential: float = 0.0
    last_spike_time:    float = float("-inf")
    fire_count:         int   = 0


@dataclass
class Synapse:
    src:    str
    dst:    str
    weight: float
    # Real per-synapse transmission delay, fused in from a direct
    # code-level comparison against Sandia National Labs' Fugu
    # framework (see Documents/neuromorphic-survey/fugu_vs_spikeling.md):
    # Fugu treats delay as core infrastructure (needed for coincidence
    # detection, temporal pattern recognition, phase-coded computation);
    # this runtime had NO delay primitive at all before this -- every
    # synapse delivered instantly within the firing neuron's own
    # _fire() call. Defaults to 0.0 (instant, delivered synchronously
    # inside _fire() exactly as before) so every existing .spk file and
    # pyspike script is 100% unaffected -- delay is opt-in, not a
    # behavior change to the default path.
    delay_ms: float = 0.0


@dataclass
class ResonatorState:
    """
    A damped harmonic oscillator tuned to `freq_hz`. Used for
    frequency-selective detection: driven by a continuous input signal,
    it only builds up amplitude when that signal contains energy near
    its own frequency -- unlike a LIF neuron, it isn't pulsed, it's
    continuously stepped with step(drive, dt).

    Validated in resonator-prototype/accuracy_benchmark.py: 99.2%
    detection accuracy vs ~65% for a naive raw-amplitude threshold, at
    telling a specific target frequency apart from distractor tones
    + noise. This is that same model, wired into the real DSL/runtime
    instead of living only in the standalone prototype script.
    """
    name:     str
    freq_hz:  float
    damping:  float
    coupling: float
    threshold: float = 0.0008  # RMS-amplitude level that counts as "detected" -- tune per use case
    gate_threshold: float = 0.00024  # |x| below this: skip the expensive energy multiply (see step())
    energy_time_constant: float = 0.0025  # seconds -- see step()'s docstring for why this
                                            # must be a TIME, not a fixed per-step decay rate
    x: float = 0.0
    v: float = 0.0
    energy_ema: float = 0.0   # exponential moving average of x^2, cheap stand-in for RMS energy

    @property
    def omega(self) -> float:
        return 2 * math.pi * self.freq_hz

    def step(self, drive: float, dt: float) -> bool:
        """Advance one timestep (symplectic Euler -- see resonator-prototype/
        resonator_bank.py for why plain Euler is numerically unstable at
        audio-range frequencies). Returns True the instant energy crosses
        `threshold` (edge-triggered, like a LIF neuron firing).

        AMPLITUDE GATING: the oscillator state (x, v) always has to be
        integrated -- a real signal could arrive at any tick, so the
        mechanics can't be skipped. But the energy_ema update's `x * x`
        term is the expensive part (a 64-bit multiply in the Verilog
        hardware backend), and most channels in a bank are sitting near
        zero amplitude most of the time (only the channels actually
        resonating with present content build any real x). When |x| is
        below `gate_threshold` -- below the noise floor that could ever
        plausibly cross `threshold` on this tick -- skip the multiply
        and just apply the EMA's own decay term (energy_ema *= 0.99,
        the cheap half of the original update). Validated to NOT regress
        detection accuracy: see resonator-prototype/accuracy_benchmark.py
        and benchmarks/BENCHMARKS.md entry #5 for the before/after
        comparison and the efficiency numbers this earns.

        SAMPLE-RATE BUG, FOUND AND FIXED: the energy EMA's decay rate
        must be derived from dt and `energy_time_constant`, not a fixed
        per-step constant. A fixed alpha=0.01 (as this used to be coded)
        implicitly bakes in whatever dt was used when that constant was
        chosen -- at dt=2.5e-5 (40kHz sample rate, used throughout the
        rest of this project), alpha=0.01 corresponds to a ~2.5ms
        averaging window. Run the SAME code at a different sample rate
        (e.g. 1MHz, needed to accurately resolve a 40kHz signal -- see
        a separate 40kHz resonator sample-rate test) without
        rescaling alpha, and the averaging window silently shrinks to
        ~0.1ms -- far shorter than this resonator's own settling time,
        which collapsed detection accuracy from ~99% to ~45% with 0%
        recall before this was caught. Deriving alpha = dt / time_constant
        each step keeps the averaging window's actual TIME duration
        constant regardless of sample rate, which is the only version of
        this that's correct at more than one specific dt."""
        omega = self.omega
        accel = -(omega ** 2) * self.x - 2 * self.damping * omega * self.v
        accel += self.coupling * drive
        self.v += accel * dt
        self.x += self.v * dt

        alpha = min(1.0, dt / self.energy_time_constant)

        # energy_ema tracks mean-square amplitude; compare its RMS
        # (sqrt) against `threshold` so the threshold is in the same
        # units as the signal amplitude itself, not amplitude-squared.
        was_above = math.sqrt(self.energy_ema) >= self.threshold
        if abs(self.x) >= self.gate_threshold:
            self.energy_ema += alpha * (self.x * self.x - self.energy_ema)
        else:
            self.energy_ema -= alpha * self.energy_ema  # decay only -- skip the x*x multiply
        now_above = math.sqrt(self.energy_ema) >= self.threshold

        return now_above and not was_above


# ─────────────────────────────────────────────
#  STDP Learning Rule
# ─────────────────────────────────────────────

# Default coupling-vs-frequency scaling when a .spk Resonator neuron
# doesn't specify an explicit `coupling=` value. Without this, a driven
# damped oscillator's steady-state amplitude falls off as ~1/omega^2, so
# higher-frequency resonators in the same bank respond far weaker than
# low-frequency ones for the same coupling constant -- this was measured
# directly in resonator-prototype/resonator_bank.py (a 1760Hz channel was
# invisible next to a 440Hz one before this fix). Scaling coupling by
# omega^2 cancels that out so every channel has comparable gain at its
# own resonance peak.
DEFAULT_RESONATOR_BASE_GAIN = 4.0e-4


class STDPLearner:
    """
    Spike-Timing Dependent Plasticity.

    If pre fires BEFORE OR AT THE SAME TIME AS post (dt >= 0): strengthen
    the synapse (LTP). If pre fires AFTER post (dt < 0): weaken the
    synapse (LTD).

    Weight change: Δw = rate × exp(-|dt| / tau) -- magnitude is greatest
    at the shortest interval in either direction (the exponential peaks
    at dt=0), which was already correct. What was wrong until 2026-08-29
    was the BRANCH the dt=0 case fell into: `if dt > 0` (strict) sent
    exact coincidence to the LTD branch, treating simultaneous pre/post
    firing -- arguably the strongest possible correlation signal two
    neurons can send -- as weakening instead of the maximum-strength
    strengthening it should be.

    Real, citable reference for the corrected behavior: Peter van der
    Made's foundational BrainChip patent, US8250011B2 ("Autonomous
    Learning Dynamic Artificial Neural Computing Device and Brain
    Inspired System"): "The synapse strength value increase is greatest
    at the shortest interval between the input pulse and the occurrence
    of an output pulse." Confirmed via git history (commit a5a2109,
    2026-06-22, this project's initial commit) that the strict `>` was
    never a deliberate design choice -- it shipped as-is on day one, was
    never revisited, and no comment anywhere discussed the dt=0 boundary
    case. Fixed to `>=` to match the reference behavior, not because the
    old value was arbitrary, but because there was no real, on-record
    reason to keep it.
    """

    def __init__(self, rate: float = 0.01, tau: float = 20.0):
        self.rate = rate
        self.tau  = tau

    def update(self, synapse: Synapse, dt: float) -> float:
        """Returns new weight after STDP update."""
        delta = self.rate * math.exp(-abs(dt) / self.tau)
        if dt >= 0:
            new_w = synapse.weight + delta        # LTP (dt==0 included, see class docstring)
        else:
            new_w = synapse.weight - delta * 0.5  # LTD (asymmetric)
        return max(0.0, min(1.0, new_w))


# ─────────────────────────────────────────────
#  Spikeling Runtime
# ─────────────────────────────────────────────

class SpikelingRuntime:
    """
    Executes a compiled Spikeling network.

    Attributes
    ----------
    neurons   : dict of name -> NeuronState
    synapses  : list of Synapse
    actions   : dict of neuron name -> command string
    learner   : STDPLearner (or None if learn not specified)
    handlers  : dict of command string -> Python callable
                Register your own callbacks with register_handler().
    """

    def __init__(self, ast):
        from compiler.compiler import SpikelingAST  # avoid circular at top level

        self.neurons: dict[str, NeuronState]       = {}
        self.resonators: dict[str, ResonatorState] = {}
        self.synapses: list[Synapse]         = []
        self.actions: dict[str, str]         = {}
        self.refractory_ms: float            = ast.refractory_ms
        self.learner: Optional[STDPLearner]  = None
        self.handlers: dict[str, Callable]   = {}
        self._spike_log: list[tuple]         = []   # (time, neuron_name)
        # Real delayed-delivery queue: (delivery_time_ms, synapse, source_fire_time).
        # Only ever populated by synapses with delay_ms > 0 -- see
        # _fire()'s propagation loop and _flush_pending_deliveries()
        # below. A plain list, not a heap: real networks built with
        # this runtime (kernel schedulers, sensor grids) run at a scale
        # where a handful of pending deliveries at once is the norm,
        # not thousands -- linear scan is simpler to read/debug and
        # its cost is negligible at that scale.
        self._pending_deliveries: list[tuple[float, Synapse, float]] = []

        # Build neuron states -- LIF/Izhikevich/AdEx go in self.neurons,
        # Resonator neurons go in self.resonators (different dynamics:
        # continuously driven, not pulsed/refractory).
        for n in ast.neurons:
            if n.neuron_type == "Resonator":
                coupling = n.coupling
                if coupling is None:
                    omega = 2 * math.pi * n.freq_hz
                    coupling = DEFAULT_RESONATOR_BASE_GAIN * (omega ** 2)
                self.resonators[n.name] = ResonatorState(
                    name     = n.name,
                    freq_hz  = n.freq_hz,
                    damping  = n.damping,
                    coupling = coupling,
                )
            else:
                self.neurons[n.name] = NeuronState(
                    name      = n.name,
                    threshold = float(n.threshold),
                    leak      = float(n.leak),
                )

        # Build synapses
        for c in ast.connections:
            self.synapses.append(Synapse(src=c.src, dst=c.dst, weight=c.weight,
                                          delay_ms=getattr(c, "delay_ms", 0.0)))

        # Build action map
        for a in ast.actions:
            self.actions[a.neuron] = a.command

        # Learning rule
        if ast.learn_rule == "STDP":
            self.learner = STDPLearner(rate=ast.learn_rate)

    # ── Public API ──────────────────────────────

    def register_handler(self, command: str, fn: Callable):
        """Bind a Python function to a command string emitted by an action."""
        self.handlers[command] = fn

    def stimulate(self, neuron_name: str, current_time_ms: float,
                  drive: float = 50.0) -> Optional[str]:
        """
        Inject current into a neuron and run one LIF tick.

        Returns the emitted action command (str) or None.
        """
        if neuron_name not in self.neurons:
            raise ValueError(f"Unknown neuron: '{neuron_name}'")

        # Real delayed synapses scheduled by a PAST _fire() call may have
        # come due by now -- deliver them before processing this call's
        # own drive, so a delayed spike that arrives exactly this tick
        # still contributes to the SAME threshold check below (needed
        # for delay-based coincidence detection to actually coincide).
        self._flush_pending_deliveries(current_time_ms)

        n       = self.neurons[neuron_name]
        elapsed = current_time_ms - n.last_spike_time

        # Refractory gate
        if elapsed < self.refractory_ms:
            return None

        # Leak toward REST (0) from either side: positive potentials decay down
        # exactly as before; negative ones (from inhibition) recover UP toward 0
        # instead of being snapped to 0, so hyperpolarization fades gradually.
        n.membrane_potential = _leak_toward_zero(n.membrane_potential, n.leak)

        # Input drive
        n.membrane_potential += drive

        # Threshold check
        command = None
        if n.membrane_potential >= n.threshold:
            command = self._fire(n, current_time_ms)

        return command

    def tick(self, current_time_ms: float):
        """
        Advance all neurons one timestep without external drive.
        Applies leak decay to every neuron, and delivers any real
        delayed-synapse spikes that have come due by this time (see
        _flush_pending_deliveries()) -- a delayed spike doesn't need an
        external stimulate() call to arrive on schedule.
        """
        self._flush_pending_deliveries(current_time_ms)
        for n in self.neurons.values():
            n.membrane_potential = _leak_toward_zero(n.membrane_potential, n.leak)

    def step_resonators(self, drive: float, dt: float, current_time_ms: float = 0.0) -> list[str]:
        """
        Advance every Resonator neuron by one timestep with the same
        shared `drive` signal (e.g. an audio sample or sensor reading).
        Returns the list of action commands emitted by resonators that
        just crossed their detection threshold this step -- same
        action/handler dispatch path as a LIF neuron firing, so game/app
        code doesn't need to know which neuron type triggered it.
        """
        fired_commands = []
        for r in self.resonators.values():
            if r.step(drive, dt):
                self._spike_log.append((current_time_ms, r.name))
                command = self.actions.get(r.name)
                if command:
                    fired_commands.append(command)
                    if command in self.handlers:
                        self.handlers[command]()
        return fired_commands

    # ── Internal mechanics ───────────────────────

    def _fire(self, n: NeuronState, t: float, _depth: int = 0) -> Optional[str]:
        """
        Reset neuron, log spike, propagate, run STDP, emit action.

        Propagation can push a downstream neuron over its own threshold —
        when that happens it fires too (recursively), so a spike can
        cascade through hidden/output layers from a single stimulus.
        `_depth` guards against runaway cascades in cyclic networks.
        """
        n.membrane_potential = 0.0
        n.last_spike_time    = t
        n.fire_count        += 1
        self._spike_log.append((t, n.name))

        # Dispatch action
        command = self.actions.get(n.name)
        if command and command in self.handlers:
            self.handlers[command]()

        # Propagate along synapses
        if _depth < 32:
            for syn in self.synapses:
                if syn.src == n.name:
                    if syn.delay_ms > 0.0:
                        # Real delayed delivery, fused in from Fugu's own
                        # first-class synaptic delay model (see the
                        # Synapse dataclass's own doc comment) -- schedule
                        # for later instead of applying now. Recorded
                        # against the SYNAPSE OBJECT itself (not a copied
                        # weight) so a live STDP update between now and
                        # delivery is still reflected when it lands, and
                        # so the eventual STDP dt calculation below has
                        # the real source fire time to work from.
                        self._pending_deliveries_list().append((t + syn.delay_ms, syn, t))
                        continue
                    self._deliver_spike(syn, downstream_fire_time=t, source_fire_time=t, _depth=_depth)

        return command

    def _deliver_spike(self, syn: "Synapse", downstream_fire_time: float,
                        source_fire_time: float, _depth: int = 0) -> None:
        """The real per-synapse delivery mechanics -- shared by _fire()'s
        instant (delay_ms == 0) path and _flush_pending_deliveries()'s
        delayed path, so both behave identically rather than maintaining
        two copies of this logic. `downstream_fire_time` is the real
        clock time delivery happens (== source_fire_time for an instant
        synapse, == source_fire_time + delay_ms for a delayed one) --
        this is the time downstream's own last_spike_time/refractory
        checks and any resulting cascade fire are stamped with, matching
        the real event-driven semantics this runtime uses throughout."""
        downstream = self.neurons[syn.dst]
        elapsed    = downstream_fire_time - downstream.last_spike_time
        if elapsed < self.refractory_ms:
            return  # downstream is refractory-locked

        # INHIBITION: a negative weight subtracts. We let the target
        # go NEGATIVE (hyperpolarize) -- that's what gives inhibition
        # real VETO power: a deficit that later excitation has to climb
        # back out of before it can fire, not just a cancel of charge
        # already there. Bounded at -threshold so one inhibitory spike
        # can't create an unrecoverable well; leak walks it back to 0.
        downstream.membrane_potential = max(
            -downstream.threshold,
            downstream.membrane_potential + syn.weight * 50.0)

        # STDP: update weight based on spike timing (measured from the
        # real source fire time, not the delivery time -- STDP is about
        # when the PRE-synaptic spike happened relative to POST, and
        # delay is a real transmission property of the synapse, not
        # something that should be invisible to the plasticity rule).
        if self.learner:
            dt         = source_fire_time - downstream.last_spike_time
            syn.weight = self.learner.update(syn, dt)

        # Cascade: downstream crossed its own threshold
        if downstream.membrane_potential >= downstream.threshold:
            self._fire(downstream, downstream_fire_time, _depth + 1)

    def _pending_deliveries_list(self) -> list:
        """Lazily-initializing accessor for self._pending_deliveries.

        REAL BUG FOUND AND FIXED HERE, TWICE, not hypothetical: at least
        two real places in this codebase construct a SpikelingRuntime via
        `SpikelingRuntime.__new__(SpikelingRuntime)` and set attributes by
        hand instead of calling __init__() -- pyspike.py's build_live()
        (fixed directly there) and test_pyspike_incremental_parity.py's
        own old-vs-new scheduling comparison harness (found by actually
        running that test, not by code review alone -- it failed with a
        real AttributeError on the first run after this feature was
        added). Rather than chase every such manual-construction site
        individually (there may be others not yet found), this accessor
        makes the pending-delivery queue self-healing: any runtime
        instance that never went through __init__ gets the attribute
        created on first real use instead of crashing."""
        if not hasattr(self, "_pending_deliveries"):
            self._pending_deliveries = []
        return self._pending_deliveries

    def _flush_pending_deliveries(self, current_time_ms: float) -> None:
        """Delivers every real scheduled delayed-synapse spike whose
        delivery time has arrived, via the SAME _deliver_spike() path
        _fire()'s own instant deliveries use. Called from both
        stimulate() and tick() so a delayed spike arrives on schedule
        regardless of which one drives the runtime forward."""
        pending = self._pending_deliveries_list()
        if not pending:
            return
        due = [d for d in pending if d[0] <= current_time_ms]
        if not due:
            return
        self._pending_deliveries = [d for d in pending if d[0] > current_time_ms]
        due.sort(key=lambda d: d[0])  # real chronological delivery order
        for delivery_time, syn, source_fire_time in due:
            self._deliver_spike(syn, downstream_fire_time=delivery_time, source_fire_time=source_fire_time)

    # ── Diagnostics ─────────────────────────────

    def state_report(self) -> str:
        lines = ["─── Spikeling Network State ───"]
        for name, n in self.neurons.items():
            action = self.actions.get(name, "—")
            lines.append(
                f"  {name:12s}  V={n.membrane_potential:6.1f}  "
                f"spikes={n.fire_count:4d}  action={action}"
            )
        if self.resonators:
            lines.append("─── Resonators ───")
            for name, r in self.resonators.items():
                action = self.actions.get(name, "—")
                lines.append(
                    f"  {name:12s}  freq={r.freq_hz:7.1f}Hz  energy={r.energy_ema:.6f}  action={action}"
                )
        lines.append("─── Synapses ───")
        for syn in self.synapses:
            lines.append(f"  {syn.src} -> {syn.dst}  w={syn.weight:.4f}")
        return "\n".join(lines)

    # ── Interactive demo ─────────────────────────

    def run_interactive(self):
        """
        Cross-platform real-time event loop.

        Key bindings (auto-discovered from neuron names):
          First neuron  → LEFT ARROW  or  'a'
          Second neuron → RIGHT ARROW or  'd'
          ESC / q       → quit
        """
        neuron_names = list(self.neurons.keys())
        key_map: dict[bytes, str] = {}

        if len(neuron_names) >= 1:
            key_map[b'a'] = neuron_names[0]
        if len(neuron_names) >= 2:
            key_map[b'd'] = neuron_names[1]

        print("╔══════════════════════════════════════════════════════════╗")
        print("║   🧠  SPIKELING RUNTIME                                 ║")
        print("╠══════════════════════════════════════════════════════════╣")
        for key, name in key_map.items():
            action = self.actions.get(name, "—")
            print(f"║   [{key.decode()}] → stimulate {name:12s}  ({action})")
        if sys.platform == "win32":
            print("║   [←/→] Arrow keys also work on Windows              ║")
        print("║   [q] or [ESC] → quit                                   ║")
        print("╚══════════════════════════════════════════════════════════╝\n")

        last_neuron: Optional[str] = None

        while True:
            if _kbhit():
                key = _getch()
                active_neuron: Optional[str] = None

                # Arrow keys (Windows escape sequence)
                if key in (b'\x00', b'\xe0'):
                    arrow = _getch()
                    if arrow == b'K':
                        active_neuron = neuron_names[0] if neuron_names else None
                    elif arrow == b'M':
                        active_neuron = neuron_names[1] if len(neuron_names) > 1 else None

                # Unix arrow keys (ESC [ A/B/C/D)
                elif key == b'\x1b':
                    next1 = _getch() if _kbhit() else b''
                    next2 = _getch() if _kbhit() else b''
                    if next1 == b'[':
                        if next2 == b'D':
                            active_neuron = neuron_names[0] if neuron_names else None
                        elif next2 == b'C':
                            active_neuron = neuron_names[1] if len(neuron_names) > 1 else None
                    else:
                        break  # bare ESC = quit

                elif key in (b'q', b'Q'):
                    break

                elif key in key_map:
                    active_neuron = key_map[key]

                if active_neuron:
                    now_ms = time.time() * 1000.0
                    self._run_benchmark_tick(active_neuron, now_ms, last_neuron)
                    last_neuron = active_neuron

            time.sleep(0.001)

        print("\n[runtime] Session ended.")
        print(self.state_report())

    def _run_benchmark_tick(self, neuron_name: str, now_ms: float,
                             last_neuron: Optional[str]):
        """Fire a neuron and print a timing comparison report."""

        # ── Engine 1: Spikeling DSL ──
        t0      = time.perf_counter_ns()
        command = self.stimulate(neuron_name, now_ms)
        t1      = time.perf_counter_ns()
        dsl_us  = (t1 - t0) / 1_000.0

        # ── Engine 2: Simulated traditional cross-correlation ──
        t2 = time.perf_counter_ns()
        _traditional_crosscorr(buf_size=1024)
        t3     = time.perf_counter_ns()
        pc_us  = (t3 - t2) / 1_000.0

        # ── Report ──
        side = "🟢 LEFT " if neuron_name == list(self.neurons.keys())[0] else "🔵 RIGHT"
        print(f"\n⚡ [SPIKE] {side} — {neuron_name}")

        if command is None:
            print(f"  ├─► [🧠 Spikeling]: Refractory lockout  ({dsl_us:.2f} µs)")
        else:
            print(f"  ├─► [🧠 Spikeling]: Action → {command}  ({dsl_us:.2f} µs)")

            # Spatial resolution hint
            if last_neuron and last_neuron != neuron_name:
                elapsed = now_ms - self.neurons[last_neuron].last_spike_time
                if elapsed < 250.0:
                    direction = "LEFT ⬅️" if last_neuron == list(self.neurons.keys())[0] else "RIGHT ➡️"
                    print(f"  │   🎯 SPATIAL: sequence verified → {direction}")

        print(f"  └─► [💻 Traditional]: 8192-byte cross-corr  ({pc_us:.2f} µs)")

        if dsl_us > 0:
            speedup = pc_us / dsl_us
            print(f"  🏆 Spikeling was {speedup:.1f}x faster")
        print("─" * 66)


# ─────────────────────────────────────────────
#  Simulated traditional processing baseline
# ─────────────────────────────────────────────

def _traditional_crosscorr(buf_size: int = 1024) -> float:
    """
    Simulate a traditional DSP cross-correlation over a stereo audio frame.
    This is the baseline we compare against in the benchmark.
    """
    left  = [math.sin(i * 0.1)        for i in range(buf_size)]
    right = [math.sin((i - 5) * 0.1)  for i in range(buf_size)]
    total = 0.0
    for lag in range(-10, 10):
        for i in range(10, buf_size - 10):
            total += left[i] * right[i + lag]
    return total
