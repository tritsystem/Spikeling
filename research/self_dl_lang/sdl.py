#!/usr/bin/env python3
"""sdl -- a ~5-construct DSL whose runtime IS a Spikeling spiking net.

A program declares sense neurons, route neurons, a dense sense->route wire
block, and mutual inhibition among the routes (winner-take-all). It compiles
to a real `pyspike.Net` -> `SpikelingRuntime` (LIF, the production runtime).

    # routing.sdl
    sense s0 s1 s2 s3 s4 s5
    route r0 r1 r2   threshold 50  leak 3
    wire  sense -> route   init 6
    wta   routes   weight -50

`Program.decide(x)` rate-codes the input vector `x` into the sense layer
over a short window and returns a `Trace` of what the net did -- which
routes fired, their membrane potentials, the winner and its margin. The
sense->route weights are exposed as a flat vector so an outer loop can
train them; `prune_dead_routes` / `spawn_hidden` let it restructure.

Nothing in here ever sees a label. The benchmark's whole question is
whether an *endogenous* signal derived from `Trace` (fire exactly one
route, decisively, no double-fire) is enough to make the program route
correctly.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from pyspike import Net  # noqa: E402  the real DSL/runtime bridge


@dataclass
class Trace:
    routes_fired: list[int]          # route indices that fired >=1 this decision
    fire_counts: np.ndarray          # per-route fire count this decision
    potentials: np.ndarray           # per-route membrane potential at window end
    winner: int                      # argmax potential (or -1 if none fired and all <=0)
    margin: float                    # winner potential minus runner-up potential

    @property
    def exactly_one(self) -> bool:
        return len(self.routes_fired) == 1

    @property
    def double_fires(self) -> int:
        return int(np.clip(self.fire_counts - 1, 0, None).sum())


class Program:
    """A compiled sdl program: the spiking runtime plus the handles an
    outer loop needs to train / restructure it."""

    def __init__(self, senses, routes, thr, leak, w_init, wta_w, window=24,
                 sense_gain=60.0, seed=0):
        self.senses, self.routes = list(senses), list(routes)
        self.n_in, self.k = len(self.senses), len(self.routes)
        self.thr, self.leak, self.wta_w = thr, leak, wta_w
        self.window, self.sense_gain = window, sense_gain
        self._rng = np.random.default_rng(seed)
        self._hidden: list[str] = []
        self._build(w_init if np.ndim(w_init) else
                    np.full((self.n_in, self.k), float(w_init)))

    # ---------------------------------------------------------------- compile
    def _build(self, W):
        net = Net(refractory_ms=0)
        for s in self.senses:
            net.neuron(s, threshold=self.thr, leak=0)
        for r in self.routes:
            net.neuron(r, threshold=self.thr, leak=self.leak)
        for h in self._hidden:
            net.neuron(h, threshold=self.thr, leak=self.leak)
        for i, s in enumerate(self.senses):
            for j, r in enumerate(self.routes):
                net.connect(s, r, weight=float(W[i, j]))
        # winner-take-all: every route inhibits every other route
        for a in self.routes:
            for b in self.routes:
                if a != b:
                    net.connect(a, b, weight=float(self.wta_w))
        for (src, dst, w) in getattr(self, "_extra_wires", []):
            net.connect(src, dst, weight=float(w))
        self.rt = net.build()
        # index the trainable sense->route synapses (skip WTA + hidden wires)
        self._w_idx = {}
        for k, syn in enumerate(self.rt.synapses):
            if syn.src in self.senses and syn.dst in self.routes:
                self._w_idx[(syn.src, syn.dst)] = k

    # ---------------------------------------------------------------- weights
    @property
    def weights(self) -> np.ndarray:
        w = np.zeros((self.n_in, self.k))
        for (s, r), k in self._w_idx.items():
            w[self.senses.index(s), self.routes.index(r)] = self.rt.synapses[k].weight
        return w

    @weights.setter
    def weights(self, W):
        for (s, r), k in self._w_idx.items():
            self.rt.synapses[k].weight = float(W[self.senses.index(s), self.routes.index(r)])

    # ---------------------------------------------------------------- run
    def reset(self):
        for n in self.rt.neurons.values():
            n.membrane_potential = 0.0
            n.last_spike_time = float("-inf")
            n.fire_count = 0
        self.rt._spike_log = []

    def decide(self, x) -> Trace:
        """Rate-code x (values in [0,1]) into the senses over up to `window`
        sub-ticks; a sense fires this tick with probability x_i. Route
        potentials integrate the weighted sense spikes; it is a first-past-
        the-post spiking race -- the window ends the instant a route crosses
        threshold, and that route is the winner. If no route crosses in
        `window` ticks the decision is 'indecisive' (no route fired)."""
        self.reset()
        x = np.clip(np.asarray(x, float), 0.0, 1.0)
        rlist = self.routes
        # per-decision rng seeded from the input, so `decide(x)` is a pure
        # function of (weights, topology, x) -- an ES/perturbation trainer
        # needs the reward to be deterministic given the input.
        drng = np.random.default_rng(int(abs(hash(x.tobytes())) % (2 ** 32)))
        t = 0.0
        won = -1
        for _ in range(self.window):
            for i, s in enumerate(self.senses):
                if drng.random() < x[i]:
                    self.rt.stimulate(s, current_time_ms=t, drive=self.sense_gain)
                t += 0.001
            fc = np.array([self.rt.neurons[r].fire_count for r in rlist])
            if fc.any():
                won = int(np.argmax(fc))
                break

        pot = np.array([self.rt.neurons[r].membrane_potential for r in rlist], float)
        fc = np.array([self.rt.neurons[r].fire_count for r in rlist], float)
        if won >= 0:
            pot[won] = self.thr                       # the winner just crossed
            others = np.delete(pot, won)
            margin = float(self.thr - others.max())   # how far ahead the winner was at the tape
            return Trace([won], fc, pot, won, margin)
        # indecisive: nobody reached threshold in `window` ticks
        winner = int(np.argmax(pot))
        runner = float(np.partition(pot, -2)[-2]) if self.k > 1 else 0.0
        return Trace([], fc, pot, winner, float(pot[winner] - runner))

    # ---------------------------------------------------------- restructuring
    def prune_dead_routes(self, window_fire_counts: np.ndarray, factor=0.5):
        """Shrink the incoming weights of any route that never fired across
        a window of decisions -- driven only by the endogenous fire trace."""
        W = self.weights
        for j in range(self.k):
            if window_fire_counts[j] == 0:
                W[:, j] *= factor
        self.weights = W

    def spawn_hidden(self, target_route: int, w=4.0, cap=3):
        """Add a hidden neuron wired from every sense into the weakest
        route -- the same move as spiking_orchestrator._maybe_spawn_specialist,
        here triggered by a low endogenous reward, never by a label."""
        if len(self._hidden) >= cap:
            return False
        h = f"h{len(self._hidden)}"
        self._hidden.append(h)
        self._extra_wires = getattr(self, "_extra_wires", [])
        for s in self.senses:
            self._extra_wires.append((s, h, w))
        self._extra_wires.append((h, self.routes[target_route], w * 2))
        W = self.weights
        self._build(W)                    # rebuild with the new neuron + wires
        return True


# ─────────────────────────────────────────────────────────────── parser
_DEFAULTS = dict(threshold=50.0, leak=2.0, init=1.5, wta=-60.0, window=32)


def compile_source(text: str, seed: int = 0) -> Program:
    senses, routes = [], []
    thr, leak, w_init, wta_w, window = (_DEFAULTS["threshold"], _DEFAULTS["leak"],
                                        _DEFAULTS["init"], _DEFAULTS["wta"], _DEFAULTS["window"])
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tok = line.split()
        head = tok[0]
        if head == "sense":
            senses += tok[1:]
        elif head == "route":
            i = 1
            while i < len(tok) and tok[i] not in ("threshold", "leak"):
                routes.append(tok[i]); i += 1
            while i < len(tok):
                if tok[i] == "threshold":
                    thr = float(tok[i + 1]); i += 2
                elif tok[i] == "leak":
                    leak = float(tok[i + 1]); i += 2
                else:
                    i += 1
        elif head == "wire":                       # `wire sense -> route init 6`
            if "init" in tok:
                w_init = float(tok[tok.index("init") + 1])
        elif head == "wta":                        # `wta routes weight -50`
            if "weight" in tok:
                wta_w = float(tok[tok.index("weight") + 1])
        elif head == "window":
            window = int(tok[1])
        else:
            raise SyntaxError(f"unknown sdl construct: {head!r}")
    if not senses or not routes:
        raise SyntaxError("an sdl program needs at least one `sense` and one `route` line")
    return Program(senses, routes, thr, leak, w_init, wta_w, window=window, seed=seed)


ROUTING_SDL = """
# the whole program -- the runtime improves its behaviour without editing this
sense s0 s1 s2 s3 s4 s5
route r0 r1 r2   threshold 50  leak 3
wire  sense -> route   init 6
wta   routes   weight -50
window 24
"""
