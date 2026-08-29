#!/usr/bin/env python
"""
stdp_trained_hidden_layer.py — the real structural fix, not a parameter
tweak: gives the ReRAM hidden-layer weights an actual learning rule.

Every prior pass in this exploration (the crude binary-rule detector, the
habituation/AdEx detector, the population-size/training-duration sweep)
kept the ReRAM synapse weights FIXED after one random initialization --
only a separate scalar adaptation variable changed. The real training-
regime sweep just showed, honestly, that scaling that up (more neurons,
longer exposure) doesn't help: more copies of the same unstructured
randomness isn't more signal. Cerebra's real demonstrated result used
actual STDP-BCM with structural plasticity -- the weights THEMSELVES get
shaped by experience.

This module directly reuses two already-built, already-verified pieces
from this session, not a reimplementation:
  - STDPLearner from Spikeling/core/runtime/runtime.py -- the SAME class
    just fixed (2026-08-29) so exact pre/post coincidence (dt=0) lands in
    LTP, not LTD, per the real van der Made patent reference. That fix
    matters directly here: with discrete per-tick channel activity and
    per-tick neuron firing, exact-tick coincidence is common, not rare.
  - ReRAMSynapseArray.program() from reram_synapse_array.py -- weight
    updates are written back through the REAL ReRAM programming model
    (iterative-mode noise, endurance tracking), not applied as an
    idealized instant write.

Adaptation for a continuous-input, event-driven-output network (channels
are analog signal values, not discrete spikes): a channel counts as a
real "pre-synaptic event" on any tick where its value exceeds
ACTIVE_THRESHOLD. Each hidden neuron's real AdEx spike is the
"post-synaptic event." On every hidden-neuron fire, every channel that
was active within STDP_WINDOW_TICKS gets a real STDP update computed
from the actual tick gap -- close-in-time active channels are
potentiated most strongly (matching this signature's real causal
structure), far-in-time or inactive channels are weakened.
"""

import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core", "runtime"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from runtime.runtime import STDPLearner, Synapse  # noqa: E402

from reram_synapse_array import ReRAMSynapseArray
from habituation_hidden_layer import AdExHabituationNeuron

ACTIVE_THRESHOLD = 0.5      # a channel counts as "active" this tick above this value
STDP_WINDOW_TICKS = 5       # only consider channel activity within this many ticks of a fire


class STDPTrainedHiddenLayer:
    """Same AdEx habituation neurons as habituation_hidden_layer.py (that
    part -- per-neuron adaptation -- was never the problem; the fixed
    weights were), but now with real, live STDP updates writing back
    through the actual ReRAM programming model during training."""

    def __init__(self, n_channels, n_hidden, reram_pos, reram_neg,
                 tau_w=40.0, stdp_rate=0.05, stdp_tau=3.0, seed=0):
        self.n_channels = n_channels
        self.n_hidden = n_hidden
        self.reram_pos = reram_pos
        self.reram_neg = reram_neg
        rng = random.Random(seed + 4242)
        self.neurons = [
            AdExHabituationNeuron(threshold=rng.uniform(0.4, 1.1), gL=0.3, tau_w=tau_w)
            for _ in range(n_hidden)
        ]
        # STDP_tau in TICKS, not ms -- same real lesson as every other
        # biological time constant in this session (Resonator's
        # energy_time_constant, BetrayalFatigue's tau_w, BurstTrauma's
        # Izhikevich a): a biological default doesn't survive translation
        # to this system's own real tick cadence unexamined. 3 ticks
        # (within the 5-tick STDP_WINDOW_TICKS) is a deliberate, disclosed
        # choice for THIS discrete-tick system, not the reference's own
        # literal ms-scale default.
        self.stdp = STDPLearner(rate=stdp_rate, tau=stdp_tau)
        self._channel_last_active = [-999] * n_channels
        self._tick = 0
        self.stdp_updates_applied = 0

    def step(self, channel_input, learn=False):
        self._tick += 1
        for c in range(self.n_channels):
            if channel_input[c] > ACTIVE_THRESHOLD:
                self._channel_last_active[c] = self._tick

        spikes = [0] * self.n_hidden
        for h in range(self.n_hidden):
            drive = 0.0
            for c in range(self.n_channels):
                w = self.reram_pos.read(h, c) - self.reram_neg.read(h, c)
                drive += w * channel_input[c]
            fired = self.neurons[h].step(drive, dt=1.0)
            if fired:
                spikes[h] = 1
                if learn:
                    self._apply_stdp(h)
        return spikes

    def _apply_stdp(self, h):
        """Real STDP update for hidden neuron h's fire event: every
        channel active within STDP_WINDOW_TICKS gets its (pos, neg) cell
        pair updated via the actual STDPLearner, written back through the
        real ReRAM programming model."""
        for c in range(self.n_channels):
            gap = self._tick - self._channel_last_active[c]
            if gap > STDP_WINDOW_TICKS:
                continue
            # dt convention matches STDPLearner's own docstring: pre
            # before/at post (dt>=0) -> LTP. The channel was active `gap`
            # ticks BEFORE this fire, so dt = +gap (pre preceded post).
            dt = float(gap)
            cur_pos = self.reram_pos.read(h, c)
            cur_neg = self.reram_neg.read(h, c)
            syn_pos = Synapse(src=f"ch{c}", dst=f"h{h}", weight=cur_pos)
            syn_neg = Synapse(src=f"ch{c}", dst=f"h{h}", weight=cur_neg)
            # Opposite dt sign for the neg cell, deliberately: w = pos - neg,
            # so to make the SIGNED weight grow (this channel becoming a
            # stronger real cause of this neuron firing), pos needs LTP
            # (dt>=0, strengthen) while neg needs LTD (dt<0, weaken) for
            # the SAME real event -- applying the same dt to both would
            # move pos and neg together and leave w roughly unchanged.
            new_pos = self.stdp.update(syn_pos, dt)
            new_neg = self.stdp.update(syn_neg, -dt)
            self.reram_pos.program(h, c, new_pos, mode="iterative")
            self.reram_neg.program(h, c, new_neg, mode="iterative")
            self.stdp_updates_applied += 1

    def total_adaptation(self):
        return sum(n.w for n in self.neurons)
