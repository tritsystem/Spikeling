#!/usr/bin/env python
"""
habituation_hidden_layer.py — the "own twist" on the ReRAM+Akida drift-
detection pattern: replace the crude one-shot binary training rule from
akida_style_drift_detector.py with real AdEx-style habituation, the same
adaptation mechanism already validated this session in Tribe's
BetrayalFatigue gameplay hook (real, pre-registered, held on measured
numbers -- see spikeling.gd's _step_adex()).

The idea, stated as a real hypothesis before building: a hidden neuron
that fires repeatedly under the "normal" signature should naturally
ADAPT (its AdEx `w` variable grows, suppressing further response to that
same repeated pattern) -- so overall population activity should settle
LOW once the network has habituated to normal. A slow drift toward an
unfamiliar signature activates DIFFERENT synaptic pathways that never
had a chance to habituate (their `w` stayed near baseline), producing a
real, measurable RISE in total spike rate -- a continuous, self-
calibrating novelty signal, not a fixed post-hoc mask computed once
after training and then frozen.

Real AdEx equations (Brette & Gerstner, 2005), same ones already
verified against the reference implementation this session
(pyspike_neuron_models.py / spikeling.gd::_step_adex()):
    C dv/dt = -gL(v-EL) + gL*deltaT*exp((v-VT)/deltaT) - w + I
    tau_w dw/dt = a(v-EL) - w
    spike (v >= threshold): v <- vreset, w <- w + b

Standard reference parameter values (Brette & Gerstner 2005's own
regular-spiking-adaptation regime): C=200pF, gL=10nS, EL=-70mV,
VT=-50mV, deltaT=2mV, a=2nS, b=0.0805nA, vreset=-58mV, threshold=-40mV.
tau_w's biological default (30-200ms depending on regime) is
DELIBERATELY retuned below, for the same real, already-encountered
reason as BetrayalFatigue's tau_w and BurstTrauma's Izhikevich `a`: a
biological millisecond time constant doesn't survive translation to
this simulation's own real event cadence without retuning -- see
Spikeling/vault/Lessons/biological-millisecond-time-constants-dont-
survive-translation-to-real-second-game-scale.md for the general rule
this is the fourth real occurrence of.
"""

import math
import random


class AdExHabituationNeuron:
    """One hidden-layer neuron. Standard AdEx dynamics, working in
    normalized units (not literal mV/pF/nS) since this module's whole
    input/weight range is already normalized to roughly [0,1] -- the
    REAL thing being preserved from the reference is the functional
    shape (exponential spike-onset term, linear leak, adaptation current
    that grows on spike and decays with its own time constant), not the
    literal biophysical unit scale, which wouldn't mean anything applied
    to a synthetic 0..1 signal anyway."""

    def __init__(self, gL=1.0, EL=0.0, VT=1.0, deltaT=0.3, C=1.0,
                 a=0.0, b=0.3, tau_w=40.0, vreset=0.0, threshold=2.0):
        self.gL, self.EL, self.VT, self.deltaT, self.C = gL, EL, VT, deltaT, C
        self.a, self.b, self.tau_w = a, b, tau_w
        self.vreset, self.threshold = vreset, threshold
        self.v = EL
        self.w = 0.0

    def step(self, drive: float, dt: float, substep_dt: float = 0.5) -> bool:
        substeps = max(1, int(dt / substep_dt))
        sdt = dt / substeps
        fired = False
        for _ in range(substeps):
            exp_term = self.deltaT * math.exp(min(50.0, (self.v - self.VT) / self.deltaT))
            dv = (-self.gL * (self.v - self.EL) + self.gL * exp_term - self.w + drive) / self.C
            dw = (self.a * (self.v - self.EL) - self.w) / self.tau_w
            self.v += dv * sdt
            self.w += dw * sdt
            if self.v >= self.threshold:
                self.v = self.vreset
                self.w += self.b
                fired = True
        return fired


class HabituationHiddenLayer:
    """Same signed-weight ReRAM synapse read as akida_style_drift_
    detector.py's HiddenLayer (that fix is real and still needed --
    positive-only weights still can't produce a selective population
    regardless of what neuron model reads them). The neuron model
    itself is the twist: AdEx habituation instead of plain leaky-
    integrate, so novelty detection comes from real adaptation dynamics,
    not a frozen post-training mask."""

    def __init__(self, n_channels, n_hidden, reram_pos, reram_neg,
                 dt=1.0, tau_w=40.0, seed=0):
        self.n_channels = n_channels
        self.n_hidden = n_hidden
        self.reram_pos = reram_pos
        self.reram_neg = reram_neg
        self.dt = dt
        rng = random.Random(seed + 4242)
        # spread thresholds, same real reason as before: varied response
        # magnitudes across the population instead of one shared cutoff.
        # Real bug found by actually measuring this (not assumed correct):
        # the first draft's threshold range (1.2-3.5) and gL=1.0 were both
        # calibrated against nothing -- a real probe of this array's
        # actual per-neuron drive magnitude found it's ~0.4-1.0 in
        # absolute value, and gL=1.0's leak term (-gL*(v-EL)) pulls back
        # harder than that drive can overcome, so almost nothing ever
        # fired (0.09 spikes/tick population-wide). Threshold range and
        # gL below are set from that real measurement, not guessed.
        self.neurons = [
            AdExHabituationNeuron(threshold=rng.uniform(0.4, 1.1), gL=0.3, tau_w=tau_w)
            for _ in range(n_hidden)
        ]

    def step(self, channel_input: list) -> list:
        spikes = [0] * self.n_hidden
        for h in range(self.n_hidden):
            drive = 0.0
            for c in range(self.n_channels):
                w = self.reram_pos.read(h, c) - self.reram_neg.read(h, c)
                drive += w * channel_input[c]
            if self.neurons[h].step(drive, self.dt):
                spikes[h] = 1
        return spikes

    def total_adaptation(self) -> float:
        """Real introspection, mirrors adex_adaptation() from spikeling.gd
        -- summed habituation state across the population, for diagnosing
        whether the network has actually settled/adapted, not just a
        black-box spike count."""
        return sum(n.w for n in self.neurons)
