#!/usr/bin/env python
"""
agreement_trained_hidden_layer.py — testing whether the SADP paper's core
idea (Cohen's-kappa population agreement instead of pairwise spike-timing)
can reduce §9's real, honest problem: the STDP-trained detector's spurious
pre-drift-trigger rate (18% at the best-found config), hypothesized to
come from over-sensitivity to a handful of coincidental exact-timing hits.

HONEST SCOPE NOTE, stated up front: this is NOT the published SADP
algorithm. SADP (arXiv:2601.08526) is SUPERVISED -- it computes kappa
between a hidden neuron's spikes and the CORRECT-CLASS OUTPUT neuron's
spikes, using real labels. This detector has no labels; it's unsupervised
novelty detection. The adaptation here takes SADP's actual mechanism
(chance-corrected population agreement over a window, replacing per-tick
timing-pair comparison) and applies it unsupervised: kappa is computed
between each CHANNEL's own binary active/inactive series and the HIDDEN
NEURON's own fire series over a rolling window, reinforcing channels
whose activity pattern agrees with this neuron's firing beyond chance --
a Hebbian-style "fire together" rule using SADP's statistic instead of
SADP's actual supervised architecture. This is a genuine "own twist",
not a claim of implementing SADP itself.

PRE-REGISTERED HYPOTHESIS (stated before running): because Cohen's kappa
is chance-corrected over an extended window rather than driven by exact-
tick coincidence, this rule should be LESS sensitive to the handful of
coincidental early exact-timing hits that plausibly cause STDP's spurious
pre-drift triggers (§9) -- i.e. a lower spurious-trigger rate at
comparable training exposure, tested on the exact same N=50 combined seed
pool and protocol as §9 for a fair, direct comparison.
DISCONFIRM: spurious rate is the same or worse than STDP's -> the
timing-precision hypothesis for §9's regression was wrong, or this
particular adaptation doesn't capture the real benefit -- report either
honestly.

Reuses AdExHabituationNeuron and ReRAMSynapseArray.program() directly,
same as every other hidden-layer variant in this folder.
"""

import random

from reram_synapse_array import ReRAMSynapseArray
from habituation_hidden_layer import AdExHabituationNeuron

ACTIVE_THRESHOLD = 0.5   # same convention as stdp_trained_hidden_layer.py
KAPPA_WINDOW = 30        # ticks per agreement-window (population-level, not per-event)
KAPPA_EPS = 1e-6


def cohens_kappa(series_a, series_b):
    """Real Cohen's kappa between two binary 0/1 series of equal length,
    same formula as the SADP paper's Eq. in Sec 3.5.2: observed agreement
    minus chance-expected agreement (from each series' own marginal
    rate), normalized by the maximum possible agreement above chance."""
    n = len(series_a)
    p_o = sum(1 for a, b in zip(series_a, series_b) if a == b) / n
    p_a = sum(series_a) / n
    p_b = sum(series_b) / n
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)
    return (p_o - p_e) / (1.0 - p_e + KAPPA_EPS)


class AgreementTrainedHiddenLayer:
    """Same AdEx habituation neuron as every other variant in this
    folder -- only the ReRAM weight-training rule differs: windowed,
    chance-corrected agreement instead of per-event timing-pair STDP."""

    def __init__(self, n_channels, n_hidden, reram_pos, reram_neg,
                 seed=0, eta=0.03, window=KAPPA_WINDOW):
        self.n_channels = n_channels
        self.n_hidden = n_hidden
        self.reram_pos = reram_pos
        self.reram_neg = reram_neg
        self.eta = eta
        self.window = window
        rng = random.Random(seed + 4242)
        self.neurons = [
            AdExHabituationNeuron(threshold=rng.uniform(0.4, 1.1), gL=0.3, tau_w=40.0)
            for _ in range(n_hidden)
        ]
        self._channel_buf = [[] for _ in range(n_channels)]
        self._fire_buf = [[] for _ in range(n_hidden)]
        self.agreement_updates_applied = 0

    def step(self, channel_input, learn=False):
        active = [1 if v > ACTIVE_THRESHOLD else 0 for v in channel_input]
        spikes = [0] * self.n_hidden
        for h in range(self.n_hidden):
            drive = 0.0
            for c in range(self.n_channels):
                w = self.reram_pos.read(h, c) - self.reram_neg.read(h, c)
                drive += w * channel_input[c]
            fired = 1 if self.neurons[h].step(drive, dt=1.0) else 0
            spikes[h] = fired
            if learn:
                self._fire_buf[h].append(fired)

        if learn:
            for c in range(self.n_channels):
                self._channel_buf[c].append(active[c])
            if len(self._channel_buf[0]) >= self.window:
                self._apply_agreement_update()
                for c in range(self.n_channels):
                    self._channel_buf[c] = []
                for h in range(self.n_hidden):
                    self._fire_buf[h] = []
        return spikes

    def _apply_agreement_update(self):
        """Real windowed update: for every (channel, hidden) pair,
        compute kappa over the just-completed window and push the signed
        weight (pos - neg) toward agreement, away from disagreement --
        same opposite-direction pos/neg push already used in
        stdp_trained_hidden_layer.py's _apply_stdp, reused here for the
        same real reason (so the SIGNED weight actually moves, not both
        cells drifting together and cancelling)."""
        for h in range(self.n_hidden):
            fire_series = self._fire_buf[h]
            if sum(fire_series) == 0:
                continue  # a neuron that never fired this window has no agreement signal
            for c in range(self.n_channels):
                kappa = cohens_kappa(self._channel_buf[c], fire_series)
                delta = self.eta * kappa
                cur_pos = self.reram_pos.read(h, c)
                cur_neg = self.reram_neg.read(h, c)
                new_pos = max(0.0, min(1.0, cur_pos + delta))
                new_neg = max(0.0, min(1.0, cur_neg - delta))
                self.reram_pos.program(h, c, new_pos, mode="iterative")
                self.reram_neg.program(h, c, new_neg, mode="iterative")
                self.agreement_updates_applied += 1
