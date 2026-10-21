"""
Spikeling Resonator bank as a reservoir for NeuroBench's Mackey-Glass chaotic
forecasting task.

Reuses the REAL Resonator class from
Documents/Spikeling/resonator-prototype/resonator_bank.py directly (not a
reimplementation) -- a damped harmonic oscillator neuron, driven each tick by
the scalar time-series value. A bank of N resonators at log-spaced natural
frequencies forms the reservoir; positions (x) across the bank at each
timestep are the reservoir-state feature vector.

Readout: ridge regression via the identical torch.linalg.lstsq call the
official EchoStateNetwork example uses (examples/mackey_glass/echo_state_network.py)
-- so any accuracy difference vs. the ESN baseline is attributable to the
RESERVOIR substrate (resonator bank vs. random recurrent tanh network), not to
a different readout mechanism. That is the one thing this comparison isolates
cleanly.

HONEST LIMITS, stated up front, not discovered later:
- The resonator bank has NO cross-unit connectivity -- each resonator is
  driven independently by the same external input. A true ESN's reservoir
  units are recurrently coupled to each other; this bank is not. This is a
  real architectural difference, not just a smaller/bigger reservoir.
- Reservoir size (200) is smaller than ESN's tuned per-tau sizes (522-966,
  from a random hyperparameter search) -- no search budget was spent tuning
  Spikeling's reservoir_size, damping, frequency range, or gain here. This is
  an UNTUNED first measurement against a TUNED baseline -- not the fair "give
  every arm a fair shot" comparison honest-benchmark requires. Reported as a
  pilot number, not a final verdict.
- Frequency range (period 6-300 ticks) was chosen from the symplectic-Euler
  stability bound in resonator_bank.py's own comments (omega*dt < ~2), not
  from any tuning against this specific task.
"""

import math
import sys
import os

sys.path.insert(0, r"C:\Users\gbran\OneDrive\Documents\Spikeling\resonator-prototype")
from resonator_bank import Resonator  # noqa: E402 -- the real module

import torch
import torch.nn as nn


class SpikelingResonatorReservoir(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_resonators: int = 200,
        min_period: float = 6.0,
        max_period: float = 300.0,
        damping: float = 0.02,
        base_gain: float = 4.0e-4,
        dt: float = 1.0,
        ridge_param: float = 1e-6,
        include_bias: bool = True,
        include_input: bool = True,
        seed_id: int = 0,
        mode: str = "autonomous",
    ):
        super().__init__()
        assert mode in ("autonomous", "single_step")
        self.in_channels = in_channels
        self.n_resonators = n_resonators
        self.dt = dt
        self.ridge_param = ridge_param
        self.include_bias = include_bias
        self.include_input = include_input
        self.mode = mode
        self.prior_prediction = None

        self.in_channels_bias = in_channels + 1 if include_bias else in_channels
        self.reservoir_extension = self.in_channels_bias if include_input else 0

        # Log-spaced natural frequencies (as cycles/tick, since this operates
        # on an abstract point-indexed series, not physical audio Hz) --
        # gain-normalized per resonator_bank.py's own documented fix so
        # high-frequency units aren't invisible next to low-frequency ones.
        periods = torch.logspace(
            math.log10(min_period), math.log10(max_period), n_resonators
        )
        freqs = 1.0 / periods
        self._bank = []
        for f in freqs.tolist():
            omega = 2 * math.pi * f
            coupling = base_gain * (omega**2)
            self._bank.append(Resonator(f"R{f:.5f}", freq_hz=f, damping=damping, coupling=coupling))

    def _reset_bank(self):
        for r in self._bank:
            r.x = 0.0
            r.v = 0.0
            r.history = []

    def _drive_bank_step(self, drive_scalar: float):
        for r in self._bank:
            r.step(drive_scalar, self.dt)
        return torch.tensor([r.x for r in self._bank], dtype=torch.float64)

    def fit(self, training_data: torch.Tensor, targets: torch.Tensor, warmup_pts: int):
        training_data = training_data.view(training_data.numel(), 1)
        warmtrain_pts = training_data.shape[0]

        if self.include_bias:
            training_data = torch.cat(
                (torch.ones((warmtrain_pts, 1), dtype=torch.float64), training_data), dim=1
            )

        self._reset_bank()
        self.reservoir_tr = torch.zeros(
            (self.n_resonators + self.reservoir_extension, warmtrain_pts), dtype=torch.float64
        )
        for i in range(warmtrain_pts):
            # sum the (bias + input) channels into one scalar drive -- the
            # resonator bank takes one scalar force per tick, unlike the
            # ESN's matrix projection into a full recurrent state
            drive = training_data[i].sum().item()
            state = self._drive_bank_step(drive)
            if self.include_input:
                self.reservoir_tr[:, i] = torch.cat((training_data[i], state), dim=0)
            else:
                self.reservoir_tr[:, i] = state

        R = self.reservoir_tr[:, warmup_pts:]
        Wout = torch.linalg.lstsq(
            (R @ R.T + self.ridge_param * torch.eye(R.shape[0], dtype=torch.float64)),
            (R @ targets),
            rcond=None,
            driver="gelsd",
        )[0].T
        self.Wout = nn.Linear(Wout.size(1), Wout.size(0), bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.Wout.weight.copy_(Wout)

    def single_forward(self, sample):
        if self.include_bias:
            sample_b = torch.cat((torch.ones((1, 1), dtype=torch.float64), sample), dim=0)
        else:
            sample_b = sample
        drive = sample_b.sum().item()
        state = self._drive_bank_step(drive)
        if self.include_input:
            x = torch.cat((sample_b.squeeze(-1), state), dim=0)
        else:
            x = state
        return self.Wout(x.unsqueeze(0))

    def forward(self, batch):
        predictions = []
        for sample in batch:
            if self.mode == "autonomous" and self.prior_prediction is not None:
                sample = self.prior_prediction
            prediction = self.single_forward(sample)
            predictions.append(prediction)
            self.prior_prediction = prediction
        self.prior_prediction = None
        return torch.stack(predictions).squeeze(1)
