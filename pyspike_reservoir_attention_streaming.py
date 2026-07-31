#!/usr/bin/env python
"""
pyspike_reservoir_attention_streaming.py -- "build something that fits":
pyspike_reservoir_attention_hybrid.py's two-hop attention needs global
backprop to TRAIN (confirmed incompatible with the real .spk runtime's
local/online STDPLearner -- see PROJECT_reservoir_attention_ternary_
spiking_hybrid.md). But that only rules out TRAINING it online. It does
NOT rule out RUNNING it online: train offline once (already done), then
deploy the FROZEN weights against a live, persistent reservoir that
steps one tick at a time -- exactly the same pattern already proven in
this portfolio (ternary readouts trained offline, deployed frozen for
real-time inference on the Arduino grid / junkyard-reservoir).

Two real differences from the offline/batched version that a genuine
streaming deployment has to solve, not paper over:

1. ReservoirBank.forward() resets x=v=0 at the start of every call --
   fine for training on independent samples, WRONG for a live stream
   where state must persist across ticks. StreamingReservoir below
   keeps x/v as instance state, stepped once per incoming sample.

2. The offline model's hop-2 shift (torch.roll, circular) only makes
   sense over a FIXED, complete T-length window. StreamingAttention
   below keeps a T-length ring buffer of [x,v] states and recomputes
   attention over that sliding window each tick -- standard
   sliding-window inference, not a retrained model.

VERIFICATION: feed the SAME T-length sequence through (a) the offline
batched model in one shot and (b) this streaming module one tick at a
time, using the identical frozen weights. If streaming inference is a
faithful deployment (not a different model), the two must agree on the
final prediction -- not "roughly," exactly to float precision.
"""
import numpy as np
import torch
import torch.nn.functional as F

from pyspike_reservoir_attention_hybrid import (
    DEVICE, T, K_LAG, M_RESERVOIR, OMEGA, DAMPING, DT, HOLD,
    ReservoirBank, ReservoirAttentionReadout, run_stage, make_task,
)


class StreamingReservoir:
    """Same physics as ReservoirBank, but state persists across step()
    calls instead of resetting each forward() -- required for a live
    input stream. Single-sample (no batch dim) to match a real deployed
    sensor stream."""

    def __init__(self, bank: ReservoirBank):
        self.bank = bank
        self.M = bank.M
        self.x = torch.zeros(self.M, device=DEVICE)
        self.v = torch.zeros(self.M, device=DEVICE)

    @torch.no_grad()
    def step(self, u_t: float):
        src, dst, w = self.bank.src, self.bank.dst, self.bank.weights
        win = self.bank.win
        x, v = self.x, self.v
        for _ in range(HOLD):
            coupling = torch.zeros(self.M, device=DEVICE)
            coupling.index_add_(0, src, w * x[dst])
            coupling.index_add_(0, dst, w * x[src])
            drive = win * u_t + coupling
            accel = -(OMEGA ** 2) * x - 2 * DAMPING * OMEGA * v + drive
            v = v + accel * DT
            x = x + v * DT
        self.x, self.v = x, v
        return torch.cat([x, v], dim=-1)  # (2*M,)


class StreamingAttentionReadout:
    """Frozen, deployed inference wrapper around a trained (offline)
    ReservoirAttentionReadout. Keeps a T-length ring buffer of reservoir
    states and recomputes the two-hop attention over that window each
    tick -- weights are NOT updated here, this is inference only."""

    def __init__(self, trained_model: ReservoirAttentionReadout, window: int = T):
        self.model = trained_model
        self.window = window
        self.buffer = []

    @torch.no_grad()
    def push_and_predict(self, state_2m):
        self.buffer.append(state_2m)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        if len(self.buffer) < self.window:
            return None  # not enough context yet -- honest, no guess
        states = torch.stack(self.buffer, dim=0).unsqueeze(0)  # (1, window, 2M)
        pred, attn1 = self.model(states)
        return pred.item(), attn1.squeeze(0).cpu().numpy()


if __name__ == "__main__":
    print("=" * 78)
    print("  STREAMING DEPLOYMENT CHECK: offline-trained weights, run live")
    print("=" * 78)

    rng = np.random.default_rng(0)
    train_u, train_y, _ = make_task(rng, 800)
    test_u, test_y, test_marker_pos = make_task(rng, 200)

    reservoir = ReservoirBank(M_RESERVOIR).to(DEVICE)
    FEAT = 2 * M_RESERVOIR

    print("\nTraining offline (identical to Stage 2 of the hybrid script)...")
    torch.manual_seed(0)
    model = ReservoirAttentionReadout(FEAT, use_ternary=False, use_spiking=False).to(DEVICE)
    nmse_offline = run_stage("offline batched (reference)", model, reservoir, train_u, train_y, test_u, test_y)
    model.eval()

    print("\nDeploying the SAME frozen weights in streaming (one-tick-at-a-time) mode...")
    n_check = 20
    max_abs_diff = 0.0
    correct_localization = 0
    for i in range(n_check):
        u_seq = test_u[i]
        stream_res = StreamingReservoir(reservoir)
        stream_attn = StreamingAttentionReadout(model, window=T)
        pred = None
        for t in range(T):
            state = stream_res.step(float(u_seq[t]))
            result = stream_attn.push_and_predict(state)
            if result is not None:
                pred, attn1 = result

        with torch.no_grad():
            batch_states = reservoir(torch.tensor(u_seq[None, :], device=DEVICE))
            offline_pred, offline_attn1 = model(batch_states)
            offline_pred = offline_pred.item()
            offline_attn1 = offline_attn1.squeeze(0).cpu().numpy()

        diff = abs(pred - offline_pred)
        max_abs_diff = max(max_abs_diff, diff)
        found_pos = int(np.argmax(attn1))
        if found_pos == test_marker_pos[i]:
            correct_localization += 1

    print(f"\nmax |streaming_pred - offline_pred| over {n_check} sequences: {max_abs_diff:.2e}")
    print(f"streaming hop-1 localization accuracy: {correct_localization}/{n_check} = {100*correct_localization/n_check:.1f}%")
    if max_abs_diff < 1e-4:
        print("VERIFIED: streaming deployment is a faithful reproduction of the offline model, not a different one.")
    else:
        print("MISMATCH: streaming and offline predictions diverge -- deployment is NOT faithful, needs a fix.")
