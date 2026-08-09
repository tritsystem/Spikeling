#!/usr/bin/env python
"""
pyspike_reservoir_attention_continuous_stream.py -- further test of the
streaming deployment (pyspike_reservoir_attention_streaming.py). That
test verified bit-exact agreement between offline and streaming
inference on ISOLATED, pre-chopped T=40 windows -- the same shape of
data the model was trained on. A real sensor deployment isn't chopped
into clean windows: it's one continuous stream with multiple marker
events over time, and the sliding window has to keep tracking whichever
marker is CURRENTLY relevant as older ones scroll toward the edge and
out of the buffer. That's a genuinely harder, untested condition.

Builds one long continuous stream (length 300) with 6 marker events at
KNOWN positions, spaced far enough apart that each has fully entered
and can be checked without overlapping another event's answer window.
Runs the persistent StreamingReservoir + StreamingAttentionReadout
across the WHOLE stream once (no resets between events, exactly like a
real always-on sensor), and checks, at each event's answer tick
(marker_pos + K_LAG), whether the model reports that event's true
target and localizes that event's marker specifically -- not a stale
one still lingering in the window.
"""
import numpy as np
import torch

from pyspike_reservoir_attention_hybrid import (
    DEVICE, T, K_LAG, M_RESERVOIR, ReservoirBank, ReservoirAttentionReadout,
    run_stage, make_task,
)
from pyspike_reservoir_attention_streaming import StreamingReservoir, StreamingAttentionReadout

if __name__ == "__main__":
    print("=" * 78)
    print("  FURTHER TEST: continuous multi-event stream (not pre-chopped windows)")
    print("=" * 78)

    rng = np.random.default_rng(0)
    train_u, train_y, _ = make_task(rng, 800)
    test_u, test_y, _ = make_task(rng, 200)
    reservoir = ReservoirBank(M_RESERVOIR).to(DEVICE)
    FEAT = 2 * M_RESERVOIR

    print("\nTraining offline (unchanged, same as the hybrid script's Stage 2)...")
    torch.manual_seed(0)
    model = ReservoirAttentionReadout(FEAT, use_ternary=False, use_spiking=False).to(DEVICE)
    run_stage("offline batched (reference)", model, reservoir, train_u, train_y, test_u, test_y)
    model.eval()

    # build one long continuous stream: 300 ticks, 6 marker events, spaced
    # 45 ticks apart so no event's [marker, marker+K_LAG] window overlaps
    # the next event's marker -- each event is independently checkable.
    STREAM_LEN = 300
    N_EVENTS = 6
    SPACING = 45
    stream_rng = np.random.default_rng(7)
    u_stream = stream_rng.uniform(-1, 1, size=STREAM_LEN).astype(np.float32)
    event_positions = [20 + i * SPACING for i in range(N_EVENTS)]
    for pos in event_positions:
        u_stream[pos] = 3.0
    event_targets = [float(u_stream[pos + K_LAG]) for pos in event_positions]
    # zero the marker AFTER recording targets, matching make_task's convention
    # (marker values already recorded above; no separate clearing needed since
    # we read u_stream[pos+K_LAG], a different index, before any mutation)

    print(f"\nStream length {STREAM_LEN}, {N_EVENTS} marker events at ticks {event_positions}")
    print(f"Running ONE continuous pass (persistent reservoir + sliding window, no resets)...\n")

    stream_res = StreamingReservoir(reservoir)
    stream_attn = StreamingAttentionReadout(model, window=T)

    results = []
    check_ticks = {pos + K_LAG: i for i, pos in enumerate(event_positions)}
    for t in range(STREAM_LEN):
        state = stream_res.step(float(u_stream[t]))
        out = stream_attn.push_and_predict(state)
        if t in check_ticks and out is not None:
            pred, attn1 = out
            ev_idx = check_ticks[t]
            true_target = event_targets[ev_idx]
            true_marker_pos = event_positions[ev_idx]
            # attn1 is over the CURRENT window (last T ticks ending at t);
            # convert the window-local argmax back to an absolute stream tick
            window_start = t - T + 1
            found_abs_pos = window_start + int(np.argmax(attn1))
            correct_loc = (found_abs_pos == true_marker_pos)
            err = abs(pred - true_target)
            results.append((ev_idx, t, true_marker_pos, found_abs_pos, correct_loc, true_target, pred, err))
            status = "OK " if correct_loc else "MISS"
            print(f"  event {ev_idx}: answer_tick={t:3d}  true_marker={true_marker_pos:3d}  "
                  f"found_marker={found_abs_pos:3d}  [{status}]  "
                  f"true_target={true_target:+.3f}  pred={pred:+.3f}  |err|={err:.4f}")

    n_correct = sum(1 for r in results if r[4])
    mean_err = np.mean([r[7] for r in results])
    print(f"\n{n_correct}/{len(results)} events correctly localized in the continuous stream")
    print(f"mean |prediction error| across events: {mean_err:.4f}")
    if n_correct == len(results):
        print("VERIFIED: streaming deployment correctly tracks the CURRENT marker across a continuous, multi-event stream.")
    else:
        print("PARTIAL: at least one event was mislocalized in continuous streaming -- a real, reportable gap vs. isolated-window inference.")
