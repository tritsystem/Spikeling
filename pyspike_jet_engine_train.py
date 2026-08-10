#!/usr/bin/env python
"""pyspike_jet_engine_train.py -- wires pyspike_native_backprop.py's fully
native (no PyTorch, no autodiff library) backprop into a real training
loop for the jet-engine pipeline, on the exact task it's used for in
production (jet_engine_gate.py / methodlm_sustained_confirmation.py /
mesh_rag_server.py's /guard/sustained-check): discriminate SUSTAINED
elevated drive from a single BLIP.

REAL METHODOLOGY, not a toy single-pair fit:
  - Multiple sustained/blip trials at VARIED, realistic drive levels
    (60-250, the same range real RV-scaled drives have used all night:
    SERVER_GUARD_RV_SCALE=800 * RV in [0.11,0.57] -> drive in [88,285]).
  - TRAIN/HELD-OUT split -- evaluated on trials never seen during
    training, the correct way to check genuine discrimination learned
    rather than memorization of one fixed pair.
  - Trained weights compared against the ORIGINAL hand-tuned .spk weights
    on the SAME held-out set -- a fair, real comparison of gradient
    descent vs a night of manual weight-sweep tuning, not assumed either
    way.
  - Trained weights saved to a NEW file
    (jet_engine_spike_pipeline_trained.spk), never overwriting the
    production .spk that mesh_rag_server.py's live routes depend on.
"""
import os
import sys

SPIKELING_ROOT = r"C:\Users\gbran\OneDrive\Documents\Spikeling"
sys.path.insert(0, SPIKELING_ROOT)
sys.path.insert(0, os.path.join(SPIKELING_ROOT, "core"))

import numpy as np

from pyspike_native_backprop import NativeSpikingGraph, compile_native, PULSE_SCALE

SPK_PATH = os.path.join(SPIKELING_ROOT, "ai-apps", "jet_engine_spike_pipeline.spk")
TRAINED_SPK_PATH = os.path.join(SPIKELING_ROOT, "ai-apps", "jet_engine_spike_pipeline_trained.spk")

T_SUSTAINED = 20
T_WINDOW = 20  # blip trials also use a 20-tick window, burst somewhere inside it --
                # makes duration the thing that must be LEARNED, not a free giveaway
DRIVE_LOW, DRIVE_HIGH = 60.0, 250.0  # real RV-scaled drive range used all night
BURST_DUR_LOW, BURST_DUR_HIGH = 1, 8  # blip burst length: 1-8 ticks, vs 20 for real
                                        # sustained -- meaningfully overlapping range,
                                        # not a trivial single-tick-vs-many gap
                                        # (this is the fix: the FIRST version made
                                        # blips exactly 1 tick against 20 sustained
                                        # ticks, a gap so large ANY positive weights
                                        # solved it -- both hand-tuned AND untrained
                                        # random init hit 100%, an uninformative test)


def make_curriculum(n_sustained, n_blip, seed):
    """Real varied trials, not one fixed pair. Returns list of
    (drive_array_spec, is_sustained) -- is_sustained is the label (True =
    should ignite, False = should not). Blip trials get a randomized burst
    DURATION (1-8 ticks) and position, not just a single fixed-width blip,
    so the network has to learn a genuine duration threshold."""
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n_sustained):
        drive_level = rng.uniform(DRIVE_LOW, DRIVE_HIGH)
        trials.append({"drive": drive_level, "sustained": True})
    for _ in range(n_blip):
        drive_level = rng.uniform(DRIVE_LOW, DRIVE_HIGH)
        burst_dur = int(rng.integers(BURST_DUR_LOW, BURST_DUR_HIGH + 1))
        start = int(rng.integers(0, T_WINDOW - burst_dur + 1))
        trials.append({"drive": drive_level, "sustained": False,
                        "burst_dur": burst_dur, "start": start})
    rng.shuffle(trials)
    return trials


def build_drive_array(net, spec):
    intake_idx = [net.name_to_idx[n] for n in ["intake1", "intake2", "intake3", "intake4"]]
    if spec["sustained"]:
        d = np.zeros((T_SUSTAINED, net.n_neurons))
        d[:, intake_idx] = spec["drive"]
    else:
        d = np.zeros((T_WINDOW, net.n_neurons))
        start, dur = spec["start"], spec["burst_dur"]
        d[start:start + dur, intake_idx] = spec["drive"]
    return d


def combustion_activity(net, drive_array):
    combustion_idx = net.name_to_idx["combustion"]
    spikes, cache = net.forward(drive_array)
    return spikes[:, combustion_idx].sum(), cache


def evaluate(net, trials):
    """Real accuracy on a set of trials: sustained trials should ignite
    (combustion > 0), blip trials should not (combustion == 0)."""
    correct = 0
    for spec in trials:
        drive = build_drive_array(net, spec)
        c, _ = combustion_activity(net, drive)
        predicted_ignite = c > 0
        if predicted_ignite == spec["sustained"]:
            correct += 1
    return correct / len(trials)


def native_adam_train_curriculum(net, train_trials, margin=1.0, epochs=15, lr=0.03):
    """FIXED from the exact-MSE-to-3.0 version: that loss pushed sustained
    trials to hit a precise numeric target, which is a harder and
    differently-shaped objective than what evaluate() actually measures
    (c>0 vs c==0). Real, measured result of the mismatch: held-out
    accuracy went 100% (untrained) -> 75% (trained) -- training made
    classification WORSE while still reducing its own (wrong) loss.

    Margin/hinge loss instead: sustained trials only need c >= margin (no
    penalty once past it, so training can't overshoot into a fragile
    razor's-edge solution); blip trials still need c <= 0 (unchanged,
    already a one-sided push since spike counts can't go negative)."""
    combustion_idx = net.name_to_idx["combustion"]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(net.weight)
    v_ = np.zeros_like(net.weight)
    step = 0
    epoch_losses = []
    for ep in range(epochs):
        tot_loss = 0.0
        for spec in train_trials:
            drive = build_drive_array(net, spec)
            c, cache = combustion_activity(net, drive)

            if spec["sustained"]:
                deficit = max(0.0, margin - c)
                loss = deficit ** 2
                d_loss_d_c = -2.0 * deficit  # 0 once c >= margin, no overshoot penalty
            else:
                loss = c ** 2
                d_loss_d_c = 2.0 * c
            tot_loss += loss

            T = drive.shape[0]
            d_loss_d_spike = np.zeros((T, net.n_neurons))
            d_loss_d_spike[:, combustion_idx] = d_loss_d_c
            grad = net.backward(cache, d_loss_d_spike)

            step += 1
            m = beta1 * m + (1 - beta1) * grad
            v_ = beta2 * v_ + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** step)
            v_hat = v_ / (1 - beta2 ** step)
            net.weight = net.weight - lr * m_hat / (np.sqrt(v_hat) + eps)
            net.weight = np.clip(net.weight, -3.0, 3.0)
        epoch_losses.append(tot_loss / len(train_trials))
    return epoch_losses


def save_trained_spk(net, original_spk_path, out_path):
    """Writes a new .spk with the SAME structure as the original (neuron
    defs, refractory) but the TRAINED weight values -- never overwrites
    the production file."""
    with open(original_spk_path, encoding="utf-8") as f:
        lines = f.readlines()

    weight_by_pair = {(s, d): w for (s, d), w in zip(net.synapse_labels, net.weight)}

    out_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("connect "):
            # connect SRC -> DST weight=W
            parts = stripped.split()
            src, dst = parts[1], parts[3]
            if (src, dst) in weight_by_pair:
                new_w = weight_by_pair[(src, dst)]
                out_lines.append(f"connect {src} -> {dst} weight={new_w:.4f}\n")
                continue
        out_lines.append(line)

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)


if __name__ == "__main__":
    print("=" * 78)
    print("  JET-ENGINE TRAINING LOOP -- fully native backprop, real curriculum")
    print("=" * 78)

    train_trials = make_curriculum(n_sustained=10, n_blip=10, seed=1)
    held_out_trials = make_curriculum(n_sustained=8, n_blip=8, seed=99)
    print(f"train: {len(train_trials)} trials, held-out: {len(held_out_trials)} trials "
          f"(drive range {DRIVE_LOW}-{DRIVE_HIGH}, real RV-scale range used all night)\n")

    # ORIGINAL hand-tuned network, evaluated on the held-out set as the real baseline
    net_original, _ = compile_native(SPK_PATH)
    acc_original = evaluate(net_original, held_out_trials)
    print(f"BASELINE (hand-tuned .spk weights, tonight's manual sweeps): "
          f"held-out accuracy = {acc_original:.1%}\n")

    # TRAIN from scratch (random init), native backprop only
    np.random.seed(2)
    net_trained, ast = compile_native(SPK_PATH)
    net_trained.weight = np.random.uniform(0.3, 1.5, size=net_trained.n_synapses)

    acc_before = evaluate(net_trained, held_out_trials)
    print(f"BEFORE training (random init): held-out accuracy = {acc_before:.1%}")

    losses = native_adam_train_curriculum(net_trained, train_trials, margin=1.0, epochs=15, lr=0.03)
    print(f"training loss by epoch: {[round(l, 3) for l in losses]}")

    acc_after = evaluate(net_trained, held_out_trials)
    print(f"AFTER training:  held-out accuracy = {acc_after:.1%}\n")

    print("=" * 78)
    print(f"COMPARISON on the SAME held-out set:")
    print(f"  hand-tuned (tonight's manual sweeps): {acc_original:.1%}")
    print(f"  gradient-trained (native backprop):   {acc_after:.1%}")
    print("=" * 78)

    save_trained_spk(net_trained, SPK_PATH, TRAINED_SPK_PATH)
    print(f"\nTrained weights saved to {TRAINED_SPK_PATH} (production .spk untouched)")
    for (s, d), w in zip(net_trained.synapse_labels, net_trained.weight):
        print(f"  {s} -> {d}  weight={w:.4f}")
