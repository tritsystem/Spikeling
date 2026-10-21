# RAF on-chip learning rule — sdk-verilog backend

`spikeling_raf_learn.v` is a synthesizable fixed-point module that **learns** —
it adjusts a set of scalar parameters to reduce a scalar loss, with **no backward
pass**, using only forward evaluations, a dither, and a bank of the same damped
resonators already in `spikeling_resonators.v`.

Origin: `Spikeling/research/raf_gradient/` (rungs 1 → 2e). That study characterised
a resonator bank in the backward path as a substrate-feasible, **sign-faithful**
learning signal for hard spiking networks. It never matches surrogate-gradient BPTT
(rung 2e: ~10–15 % R² short, because it measures the *true* gradient of a hard
network, which is a worse descent direction than the biased surrogate) — but a chip
has no backward pass, so the comparison there is **"vs no on-chip learning at all,"**
which it wins.

## The rule

One gradient step = one **measurement window** of `WINDOW` clock ticks. Per tick `t`:

```
dref_j(t)     = sine_lut[phase_j >> (PHASE_BITS-8)]        # per-channel DDS dither reference
param_out_j   = param_j + (DITHER_AMP * dref_j) >> FRAC    # the app forward-evaluates THESE
drive(t)      = (loss(t) - loss_ema) << DRIVE_GAIN_SHIFT   # DC-removed scalar loss, one shared wire
<resonator_j stepped by the shared drive>                  # exact spikeling_resonators.v math
if t >= SETTLE:  vref_acc_j += (v_j * dref_j) >> FRAC      # velocity–reference correlation (rung 1b)
phase_j      += PINC_j
```

At `t == WINDOW-1` (SIGN mode — the default):

```
param_j -= STEP * sign(vref_acc_j)     # signSGD: fixed step, NO multiplier in the update path
<reset x, v, energy_ema, vref_acc, win_ctr; pulse `updated`>
```

Why velocity, not energy: a resonator driven at its own resonance has **velocity in
phase with the drive**, so `sign(<v_j · dref_j>)` recovers whether the loss rose or
fell when `param_j` went up. `energy_ema` (which squares `x`) is sign-blind — it is
still computed (free, shared with the detector path) and is available for a SCALED
update mode, but rungs 2c/2d showed `raf_sign ≈ raf_full`, so SIGN is the recommended
hardware default and the only mode the generated `.v` implements.

## Interface

```
spikeling_raf_learn #(N, WINDOW, SETTLE, STEP_SHIFT, ...) (
    input  clk, rst,
    input  signed [31:0] loss,             // Q14.18 scalar loss of param_out, from the app
    output signed [31:0] param_out [0:N-1], // Q14.18 dithered params — the app evaluates THESE
    output signed [31:0] param     [0:N-1], // Q14.18 current learned values
    output               updated            // 1-cycle pulse each time param is updated
);
```

The application (an SNN forward path, a controller, a physical readout) computes
`loss` from `param_out` and feeds it back. One-cycle loop latency is fine — a fixed
phase lag rotates the I/Q slightly but does not flip the velocity-phase sign for small
lag (rung 1b).

## What "param" is is the app's choice

The module treats every parameter identically: a scalar knob dithered at `f_j` with a
resonator reading the shared loss. Whether knob `j` is a hidden-neuron bias current, a
synaptic weight, a threshold, or a phase-shifter voltage is entirely the app's wiring.
- **Node perturbation** (rung 2b/2c): the `N` knobs are hidden-unit biases;
  `ΔW1_ij = -lr · sign(vref_acc_j) · x_i` is done in app logic from the `updated`
  pulse and the locally-available presynaptic activity `x_i`.
- **Weight perturbation** (rung 2e `raf_weights`): the `N` knobs are the weights
  themselves — unbiased, but costs `n_in ×` more frequencies.

## Fixed-point

Identical to `spikeling_resonators.v`: Q14.18 for `x`, `v`, `param`, `dref`;
Q28.36 in a 64-bit register for `energy_ema`; `dt = 2^-15`; energy decay is
`>> ALPHA_SHIFT` (derived from `dt` and the 2.5 ms time constant, not hand-picked —
see the resonator generator docstring for the sample-rate bug this avoids).

**Two overflow traps found while building this (both fixed in `gen_raf_learn.py`):**
1. `DITHER_AMP * dref` feeds a 32-bit add, so the product was truncated to 32 bits
   *before* the `>> FRAC` — it overflows near the sine peaks and **sign-inverts the
   dither**, which silently inverts the learning direction. Fixed with an explicit
   `wire signed [2*WIDTH-1:0] dith_full`.
2. `loss_ema` must be seeded with a **blocking** assign on the first cycle
   (`loss_ema = loss`), or cycle 1 sees `err = loss - 0`, a large transient kick.

The DDS phase accumulator is `PHASE_BITS = 28` wide specifically so
`round(f · dt · 2^28)` has sub-Hz error at the lowest channel — a narrow accumulator
quantises the low-frequency dither off its resonator's tuned frequency and flips the
readout sign.

## Verification

`python test_raf_learn.py`:
1. **bit-parity** — `spikeling_raf_learn.v` under Icarus Verilog vs the Python twin
   `spikeling_raf_learn_model.py::RAFLearnCore` produce **byte-identical** learned-
   parameter trajectories on the same closed-loop task (40 windows, 0 mismatches).
2. **convergence** — quadratic loss `L = Σ(param_j − target_j)²`, 4 channels dithered
   simultaneously, `max|param − target|` → **0.009** (Q14.18 tol 0.05) in 200 windows.

Icarus needs `-g2012` (packed-array ports), same as `spikeling_resonators.v`.

## Cost (per channel, Q14.18)

| resource | count |
|---|---|
| 32×32 multipliers | 4 — three resonator (`K1·x`, `K2·v`, `K3·drive`) + one correlation (`v·dref`) |
| + 1 shared | dither (`DITHER_AMP·dref`, 32×32→wide) |
| adders | ~5 |
| state | 2×32 (x,v) + 64 (energy) + 28 (phase) + 48 (vref_acc) ≈ 220 bits |
| ROM | one shared 256×32 sine LUT for the whole bank |
| update path | **no multiplier** (SIGN mode) — a sign test + add |

For `N = 8`: ~33 multipliers, ~1.8 kbit state, one small ROM. Modest for any FPGA;
plausible as a learning coprocessor block next to a neuromorphic array.

## Honest limits (carried from the rung ladder)

- **~10–15 % R² / a few-% accuracy below surrogate BPTT**, and no amount of tuning
  closes it (rung 2e) — it is the price of measuring the true gradient instead of
  assuming a surrogate. Only worth it where BPTT can't run.
- **Window latency**: one parameter update per `WINDOW` (~`SETTLE` + a few resonator
  settling times) ticks. At `dt = 2^-15` and `WINDOW = 4000` that's ~122 ms of
  wall-clock per gradient step in the sim's time base — set `WINDOW`/`SETTLE` to your
  clock and resonator Q.
- **Frequency budget**: rung 1 P3 saw the multiplexed readout degrade past ~32
  simultaneous channels per bank. Beyond that, time-multiplex subsets or use multiple
  banks.
- **Multi-layer credit assignment** beyond one hidden layer + linear readout is
  untested here — the global `loss`/error broadcast is assumed; wiring it across
  layers on real hardware is an open engineering question.
- SIGN mode has no natural stopping criterion — it dithers around the optimum at ±1
  `STEP`. Freeze on an external "converged" signal or decay `STEP_SHIFT`.

## Regenerate for a different bank

```
python gen_raf_learn.py 200 340 560 900 1500 2400     # 6 channels at these dither freqs
iverilog -g2012 -o raf_learn.vvp spikeling_raf_learn.v tb_raf_learn.v && vvp raf_learn.vvp
```

Pick dither frequencies spread across a band, mutually non-harmonic where possible
(intermodulation `f_i ± f_j` landing on `f_k` is cross-talk). The generator bakes each
channel's resonator constants and DDS increment from its frequency.
