#!/usr/bin/env python3
"""Stage 4 of device idea #17, extended to the actual M=128 tested in Stages
1-2 (test_fibonacci_chain_coupling.py proved the MECHANISM at N=8; this
extends the same three checks to the real tested scale).

SCOPE, stated up front: this re-runs Stage 4's own hardware-trajectory-fidelity
check (does the synthesized Fibonacci-LUT Verilog module match a Python
fixed-point emulator, bounded and correct) at M=128 -- it does NOT re-run
Stage 1/2's decode-task NMSE inside Verilog (that would additionally require
porting the trained ridge decoder and the outer/inner HOLD-tick task
structure into hardware, a materially bigger, separate undertaking not
attempted here). Physical resonator params (freq=220Hz, damping=0.05) match
chain_reservoir.spk's existing convention and the Verilog backend's own
native real-time clocking (dt=2^-DT_SHIFT), NOT the abstract dt=0.05 used by
the Python reservoir task -- consistent with what the N=8 version of this
test already did.

PRE-REGISTERED: no prediction on numeric error magnitude at this scale (first
time M=128 has been synthesized to Verilog). Capability bar, same as N=8:
(A) LUT confirmed (127 entries, not 2-constant parity). (B) bounded, no
overflow, tracks float reference. (C) simulated hardware x[] EXACTLY matches
the Python fixed-point emulator's x[] at all 128 channels.
"""
import os, subprocess, sys, time, importlib.util
import numpy as np
from spikeling_resonator_verilog import (
    parse_resonators, generate_fibonacci_coupled_module, to_fixed, FRAC_BITS, WIDTH, DT_SHIFT,
    DEFAULT_RESONATOR_GATE_THRESHOLD,
)

HERE = os.path.dirname(os.path.abspath(__file__))
IVERILOG = r"C:\iverilog\bin\iverilog.exe"
VVP = r"C:\iverilog\bin\vvp.exe"

spec = importlib.util.spec_from_file_location(
    "fib_m128", os.path.join(HERE, "..", "research", "fib_noise_combined_check_M128.py"))
fib_m128 = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = [sys.argv[0]]
spec.loader.exec_module(fib_m128)   # reuse, not re-derive, fib_K
sys.argv = _argv

SPK = "chain_reservoir_M128.spk"
OUT_V = "spikeling_fib_chain_M128.v"

resonators, actions = parse_resonators(SPK)
N = len(resonators)
print(f"\nParsed {N} resonators from {SPK}")

K = fib_m128.fib_K(N)                         # EXACT same construction as the M=128 keystone test
bonds = np.array([K[i, i + 1] for i in range(N - 1)])
print(f"Real Fibonacci bonds (N={N}, {N-1} bonds), first 10: {np.round(bonds[:10], 4).tolist()}")

t0 = time.time()
generate_fibonacci_coupled_module(resonators, OUT_V, bonds.tolist())
print(f"Verilog generation: {time.time()-t0:.1f}s, wrote {OUT_V} ({os.path.getsize(OUT_V)} bytes)")
text = open(OUT_V, encoding="utf-8").read()

# ---------------------------------------------------------------------------
# (A) STORAGE CLAIM
# ---------------------------------------------------------------------------
n_bond_case_lines = text.count(": bond_of = BOND_")
has_parity_select = "j[0] == 1'b0" in text
print("\n--- (A) storage claim ---")
print(f"  bond LUT entries: {n_bond_case_lines} (expected N-1={N-1})")
print(f"  2-constant parity select present (would be WRONG): {has_parity_select}")
a_pass = n_bond_case_lines == N - 1 and not has_parity_select
print(f"  (A) {'CONFIRMED' if a_pass else 'NOT CONFIRMED -- investigate before trusting B/C'}\n")

# ---------------------------------------------------------------------------
# (B) FIXED-POINT-VS-FLOAT
# ---------------------------------------------------------------------------
def q(val): return to_fixed(val, FRAC_BITS)
def mul_shift(a, b, shift=FRAC_BITS): return (a * b) >> shift

K1 = [q(r.k1) for r in resonators]
K2 = [q(r.k2) for r in resonators]
K3 = [q(r.k3) for r in resonators]
dt = 2.0 ** -DT_SHIFT
BOND_Q = [q(float(b) * dt) for b in bonds]

def fixed_point_step(x, v, drive_q):
    x_new, v_new = [0] * N, [0] * N
    for i in range(N):
        k1x = mul_shift(K1[i], x[i]); k2v = mul_shift(K2[i], v[i]); k3d = mul_shift(K3[i], drive_q)
        cleft = mul_shift(BOND_Q[i - 1], x[i - 1]) if i > 0 else 0
        cright = mul_shift(BOND_Q[i], x[i + 1]) if i < N - 1 else 0
        vn = v[i] - k1x - k2v + k3d + cleft + cright
        xn = x[i] + (vn >> DT_SHIFT)
        v_new[i], x_new[i] = vn, xn
    return x_new, v_new

def float_step(x, v, drive, freq_hz, damping):
    x_new, v_new = np.zeros(N), np.zeros(N)
    omega = 2 * np.pi * freq_hz
    for i in range(N):
        k1x = (omega ** 2) * dt * x[i]; k2v = (2 * damping * omega) * dt * v[i]
        k3d = resonators[i].coupling * dt * drive
        cleft = float(bonds[i - 1]) * dt * x[i - 1] if i > 0 else 0
        cright = float(bonds[i]) * dt * x[i + 1] if i < N - 1 else 0
        vn = v[i] - k1x - k2v + k3d + cleft + cright
        xn = x[i] + vn * dt
        v_new[i], x_new[i] = vn, xn
    return x_new, v_new

T = 3000
rng = np.random.default_rng(0)
t_axis = np.arange(T) * dt
drive_signal = 1.0 * np.sin(2 * np.pi * 220 * t_axis) + 0.1 * rng.standard_normal(T)

x_fp = [0] * N; v_fp = [0] * N
x_fl = np.zeros(N); v_fl = np.zeros(N)
X_FP_max = 0.0
drive_q_seq = []
overflowed = False
t0 = time.time()
for t in range(T):
    drive_q = q(float(drive_signal[t])); drive_q_seq.append(drive_q)
    x_fp, v_fp = fixed_point_step(x_fp, v_fp, drive_q)
    x_fl, v_fl = float_step(x_fl, v_fl, drive_signal[t], resonators[0].freq_hz, resonators[0].damping)
    xfp_float = [xi / (1 << FRAC_BITS) for xi in x_fp]
    X_FP_max = max(X_FP_max, max(abs(v) for v in xfp_float))
    if any(abs(xi) > (1 << (WIDTH - 1)) for xi in x_fp): overflowed = True
print(f"--- (B) fixed-point-vs-float, {T}-step run, N={N} ({time.time()-t0:.1f}s) ---")
print(f"  overflow at any step: {overflowed}")
print(f"  max |x| reached: {X_FP_max:.5f} (bounded: {X_FP_max < 100})")
final_err = np.abs(np.array([xi / (1 << FRAC_BITS) for xi in x_fp]) - x_fl)
rel_err = final_err / (np.max(np.abs(x_fl)) + 1e-12)
print(f"  final-step max relative error: {np.max(rel_err):.4%}   mean: {np.mean(rel_err):.4%}")
b_pass = (not overflowed) and X_FP_max < 100 and np.max(rel_err) < 0.05
print(f"  (B) {'PASS' if b_pass else 'FAIL/MARGINAL'}\n")

# ---------------------------------------------------------------------------
# (C) ACTUAL HARDWARE SIMULATION at M=128
# ---------------------------------------------------------------------------
print("--- (C) actual Verilog simulation at M=128 (iverilog + vvp) ---")
hex_path = "fib_drive_M128.hex"
with open(hex_path, "w") as f:
    for dq in drive_q_seq:
        f.write(format(dq & 0xFFFFFFFF, "08x") + "\n")

tb_path = "tb_fib_chain_M128.v"
with open(tb_path, "w") as f:
    f.write(f"""`timescale 1ns/1ps
module tb;
    localparam N = {N};
    localparam WIDTH = {WIDTH};
    localparam STEPS = {T};
    reg clk = 0;
    reg rst = 1;
    reg signed [WIDTH-1:0] drive_mem [0:STEPS-1];
    reg signed [WIDTH-1:0] drive;
    wire [N-1:0] detected;

    spikeling_resonators dut (.clk(clk), .rst(rst), .drive(drive), .detected(detected));
    always #5 clk = ~clk;

    integer t;
    initial begin
        $readmemh("{hex_path}", drive_mem);
        @(negedge clk); rst = 0;
        for (t = 0; t < STEPS; t = t + 1) begin
            drive = drive_mem[t];
            @(negedge clk);
        end
        for (t = 0; t < N; t = t + 1) begin
            $display("XFINAL %0d %0d", t, $signed(dut.x[t]));
        end
        $finish;
    end
endmodule
""")

sim_out = "fib_chain_M128_sim.exe"
t0 = time.time()
r1 = subprocess.run([IVERILOG, "-g2012", "-o", sim_out, OUT_V, tb_path], cwd=HERE, capture_output=True, text=True)
print(f"  iverilog compile: {'OK' if r1.returncode == 0 else 'FAILED'} ({time.time()-t0:.1f}s)")
if r1.returncode != 0:
    print(r1.stdout[-3000:]); print(r1.stderr[-3000:])
    print("  (C) FAILED -- compile error, see above.")
else:
    t0 = time.time()
    r2 = subprocess.run([VVP, sim_out], cwd=HERE, capture_output=True, text=True, timeout=600)
    print(f"  vvp run: {'OK' if r2.returncode == 0 else 'FAILED'} ({time.time()-t0:.1f}s)")
    xfinal_sim = {}
    for line in r2.stdout.splitlines():
        if line.startswith("XFINAL"):
            _, idx, val = line.split()
            xfinal_sim[int(idx)] = int(val)
    if len(xfinal_sim) != N:
        print(f"  (C) FAILED -- expected {N} XFINAL lines, got {len(xfinal_sim)}.")
        print(r2.stdout[-2000:]); print(r2.stderr[-2000:])
    else:
        sim_vals = [xfinal_sim[i] for i in range(N)]
        emu_vals = x_fp
        n_mismatch = sum(1 for a, b in zip(sim_vals, emu_vals) if a != b)
        max_abs_diff = max(abs(a - b) for a, b in zip(sim_vals, emu_vals))
        print(f"  channels compared: {N}")
        print(f"  channels with EXACT integer match: {N - n_mismatch}/{N}")
        print(f"  max abs diff (fixed-point units) across any mismatched channel: {max_abs_diff}")
        print(f"  first 8 sim  values: {sim_vals[:8]}")
        print(f"  first 8 emu  values: {emu_vals[:8]}")
        c_pass = n_mismatch == 0
        print(f"  (C) {'PASS -- all 128 channels exact integer match' if c_pass else f'{n_mismatch} channel(s) mismatched'}")

print("\n--- overall verdict, M=128 ---")
print(f"  (A) LUT confirmed: {a_pass}")
print(f"  (B) fixed-point tracks float: {b_pass}")
print("  (C) see above.")
