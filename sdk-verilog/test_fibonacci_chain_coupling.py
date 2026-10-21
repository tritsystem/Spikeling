#!/usr/bin/env python3
"""Stage 4 of device idea #17's build plan: validate generate_fibonacci_coupled_module()
(the new per-bond-LUT coupling generator, added because generate_coupled_module()'s
existing SSH 2-constant parity select is WRONG for a real Fibonacci word -- confirmed
by construction, Fibonacci words are non-periodic).

THREE checks, going one further than test_chain_coupling.py's prior SSH validation
(which explicitly couldn't do part C -- "no iverilog installed here" -- iverilog IS
now installed at C:\\iverilog):

  (A) STORAGE CLAIM -- the generated file uses a real N-1-entry bond LUT, not a
      2-constant parity select (parity would be WRONG here).
  (B) FIXED-POINT-VS-FLOAT -- Python integer emulator of the exact Verilog ops,
      vs a floating-point reference, both using the REAL Fibonacci bonds.
  (C) ACTUAL HARDWARE SIMULATION -- compile the generated .v with iverilog, run it
      with vvp, and compare its own printed final register values against the
      Python fixed-point emulator's (B) -- bit-exact expected, since both implement
      identical integer operations. This is the real conformance check (B) alone
      cannot provide: it proves the SYNTHESIZABLE VERILOG itself behaves as
      predicted, not just a Python model of what it should do.

PRE-REGISTERED: no prediction on numeric error magnitude (first time this path is
built or run). Capability bar: (A) LUT confirmed, not parity. (B) bounded, no
overflow, tracks float reference. (C) simulated hardware x[] EXACTLY matches the
Python fixed-point emulator's x[] (both are integer arithmetic -- any mismatch is
a real bug, not expected rounding noise).

N=8, matching chain_reservoir.spk's existing convention (same file used by
test_chain_coupling.py) -- explicitly NOT the full M=128 tested in Stage 1/2.
"""
import os, subprocess, sys, importlib.util
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
spec.loader.exec_module(fib_m128)   # re-runs the M=128 test too (verbose); harmless, real reuse not re-derivation
sys.argv = _argv

SPK = "chain_reservoir.spk"
OUT_V = "spikeling_fib_chain.v"

resonators, actions = parse_resonators(SPK)
N = len(resonators)
print(f"\nParsed {N} resonators from {SPK}")

K8 = fib_m128.fib_K(N)                       # EXACT same construction as the M=128 keystone test, just N=8
bonds = np.array([K8[i, i + 1] for i in range(N - 1)])
print(f"Real Fibonacci bonds (N={N}, {N-1} bonds): {np.round(bonds, 4).tolist()}")
word_prefix = fib_m128.LONG_WORD[:N - 1]
print(f"Fibonacci word prefix used: {word_prefix}  (NOT period-2 -- e.g. positions 3-4 are both 'A')")

generate_fibonacci_coupled_module(resonators, OUT_V, bonds.tolist())
text = open(OUT_V, encoding="utf-8").read()

# ---------------------------------------------------------------------------
# (A) STORAGE CLAIM
# ---------------------------------------------------------------------------
n_bond_case_lines = text.count(": bond_of = BOND_")   # real numbered entries only, excludes the "default:" line
has_parity_select = "j[0] == 1'b0" in text
print("\n--- (A) storage claim, measured on the generated .v file ---")
print(f"  bond case-statement (LUT) entries: {n_bond_case_lines} (expected N-1={N-1})")
print(f"  2-constant parity select present (would be WRONG here): {has_parity_select}")
if n_bond_case_lines == N - 1 and not has_parity_select:
    print(f"  CONFIRMED: real {N-1}-entry Fibonacci bond LUT emitted, not an incorrect parity select.\n")
else:
    print("  CLAIM NOT CONFIRMED -- investigate before trusting parts B/C.\n")

# ---------------------------------------------------------------------------
# (B) FIXED-POINT-VS-FLOAT, using the REAL Fibonacci bonds
# ---------------------------------------------------------------------------
def q(val): return to_fixed(val, FRAC_BITS)
def mul_shift(a, b, shift=FRAC_BITS): return (a * b) >> shift

GATE_THRESH = q(DEFAULT_RESONATOR_GATE_THRESHOLD)
K1 = [q(r.k1) for r in resonators]
K2 = [q(r.k2) for r in resonators]
K3 = [q(r.k3) for r in resonators]
dt = 2.0 ** -DT_SHIFT
BOND_Q = [q(float(b) * dt) for b in bonds]

def bond_of(j): return BOND_Q[j]

def fixed_point_step(x, v, drive_q):
    x_new, v_new = [0] * N, [0] * N
    for i in range(N):
        k1x = mul_shift(K1[i], x[i]); k2v = mul_shift(K2[i], v[i]); k3d = mul_shift(K3[i], drive_q)
        cleft = mul_shift(bond_of(i - 1), x[i - 1]) if i > 0 else 0
        cright = mul_shift(bond_of(i), x[i + 1]) if i < N - 1 else 0
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
X_FP = np.zeros((T, N)); X_FL = np.zeros((T, N))
drive_q_seq = []
overflowed = False
for t in range(T):
    drive_q = q(float(drive_signal[t])); drive_q_seq.append(drive_q)
    x_fp, v_fp = fixed_point_step(x_fp, v_fp, drive_q)
    x_fl, v_fl = float_step(x_fl, v_fl, drive_signal[t], resonators[0].freq_hz, resonators[0].damping)
    X_FP[t] = [xi / (1 << FRAC_BITS) for xi in x_fp]
    X_FL[t] = x_fl
    if any(abs(xi) > (1 << (WIDTH - 1)) for xi in x_fp): overflowed = True

print("--- (B) fixed-point-vs-float numeric cross-check, driven 3000-step run ---")
print(f"  fixed-point register overflow at any step: {overflowed}")
max_abs = np.max(np.abs(X_FP))
print(f"  max |x| (fixed-point trajectory): {max_abs:.5f} (bounded: {max_abs < 100})")
err = X_FP - X_FL
rel_err = np.abs(err) / (np.max(np.abs(X_FL)) + 1e-12)
print(f"  max relative error: {np.max(rel_err):.4%}   mean relative error: {np.mean(rel_err):.4%}")
b_pass = (not overflowed) and max_abs < 100 and np.max(rel_err) < 0.05
print(f"  (B) {'PASS' if b_pass else 'FAIL/MARGINAL'}\n")

# ---------------------------------------------------------------------------
# (C) ACTUAL HARDWARE SIMULATION -- iverilog + vvp, real execution
# ---------------------------------------------------------------------------
print("--- (C) actual Verilog simulation (iverilog + vvp) ---")
hex_path = "fib_drive.hex"
with open(hex_path, "w") as f:
    for dq in drive_q_seq:
        f.write(format(dq & 0xFFFFFFFF, "08x") + "\n")

tb_path = "tb_fib_chain.v"
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

sim_out = "fib_chain_sim.exe"
compile_cmd = [IVERILOG, "-g2012", "-o", sim_out, OUT_V, tb_path]
r1 = subprocess.run(compile_cmd, cwd=HERE, capture_output=True, text=True)
print("  iverilog compile:", "OK" if r1.returncode == 0 else "FAILED")
if r1.returncode != 0:
    print(r1.stdout); print(r1.stderr)
    print("  (C) FAILED -- could not compile; see errors above.")
else:
    r2 = subprocess.run([VVP, sim_out], cwd=HERE, capture_output=True, text=True)
    print("  vvp run:", "OK" if r2.returncode == 0 else "FAILED")
    xfinal_sim = {}
    for line in r2.stdout.splitlines():
        if line.startswith("XFINAL"):
            _, idx, val = line.split()
            xfinal_sim[int(idx)] = int(val)
    if len(xfinal_sim) != N:
        print(f"  (C) FAILED -- expected {N} XFINAL lines, got {len(xfinal_sim)}. Raw stdout:")
        print(r2.stdout[-2000:])
    else:
        sim_vals = [xfinal_sim[i] for i in range(N)]
        emu_vals = x_fp   # Python fixed-point emulator's own final x[] after T steps
        print(f"  simulated Verilog x[] (raw fixed-point ints): {sim_vals}")
        print(f"  Python emulator x[]   (raw fixed-point ints): {emu_vals}")
        exact_match = sim_vals == emu_vals
        print(f"  (C) EXACT MATCH between real simulated hardware and Python emulator: {exact_match}")
        if exact_match:
            print("  (C) PASS -- the actual synthesizable Verilog reproduces the predicted integer trajectory bit-exactly.")
        else:
            diffs = [abs(a - b) for a, b in zip(sim_vals, emu_vals)]
            print(f"  (C) MISMATCH -- per-channel abs diff (fixed-point units): {diffs}")

print("\n--- overall verdict ---")
print(f"  (A) LUT confirmed: {n_bond_case_lines == N - 1 and not has_parity_select}")
print(f"  (B) fixed-point tracks float: {b_pass}")
print("  (C) see above -- this is the real hardware-execution check.")
