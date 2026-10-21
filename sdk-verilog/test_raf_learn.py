"""
test_raf_learn.py -- verifies the RAF on-chip learning rule two ways:

  1. bit-parity: the Verilog (spikeling_raf_learn.v) and the Python twin
     (spikeling_raf_learn_model.py::RAFLearnCore) produce byte-identical
     learned-parameter trajectories on the same closed-loop task.
  2. convergence: both drive a quadratic loss L = sum_j (param_j - target_j)^2
     to below tolerance within a fixed window budget.

Needs Icarus Verilog (iverilog -g2012 / vvp) on PATH for part 1; part 2 runs
model-only if iverilog is absent.

    python test_raf_learn.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

from spikeling_raf_learn_model import RAFLearnCore, q, f2, FRAC

FREQS = [311.0, 523.0, 881.0, 1483.0]
TARGETS = [0.35, -0.30, 0.20, -0.15]
WINDOW, SETTLE, STEP_SHIFT = 4000, 2000, 10


def model_run(n_windows):
    c = RAFLearnCore(FREQS, [0.05] * len(FREQS), [0.0] * len(FREQS),
                     window=WINDOW, settle=SETTLE, step_shift=STEP_SHIFT, mode="SIGN")
    rows, w = [], 0
    while w < n_windows:
        po = c.param_out()
        acc = 0
        for j, t in enumerate(TARGETS):
            d = po[j] - q(t)
            acc += (d * d) >> FRAC
        if c.tick(acc):
            w += 1
            rows.append(tuple(c.param))
    return rows


def verilog_run(n_windows):
    iv = shutil.which("iverilog") or shutil.which("iverilog", path=r"C:\iverilog\bin")
    vv = shutil.which("vvp") or shutil.which("vvp", path=r"C:\iverilog\bin")
    if not (iv and vv):
        return None
    tgt_hex = [f"32'sh{(q(t) & 0xffffffff):08x}" for t in TARGETS]
    tb = f"""`timescale 1ns/1ps
module tb;
  localparam N={len(FREQS)}, WIDTH=32, FRAC={FRAC}, WINDOW={WINDOW}, NWIN={n_windows};
  reg clk=0, rst=1; reg signed [WIDTH-1:0] loss;
  wire signed [WIDTH-1:0] param_out [0:N-1]; wire signed [WIDTH-1:0] param [0:N-1]; wire updated;
  spikeling_raf_learn dut(.clk(clk),.rst(rst),.loss(loss),.param_out(param_out),.param(param),.updated(updated));
  always #5 clk=~clk;
  reg signed [WIDTH-1:0] target [0:N-1];
  initial begin {' '.join(f'target[{i}]={h};' for i, h in enumerate(tgt_hex))} end
  integer i,t,w; reg signed [63:0] d,sq,acc;
  initial begin
    @(negedge clk); rst=0; w=0;
    for (t=0;t<NWIN*WINDOW+20;t=t+1) begin
      acc=0; for(i=0;i<N;i=i+1) begin d=param_out[i]-target[i]; sq=(d*d)>>>FRAC; acc=acc+sq; end
      loss=acc[WIDTH-1:0];
      @(negedge clk);
      if (updated) begin w=w+1; $display("W %0d %0d %0d %0d %0d", w, param[0],param[1],param[2],param[3]); end
    end
    $finish;
  end
endmodule
"""
    d = tempfile.mkdtemp()
    tbf = os.path.join(d, "tb.v")
    vvpf = os.path.join(d, "a.vvp")
    open(tbf, "w").write(tb)
    subprocess.run([iv, "-g2012", "-o", vvpf, "spikeling_raf_learn.v", tbf], check=True)
    out = subprocess.run([vv, vvpf], capture_output=True, text=True, check=True).stdout
    shutil.rmtree(d, ignore_errors=True)
    rows = []
    for ln in out.splitlines():
        p = ln.split()
        if p and p[0] == "W":
            rows.append(tuple(int(x) for x in p[2:6]))
    return rows


def main():
    N_PARITY, N_CONV = 40, 200
    if not os.path.exists("spikeling_raf_learn.v"):
        print("spikeling_raf_learn.v missing -- run: python gen_raf_learn.py")
        sys.exit(1)

    m = model_run(N_PARITY)
    v = verilog_run(N_PARITY)
    if v is None:
        print("[1] bit-parity: SKIPPED (no iverilog/vvp on PATH)")
    else:
        mismatch = sum(1 for a, b in zip(m, v) if a != b) + abs(len(m) - len(v))
        for i, (a, b) in enumerate(zip(m, v), 1):
            if a != b:
                print(f"    win {i}: model {a}  verilog {b}")
        print(f"[1] bit-parity: {len(m)} windows, {mismatch} mismatches -- "
              + ("PASS" if mismatch == 0 else "FAIL"))
        if mismatch:
            sys.exit(1)

    conv = model_run(N_CONV)
    fin = [f2(x) for x in conv[-1]]
    err = max(abs(fin[j] - TARGETS[j]) for j in range(len(TARGETS)))
    print(f"[2] convergence: max|param-target| after {N_CONV} windows = {err:.4f}  "
          + ("PASS" if err < 0.05 else "FAIL"))
    sys.exit(0 if err < 0.05 else 1)


if __name__ == "__main__":
    main()
