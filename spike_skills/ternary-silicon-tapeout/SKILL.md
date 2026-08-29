---
name: ternary-silicon-tapeout
description: Use when working on the ternary-silicon-tapeout project's Verilog, TinyTapeout/SkyWater submission, or balanced-ternary logic design.
keywords: ternary-silicon-tapeout, TinyTapeout, SkyWater, SKY130, balanced ternary, half-adder
---

# Ternary silicon tapeout

Real balanced-ternary half-adder built in Verilog
(`ternary_half_adder.v` + testbench), 2-bit-per-trit encoding (10 = -1,
00 = 0, 01 = +1, 11 = invalid), verified 16/16 via Icarus Verilog
simulation -- not just written, actually simulated and checked.

Scope boundary that matters: Tier 1 (balanced ternary logic in standard
CMOS) is achievable at accessible budget via TinyTapeout/SkyWater SKY130.
Tier 2 (true multi-valued voltage-level ternary requiring CNTFETs or
memristors) is NOT achievable at accessible budget -- don't scope new
tapeout work into Tier 2 territory without flagging that constraint
explicitly.
