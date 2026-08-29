---
name: spiking-orchestrator-routing
description: Use when modifying spiking_orchestrator.py's neuron cascade, tool-tier routing, or specialist spawning.
keywords: spiking_orchestrator, neuron cascade, tool tier, spike_tools.yaml, refractory
---

# Spiking orchestrator routing

This is the real, verified LIF-neuron-cascade agent router --
`test_pyspike_orchestrator_parity.py` (10/10) proves the pyspike-built
brain matches the original .spk-parsed brain exactly, including the
correction loop and refractory behavior. Re-run it after ANY change
here, not just changes that look related.

Tool-tier resolution is config-driven via `spike_tool_gateway.py` +
`spike_tools.yaml` (4 tiers: review / research_review / research /
code) -- config is the sole source of truth, "auto" is a hard error, no
ambient-credential fallback. Don't reintroduce implicit auto-detection;
that's the exact failure class this replaced.

Real ordering gotcha found by testing: handlers run synchronously as
neurons fire, and the runtime runs a fired neuron's action BEFORE
propagating its spike -- so a downstream neuron firing from the SAME
cascade as its own trigger can hit that trigger's still-refractory-
locked state and silently skip a fire that should happen (found: a
dynamically-spawned specialist's synapse into Reviewer tried to fire it
at the same instant Reviewer was still refractory from firing moments
earlier, so the review silently never happened twice when it should
have).
