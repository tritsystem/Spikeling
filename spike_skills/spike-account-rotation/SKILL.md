---
name: spike-account-rotation
description: Use when working on spike.py's AccountRotator (dual-Anthropic-account rotation) or its real-run fallback behavior.
keywords: AccountRotator, SPIKE_ANTHROPIC_KEY, account rotation
---

# Spike account rotation

`AccountRotator` cycles across `SPIKE_ANTHROPIC_KEY_1/2` when set. Real
fixed bug worth remembering: an early version incorrectly REQUIRED
rotation keys to be set for ANY real (non-dry-run) run, silently
falling back to dry-run otherwise -- wrong, since the underlying
`spiking_orchestrator.py --real` flag works fine with just the
machine's already-logged-in Claude CLI session, no special keys needed.
Fixed so `real=True` without rotation keys still runs for real on
ambient auth; rotation is a strictly additive opt-in on top, never a
requirement.
