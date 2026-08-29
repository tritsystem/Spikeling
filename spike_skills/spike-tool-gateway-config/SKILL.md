---
name: spike-tool-gateway-config
description: Use when adding a new tool category or provider to Spike's own tool-gateway config (spike_tools.yaml).
keywords: spike_tools.yaml, spike_tool_gateway, config always wins
---

# Spike tool gateway config discipline

`spike_tools.yaml` is the sole source of truth for tool-tier routing --
modeled directly on Nous Portal's real "config always wins over ambient
credentials" rule. Never add an "auto" value (hard-rejected at load
time by design) and never add an ambient-credential-detection fallback
path -- that reintroduces the exact failure class Hermes hit this
portfolio with (an implicit "auto" resolution silently falling through
to a stale hardcoded default nobody configured on purpose). A new
category is a real error to resolve until it's explicitly added here,
not a silent default.
