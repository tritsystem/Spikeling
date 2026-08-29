---
name: observe-docker-workaround
description: Use when mcp__observe__* tools aren't loaded in a session but OBSERVE's search/query capability is still needed.
keywords: OBSERVE Docker, trit_mcp_server, docker run --entrypoint
---

# OBSERVE Docker CLI workaround

When the `mcp__observe__*` tools aren't loaded, call
`trit_mcp_server.py` directly via
`docker run --entrypoint python trit_mcp_server.py ...` instead of
giving up on OBSERVE's capability for that session. Confirmed working,
and has found real bugs this way -- a genuine fallback, not a
degraded/theoretical one.
