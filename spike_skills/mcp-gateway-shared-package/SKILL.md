---
name: mcp-gateway-shared-package
description: Use when extending audit logging or rate limiting to a new MCP server, or modifying mcp-gateway's audit_log.py / rate_limiter.py / wrap.py.
keywords: mcp-gateway, audit_log.py, rate_limiter.py, wrap.py, audit logging, rate limiting
---

# mcp-gateway

Shared audit-log/rate-limit package used by OBSERVE's and Spikeling's
MCP servers. The real point of this package existing at all: gating
(audit logging, rate limiting) should be ONE shared implementation
wired into every MCP server in this portfolio, not reimplemented per
repo. Before writing gating logic for a new MCP server, check whether
mcp-gateway already has the primitive rather than rebuilding it
locally.
