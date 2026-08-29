---
name: observe-ternary-search
description: Use when working on OBSERVE / 012-ternary's semantic code search, ternary compression, or the trit_mcp_server.py MCP server.
keywords: OBSERVE, 012-ternary, trit_mcp_server, ternary compression, chunk provenance, entanglement database
---

# OBSERVE / 012-ternary

Semantic code search engine with ternary compression, a chunk
provenance/lineage layer (chunk_provenance.py), hybrid search, an
incremental indexer, and an entanglement (cross-project relationship)
database.

Workaround for when `mcp__observe__*` tools aren't loaded in a session:
call `trit_mcp_server.py` directly via
`docker run --entrypoint python ...` -- confirmed working, has found
real bugs this way, not just a theoretical fallback.

Before claiming something is a new finding anywhere in this portfolio,
run `research find "<topic>"` (the research-index CLI) against the whole
memory folder first -- it exists specifically to stop re-deriving a
finding by hand that's already been made and recorded.
