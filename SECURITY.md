# Security Policy

## Reporting a vulnerability

Email **gbranaa4@gmail.com** with "SECURITY" in the subject. Do not open a
public issue for a security report.

Include what you found, how to reproduce it, and the impact. You'll get an
acknowledgement within a few days. Single-maintainer project, no bug-bounty
budget — you get a fix, changelog credit if you want it, and a straight answer.

## Scope

Worth reporting:

- the `.spk` compiler or runtime executing arbitrary code from a crafted
  network file (it should compile and simulate, not run host code)
- the generated **C** / **Verilog** emitting something unsafe from attacker-
  controlled `.spk` input
- the MCP server (`mcp_server.py`) exposing more than its documented tools, or
  accepting input that escapes the intended sandbox
- path traversal in the compiler's output-file emission

## Supported versions

The latest tag on `main` is supported. Earlier tags are not patched.
