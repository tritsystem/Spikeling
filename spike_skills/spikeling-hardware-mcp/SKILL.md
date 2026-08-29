---
name: spikeling-hardware-mcp
description: Use when running parametric real-hardware experiments through Spikeling's SensorAdapter/BaselineDeviation framework or mcp_server.py.
keywords: SensorAdapter, BaselineDeviation, mcp_server.py, hardware sensor-adapter
---

# Spikeling hardware MCP framework

SensorAdapter/BaselineDeviation hardware layer with a real/simulated
split that matters: acoustic, system_telemetry, and video adapters are
REAL; emg and environmental adapters are SIMULATED. Don't treat an emg
or environmental reading as physically measured without checking which
mode it's actually running in.

Exposed as an MCP server (mcp_server.py) specifically so a new one-off
hardware experiment script doesn't need to be written from scratch --
reach for this framework first.
