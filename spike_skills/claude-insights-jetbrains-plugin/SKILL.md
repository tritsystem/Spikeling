---
name: claude-insights-jetbrains-plugin
description: Use when working on the Claude Insights JetBrains/PyCharm plugin.
keywords: Claude Insights, JetBrains plugin, PyCharm plugin
---

# Claude Insights JetBrains plugin

Real, working PyCharm plugin, confirmed end-to-end (right-click code ->
inline Claude review). Two real Windows-specific bugs were found and
fixed: GUI-app PATH resolution (a GUI-launched process doesn't inherit
the same PATH a terminal session would), and cmd.exe mangling multi-
line CLI arguments -- fixed by passing the payload via stdin instead of
as a command-line argument. Separately, computer-use hit a real,
persistent PyCharm rendering wall during testing -- not every
verification path works for every IDE; that limitation is real, not a
one-off flake.
