#!/usr/bin/env python
"""
test_spike_tool_gateway.py — real verification of spike_tool_gateway.py's
central claim: config is the ONLY source of truth, nothing else can
influence resolution. Uses real temp config files, not mocks, and a real
environment variable to prove it has zero effect.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spike_tool_gateway import (  # noqa: E402
    load_tool_config, resolve_tool_provider, ToolGatewayError,
)

PASS = 0
FAIL = 0


def check(label, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"PASS  {label}")
    else:
        FAIL += 1
        print(f"FAIL  {label}")


def write_config(text: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False,
                                     encoding="utf-8")
    f.write(text)
    f.close()
    return f.name


# 1. Real category resolves to the real configured provider.
p = write_config("code_execution:\n  provider: claude_code\n")
check("configured category resolves to its real provider",
      resolve_tool_provider("code_execution", p) == "claude_code")
os.unlink(p)

# 2. Changing the config file's value changes the resolved result -- the
# actual "config is the source of truth" claim, not just "it reads once".
p = write_config("web_search:\n  provider: providerA\n")
r1 = resolve_tool_provider("web_search", p)
with open(p, "w", encoding="utf-8") as f:
    f.write("web_search:\n  provider: providerB\n")
r2 = resolve_tool_provider("web_search", p)
check("changing config value changes the resolved result",
      r1 == "providerA" and r2 == "providerB")
os.unlink(p)

# 3. THE central claim: an environment variable that would tempt an
# "auto"-style implementation to auto-detect a different provider has
# ZERO effect on resolution. This is the exact failure mode
# spike_tool_gateway.py exists to refuse to reintroduce.
p = write_config("code_execution:\n  provider: claude_code\n")
os.environ["SPIKE_TEST_FAKE_CREDENTIAL"] = "sk-something-that-should-be-ignored"
os.environ["OPENROUTER_API_KEY"] = "sk-or-fake-should-also-be-ignored"
r = resolve_tool_provider("code_execution", p)
del os.environ["SPIKE_TEST_FAKE_CREDENTIAL"]
del os.environ["OPENROUTER_API_KEY"]
check("ambient credentials in the environment have zero effect on resolution",
      r == "claude_code")
os.unlink(p)

# 4. "auto" is rejected at load time, not silently accepted.
p = write_config("code_execution:\n  provider: auto\n")
try:
    load_tool_config(p)
    check('"auto" provider value is rejected', False)
except ToolGatewayError:
    check('"auto" provider value is rejected', True)
os.unlink(p)

# 5. An unconfigured category is a real error, not a silent default.
p = write_config("code_execution:\n  provider: claude_code\n")
try:
    resolve_tool_provider("image_gen", p)
    check("unconfigured category raises, does not silently default", False)
except ToolGatewayError:
    check("unconfigured category raises, does not silently default", True)
os.unlink(p)

# 6. A missing config file is a real, clear error, not an empty-dict fallback.
try:
    load_tool_config("/nonexistent/path/spike_tools_does_not_exist.yaml")
    check("missing config file raises, does not silently return {}", False)
except ToolGatewayError:
    check("missing config file raises, does not silently return {}", True)

# 7. The real, shipped spike_tools.yaml actually loads and has the 4 real
# tool tiers now wired into spiking_orchestrator.py's _run_real_agent().
real_cfg = load_tool_config()
check("real spike_tools.yaml has exactly the 4 real tool tiers",
      set(real_cfg.keys()) == {"review", "research_review", "research", "code"})
check("each tier resolves to its real voice_commands.py tool-list attribute name",
      real_cfg == {
          "review": "REVIEW_TOOLS",
          "research_review": "RESEARCH_REVIEW_TOOLS",
          "research": "RESEARCH_CODE_TOOLS",
          "code": "CLAUDE_CODE_TOOLS",
      })

print(f"\n=== RESULT: {PASS} passed, {FAIL} failed, {PASS + FAIL} total ===")
print("OVERALL:", "PASS" if FAIL == 0 else "FAIL")
sys.exit(0 if FAIL == 0 else 1)
