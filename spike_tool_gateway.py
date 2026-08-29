#!/usr/bin/env python
"""
spike_tool_gateway.py — Spike's tool-category resolver, modeled directly on
Nous Portal's real Tool Gateway architecture (checked against its actual
docs, not assumed -- see vault/Projects/spike-tool-gateway-and-skills.md).

THE ONE RULE THIS FILE EXISTS TO ENFORCE, AS CODE NOT CONVENTION:
    Config is the ONLY source of truth for which provider handles a tool
    category. No ambient-credential auto-detection. No implicit fallback
    chain. No "auto" resolution that silently falls through to a hardcoded
    default nobody can see by reading one file.

WHY THIS IS A HARD RULE HERE, NOT A STYLE PREFERENCE:
    This session spent real, extensive effort chasing exactly the failure
    mode this rule prevents: Hermes' own auxiliary-task routing had an
    "auto" resolution that silently fell through to a dead hardcoded
    OpenRouter model (stealth/ox-alpha, retired, confirmed via OpenRouter's
    own live catalog) whenever the primary provider failed -- and the user
    could not tell this was happening from config.yaml alone; it took
    reading auxiliary_client.py's actual source to find. Nous Portal's own
    docs independently state the same rule for a different reason (a
    FAL_KEY sitting in .env is ignored while image_gen.provider: nous is
    set) -- config always wins, credential presence is irrelevant. Spike
    adopts this as a hard constraint, enforced by refusing to load a config
    that violates it, not merely as a documented intention that code can
    quietly drift away from.

WHAT THIS DOES NOT DO YET:
    Actually route a tool CALL anywhere. This is deliberately just the
    resolver -- category -> provider name, config-only, real error on
    anything unconfigured or set to "auto". Wiring resolved providers to
    real tool execution is a separate, later step (see the vault scope
    doc's Phase 0 boundary -- this IS Phase 0, not more than it).
"""

import os

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "spike_tool_gateway.py needs PyYAML (pip install pyyaml) -- not "
        "silently falling back to a hand-rolled parser, that would be "
        "exactly the kind of implicit behavior this file exists to avoid."
    ) from e

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "spike_tools.yaml")


class ToolGatewayError(Exception):
    """Raised for any config problem -- unconfigured category, malformed
    entry, or a rejected "auto" value. Always a real, named error, never a
    silent default."""


def load_tool_config(path: str = CONFIG_PATH) -> dict:
    """Load and validate spike_tools.yaml. Real validation, not just a
    YAML parse -- every category must have a non-empty string "provider"
    that is not the literal "auto"."""
    if not os.path.isfile(path):
        raise ToolGatewayError(
            f"No tool config found at {path}. Spike will not guess a "
            f"default provider for any category -- create the file first."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or not raw:
        raise ToolGatewayError(f"{path} is empty or not a mapping of categories.")

    validated: dict[str, str] = {}
    for category, entry in raw.items():
        if not isinstance(entry, dict) or "provider" not in entry:
            raise ToolGatewayError(
                f"Category '{category}' in {path} has no 'provider' key. "
                f"Every category must explicitly name one."
            )
        provider = entry["provider"]
        if not isinstance(provider, str) or not provider.strip():
            raise ToolGatewayError(
                f"Category '{category}'s provider is empty or not a string."
            )
        if provider.strip().lower() == "auto":
            raise ToolGatewayError(
                f"Category '{category}' is set to \"auto\" -- rejected on "
                f"purpose. Name the real provider explicitly; see this "
                f"file's module docstring for why \"auto\" is exactly the "
                f"failure mode Spike's tool gateway refuses to reintroduce."
            )
        validated[category] = provider.strip()

    return validated


def resolve_tool_provider(category: str, config_path: str = CONFIG_PATH) -> str:
    """The one real entry point. Returns the configured provider for
    `category`. Reads ONLY the config file -- no environment variable, no
    credential-presence check, no fallback chain is consulted, ever. An
    unconfigured category is a real error, not a silent guess."""
    config = load_tool_config(config_path)
    if category not in config:
        raise ToolGatewayError(
            f"No provider configured for category '{category}' in "
            f"{config_path}. Known categories: {sorted(config.keys())}. "
            f"Add it explicitly -- Spike does not guess."
        )
    return config[category]


if __name__ == "__main__":
    import sys
    cfg = load_tool_config()
    print(f"Loaded {len(cfg)} categor{'y' if len(cfg) == 1 else 'ies'} from {CONFIG_PATH}:")
    for cat, provider in sorted(cfg.items()):
        print(f"  {cat:20s} -> {provider}")
    if len(sys.argv) > 1:
        cat = sys.argv[1]
        try:
            print(f"\nresolve_tool_provider({cat!r}) = {resolve_tool_provider(cat)!r}")
        except ToolGatewayError as e:
            print(f"\nresolve_tool_provider({cat!r}) raised ToolGatewayError: {e}")
