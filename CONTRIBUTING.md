# Contributing

Spikeling is a DSL + runtime for spiking neural networks (`.spk` → Python / C /
Verilog / GDScript), maintained by one person. The canonical package is
[`core/`](core/README.md) — start there.

## Ground rules

1. **`core/` is the source of truth.** Changes to the compiler, runtime, or
   encoder need a test and must keep `python -m core --no-interactive` green.
2. **Backends must agree.** If you change semantics, the Python, C, Verilog and
   GDScript backends should still produce the same spike behaviour for the same
   `.spk` — say which backends you checked.
3. **Measured claims only.** The README's status column ("falsified as a
   differentiator", "no throughput advantage found", etc.) is deliberate. A
   negative result is a fine PR if it's measured and written up honestly.
4. **Disclose AI assistance** if you used it — a line in the PR is enough.

## Setup

```bash
git clone https://github.com/tritsystem/Spikeling
cd Spikeling
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m core --no-interactive                   # smoke test
```

The C / Verilog / GDScript backends and the `ai-apps/` assistants have their own
toolchains — see the README in each folder.

## Before a PR

```bash
python -m core --no-interactive
python -m pytest -q test_pyspike_encoding.py test_pyspike_*_parity.py
```

Many suites need optional hardware or model dependencies and won't collect in a
bare environment — that's expected; run the ones relevant to your change.

- One logical change per PR.
- Explain what you measured and which backends you verified.

## Security

Don't open a public issue — see [SECURITY.md](SECURITY.md).
