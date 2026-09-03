"""
spikeling/__main__.py
=====================
Entry point for the Spikeling neuromorphic runtime.

Usage:
  python -m spikeling                               # reflex demo (sound localizer)
  python -m spikeling --train                       # interactive trainer
  python -m spikeling --train path/to/network.spk  # train a specific network
  python -m spikeling path/to/network.spk           # run any network interactively
  python -m spikeling --no-interactive              # compile only
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from compiler.compiler import compile_file
from runtime.runtime   import SpikelingRuntime


def main():
    args         = sys.argv[1:]
    train_mode   = "--train" in args
    no_ui        = "--no-interactive" in args
    spk_args     = [a for a in args if not a.startswith("--")]

    # Default networks
    base = os.path.dirname(__file__)
    default_demo    = os.path.join(base, "examples", "sound_localizer.spk")
    default_learner = os.path.join(base, "examples", "word_learner.spk")

    if train_mode:
        spk_path = spk_args[0] if spk_args else default_learner
    else:
        spk_path = spk_args[0] if spk_args else default_demo

    if not os.path.exists(spk_path):
        print(f"[spikeling] error: file not found - {spk_path}")
        sys.exit(1)

    print(f"[spikeling] compiling {spk_path} ...")
    output_dir = os.path.dirname(os.path.abspath(spk_path))
    ast        = compile_file(spk_path, output_dir=output_dir)
    rt         = SpikelingRuntime(ast)

    if no_ui:
        print("[spikeling] compiled successfully (--no-interactive mode)")
        print(rt.state_report())
        return

    if train_mode:
        # Wire in the trainer
        from trainer.trainer import SpikelingTrainer
        memory_path = os.path.join(output_dir, "spikeling_memory.json")
        trainer     = SpikelingTrainer(rt, memory_path=memory_path)
        trainer.run_interactive()
    else:
        # Reflex / benchmark demo
        for action in ast.actions:
            cmd = action.command
            rt.register_handler(cmd, lambda c=cmd: None)
        rt.run_interactive()


if __name__ == "__main__":
    main()
