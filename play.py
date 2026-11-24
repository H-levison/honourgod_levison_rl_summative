"""
play.py -- watch the best trained agent run the desk.

    uv run play.py                      best available agent, 3D scene
    uv run play.py --policy ppo         a specific algorithm
    uv run play.py --policy heuristic   the rule-based reference
    uv run play.py --fps 2.5 --seed 7

This is a thin convenience wrapper around `uv run main.py demo`, which accepts
the same options. It exists because watching the trained agent is the single
thing most people want to do with this repository, and it should not require
knowing a subcommand.

Every other operation -- training, sweeps, the headless scoreboard -- lives in
main.py.
"""

from __future__ import annotations

import sys

from main import main

if __name__ == "__main__":
    raise SystemExit(main(["demo", *sys.argv[1:]]))
