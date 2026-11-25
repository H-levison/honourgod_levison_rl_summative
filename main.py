"""
Cross-Border Liquidity Desk — entry point.

    uv run main.py                        3D demo of the best available policy
    uv run main.py demo --policy ppo      3D demo of a specific trained agent
    uv run main.py demo --policy heuristic --fps 6
    uv run main.py evaluate               headless scoreboard over N episodes
    uv run main.py train --algo dqn       train one algorithm
    uv run main.py sweep --algo ppo       run that algorithm's hyperparameter grid

Run with no arguments and it renders the best policy it can find, falling back
to the rule-based desk operator if nothing has been trained yet, so the project
is demonstrable straight after `uv sync`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from environment import ACTION_NAMES, DeskConfig, LiquidityDeskEnv
from training.baselines import BASELINES, make_baseline

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

ALGOS = ("dqn", "reinforce", "ppo", "a2c")


# --------------------------------------------------------------------------- #
# Policy resolution
# --------------------------------------------------------------------------- #

def _checkpoint_path(algo: str) -> Path:
    family = "dqn" if algo == "dqn" else "pg"
    return MODELS_DIR / family / f"{algo}_best.pt"


def resolve_policy(name: str):
    """
    Build a callable policy from a name.

    Accepts any baseline name, any algorithm name, or "best" which prefers a
    trained checkpoint and falls back to the heuristic operator.
    """
    key = name.lower()

    if key in BASELINES:
        return make_baseline(key)

    if key == "best":
        trained = [a for a in ALGOS if _checkpoint_path(a).exists()]
        if not trained:
            print(
                "[main] no trained checkpoints found under models/ -- "
                "falling back to the rule-based heuristic desk operator.\n"
                "[main] train an agent first with: uv run main.py sweep --algo ppo",
                file=sys.stderr,
            )
            return make_baseline("heuristic")
        # pick by recorded evaluation score
        from training.common import load_policy, read_checkpoint_score

        best = max(trained, key=lambda a: read_checkpoint_score(_checkpoint_path(a)))
        print(f"[main] best available checkpoint: {best}")
        return load_policy(_checkpoint_path(best))

    if key in ALGOS:
        path = _checkpoint_path(key)
        if not path.exists():
            raise SystemExit(
                f"no checkpoint for {key!r} at {path}.\n"
                f"Train it first:  uv run main.py sweep --algo {key}"
            )
        from training.common import load_policy

        return load_policy(path)

    raise SystemExit(
        f"unknown policy {name!r}. "
        f"Choose from: {', '.join(sorted(BASELINES) + list(ALGOS) + ['best'])}"
    )


# --------------------------------------------------------------------------- #
# Terminal reporting
# --------------------------------------------------------------------------- #

BANNER = """
+--------------------------------------------------------------------------+
|  CROSS-BORDER LIQUIDITY DESK  --  reinforcement learning agent           |
|                                                                          |
|  Pools     NGN / RWF / KES fiat inventory + USDT settlement rail         |
|  Actions   13 discrete: 6 rebalances, 2 spread moves, 3 corridor         |
|            pauses, resume-all, hold                                      |
|  Reward    spread revenue - rebalance fees - FX mark-to-market           |
|            - stockout penalty - drawdown penalty                         |
|  Ends      insolvency below 85% of opening equity, or end of the week    |
+--------------------------------------------------------------------------+
"""

STEP_HEADER = (
    f"{'step':>5}  {'action':<14} {'equity($)':>11} {'p&l':>8} "
    f"{'dd':>7} {'spread':>7} {'fill':>6} {'stock':>6} {'reward':>8}"
)


def step_line(info: dict, reward: float) -> str:
    return (
        f"{info['t']:>5}  "
        f"{info['action_name']:<14} "
        f"{info['equity_usd']:>11,.0f} "
        f"{(info['equity_ratio'] - 1) * 100:>+7.2f}% "
        f"{info['drawdown'] * 100:>6.2f}% "
        f"{info['spread'] * 100:>6.2f}% "
        f"{info['n_filled']:>6} "
        f"{info['n_unfilled']:>6} "
        f"{reward:>+8.3f}"
    )


def episode_summary(info: dict, terminated: bool) -> str:
    verdict = "INSOLVENT" if terminated else "SURVIVED"
    return (
        f"\n  {verdict}   "
        f"equity ${info['equity_usd']:,.0f} "
        f"({(info['equity_ratio'] - 1) * 100:+.2f}%)   "
        f"revenue ${info['cum_revenue_usd']:,.0f}   "
        f"fees ${info['cum_fees_usd']:,.0f}   "
        f"filled {info['n_filled']}   "
        f"stockouts {info['n_unfilled']}   "
        f"declined {info['n_declined']}   "
        f"fill rate {info['fill_rate'] * 100:.1f}%"
    )


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #

def cmd_demo(args: argparse.Namespace) -> int:
    from environment.rendering import DeskRenderer

    policy = resolve_policy(args.policy)
    policy_name = getattr(policy, "name", args.policy)

    env = LiquidityDeskEnv(DeskConfig(), render_mode="state")
    renderer = DeskRenderer(open_browser=not args.no_browser)

    print(BANNER)
    url = renderer.start()
    print(f"  policy      {policy_name}")
    print(f"  scene       {url}")
    print(f"  speed       {args.fps} steps/sec")
    print(f"  episodes    {args.episodes}")
    print("\n  Waiting for the browser scene to connect...", flush=True)

    if renderer.wait_for_viewer(timeout=args.timeout):
        print("  Viewer connected.\n")
    else:
        print(
            f"  No viewer after {args.timeout:.0f}s. Open this URL manually:\n"
            f"    {url}\n"
            "  Continuing anyway.\n"
        )

    delay = 1.0 / max(args.fps, 0.1)
    returns: list[float] = []

    try:
        # episodes == 0 loops until interrupted, so the scene is never left
        # frozen on a finished episode with nothing moving
        ep = 0
        while args.episodes == 0 or ep < args.episodes:
            seed = args.seed + ep
            obs, _ = env.reset(seed=seed)
            total = 0.0

            label = f"{ep + 1}/{args.episodes}" if args.episodes else f"{ep + 1}"
            print(f"  --- episode {label}  (seed {seed}) ---")
            print(STEP_HEADER)

            while True:
                # Block here while the viewer has playback paused, so the
                # simulation stops rather than running ahead unseen.
                renderer.controls.wait_for_turn()

                # Capture the policy's scores for THIS observation before the
                # environment advances, so the panel shows the inputs to the
                # decision rather than the state that followed it.
                scorer = getattr(policy, "scores", None)
                scores = list(scorer(obs)) if callable(scorer) else None
                kind = getattr(policy, "score_kind", None)

                action = int(policy(obs))
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward

                state = env.render_state()
                state["policy"] = policy_name
                state["policy_scores"] = scores
                state["policy_score_kind"] = kind
                state["playback"] = renderer.controls.snapshot()
                state["episode_index"] = ep + 1
                state["episode_return"] = total
                state["done"] = terminated or truncated
                state["verdict"] = "insolvent" if terminated else "complete"
                renderer.push(state)

                if info["t"] % args.log_every == 0 or terminated or truncated:
                    print(step_line(info, reward), flush=True)

                if terminated or truncated:
                    print(episode_summary(info, terminated))
                    print(f"  episode return {total:+.2f}\n")
                    returns.append(total)
                    time.sleep(args.hold)
                    break

                time.sleep(delay / max(renderer.controls.speed, 0.01))

            ep += 1

        if len(returns) > 1:
            print(
                f"  mean return over {len(returns)} episodes: "
                f"{np.mean(returns):+.2f} (sd {np.std(returns):.2f})"
            )

        print("  Demo finished. The scene stays live -- Ctrl+C to exit.")
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        renderer.stop()
        env.close()

    return 0


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #

def rollout(env: LiquidityDeskEnv, policy, seed: int) -> dict:
    """One episode. Returns scoreboard metrics plus the action histogram."""
    # Stochastic policies are re-keyed from the same episode seed, so the whole
    # scoreboard is reproducible and not just the environment half of it.
    reseed = getattr(policy, "reseed", None)
    if callable(reseed):
        reseed(seed)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    actions = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    while True:
        action = int(policy(obs))
        actions[action] += 1
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            return {
                "return": total,
                "equity_ratio": info["equity_ratio"],
                "drawdown": info["drawdown"],
                "filled": info["n_filled"],
                "stockouts": info["n_unfilled"],
                "declined": info["n_declined"],
                "revenue": info["cum_revenue_usd"],
                "fees": info["cum_fees_usd"],
                "fill_rate": info["fill_rate"],
                "insolvent": float(terminated),
                "steps": info["t"],
                "actions": actions,
            }


def cmd_evaluate(args: argparse.Namespace) -> int:
    names = args.policies or list(BASELINES) + [
        a for a in ALGOS if _checkpoint_path(a).exists()
    ]
    env = LiquidityDeskEnv(DeskConfig())

    print(f"\nEvaluating {len(names)} policies over {args.episodes} episodes "
          f"(seeds {args.seed}..{args.seed + args.episodes - 1})\n")
    header = (
        f"{'policy':<14} {'return':>16} {'equity':>9} {'maxDD':>7} "
        f"{'fills':>6} {'stock':>6} {'decl':>6} {'rev$':>8} {'fees$':>7}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name in names:
        policy = resolve_policy(name)
        stats = [rollout(env, policy, args.seed + i) for i in range(args.episodes)]
        agg = {
            k: np.array([s[k] for s in stats])
            for k in stats[0] if k != "actions"
        }
        agg["actions"] = np.stack([s["actions"] for s in stats]).mean(axis=0)
        rows.append((name, getattr(policy, "name", name), agg))
        print(
            f"{getattr(policy, 'name', name):<14} "
            f"{agg['return'].mean():>+9.2f} +/-{agg['return'].std():>5.1f} "
            f"{(agg['equity_ratio'].mean() - 1) * 100:>+8.2f}% "
            f"{agg['drawdown'].mean() * 100:>6.2f}% "
            f"{agg['filled'].mean():>6.0f} "
            f"{agg['stockouts'].mean():>6.1f} "
            f"{agg['declined'].mean():>6.0f} "
            f"{agg['revenue'].mean():>8.0f} "
            f"{agg['fees'].mean():>7.0f}",
            flush=True,
        )

    # Paired comparison. Every policy is run on the identical seed set, so the
    # per-seed difference cancels the common FX shock -- which is by far the
    # largest variance term and the one no policy can control. The unpaired
    # standard deviations above are dominated by it and understate the real
    # separation between policies.
    ref_name, ref_label, ref_agg = next(
        (r for r in rows if r[0] == "hold"), rows[0]
    )
    others = [r for r in rows if r[0] != ref_name]
    if others:
        print(f"\nPaired difference vs {ref_label} (same seeds, n={args.episodes}):")
        n = args.episodes
        for name, label, agg in others:
            diff = agg["return"] - ref_agg["return"]
            se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
            t = diff.mean() / se if se > 0 else float("nan")
            verdict = "significant" if abs(t) > 2.0 else "not significant"
            print(
                f"  {label:<14} {diff.mean():>+8.2f} +/- {se:<5.2f} (SE)   "
                f"t = {t:>+6.2f}   {verdict}"
            )

    if args.actions:
        print("\nMean action counts per episode:")
        width = max(len(n) for n in ACTION_NAMES)
        print(f"  {'action':<{width}} " + "".join(f"{r[1][:9]:>10}" for r in rows))
        for a, aname in enumerate(ACTION_NAMES):
            cells = "".join(f"{r[2]['actions'][a]:>10.1f}" for r in rows)
            print(f"  {aname:<{width}} {cells}")

    if args.csv:
        _write_evaluation_csv(Path(args.csv), rows, args.episodes, args.seed)
        print(f"\nwrote {args.csv}")

    best = max(rows, key=lambda r: r[2]["return"].mean())
    print(f"\nBest mean return: {best[1]} ({best[2]['return'].mean():+.2f})\n")
    env.close()
    return 0


def _write_evaluation_csv(path: Path, rows, episodes: int, seed: int) -> None:
    """
    Persist the held-out scoreboard so the figures and the report quote one
    source rather than transcribed numbers.

    The paired column is the meaningful uncertainty. Every policy is run on the
    identical seed set, so differencing against the hold baseline cancels the
    common FX shock -- by far the largest variance term, and the one no policy
    can control. The unpaired standard deviation understates the separation.
    """
    import csv as _csv

    ref = next((r for r in rows if r[0] == "hold"), rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow([
            "policy", "label", "episodes", "seed_base", "return_mean",
            "return_std", "paired_diff_vs_hold", "paired_se", "paired_t",
            "equity_pct", "drawdown_pct", "filled", "stockouts", "declined",
            "revenue_usd", "fees_usd", "insolvency_rate",
        ] + [f"act_{n.lower()}" for n in ACTION_NAMES])
        for name, label, agg in rows:
            diff = agg["return"] - ref[2]["return"]
            se = diff.std(ddof=1) / np.sqrt(episodes) if episodes > 1 else float("nan")
            tstat = diff.mean() / se if se > 0 else float("nan")
            w.writerow([
                name, label, episodes, seed,
                f"{agg['return'].mean():.4f}",
                f"{agg['return'].std(ddof=1):.4f}",
                f"{diff.mean():.4f}", f"{se:.4f}", f"{tstat:.4f}",
                f"{(agg['equity_ratio'].mean() - 1) * 100:.4f}",
                f"{agg['drawdown'].mean() * 100:.4f}",
                f"{agg['filled'].mean():.2f}",
                f"{agg['stockouts'].mean():.2f}",
                f"{agg['declined'].mean():.2f}",
                f"{agg['revenue'].mean():.1f}",
                f"{agg['fees'].mean():.1f}",
                f"{agg['insolvent'].mean():.4f}",
            ] + [
                # Share of the episode spent on each action, so policies with
                # different episode lengths stay comparable.
                f"{c / max(agg['steps'].mean(), 1.0):.6f}"
                for c in agg["actions"]
            ])


# --------------------------------------------------------------------------- #
# train / sweep
# --------------------------------------------------------------------------- #

def cmd_train(args: argparse.Namespace) -> int:
    from training.runner import train_one

    train_one(args.algo, seed=args.seed, overrides=_parse_overrides(args.set))
    return 0


def cmd_replicate(args: argparse.Namespace) -> int:
    from training.runner import run_replication

    run_replication(
        seeds=tuple(range(args.seeds)),
        episodes=args.episodes,
        eval_seed_base=args.eval_seed,
    )
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from training.runner import run_sweep

    run_sweep(args.algo, limit=args.limit, seed=args.seed)
    return 0


def _parse_overrides(pairs: list[str] | None) -> dict:
    out: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            value: object = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = {"true": True, "false": False}.get(raw.lower(), raw)
        out[key] = value
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Cross-border liquidity desk: RL training and 3D demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # --- demo ---------------------------------------------------------------
    d = sub.add_parser("demo", help="render a policy in the 3D scene")
    d.add_argument("--policy", default="best",
                   help="best | heuristic | random | hold | dqn | reinforce | ppo | a2c")
    d.add_argument("--fps", type=float, default=3.0,
                   help="simulation steps per second (default 3, ~2.8 min per episode)")
    d.add_argument("--episodes", type=int, default=0,
                   help="number of episodes; 0 (the default) loops until Ctrl+C "
                        "so the scene always has motion in it")
    d.add_argument("--seed", type=int, default=2024)
    d.add_argument("--log-every", type=int, default=1,
                   help="print a terminal line every N steps")
    d.add_argument("--timeout", type=float, default=30.0,
                   help="seconds to wait for the browser to connect")
    d.add_argument("--hold", type=float, default=4.0,
                   help="seconds to pause on the end-of-episode verdict")
    d.add_argument("--no-browser", action="store_true",
                   help="do not open a browser automatically")
    d.set_defaults(func=cmd_demo)

    # --- evaluate -----------------------------------------------------------
    e = sub.add_parser("evaluate", help="headless scoreboard across policies")
    e.add_argument("policies", nargs="*", help="policy names (default: all available)")
    e.add_argument("--episodes", type=int, default=30)
    e.add_argument("--seed", type=int, default=9000)
    e.add_argument("--csv", metavar="PATH",
                   help="write the scoreboard to PATH for the report "
                        "and figures")
    e.add_argument("--actions", action="store_true",
                   help="also print the mean action histogram per policy")
    e.set_defaults(func=cmd_evaluate)

    # --- train --------------------------------------------------------------
    t = sub.add_parser("train", help="train a single configuration")
    t.add_argument("--algo", required=True, choices=ALGOS)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override a hyperparameter, repeatable")
    t.set_defaults(func=cmd_train)

    # --- replicate ----------------------------------------------------------
    rp = sub.add_parser(
        "replicate",
        help="retrain each algorithm's best configuration across several seeds",
    )
    rp.add_argument("--seeds", type=int, default=3,
                    help="number of training seeds per configuration")
    rp.add_argument("--episodes", type=int, default=100,
                    help="held-out episodes used to score each checkpoint")
    rp.add_argument("--eval-seed", type=int, default=20_000,
                    help="first held-out seed, disjoint from model selection")
    rp.set_defaults(func=cmd_replicate)

    # --- sweep --------------------------------------------------------------
    s = sub.add_parser("sweep", help="run an algorithm's hyperparameter grid")
    s.add_argument("--algo", required=True, choices=ALGOS)
    s.add_argument("--limit", type=int, default=None,
                   help="run only the first N configurations")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_sweep)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # bare `uv run main.py` -> demo the best policy available
        args = parser.parse_args(["demo"])

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
