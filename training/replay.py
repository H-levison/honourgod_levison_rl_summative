"""
Animated replay of a single episode, written as a GIF.

The 3D scene is the primary visualisation, but it needs a browser and a running
process. This renders one real episode of a trained policy to a self-contained
animation so the repository's front page can show the agent working, and so a
reader with no environment can still see what the policy does over a week.

Everything plotted is taken from an actual rollout -- no illustration, no
smoothing beyond what is stated on the axes.

    uv run python -m training.replay                  best policy, default seed
    uv run python -m training.replay --policy ppo --seed 20003
    uv run python -m training.replay --out docs/replay.gif --fps 18
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from environment import ACTION_NAMES, CURRENCIES, DeskConfig, LiquidityDeskEnv
from training.common import ROOT

POOL_COLOURS = ("#22c55e", "#6366f1", "#f59e0b", "#06b6d4")
INK = "#e5e7eb"
DIM = "#9ca3af"
PANEL = "#111827"


def rollout(policy, seed: int) -> dict[str, np.ndarray]:
    """One episode, keeping everything the animation needs."""
    env = LiquidityDeskEnv(DeskConfig(), render_mode="state")
    obs, _ = env.reset(seed=seed)
    frames: list[dict] = []
    while True:
        action = int(policy(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        s = env.render_state()
        frames.append({
            "t": s["t"],
            "pools": np.array(s["pools_usd"], dtype=float),
            "equity": float(s["equity_usd"]),
            "equity_ratio": float(s["equity_usd"] / s["initial_equity_usd"] - 1.0),
            "pressure": np.array(s["pressure"], dtype=float),
            "action": s["action_name"],
            "devaluation": bool(any(s["devaluation"])),
            "paused": list(s["paused"]),
            "drawdown": float(s["drawdown"]),
        })
        if terminated or truncated:
            break
    env.close()
    out = {k: np.array([f[k] for f in frames]) for k in
           ("t", "equity", "equity_ratio", "drawdown", "devaluation")}
    out["pools"] = np.stack([f["pools"] for f in frames])
    out["pressure"] = np.stack([f["pressure"] for f in frames])
    out["action"] = np.array([f["action"] for f in frames])
    out["terminated"] = np.array(terminated)
    return out


def build(data: dict, out_path: Path, fps: int, stride: int) -> None:
    idx = list(range(0, len(data["t"]), stride))
    if idx[-1] != len(data["t"]) - 1:
        idx.append(len(data["t"]) - 1)

    fig = plt.figure(figsize=(9.0, 4.6), facecolor="#0b0f17")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.5], height_ratios=[1.0, 1.0],
                          hspace=0.45, wspace=0.26,
                          left=0.07, right=0.975, top=0.84, bottom=0.11)
    ax_pool = fig.add_subplot(gs[:, 0])
    ax_eq = fig.add_subplot(gs[0, 1])
    ax_press = fig.add_subplot(gs[1, 1])

    for ax in (ax_pool, ax_eq, ax_press):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color("#1f2937")
        ax.tick_params(colors=DIM, labelsize=7.5)
        ax.grid(alpha=0.14, color="#374151")

    # --- pools -------------------------------------------------------------
    cap = data["pools"].max() * 1.12
    bars = ax_pool.bar(range(4), data["pools"][0], color=POOL_COLOURS, width=0.62)
    ax_pool.set_ylim(0, cap)
    ax_pool.set_xticks(range(4))
    ax_pool.set_xticklabels(CURRENCIES, color=INK, fontsize=9)
    ax_pool.set_ylabel("pool value (USD)", color=DIM, fontsize=8.5)
    ax_pool.set_title("Inventory", color=INK, fontsize=10, pad=7)
    pool_labels = [
        ax_pool.text(i, 0, "", ha="center", va="bottom", color=INK, fontsize=7.5)
        for i in range(4)
    ]

    # --- equity ------------------------------------------------------------
    steps = data["t"]
    ax_eq.set_xlim(0, steps[-1])
    pad = max(0.004, float(np.abs(data["equity_ratio"]).max()) * 1.25)
    ax_eq.set_ylim(-pad * 100, pad * 100)
    ax_eq.axhline(0, color="#4b5563", lw=0.9)
    # The insolvency bar only belongs on the chart when the week came near it;
    # drawing it off-scale strands the label outside the axes.
    if pad * 100 > 13.0:
        ax_eq.axhline(-15, color="#f43f5e", lw=0.9, ls="--")
        ax_eq.text(steps[-1], -15, " insolvency ", color="#f43f5e", fontsize=6.5,
                   va="bottom", ha="right")
    else:
        ax_eq.text(0.985, 0.06, "insolvency at −15%, not approached",
                   transform=ax_eq.transAxes, color="#6b7280", fontsize=6.5,
                   ha="right", va="bottom")
    (eq_line,) = ax_eq.plot([], [], color="#22d3ee", lw=1.6)
    ax_eq.set_ylabel("equity vs. open (%)", color=DIM, fontsize=8.5)
    ax_eq.set_title("Profit and loss", color=INK, fontsize=10, pad=7)

    # --- pressure ----------------------------------------------------------
    ax_press.set_xlim(0, steps[-1])
    ax_press.set_ylim(0, max(0.6, float(data["pressure"].max()) * 1.15))
    press_lines = [
        ax_press.plot([], [], color=POOL_COLOURS[i], lw=1.3,
                      label=CURRENCIES[i])[0]
        for i in range(3)
    ]
    ax_press.set_ylabel("devaluation pressure", color=DIM, fontsize=8.5)
    ax_press.set_xlabel("step in the trading week", color=DIM, fontsize=8.5)
    leg = ax_press.legend(fontsize=7, ncol=3, loc="upper left", frameon=False)
    for txt in leg.get_texts():
        txt.set_color(DIM)

    title = fig.text(0.07, 0.945, "", color=INK, fontsize=11.5, fontweight="bold")
    action = fig.text(0.07, 0.895, "", color="#22d3ee", fontsize=9,
                      family="monospace")
    flash = fig.text(0.975, 0.895, "", color="#f43f5e", fontsize=9,
                     fontweight="bold", ha="right")

    def draw(k: int):
        i = idx[k]
        for b, v, lab in zip(bars, data["pools"][i], pool_labels):
            b.set_height(v)
            lab.set_y(v + cap * 0.015)
            lab.set_text(f"${v/1000:,.1f}k")
        eq_line.set_data(steps[: i + 1], data["equity_ratio"][: i + 1] * 100)
        for j, ln in enumerate(press_lines):
            ln.set_data(steps[: i + 1], data["pressure"][: i + 1, j])

        title.set_text(
            f"step {int(data['t'][i]):>3} / {int(steps[-1])}    "
            f"equity ${data['equity'][i]:,.0f}    "
            f"{data['equity_ratio'][i] * 100:+.2f}%"
        )
        action.set_text(data["action"][i].replace("_", " "))
        # A devaluation is a single-frame event; hold the marker briefly so it
        # is visible at playback speed.
        window = data["devaluation"][max(0, i - 6): i + 1]
        flash.set_text("DEVALUATION" if window.any() else "")
        return [*bars, eq_line, *press_lines, *pool_labels, title, action, flash]

    anim = FuncAnimation(fig, draw, frames=len(idx), interval=1000 / fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=84,
              savefig_kwargs={"facecolor": "#0b0f17"})
    plt.close(fig)
    size = out_path.stat().st_size / 1024
    print(f"  {out_path.relative_to(ROOT)}  ({len(idx)} frames, {size:,.0f} KB)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="best")
    ap.add_argument("--seed", type=int, default=20003)
    ap.add_argument("--out", default="report/figures/replay.gif")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4,
                    help="keep every Nth step, to hold the file size down")
    args = ap.parse_args(argv)

    import sys
    sys.path.insert(0, str(ROOT))
    from main import resolve_policy

    policy = resolve_policy(args.policy)
    print(f"replaying {getattr(policy, 'name', args.policy)} on seed {args.seed}")
    data = rollout(policy, args.seed)
    print(f"  {len(data['t'])} steps, final equity {data['equity_ratio'][-1] * 100:+.2f}%")
    build(data, ROOT / args.out, args.fps, args.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
