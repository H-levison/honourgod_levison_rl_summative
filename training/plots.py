"""
Figures for the report.

Produces exactly the visualisations the Discussion & Analysis criterion names:

    1. comparative rewards across all four algorithms
    2. the DQN objective (Q-value loss)
    3. policy-gradient entropy
    4. convergence, on held-out evaluation seeds
    5. generalisation: trained agents against the rule-based references

plus an action-distribution figure, because a score says the agent did well
while an action histogram says what it actually learned to do.

Everything is drawn from the per-run CSVs in logs/runs/, so the figures and the
numbers quoted in the report come from the same source.

    uv run python -m training.plots
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environment import ACTION_NAMES
from training.common import LOGS_DIR, ROOT

FIGURES_DIR = ROOT / "report" / "figures"
ALGOS = ("dqn", "reinforce", "ppo", "a2c")

COLOURS = {
    "dqn": "#2563eb",
    "reinforce": "#dc2626",
    "ppo": "#16a34a",
    "a2c": "#ea580c",
}

BASELINE_GREY = "#94a3b8"
LABELS = {
    "dqn": "DQN",
    "reinforce": "REINFORCE",
    "ppo": "PPO",
    "a2c": "A2C",
}

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_runs(algo: str) -> dict[str, pd.DataFrame]:
    """Every run of one algorithm, keyed by configuration tag."""
    out: dict[str, pd.DataFrame] = {}
    d = LOGS_DIR / "runs" / algo
    if not d.exists():
        return out
    for path in sorted(d.glob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        out[path.stem.replace("_seed0", "")] = df
    return out


def load_summary(algo: str) -> pd.DataFrame | None:
    path = LOGS_DIR / "sweeps" / f"{algo}_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def best_tag(algo: str) -> str | None:
    s = load_summary(algo)
    if s is None or s.empty:
        return None
    return str(s.iloc[0]["tag"])


def _series(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Frames and values for a column, dropping rows where it is blank."""
    if col not in df.columns:
        return np.array([]), np.array([])
    sub = df[["frames", col]].apply(pd.to_numeric, errors="coerce").dropna()
    return sub["frames"].to_numpy(), sub[col].to_numpy()


def _smooth(y: np.ndarray, window: int = 21) -> np.ndarray:
    """
    Centred rolling mean that leaves the endpoints alone.

    A convolution with mode="same" zero-pads beyond the data, which drags the
    first and last few points toward zero and fabricates a sharp dip at both
    ends of every curve -- easy to misread as the run collapsing. A rolling
    mean with min_periods=1 averages over whatever window is actually
    available instead.
    """
    if y.size < 3:
        return y
    w = min(window, max(3, y.size // 5))
    return (
        pd.Series(y).rolling(w, min_periods=1, center=True).mean().to_numpy()
    )


# --------------------------------------------------------------------------- #
# 1. comparative rewards
# --------------------------------------------------------------------------- #

def fig_comparative_rewards() -> None:
    fig, (ax_all, ax_best) = plt.subplots(1, 2, figsize=(11, 4.0))

    for algo in ALGOS:
        runs = load_runs(algo)
        if not runs:
            continue
        for df in runs.values():
            x, y = _series(df, "episode_return_mean")
            if x.size:
                ax_all.plot(x, _smooth(y), color=COLOURS[algo], alpha=0.16, lw=0.8)

        tag = best_tag(algo)
        if tag and tag in runs:
            x, y = _series(runs[tag], "episode_return_mean")
            if x.size:
                ax_all.plot(x, _smooth(y), color=COLOURS[algo], lw=1.9,
                            label=f"{LABELS[algo]} ({tag})")
                ax_best.plot(x, _smooth(y), color=COLOURS[algo], lw=1.9,
                             label=LABELS[algo])

    for ax, title in (
        (ax_all, "All 12 configurations per algorithm\n(bold = best)"),
        (ax_best, "Best configuration per algorithm"),
    ):
        ax.set_xlabel("environment frames")
        ax.set_ylabel("episode return (training)")
        ax.set_title(title, fontsize=9.5)
        ax.legend(fontsize=7.5)
        ax.axhline(0, color="#64748b", lw=0.7, ls=":")

    fig.suptitle("Comparative training reward", fontsize=11, y=1.0)
    _save(fig, "01_comparative_rewards.png")


# --------------------------------------------------------------------------- #
# 2. DQN objective
# --------------------------------------------------------------------------- #

def fig_dqn_objective() -> None:
    runs = load_runs("dqn")
    if not runs:
        return
    fig, (ax_loss, ax_eps) = plt.subplots(1, 2, figsize=(11, 3.8))

    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(runs)))
    for colour, (tag, df) in zip(cmap, sorted(runs.items())):
        x, y = _series(df, "loss_qvalue")
        if x.size:
            ax_loss.plot(x, _smooth(y), color=colour, lw=1.1, label=tag)
        xe, ye = _series(df, "epsilon")
        if xe.size:
            ax_eps.plot(xe, ye, color=colour, lw=1.1)

    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("environment frames")
    ax_loss.set_ylabel("Q-value loss (L2, log scale)")
    ax_loss.set_title("DQN objective", fontsize=9.5)

    ax_eps.set_xlabel("environment frames")
    ax_eps.set_ylabel("epsilon")
    ax_eps.set_title("Exploration schedule", fontsize=9.5)
    # Only three distinct schedules appear in the grid, so most of these
    # twelve curves lie exactly on top of one another. Say so, rather than
    # leaving the reader to wonder where the other nine went.
    ax_eps.text(
        0.98, 0.96,
        "10 of 12 configurations share the baseline schedule,\nso those curves coincide",
        transform=ax_eps.transAxes, ha="right", va="top",
        fontsize=6.5, color="#475569",
    )

    # One legend for both panels, below the figure where it covers no data.
    handles, labels = ax_loss.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02),
        ncol=6, fontsize=6.5, columnspacing=1.4, handlelength=1.6,
    )
    _save(fig, "02_dqn_objective.png")


# --------------------------------------------------------------------------- #
# 3. policy-gradient entropy
# --------------------------------------------------------------------------- #

def fig_pg_entropy() -> None:
    pg = [a for a in ALGOS if a != "dqn"]
    fig, axes = plt.subplots(1, len(pg), figsize=(11, 4.4), sharey=True)
    uniform = np.log(len(ACTION_NAMES))

    for ax, algo in zip(np.atleast_1d(axes), pg):
        runs = load_runs(algo)
        cmap = plt.cm.plasma(np.linspace(0, 0.85, max(len(runs), 1)))
        for colour, (tag, df) in zip(cmap, sorted(runs.items())):
            x, y = _series(df, "entropy")
            if x.size:
                ax.plot(x, _smooth(y), color=colour, lw=1.1, label=tag)
        ax.axhline(uniform, color="#334155", ls="--", lw=0.9)
        ax.text(0.02, uniform, " uniform  ln(13)", va="bottom", fontsize=6.5,
                color="#334155", transform=ax.get_yaxis_transform())
        ax.set_title(LABELS[algo], fontsize=9.5)
        ax.set_xlabel("environment frames")
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.22),
            ncol=2, fontsize=5.8, columnspacing=1.0, handlelength=1.5,
        )
    np.atleast_1d(axes)[0].set_ylabel("policy entropy (nats)")

    fig.suptitle(
        "Policy entropy: exploration collapse across the policy-gradient family",
        fontsize=10.5, y=1.02,
    )
    _save(fig, "03_pg_entropy.png")


# --------------------------------------------------------------------------- #
# 4. convergence on held-out seeds
# --------------------------------------------------------------------------- #

def fig_convergence() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for algo in ALGOS:
        runs = load_runs(algo)
        curves = []
        for df in runs.values():
            x, y = _series(df, "eval_return_mean")
            if x.size > 2:
                curves.append((x, y))
        if not curves:
            continue
        grid = np.linspace(
            0, max(c[0].max() for c in curves), 40,
        )
        stacked = np.vstack([np.interp(grid, x, y) for x, y in curves])
        med = np.median(stacked, axis=0)
        lo, hi = np.percentile(stacked, [25, 75], axis=0)
        ax.plot(grid, med, color=COLOURS[algo], lw=2.0, label=LABELS[algo])
        ax.fill_between(grid, lo, hi, color=COLOURS[algo], alpha=0.15)

    ax.axhline(0, color="#64748b", lw=0.7, ls=":")
    ax.set_xlabel("environment frames")
    ax.set_ylabel("held-out evaluation return")
    ax.set_title(
        "Convergence on held-out seeds\nmedian across 12 configurations, "
        "shaded interquartile range",
        fontsize=9.5,
    )
    ax.legend()
    _save(fig, "04_convergence.png")


# --------------------------------------------------------------------------- #
# 5. generalisation scoreboard
# --------------------------------------------------------------------------- #

def load_final_evaluation() -> pd.DataFrame | None:
    """The held-out scoreboard written by `main.py evaluate --csv`."""
    path = LOGS_DIR / "final_evaluation.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def fig_generalisation() -> None:
    """
    Every policy on one identical held-out seed set.

    The trained agents' bars come from the same evaluation run as the
    rule-based references, so the comparison is paired rather than assembled
    from separately-scored runs.
    """
    final = load_final_evaluation()
    names, scores, colours, episodes = [], [], [], None
    seed0 = None

    if final is not None and not final.empty:
        episodes = int(final["episodes"].iloc[0])
        seed0 = int(final["seed_base"].iloc[0])
        tags = {a: best_tag(a) for a in ALGOS}
        for _, row in final.iterrows():
            key = str(row["policy"]).lower()
            if key in ALGOS:
                tag = tags.get(key)
                names.append(f"{LABELS[key]}\n{tag}" if tag else LABELS[key])
                colours.append(COLOURS[key])
            else:
                names.append(str(row["label"]))
                colours.append(BASELINE_GREY)
            scores.append(float(row["return_mean"]))
    else:
        # No scoreboard yet -- fall back to each sweep's own final evaluation.
        for algo in ALGOS:
            s = load_summary(algo)
            if s is None or s.empty:
                continue
            names.append(f"{LABELS[algo]}\n{s.iloc[0]['tag']}")
            scores.append(float(s.iloc[0]["score"]))
            colours.append(COLOURS[algo])

    if not names:
        return

    order = np.argsort(scores)
    names = [names[i] for i in order]
    scores = [scores[i] for i in order]
    colours = [colours[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.8, 0.52 * len(names) + 1.6))
    bars = ax.barh(names, scores, color=colours, height=0.62)

    # Keep every label inside the axes: pad the limits by a fixed fraction of
    # the span, then write each value on the far side of its own bar.
    span = max(scores) - min(scores) or 1.0
    pad = 0.12 * span
    ax.set_xlim(min(min(scores), 0) - pad, max(max(scores), 0) + pad)
    offset = 0.015 * span
    for bar, value in zip(bars, scores):
        ax.text(
            value + (offset if value >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}", va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8, fontweight="bold",
        )

    ax.axvline(0, color="#334155", lw=0.9)
    seeds = (f", {episodes} unseen seeds ({seed0}–{seed0 + episodes - 1})"
             if episodes and seed0 is not None else "")
    ax.set_xlabel(f"mean episode return{seeds}")
    ax.set_title(
        "Generalisation: trained agents vs. rule-based references",
        fontsize=9.5,
    )
    ax.tick_params(axis="y", labelsize=8)
    _save(fig, "05_generalisation.png")


# --------------------------------------------------------------------------- #
# 6. what each agent learned to do
# --------------------------------------------------------------------------- #

def fig_action_distribution() -> None:
    """
    What each policy actually does when deployed.

    The shares come from the held-out evaluation, which runs the greedy policy,
    rather than from the training logs, which record actions taken while still
    exploring. The rule-based references are included because the interesting
    result is not any single row but the contrast between them.
    """
    final = load_final_evaluation()
    cols = [f"act_{n.lower()}" for n in ACTION_NAMES]
    rows, labels, is_agent = [], [], []

    if final is not None and set(cols).issubset(final.columns):
        tags = {a: best_tag(a) for a in ALGOS}
        for _, row in final.iterrows():
            key = str(row["policy"]).lower()
            rows.append(row[cols].to_numpy(dtype=float))
            if key in ALGOS and tags.get(key):
                labels.append(f"{LABELS[key]}  ({tags[key]})")
            else:
                labels.append(str(row["label"]))
            is_agent.append(key in ALGOS)
    else:
        # Fall back to the training logs if no scoreboard has been written.
        for algo in ALGOS:
            runs, tag = load_runs(algo), best_tag(algo)
            if not tag or tag not in runs:
                continue
            df = runs[tag]
            present = [c for c in cols if c in df.columns]
            if not present:
                continue
            tail = df[present].apply(pd.to_numeric, errors="coerce").dropna()
            if tail.empty:
                continue
            counts = tail.iloc[-1].to_numpy(dtype=float)
            total = counts.sum()
            rows.append(counts / total if total else counts)
            labels.append(f"{LABELS[algo]}  ({tag})")
            is_agent.append(True)

    if not rows:
        return

    data = np.vstack(rows)
    fig, ax = plt.subplots(figsize=(10.5, 0.46 * len(labels) + 2.4))
    im = ax.imshow(data, cmap="magma", aspect="auto", vmin=0)

    ax.set_xticks(range(len(ACTION_NAMES)))
    ax.set_xticklabels([n.replace("_", " ") for n in ACTION_NAMES],
                       rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    # Learned policies in black, rule-based references in grey, so the two
    # kinds of row are distinguishable at a glance.
    for tick, agent in zip(ax.get_yticklabels(), is_agent):
        tick.set_color("#0f172a" if agent else "#64748b")
    ax.grid(False)

    hi = data.max() * 0.6
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] > 0.02:
                ax.text(j, i, f"{data[i, j] * 100:.0f}", ha="center",
                        va="center", fontsize=6.5,
                        color="white" if data[i, j] < hi else "black")

    cb = fig.colorbar(im, ax=ax, label="share of episode (%)", pad=0.015)
    cb.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}")
    )
    ax.set_title(
        "What each policy does when deployed (% of steps, greedy, 200 held-out seeds)",
        fontsize=9.5,
    )
    # RESUME ALL is inert unless a corridor is paused, so a large share of it
    # is idling rather than managing. Say so on the figure, because the column
    # otherwise reads as activity.
    fig.text(
        0.5, -0.02,
        "RESUME ALL clears pause flags, so with nothing paused it is a second "
        "no-op \u2014 much of that column is idling, not activity.",
        ha="center", fontsize=7, color="#475569",
    )
    _save(fig, "06_action_distribution.png")



# --------------------------------------------------------------------------- #
# 7. seed replication
# --------------------------------------------------------------------------- #

def load_replication() -> pd.DataFrame | None:
    path = LOGS_DIR / "seed_replication.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def fig_seed_replication() -> None:
    """
    Every seed as its own point, so the reader can see the spread rather than
    trust a mean. Reference lines are the rule-based policies on the same seeds.
    """
    rep = load_replication()
    if rep is None or rep.empty:
        return
    final = load_final_evaluation()

    order = [a for a in ALGOS if a in set(rep["algo"])]
    order.sort(key=lambda a: rep.loc[rep["algo"] == a, "held_out_score"].mean())

    fig, ax = plt.subplots(figsize=(7.8, 0.52 * len(order) + 1.9))

    # rule-based references as vertical lines
    if final is not None:
        for name, style in (("hold", ":"), ("heuristic", "--")):
            row = final[final["policy"].str.lower() == name]
            if row.empty:
                continue
            x = float(row["return_mean"].iloc[0])
            ax.axvline(x, color="#64748b", ls=style, lw=1.2, zorder=1)
            # Axes-fraction y keeps the label clear of the title and of the
            # topmost row, whatever the number of algorithms.
            ax.text(x, 0.015, f"{row['label'].iloc[0]} ",
                    transform=ax.get_xaxis_transform(),
                    rotation=90, va="bottom", ha="right",
                    fontsize=7.5, color="#475569")

    for y, algo in enumerate(order):
        vals = rep.loc[rep["algo"] == algo, "held_out_score"].to_numpy(dtype=float)
        mean = vals.mean()
        colour = COLOURS[algo]
        # the range, then each seed, then the mean on top
        ax.plot([vals.min(), vals.max()], [y, y], color=colour, lw=2.2,
                alpha=0.35, solid_capstyle="round", zorder=2)
        ax.scatter(vals, [y] * len(vals), s=34, color=colour,
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter([mean], [y], s=150, marker="|", color=colour,
                   linewidth=2.4, zorder=4)
        sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
        ax.text(vals.max() + 4, y, f"{mean:+.1f} ± {sd:.1f}",
                va="center", fontsize=8, fontweight="bold", color=colour)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(
        [f"{LABELS[a]}\n{rep.loc[rep['algo'] == a, 'tag'].iloc[0]}" for a in order],
        fontsize=8,
    )
    ax.axvline(0, color="#334155", lw=0.9, zorder=1)
    n_seeds = int(rep.groupby("algo").size().max())
    ax.set_xlabel(
        f"mean episode return on the held-out set "
        f"(one point per training seed, n={n_seeds})"
    )
    ax.set_title(
        "Seed replication: does the ranking survive a change of training seed?",
        fontsize=9.5,
    )
    ax.margins(x=0.16)
    _save(fig, "07_seed_replication.png")


# --------------------------------------------------------------------------- #

def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")


def make_all(baselines: dict[str, float] | None = None) -> None:
    print("writing figures:")
    fig_comparative_rewards()
    fig_dqn_objective()
    fig_pg_entropy()
    fig_convergence()
    fig_generalisation()
    fig_seed_replication()
    fig_action_distribution()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build the report figures.")
    p.add_argument("--baselines", type=str, default=None,
                   help='JSON of reference scores, e.g. \'{"Heuristic": -80.5}\'')
    args = p.parse_args()
    make_all(json.loads(args.baselines) if args.baselines else None)
