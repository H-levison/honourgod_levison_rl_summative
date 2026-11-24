"""
Report-ready tables, generated from the same CSVs as the figures.

The report quotes a lot of numbers. Transcribing them by hand is how a report
ends up disagreeing with its own appendix, so every table the report needs is
produced here and written to `report/tables.md` for pasting in.

    uv run python -m training.tables

Produces:
    1. the hyperparameter grid, per algorithm, ranked        (Criterion 4)
    2. the held-out scoreboard with paired statistics        (Criterion 5)
    3. a one-line-per-algorithm summary for the abstract
"""

from __future__ import annotations

import pandas as pd

from training.common import LOGS_DIR, ROOT

ALGOS = ("dqn", "reinforce", "ppo", "a2c")
LABELS = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}

OUT = ROOT / "report" / "tables.md"


_PAUSE_COLS = ("act_pause_ngn", "act_pause_rwf", "act_pause_kes")


def _idle_share(row) -> float:
    """
    Conservative floor on the share of steps that change nothing.

    HOLD is unambiguous. RESUME_ALL is a no-op whenever no corridor is paused,
    and a resume can only act if a pause preceded it, so the excess of resumes
    over pauses is guaranteed-inert. Both quantities are shares of the episode.
    """
    if "act_hold" not in row:
        return float("nan")
    hold = float(row["act_hold"])
    resume = float(row.get("act_resume_all", 0.0))
    pause = sum(float(row.get(c, 0.0)) for c in _PAUSE_COLS)
    return hold + max(0.0, resume - pause)


def _read(path):
    return pd.read_csv(path) if path.exists() else None


def hyperparameter_tables() -> str:
    """One ranked table per algorithm, with the hypothesis each run tested."""
    out = ["## Hyperparameter study\n"]
    for algo in ALGOS:
        df = _read(LOGS_DIR / "sweeps" / f"{algo}_summary.csv")
        if df is None or df.empty:
            continue
        df = df.sort_values("score", ascending=False)
        out.append(f"\n### {LABELS[algo]} — {len(df)} configurations\n")
        out.append(
            "| Rank | Configuration | Score | Equity | Max DD | Stockouts | "
            "Insolvency | Hypothesis tested |"
        )
        out.append("|---:|---|---:|---:|---:|---:|---:|---|")
        for rank, (_, r) in enumerate(df.iterrows(), start=1):
            best = "**" if rank == 1 else ""
            out.append(
                f"| {rank} | {best}`{r['tag']}`{best} "
                f"| {best}{r['score']:+.2f}{best} "
                f"| {(r['equity_ratio'] - 1) * 100:+.2f}% "
                f"| {r['drawdown'] * 100:.2f}% "
                f"| {r['stockouts']:.1f} "
                f"| {r['insolvency_rate'] * 100:.0f}% "
                f"| {r['note']} |"
            )
        spread = df["score"].max() - df["score"].min()
        out.append(
            f"\nSpread between best and worst configuration: "
            f"**{spread:.1f} points** "
            f"(`{df.iloc[0]['tag']}` {df['score'].max():+.2f} vs "
            f"`{df.iloc[-1]['tag']}` {df['score'].min():+.2f}).\n"
        )
    return "\n".join(out)


def scoreboard_table() -> str:
    """The held-out comparison, which is the result the report argues from."""
    df = _read(LOGS_DIR / "final_evaluation.csv")
    if df is None or df.empty:
        return ""

    episodes = int(df["episodes"].iloc[0])
    seed = int(df["seed_base"].iloc[0])
    df = df.sort_values("return_mean", ascending=False)

    out = [
        "\n## Held-out evaluation\n",
        f"All policies on the identical seed set "
        f"{seed}..{seed + episodes - 1} (n = {episodes}).\n",
        "| Policy | Mean return | Equity | Max DD | Stockouts | Fees $ | "
        "Revenue $ | Idle | vs `hold` (paired) | t |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        is_hold = str(r["policy"]).lower() == "hold"
        paired = "—" if is_hold else f"{r['paired_diff_vs_hold']:+.2f} ± {r['paired_se']:.2f}"
        tcell = "—" if is_hold else f"{r['paired_t']:+.2f}"
        if not is_hold and abs(float(r["paired_t"])) > 2.0:
            tcell = f"**{tcell}**"
        out.append(
            f"| {r['label']} | {r['return_mean']:+.2f} "
            f"| {r['equity_pct']:+.2f}% | {r['drawdown_pct']:.2f}% "
            f"| {r['stockouts']:.1f} | {r['fees_usd']:,.0f} "
            f"| {r['revenue_usd']:,.0f} | {_idle_share(r) * 100:.0f}% "
            f"| {paired} | {tcell} |"
        )
    out.append(
        "\nPaired because the FX path is common to every policy on a given "
        "seed and controllable by none of them; differencing removes it. "
        "Bold *t* exceeds |2|, the significance bar used throughout.\n"
        "\n*Idle* is a conservative floor on how often a policy does nothing. "
        "`RESUME_ALL` clears every pause flag, so with nothing paused it is a "
        "second no-op, identical in effect to `HOLD` -- and the "
        "policy-gradient agents settled on it rather than on `HOLD`. A resume "
        "can only act if a pause preceded it, so "
        "`idle >= HOLD + max(0, resume - pause)`. Read without this "
        "correction the action histogram suggests REINFORCE spends its week "
        "managing corridors, when most of its steps change nothing.\n"
    )
    return "\n".join(out)


def headline_table() -> str:
    """Best configuration per algorithm, for the abstract and conclusion."""
    final = _read(LOGS_DIR / "final_evaluation.csv")
    out = [
        "\n## Headline: best configuration per algorithm\n",
        "| Algorithm | Best configuration | Sweep score (40 ep) | "
        "Held-out score (200 ep) | vs `hold` |",
        "|---|---|---:|---:|---:|",
    ]
    for algo in ALGOS:
        df = _read(LOGS_DIR / "sweeps" / f"{algo}_summary.csv")
        if df is None or df.empty:
            continue
        top = df.sort_values("score", ascending=False).iloc[0]
        held, paired = "—", "—"
        if final is not None:
            row = final[final["policy"].str.lower() == algo]
            if not row.empty:
                held = f"{float(row['return_mean'].iloc[0]):+.2f}"
                paired = (
                    f"{float(row['paired_diff_vs_hold'].iloc[0]):+.2f} "
                    f"(t = {float(row['paired_t'].iloc[0]):+.2f})"
                )
        out.append(
            f"| {LABELS[algo]} | `{top['tag']}` | {top['score']:+.2f} "
            f"| {held} | {paired} |"
        )
    out.append(
        "\nThe two score columns differ because the sweep ranks configurations "
        "on 40 episodes while the held-out column re-scores the winner on 200. "
        "The gap is a measurement of evaluation noise, not a discrepancy: "
        "40 episodes is enough to rank, not enough to report.\n"
    )
    return "\n".join(out)


def main() -> int:
    parts = [
        "# Generated tables\n",
        "Produced by `uv run python -m training.tables` from the CSVs in "
        "`logs/`. Do not edit by hand -- regenerate instead, so the report and "
        "its evidence cannot drift apart.\n",
        headline_table(),
        scoreboard_table(),
        hyperparameter_tables(),
    ]
    text = "\n".join(p for p in parts if p)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(text):,} characters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
