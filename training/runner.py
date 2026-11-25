"""
Run orchestration: single configurations, full grids, and the whole study.

Runs execute in separate processes. Measured on this project a GPU is worth
only about 6% of end-to-end wall-clock (the environment, not the gradients, is
the bottleneck), while 16 logical cores running six single-threaded workers is
close to a six-fold speedup. So the parallelism here is across runs, not inside
them.

Each run writes `logs/runs/{algo}/{tag}_seed{n}.csv` plus a metadata file, and
each grid writes `logs/sweeps/{algo}_summary.csv` -- the raw material for the
hyperparameter table in the report.
"""

from __future__ import annotations

import csv
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import torch

from training import hparams
from training.common import (
    LOGS_DIR,
    RunLogger,
    checkpoint_path,
    evaluate_policy,
    load_policy,
    save_checkpoint,
)

DEFAULT_WORKERS = 6

SUMMARY_COLUMNS = (
    "algo", "tag", "seed", "score", "equity_ratio", "drawdown", "stockouts",
    "insolvency_rate", "frames", "wall_seconds", "note",
)


def _trainer(algo: str):
    """Imported lazily so a worker process only loads what it needs."""
    if algo == "dqn":
        from training.dqn_training import train_dqn
        return train_dqn
    from training import pg_training
    return {
        "ppo": pg_training.train_ppo,
        "a2c": pg_training.train_a2c,
        "reinforce": pg_training.train_reinforce,
    }[algo]


def _align_frames(cfg: dict[str, Any]) -> dict[str, Any]:
    """Round the frame budget to a whole number of batches."""
    fpb = int(cfg["frames_per_batch"])
    total = int(cfg["total_frames"])
    cfg["total_frames"] = max(fpb, (total // fpb) * fpb)
    return cfg


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #

def train_one(
    algo: str,
    tag: str | None = None,
    seed: int = 0,
    overrides: dict[str, Any] | None = None,
    progress: bool = True,
    save_best: bool = True,
) -> dict[str, Any]:
    cfg = hparams.config(algo, tag) if tag else hparams.baseline_config(algo)
    cfg.update(overrides or {})
    cfg = _align_frames(cfg)

    logger = RunLogger(algo, cfg["tag"], seed).start(cfg)
    try:
        result = _trainer(algo)(cfg, seed=seed, logger=logger, progress=progress)
    except Exception:
        logger.close()
        raise

    # final held-out evaluation, more episodes than the in-training checks
    path = checkpoint_path(algo, f"{cfg['tag']}_seed{seed}")
    save_checkpoint(
        path,
        algo=algo, tag=cfg["tag"], seed=seed, kind=result["kind"],
        hidden=result["hidden"], state_dict=result["state_dict"],
        score=result["score"], config=cfg,
    )
    final = evaluate_policy(load_policy(path), episodes=40)
    summary = final.summary()
    summary.update(
        score=float(final.returns.mean()),
        frames=result["frames"],
        wall_seconds=result["wall_seconds"],
    )
    logger.finish(summary)

    if save_best:
        best = checkpoint_path(algo, "best")
        from training.common import read_checkpoint_score
        if not best.exists() or summary["score"] > read_checkpoint_score(best):
            save_checkpoint(
                best,
                algo=algo, tag=cfg["tag"], seed=seed, kind=result["kind"],
                hidden=result["hidden"], state_dict=result["state_dict"],
                score=summary["score"], config=cfg, summary=summary,
            )

    return {
        "algo": algo,
        "tag": cfg["tag"],
        "seed": seed,
        "score": summary["score"],
        "equity_ratio": summary["eval_equity_ratio"],
        "drawdown": summary["eval_drawdown"],
        "stockouts": summary["eval_stockouts"],
        "insolvency_rate": summary["eval_insolvency_rate"],
        "frames": result["frames"],
        "wall_seconds": result["wall_seconds"],
        "note": cfg.get("note", ""),
    }


def _worker(args: tuple[str, str, int, dict | None]) -> dict[str, Any]:
    algo, tag, seed, overrides = args
    torch.set_num_threads(1)
    try:
        return train_one(
            algo, tag=tag, seed=seed, overrides=overrides,
            progress=False, save_best=False,
        )
    except Exception as exc:
        return {
            "algo": algo, "tag": tag, "seed": seed, "score": float("nan"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #

def run_sweep(
    algo: str,
    limit: int | None = None,
    seed: int = 0,
    workers: int = DEFAULT_WORKERS,
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    configs = hparams.configs(algo)
    if limit:
        configs = configs[:limit]

    jobs = [(algo, c["tag"], seed, overrides) for c in configs]
    print(
        f"\n{algo.upper()} sweep: {len(jobs)} configurations, "
        f"{workers} parallel workers\n" + "-" * 68
    )

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, j): j for j in jobs}
        for n, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            rows.append(row)
            if "error" in row:
                print(f"  [{n:>2}/{len(jobs)}] {row['tag']:<22} FAILED  {row['error']}")
            else:
                print(
                    f"  [{n:>2}/{len(jobs)}] {row['tag']:<22} "
                    f"score {row['score']:>+8.2f}  "
                    f"equity {(row['equity_ratio'] - 1) * 100:>+6.2f}%  "
                    f"stockouts {row['stockouts']:>5.1f}  "
                    f"{row['wall_seconds']:>5.0f}s",
                    flush=True,
                )

    _write_summary(algo, rows)
    _promote_best(algo, rows, seed)
    return rows


def _write_summary(algo: str, rows: list[dict[str, Any]]) -> None:
    out = LOGS_DIR / "sweeps"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{algo}_summary.csv"
    ok = [r for r in rows if "error" not in r]
    ok.sort(key=lambda r: r["score"], reverse=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in ok:
            w.writerow(r)
    print(f"\n  summary -> {path}")
    if ok:
        best = ok[0]
        print(f"  best: {best['tag']} ({best['score']:+.2f})")
    failed = [r for r in rows if "error" in r]
    if failed:
        print(f"  {len(failed)} run(s) failed: {', '.join(r['tag'] for r in failed)}")


def _promote_best(algo: str, rows: list[dict[str, Any]], seed: int) -> None:
    """Copy the winning configuration's checkpoint to {algo}_best.pt."""
    ok = [r for r in rows if "error" not in r and np.isfinite(r["score"])]
    if not ok:
        return
    best = max(ok, key=lambda r: r["score"])
    src = checkpoint_path(algo, f"{best['tag']}_seed{best['seed']}")
    if not src.exists():
        return
    blob = torch.load(src, map_location="cpu", weights_only=False)
    save_checkpoint(
        checkpoint_path(algo, "best"),
        algo=algo, tag=blob["tag"], seed=blob["seed"], kind=blob["kind"],
        hidden=blob["hidden"], state_dict=blob["state_dict"],
        score=best["score"], config=blob["config"], summary=best,
    )


def run_all(workers: int = DEFAULT_WORKERS, seed: int = 0) -> None:
    for algo in hparams.ALGOS:
        run_sweep(algo, seed=seed, workers=workers)

# --------------------------------------------------------------------------- #
# Seed replication
# --------------------------------------------------------------------------- #

REPLICATION_COLUMNS = (
    "algo", "tag", "seed", "held_out_score", "equity_ratio", "drawdown",
    "stockouts", "insolvency_rate",
)


def best_tag(algo: str) -> str | None:
    """The winning configuration from this algorithm's grid."""
    path = LOGS_DIR / "sweeps" / f"{algo}_summary.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    return max(rows, key=lambda r: float(r["score"]))["tag"]


def run_replication(
    seeds: tuple[int, ...] = (0, 1, 2),
    workers: int = DEFAULT_WORKERS,
    episodes: int = 100,
    eval_seed_base: int = 20_000,
) -> list[dict[str, Any]]:
    """
    Retrain each algorithm's best configuration across several seeds.

    A configuration that only ever ran once cannot tell a better algorithm from
    a luckier initialisation. Checkpoints that already exist are reused rather
    than retrained, and every checkpoint -- old and new -- is then scored on the
    same held-out seed range, disjoint from both training and model selection.
    """
    targets = [(a, best_tag(a)) for a in hparams.ALGOS]
    targets = [(a, tg) for a, tg in targets if tg]
    if not targets:
        print("no sweep summaries found -- run the sweeps first")
        return []

    jobs = [
        (algo, tag, seed, None)
        for algo, tag in targets
        for seed in seeds
        if not checkpoint_path(algo, f"{tag}_seed{seed}").exists()
    ]
    total = len(targets) * len(seeds)
    print(
        f"\nSeed replication: {len(targets)} configurations x {len(seeds)} seeds "
        f"= {total} runs\n  {len(jobs)} to train, "
        f"{total - len(jobs)} already on disk\n" + "-" * 68
    )

    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, j): j for j in jobs}
            for n, fut in enumerate(as_completed(futures), start=1):
                row = fut.result()
                status = (row["error"] if "error" in row
                          else f"score {row['score']:+8.2f}")
                print(f"  [{n:>2}/{len(jobs)}] {row['tag']:<20} "
                      f"seed {row['seed']}  {status}", flush=True)

    # Re-score every checkpoint on the same held-out seeds, so the spread
    # reported is a spread over training seeds and nothing else.
    print("\n  scoring all checkpoints on held-out seeds "
          f"{eval_seed_base}..{eval_seed_base + episodes - 1}")
    rows: list[dict[str, Any]] = []
    for algo, tag in targets:
        for seed in seeds:
            path = checkpoint_path(algo, f"{tag}_seed{seed}")
            if not path.exists():
                continue
            res = evaluate_policy(
                load_policy(path), episodes=episodes, seed_base=eval_seed_base
            )
            s = res.summary()
            rows.append({
                "algo": algo, "tag": tag, "seed": seed,
                "held_out_score": float(res.returns.mean()),
                "equity_ratio": s["eval_equity_ratio"],
                "drawdown": s["eval_drawdown"],
                "stockouts": s["eval_stockouts"],
                "insolvency_rate": s["eval_insolvency_rate"],
            })

    out = LOGS_DIR / "seed_replication.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REPLICATION_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {out}\n")

    for algo, tag in targets:
        vals = [r["held_out_score"] for r in rows if r["algo"] == algo]
        if vals:
            arr = np.array(vals)
            print(f"  {algo.upper():<10} {tag:<20} "
                  f"{arr.mean():+8.2f} +/- {arr.std(ddof=1) if len(arr) > 1 else 0.0:5.2f} "
                  f"(n={len(arr)}, min {arr.min():+.1f}, max {arr.max():+.1f})")
    return rows
