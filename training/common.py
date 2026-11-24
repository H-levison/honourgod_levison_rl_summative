"""
Shared training infrastructure: networks, logging, evaluation, checkpoints.

Everything here is deliberately algorithm-agnostic. The four trainers differ
only in their loss and their data collection; they share this environment
factory, this network builder and this evaluation loop, which is what makes the
comparison in the report a fair one rather than four separately-tuned pipelines.

The logging schema is fixed and complete for every run, including columns an
individual algorithm does not populate. The report needs five specific figures
-- comparative rewards, DQN loss, policy-gradient entropy, convergence, and
generalisation -- and discovering a missing column after a 48-run sweep means
running the sweep again.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from environment import ACTION_NAMES, N_ACTIONS, OBS_DIM, DeskConfig, LiquidityDeskEnv

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"
MODELS_DIR = ROOT / "models"

# Training and evaluation seeds never overlap, so the evaluation columns are
# genuine held-out generalisation evidence rather than training performance.
TRAIN_SEED_BASE = 0
EVAL_SEED_BASE = 9_000


# --------------------------------------------------------------------------- #
# Logging schema
# --------------------------------------------------------------------------- #

BASE_COLUMNS: tuple[str, ...] = (
    "iteration",
    "frames",
    "time_s",
    # on-policy rollout statistics
    "episode_return_mean",
    "episode_return_std",
    "episode_len_mean",
    # optimisation
    "loss_total",
    "loss_policy",
    "loss_critic",
    "loss_qvalue",
    "entropy",
    "grad_norm",
    "lr",
    "epsilon",
    "explained_variance",
    # periodic held-out evaluation
    "eval_return_mean",
    "eval_return_std",
    "eval_equity_ratio",
    "eval_drawdown",
    "eval_stockouts",
    "eval_fill_rate",
    "eval_insolvency_rate",
)

ACTION_COLUMNS: tuple[str, ...] = tuple(f"act_{name.lower()}" for name in ACTION_NAMES)
ALL_COLUMNS: tuple[str, ...] = BASE_COLUMNS + ACTION_COLUMNS


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "uncommitted"
    except Exception:
        return "unknown"


class RunLogger:
    """One CSV plus one metadata file per training run."""

    def __init__(self, algo: str, tag: str, seed: int, root: Path = LOGS_DIR) -> None:
        self.algo = algo
        self.tag = tag
        self.seed = seed
        self.dir = root / "runs" / algo
        self.dir.mkdir(parents=True, exist_ok=True)
        stem = f"{tag}_seed{seed}"
        self.csv_path = self.dir / f"{stem}.csv"
        self.meta_path = self.dir / f"{stem}.meta.json"
        self._fh = None
        self._writer: csv.DictWriter | None = None
        self._t0 = time.perf_counter()
        self.rows = 0

    def start(self, config: dict[str, Any]) -> RunLogger:
        self.meta_path.write_text(
            json.dumps(
                {
                    "algo": self.algo,
                    "tag": self.tag,
                    "seed": self.seed,
                    "config": _jsonable(config),
                    "git_sha": git_sha(),
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "torch": torch.__version__,
                    "cuda": torch.cuda.is_available(),
                    "device_name": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else platform.processor()
                    ),
                    "obs_dim": OBS_DIM,
                    "n_actions": N_ACTIONS,
                    "train_seed_base": TRAIN_SEED_BASE,
                    "eval_seed_base": EVAL_SEED_BASE,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._fh = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=ALL_COLUMNS, restval="")
        self._writer.writeheader()
        return self

    def log(self, **row: Any) -> None:
        if self._writer is None:
            raise RuntimeError("RunLogger.start() must be called before log()")
        unknown = set(row) - set(ALL_COLUMNS)
        if unknown:
            raise KeyError(f"unknown log columns {sorted(unknown)}")
        row.setdefault("time_s", round(time.perf_counter() - self._t0, 2))
        self._writer.writerow({k: _scalar(v) for k, v in row.items()})
        self._fh.flush()
        self.rows += 1

    def finish(self, summary: dict[str, Any]) -> None:
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        meta["summary"] = _jsonable(summary)
        meta["wall_seconds"] = round(time.perf_counter() - self._t0, 1)
        meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _scalar(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().mean().item()
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float):
        return round(v, 6)
    return v


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# --------------------------------------------------------------------------- #
# Devices and seeding
# --------------------------------------------------------------------------- #

def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #

def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden: Sequence[int] = (256, 256),
    activation: type[nn.Module] = nn.Tanh,
    final_gain: float = 0.01,
) -> nn.Sequential:
    """
    A plain MLP with orthogonal initialisation.

    Orthogonal init with a small final-layer gain is the standard recipe for
    policy heads: it starts the action distribution close to uniform, so the
    agent explores all 13 actions before committing. Without it a randomly
    initialised head can begin heavily biased toward one action and the
    entropy coefficient has to undo that before any learning happens.
    """
    dims = [in_dim, *hidden]
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        linear = nn.Linear(a, b)
        nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers += [linear, activation()]
    head = nn.Linear(dims[-1], out_dim)
    nn.init.orthogonal_(head.weight, gain=final_gain)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# Environment factory
# --------------------------------------------------------------------------- #

def make_env(seed: int | None = None, **kwargs: Any) -> LiquidityDeskEnv:
    """Every algorithm and every baseline builds the environment through here."""
    env = LiquidityDeskEnv(DeskConfig(), **kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_torchrl_env(seed: int | None = None, device: str | torch.device = "cpu"):
    """
    The same environment, wrapped for TorchRL.

    `categorical_action_encoding=True` keeps actions as plain integers rather
    than one-hot vectors, matching the environment's own `step(int)` contract
    and the `Discrete(13)` action space.

    `RewardSum` is what makes episode returns observable to the trainers: it
    accumulates reward across each episode so a completed trajectory carries
    its total, which is the headline metric in the comparative-rewards figure.
    """
    from torchrl.envs import (
        Compose, DoubleToFloat, GymWrapper, RewardSum, StepCounter, TransformedEnv,
    )

    base = GymWrapper(
        LiquidityDeskEnv(DeskConfig()),
        categorical_action_encoding=True,
        device=device,
    )
    env = TransformedEnv(base, Compose(DoubleToFloat(), StepCounter(), RewardSum()))
    if seed is not None:
        env.set_seed(seed)
    return env


def single_threaded() -> None:
    """
    Pin each process to one BLAS thread.

    The sweep runs several configurations concurrently. Measured on this
    project, GPU gradient steps are only 1.4x CPU for a 35->256->256->13 MLP
    and the environment is the real bottleneck (41 ms of env stepping against
    10 ms of gradients per DQN iteration), so the throughput win comes from
    running many processes rather than from the accelerator. Letting each
    process grab every core would make them fight.
    """
    torch.set_num_threads(1)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@dataclass
class EvalResult:
    returns: np.ndarray
    equity_ratio: np.ndarray
    drawdown: np.ndarray
    stockouts: np.ndarray
    fill_rate: np.ndarray
    insolvent: np.ndarray
    actions: np.ndarray = field(default_factory=lambda: np.zeros(N_ACTIONS))

    def as_log_columns(self) -> dict[str, Any]:
        cols: dict[str, Any] = {
            "eval_return_mean": float(self.returns.mean()),
            "eval_return_std": float(self.returns.std()),
            "eval_equity_ratio": float(self.equity_ratio.mean()),
            "eval_drawdown": float(self.drawdown.mean()),
            "eval_stockouts": float(self.stockouts.mean()),
            "eval_fill_rate": float(self.fill_rate.mean()),
            "eval_insolvency_rate": float(self.insolvent.mean()),
        }
        total = max(self.actions.sum(), 1.0)
        for name, count in zip(ACTION_COLUMNS, self.actions):
            cols[name] = float(count / total)
        return cols

    def summary(self) -> dict[str, Any]:
        return {
            "eval_return_mean": float(self.returns.mean()),
            "eval_return_std": float(self.returns.std()),
            "eval_equity_ratio": float(self.equity_ratio.mean()),
            "eval_drawdown": float(self.drawdown.mean()),
            "eval_stockouts": float(self.stockouts.mean()),
            "eval_insolvency_rate": float(self.insolvent.mean()),
            "action_distribution": {
                name: float(c / max(self.actions.sum(), 1.0))
                for name, c in zip(ACTION_NAMES, self.actions)
            },
        }


def evaluate_policy(
    policy: Callable[[np.ndarray], int],
    episodes: int = 20,
    seed_base: int = EVAL_SEED_BASE,
    env: LiquidityDeskEnv | None = None,
) -> EvalResult:
    """
    Run a greedy policy over held-out seeds.

    Shared by every algorithm and every baseline so that a number in the report
    means the same thing wherever it came from.
    """
    own = env is None
    env = env or make_env()
    out: dict[str, list[float]] = {
        k: [] for k in
        ("returns", "equity_ratio", "drawdown", "stockouts", "fill_rate", "insolvent")
    }
    actions = np.zeros(N_ACTIONS, dtype=np.float64)

    for i in range(episodes):
        obs, _ = env.reset(seed=seed_base + i)
        total = 0.0
        while True:
            a = int(policy(obs))
            actions[a] += 1
            obs, reward, terminated, truncated, info = env.step(a)
            total += reward
            if terminated or truncated:
                out["returns"].append(total)
                out["equity_ratio"].append(info["equity_ratio"])
                out["drawdown"].append(info["drawdown"])
                out["stockouts"].append(info["n_unfilled"])
                out["fill_rate"].append(info["fill_rate"])
                out["insolvent"].append(float(terminated))
                break

    if own:
        env.close()

    return EvalResult(
        returns=np.array(out["returns"]),
        equity_ratio=np.array(out["equity_ratio"]),
        drawdown=np.array(out["drawdown"]),
        stockouts=np.array(out["stockouts"]),
        fill_rate=np.array(out["fill_rate"]),
        insolvent=np.array(out["insolvent"]),
        actions=actions,
    )


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #

def checkpoint_path(algo: str, tag: str = "best") -> Path:
    family = "dqn" if algo == "dqn" else "pg"
    d = MODELS_DIR / family
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{algo}_{tag}.pt"


def save_checkpoint(
    path: Path,
    *,
    algo: str,
    tag: str,
    seed: int,
    kind: str,                 # "qvalue" (argmax Q) or "policy" (argmax logits)
    hidden: Sequence[int],
    state_dict: dict[str, Any],
    score: float,
    config: dict[str, Any],
    summary: dict[str, Any] | None = None,
) -> None:
    if kind not in {"qvalue", "policy"}:
        raise ValueError(f"kind must be 'qvalue' or 'policy', got {kind!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algo": algo,
            "tag": tag,
            "seed": seed,
            "kind": kind,
            "obs_dim": OBS_DIM,
            "n_actions": N_ACTIONS,
            "hidden": list(hidden),
            "state_dict": state_dict,
            "score": float(score),
            "config": _jsonable(config),
            "summary": _jsonable(summary or {}),
            "git_sha": git_sha(),
        },
        path,
    )


def read_checkpoint_score(path: Path) -> float:
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return float(blob.get("score", float("-inf")))
    except Exception:
        return float("-inf")


class GreedyNetPolicy:
    """A saved network wrapped as a callable the demo and evaluator can use."""

    def __init__(self, net: nn.Module, name: str) -> None:
        self.net = net.eval()
        self.name = name

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return int(self.net(x).argmax(dim=-1).item())


def load_policy(path: Path) -> GreedyNetPolicy:
    """
    Rebuild a greedy policy from a checkpoint.

    Both checkpoint kinds reduce to argmax over the network's output: the
    Q-network's action values for DQN, the policy logits for the policy
    gradient methods. Greedy evaluation of a stochastic policy is the standard
    choice and keeps the comparison against DQN like for like.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob["obs_dim"] != OBS_DIM or blob["n_actions"] != N_ACTIONS:
        raise ValueError(
            f"checkpoint {path.name} was trained on a different environment "
            f"(obs {blob['obs_dim']} vs {OBS_DIM}, "
            f"actions {blob['n_actions']} vs {N_ACTIONS}). Retrain it."
        )
    net = build_mlp(blob["obs_dim"], blob["n_actions"], blob["hidden"])
    net.load_state_dict(blob["state_dict"])
    return GreedyNetPolicy(net, name=blob["algo"].upper())
