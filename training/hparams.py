"""
Hyperparameter grids: twelve configurations per algorithm, forty-eight total.

Each entry carries a `note` stating the hypothesis it tests. The grids are not
random jitter -- every configuration moves a parameter that theory says should
matter for this specific environment, so the report can explain *why* a run
behaved the way it did rather than just tabulating numbers.

The environment's defining difficulty is delayed credit assignment: a rebalance
fee is paid now to avoid a stockout tens of steps later. That single fact drives
most of the grid design below -- discount factor, bootstrap length, GAE lambda
and baseline choice are all varied because they all control how far reward
signal propagates backwards.
"""

from __future__ import annotations

import copy
from typing import Any

# --------------------------------------------------------------------------- #
# Shared defaults
# --------------------------------------------------------------------------- #

SHARED: dict[str, Any] = {
    "total_frames": 300_000,
    "hidden": (256, 256),
    "gamma": 0.99,
    "max_grad_norm": 1.0,
    "eval_episodes": 20,
    "eval_every_frames": 25_000,
    "device": "auto",
}


# --------------------------------------------------------------------------- #
# DQN  --  value based
# --------------------------------------------------------------------------- #

DQN_BASE: dict[str, Any] = {
    "frames_per_batch": 256,
    "lr": 2.5e-4,
    "buffer_size": 200_000,
    "batch_size": 256,
    "n_optim": 4,                 # gradient steps per collected batch
    "init_random_frames": 10_000,
    "eps_init": 1.0,
    "eps_end": 0.05,
    "eps_annealing_frames": 120_000,
    "target_update_tau": 0.005,   # soft update; 1.0 means hard copy
    "target_update_period": 1,
    "double_dqn": True,
}

DQN_GRID: list[dict[str, Any]] = [
    {"tag": "d01-baseline", "note": "Reference settings for the family."},

    {"tag": "d02-lr-high", "lr": 1e-3,
     "note": "Faster value updates. Expect quicker early progress but noisier "
             "Q-estimates once FX jumps start appearing."},

    {"tag": "d03-lr-low", "lr": 5e-5,
     "note": "Slow, stable bootstrapping. Tests whether d02's instability is "
             "really a learning-rate effect."},

    {"tag": "d04-gamma-short", "gamma": 0.95,
     "note": "Shortens the effective horizon to ~20 steps. Should under-value "
             "rebalances whose payoff is a stockout avoided much later."},

    {"tag": "d05-gamma-long", "gamma": 0.995,
     "note": "Horizon ~200 steps, close to the episode length. Better credit "
             "assignment, but higher variance in the bootstrap target."},

    {"tag": "d06-buffer-small", "buffer_size": 20_000,
     "note": "Replay holds roughly 40 episodes. Tests sensitivity to stale "
             "transitions drawn from a non-stationary FX process."},

    {"tag": "d07-target-hard", "target_update_tau": 1.0,
     "target_update_period": 1_000,
     "note": "Hard periodic target copy instead of Polyak averaging, the "
             "original DQN scheme."},

    {"tag": "d08-no-double", "double_dqn": False,
     "note": "Isolates Double-DQN's contribution. Expect optimistic value "
             "bias, visible as overconfident holding during devaluations."},

    {"tag": "d09-batch-large", "batch_size": 1024, "n_optim": 2,
     "note": "Lower-variance gradients per update, fewer updates. Tests the "
             "batch-size / update-count trade-off at fixed compute."},

    {"tag": "d10-explore-long", "eps_annealing_frames": 240_000,
     "note": "Exploration for 80% of training. Stockouts are rare events, so "
             "sustained exploration may be needed to ever see them."},

    {"tag": "d11-explore-short", "eps_annealing_frames": 30_000,
     "eps_end": 0.01,
     "note": "Commits to greedy behaviour early. Risks locking in a "
             "do-nothing policy before the cost of stockouts is learned."},

    {"tag": "d12-net-wide", "hidden": (512, 512), "lr": 1.5e-4,
     "note": "More capacity for the 35-dim observation, with the learning "
             "rate reduced to keep the larger network stable."},
]


# --------------------------------------------------------------------------- #
# REINFORCE  --  Monte Carlo policy gradient
# --------------------------------------------------------------------------- #

REINFORCE_BASE: dict[str, Any] = {
    "episodes_per_update": 8,     # full episodes; REINFORCE needs complete returns
    "lr": 3e-4,
    "entropy_coef": 0.01,
    "use_baseline": True,         # learned value baseline, not plain REINFORCE
    "baseline_lr": 1e-3,
    "normalize_returns": True,
    "frames_per_batch": 4_000,
}

REINFORCE_GRID: list[dict[str, Any]] = [
    {"tag": "r01-baseline", "note": "Reference settings for the family."},

    {"tag": "r02-no-baseline", "use_baseline": False,
     "note": "Textbook REINFORCE. Expect the highest gradient variance of any "
             "run in the study -- this is the control that shows why baselines "
             "exist."},

    {"tag": "r03-no-normalize", "normalize_returns": False,
     "note": "Raw returns. Reward scale varies with FX regime, so this should "
             "destabilise the policy when volatility clusters."},

    {"tag": "r04-lr-high", "lr": 1e-3,
     "note": "Large steps on a high-variance gradient estimate. Tests whether "
             "the policy can survive its own update noise."},

    {"tag": "r05-lr-low", "lr": 1e-4,
     "note": "Small steps to average out variance, at the cost of sample "
             "efficiency over a fixed frame budget."},

    {"tag": "r06-batch-large", "episodes_per_update": 24, "frames_per_batch": 12_000,
     "note": "Triples the episodes per gradient step. The cleanest available "
             "lever on Monte Carlo variance."},

    {"tag": "r07-batch-small", "episodes_per_update": 2, "frames_per_batch": 1_000,
     "note": "Frequent but very noisy updates. Expect erratic learning curves."},

    {"tag": "r08-entropy-high", "entropy_coef": 0.05,
     "note": "Strong exploration pressure. Should keep the agent trying "
             "rebalances it would otherwise abandon early."},

    {"tag": "r09-entropy-zero", "entropy_coef": 0.0,
     "note": "No entropy bonus. Watch for premature collapse onto HOLD, the "
             "action with the lowest immediate cost."},

    {"tag": "r10-gamma-short", "gamma": 0.95,
     "note": "Short horizon on a Monte Carlo estimator, which already sees the "
             "whole episode. Tests how much the discount alone matters."},

    {"tag": "r11-gamma-undiscounted", "gamma": 1.0,
     "note": "No discounting at all. Correct for a fixed-length episode, but "
             "maximal return variance."},

    {"tag": "r12-net-small", "hidden": (128, 128), "lr": 5e-4,
     "note": "A smaller policy generalises more and overfits the FX noise "
             "less; paired with a higher learning rate."},
]


# --------------------------------------------------------------------------- #
# PPO  --  clipped surrogate policy gradient
# --------------------------------------------------------------------------- #

PPO_BASE: dict[str, Any] = {
    "frames_per_batch": 4_000,
    "lr": 3e-4,
    "clip_epsilon": 0.2,
    "ppo_epochs": 10,
    "minibatch_size": 256,
    "entropy_coef": 0.01,
    "critic_coef": 0.5,
    "lmbda": 0.95,
}

PPO_GRID: list[dict[str, Any]] = [
    {"tag": "p01-baseline", "note": "Reference settings for the family."},

    {"tag": "p02-clip-tight", "clip_epsilon": 0.05,
     "note": "Very conservative trust region. Slow but should be the most "
             "monotonic curve in the study."},

    {"tag": "p03-clip-loose", "clip_epsilon": 0.4,
     "note": "Weak constraint, approaching unclipped policy gradient. Tests "
             "whether clipping is what keeps PPO stable here."},

    {"tag": "p04-epochs-few", "ppo_epochs": 3,
     "note": "Less reuse per batch. Lower off-policy drift, worse sample "
             "efficiency."},

    {"tag": "p05-epochs-many", "ppo_epochs": 30,
     "note": "Heavy reuse of each batch. Expect the ratio to leave the trust "
             "region and clipping to bind on most minibatches."},

    {"tag": "p06-lambda-low", "lmbda": 0.8,
     "note": "GAE closer to one-step TD: lower variance, more bias. Should "
             "blur credit for rebalances that pay off slowly."},

    {"tag": "p07-lambda-one", "lmbda": 1.0,
     "note": "GAE becomes the full Monte Carlo advantage. Unbiased, high "
             "variance -- the direct bridge to the REINFORCE family."},

    {"tag": "p08-batch-large", "frames_per_batch": 16_000, "minibatch_size": 1_024,
     "note": "Four times the data per policy update. Better advantage "
             "estimates, fewer updates within the frame budget."},

    {"tag": "p09-batch-small", "frames_per_batch": 1_000, "minibatch_size": 128,
     "note": "Frequent updates on short rollouts that rarely span a full "
             "devaluation episode."},

    {"tag": "p10-entropy-high", "entropy_coef": 0.05,
     "note": "Keeps the action distribution broad for longer, counteracting "
             "early collapse onto HOLD."},

    {"tag": "p11-lr-high", "lr": 1e-3,
     "note": "Tests whether PPO's clipping absorbs an aggressive learning "
             "rate that would break REINFORCE outright."},

    {"tag": "p12-gamma-long", "gamma": 0.995, "lmbda": 0.98,
     "note": "Long horizon with long GAE traces, matched to the 500-step "
             "episode. The configuration theory favours for this task."},
]


# --------------------------------------------------------------------------- #
# A2C  --  synchronous advantage actor critic
# --------------------------------------------------------------------------- #

A2C_BASE: dict[str, Any] = {
    "frames_per_batch": 512,      # short n-step rollouts, A2C's signature
    "lr": 7e-4,
    "entropy_coef": 0.01,
    "critic_coef": 0.5,
    "lmbda": 0.95,
}

A2C_GRID: list[dict[str, Any]] = [
    {"tag": "a01-baseline", "note": "Reference settings for the family."},

    {"tag": "a02-nstep-short", "frames_per_batch": 128,
     "note": "Very short bootstrap. Fast, biased, and likely blind to delayed "
             "stockout costs."},

    {"tag": "a03-nstep-long", "frames_per_batch": 2_048,
     "note": "Long rollouts approach PPO's batch regime without the clipped "
             "objective -- a clean isolation of what clipping buys."},

    {"tag": "a04-lr-high", "lr": 2e-3,
     "note": "A2C has no trust region, so this should be visibly less stable "
             "than PPO at a comparable learning rate."},

    {"tag": "a05-lr-low", "lr": 2e-4,
     "note": "Conservative steps to compensate for the missing trust region."},

    {"tag": "a06-critic-weak", "critic_coef": 0.1,
     "note": "Under-trained critic means a poor baseline, pushing A2C back "
             "toward REINFORCE-like variance."},

    {"tag": "a07-critic-strong", "critic_coef": 1.0,
     "note": "Critic dominates the shared gradient. Tests whether value "
             "accuracy or policy improvement is the binding constraint."},

    {"tag": "a08-entropy-high", "entropy_coef": 0.05,
     "note": "Sustained exploration across the 13 discrete actions."},

    {"tag": "a09-entropy-zero", "entropy_coef": 0.0,
     "note": "Expect the fastest policy collapse of any actor-critic run."},

    {"tag": "a10-lambda-zero", "lmbda": 0.0,
     "note": "Pure one-step TD advantage, classical A2C. Maximum bias, "
             "minimum variance."},

    {"tag": "a11-gamma-long", "gamma": 0.995,
     "note": "Extends the horizon while keeping rollouts short, stressing the "
             "critic's bootstrap accuracy."},

    {"tag": "a12-net-wide", "hidden": (512, 512), "lr": 3e-4,
     "note": "More capacity in the shared trunk, with a reduced learning "
             "rate."},
]


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

_FAMILIES: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {
    "dqn": (DQN_BASE, DQN_GRID),
    "reinforce": (REINFORCE_BASE, REINFORCE_GRID),
    "ppo": (PPO_BASE, PPO_GRID),
    "a2c": (A2C_BASE, A2C_GRID),
}

ALGOS: tuple[str, ...] = tuple(_FAMILIES)


def configs(algo: str) -> list[dict[str, Any]]:
    """Fully merged configurations for one algorithm, in grid order."""
    key = algo.lower()
    if key not in _FAMILIES:
        raise KeyError(f"unknown algorithm {algo!r}; choose from {ALGOS}")

    base, grid = _FAMILIES[key]
    out: list[dict[str, Any]] = []
    for entry in grid:
        cfg = copy.deepcopy(SHARED)
        cfg.update(copy.deepcopy(base))
        cfg.update(copy.deepcopy(entry))
        cfg["algo"] = key
        out.append(cfg)
    return out


def config(algo: str, tag: str) -> dict[str, Any]:
    """One configuration by tag, or the algorithm's baseline if tag is None."""
    for cfg in configs(algo):
        if cfg["tag"] == tag:
            return cfg
    raise KeyError(f"no configuration tagged {tag!r} for {algo!r}")


def baseline_config(algo: str) -> dict[str, Any]:
    return configs(algo)[0]


def all_configs() -> list[dict[str, Any]]:
    return [cfg for algo in ALGOS for cfg in configs(algo)]


def summary_table() -> str:
    """Plain-text grid listing, handy when writing up the report."""
    lines = []
    for algo in ALGOS:
        entries = configs(algo)
        lines.append(f"\n{algo.upper()}  ({len(entries)} runs)")
        lines.append("-" * 72)
        for cfg in entries:
            lines.append(f"  {cfg['tag']:<22} {cfg['note']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary_table())
    print(f"\nTotal runs: {len(all_configs())}")
