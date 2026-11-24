"""
Policy-gradient arm of the comparison: REINFORCE, PPO and A2C.

All three share an actor, a critic, a collector and an advantage estimator, and
differ only in their loss and how each batch is consumed:

    REINFORCE  whole episodes, Monte Carlo advantage (GAE with lambda = 1),
               one gradient step per batch
    A2C        short bootstrapped rollouts, GAE, one gradient step per batch
    PPO        same rollouts, GAE, many epochs of clipped minibatch updates

Keeping the actor, critic, optimiser and evaluation identical across the three
is deliberate: it means a difference in the results is attributable to the
objective rather than to incidental tuning, which is what the report needs in
order to say anything about clipping, baselines or advantage estimation.

A note on the REINFORCE baseline switch. Textbook REINFORCE has no baseline at
all. Rather than special-casing the loss, the `r02-no-baseline` configuration
keeps the same critic but zeroes it and freezes it, so GAE with lambda = 1
returns the raw discounted return as the advantage. That is exactly the
unbaselined estimator, and it keeps every other moving part identical.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torch.distributions import Categorical
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss
from torchrl.objectives.value import GAE

from environment import N_ACTIONS, OBS_DIM
from training.common import (
    EVAL_SEED_BASE,
    GreedyNetPolicy,
    RunLogger,
    build_mlp,
    evaluate_policy,
    make_torchrl_env,
    resolve_device,
    set_seed,
    single_threaded,
)


# --------------------------------------------------------------------------- #
# Shared actor / critic
# --------------------------------------------------------------------------- #

def build_actor_critic(cfg: dict[str, Any], action_spec, device):
    """One actor-critic construction for all three policy-gradient methods."""
    policy_net = build_mlp(
        OBS_DIM, N_ACTIONS, cfg["hidden"], activation=nn.Tanh, final_gain=0.01
    ).to(device)
    policy_module = TensorDictModule(
        policy_net, in_keys=["observation"], out_keys=["logits"]
    )
    actor = ProbabilisticActor(
        policy_module,
        in_keys=["logits"],
        spec=action_spec,
        distribution_class=Categorical,
        return_log_prob=True,
        default_interaction_type="random",
    ).to(device)

    value_net = build_mlp(
        OBS_DIM, 1, cfg["hidden"], activation=nn.Tanh, final_gain=1.0
    ).to(device)
    critic = ValueOperator(value_net, in_keys=["observation"]).to(device)

    return policy_net, actor, value_net, critic


def _freeze_zero_critic(value_net: nn.Module) -> None:
    """Zero and freeze the critic, turning GAE(lambda=1) into a raw return."""
    with torch.no_grad():
        for p in value_net.parameters():
            p.zero_()
    for p in value_net.parameters():
        p.requires_grad_(False)


# --------------------------------------------------------------------------- #
# Generic training loop
# --------------------------------------------------------------------------- #

def _run(
    algo: str,
    cfg: dict[str, Any],
    seed: int,
    logger: RunLogger | None,
    progress: bool,
    build_loss: Callable,
    consume: Callable,
    use_baseline: bool = True,
) -> dict[str, Any]:
    single_threaded()
    device = resolve_device(cfg.get("device", "auto"))
    set_seed(seed)

    proto = make_torchrl_env(seed=seed, device=device)
    policy_net, actor, value_net, critic = build_actor_critic(
        cfg, proto.action_spec, device
    )

    if not use_baseline:
        _freeze_zero_critic(value_net)

    lmbda = 1.0 if algo == "reinforce" else float(cfg["lmbda"])
    advantage = GAE(
        gamma=float(cfg["gamma"]),
        lmbda=lmbda,
        value_network=critic,
        average_gae=bool(cfg.get("normalize_returns", True)),
    )

    loss_module = build_loss(cfg, actor, critic)

    trainable = [p for p in loss_module.parameters() if p.requires_grad]
    optim = torch.optim.Adam(trainable, lr=float(cfg["lr"]))

    collector = Collector(
        lambda: make_torchrl_env(seed=seed, device=device),
        actor,
        frames_per_batch=int(cfg["frames_per_batch"]),
        total_frames=int(cfg["total_frames"]),
        device=device,
    )

    greedy = GreedyNetPolicy(policy_net, name=algo.upper())
    best_score = -np.inf
    best_state: dict[str, Any] | None = None
    frames = 0
    next_eval = cfg["eval_every_frames"]
    t0 = time.perf_counter()

    for iteration, data in enumerate(collector):
        frames += data.numel()

        with torch.no_grad():
            advantage(data)

        stats = consume(cfg, data, loss_module, optim)

        collector.update_policy_weights_()

        row: dict[str, Any] = {
            "iteration": iteration,
            "frames": frames,
            "lr": float(cfg["lr"]),
            **stats,
        }

        done = data["next", "done"].reshape(-1)
        if done.any():
            finished = data["next", "episode_reward"].reshape(-1)[done]
            row["episode_return_mean"] = float(finished.mean())
            row["episode_return_std"] = (
                float(finished.std()) if finished.numel() > 1 else 0.0
            )
            steps = data["next", "step_count"].reshape(-1)[done].float()
            row["episode_len_mean"] = float(steps.mean())

        # explained variance of the critic: how much of the return it captures
        if use_baseline:
            with torch.no_grad():
                target = data["value_target"].reshape(-1)
                pred = data["state_value"].reshape(-1)
                var = target.var()
                row["explained_variance"] = (
                    float(1.0 - (target - pred).var() / var) if var > 0 else ""
                )

        if frames >= next_eval:
            next_eval += cfg["eval_every_frames"]
            result = evaluate_policy(
                greedy, episodes=cfg["eval_episodes"], seed_base=EVAL_SEED_BASE
            )
            row.update(result.as_log_columns())
            score = float(result.returns.mean())
            if score > best_score:
                best_score = score
                best_state = {
                    k: v.detach().cpu().clone() for k, v in policy_net.state_dict().items()
                }
            if progress:
                print(
                    f"    [{algo} {cfg['tag']}] {frames:>7}/{cfg['total_frames']} frames "
                    f"eval {score:+8.2f}  H {stats.get('entropy', float('nan')):.3f}  "
                    f"{frames / max(time.perf_counter() - t0, 1e-9):,.0f} fps",
                    flush=True,
                )

        if logger is not None:
            logger.log(**row)

    collector.shutdown()

    if best_state is None:
        best_state = {
            k: v.detach().cpu().clone() for k, v in policy_net.state_dict().items()
        }
        best_score = float(
            evaluate_policy(greedy, episodes=cfg["eval_episodes"]).returns.mean()
        )

    return {
        "algo": algo,
        "kind": "policy",
        "state_dict": best_state,
        "score": best_score,
        "hidden": list(cfg["hidden"]),
        "frames": frames,
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }


def _step_once(cfg, loss_td, loss_module, optim, keys):
    total = sum(loss_td[k] for k in keys if k in loss_td.keys())
    optim.zero_grad(set_to_none=True)
    total.backward()
    grad = torch.nn.utils.clip_grad_norm_(
        [p for p in loss_module.parameters() if p.requires_grad],
        float(cfg["max_grad_norm"]),
    )
    optim.step()
    return float(total.detach()), float(grad)


# --------------------------------------------------------------------------- #
# PPO
# --------------------------------------------------------------------------- #

def _ppo_loss(cfg, actor, critic):
    return ClipPPOLoss(
        actor,
        critic,
        clip_epsilon=float(cfg["clip_epsilon"]),
        entropy_bonus=True,
        entropy_coeff=float(cfg["entropy_coef"]),
        critic_coeff=float(cfg["critic_coef"]),
        normalize_advantage=True,
    )


def _ppo_consume(cfg, data, loss_module, optim):
    """Many epochs of clipped minibatch updates over the same batch."""
    flat = data.reshape(-1)
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(flat.numel()),
        sampler=SamplerWithoutReplacement(),
        batch_size=int(cfg["minibatch_size"]),
    )
    buffer.extend(flat)

    n_mb = max(flat.numel() // int(cfg["minibatch_size"]), 1)
    losses, grads, ents, crits, objs = [], [], [], [], []

    for _ in range(int(cfg["ppo_epochs"])):
        for _ in range(n_mb):
            mb = buffer.sample()
            loss_td = loss_module(mb)
            total, grad = _step_once(
                cfg, loss_td, loss_module, optim,
                ("loss_objective", "loss_critic", "loss_entropy"),
            )
            losses.append(total)
            grads.append(grad)
            objs.append(float(loss_td["loss_objective"].detach()))
            if "loss_critic" in loss_td.keys():
                crits.append(float(loss_td["loss_critic"].detach()))
            if "entropy" in loss_td.keys():
                ents.append(float(loss_td["entropy"].detach().mean()))

    return {
        "loss_total": float(np.mean(losses)),
        "loss_policy": float(np.mean(objs)),
        "loss_critic": float(np.mean(crits)) if crits else "",
        "entropy": float(np.mean(ents)) if ents else "",
        "grad_norm": float(np.mean(grads)),
    }


# --------------------------------------------------------------------------- #
# A2C
# --------------------------------------------------------------------------- #

def _a2c_loss(cfg, actor, critic):
    return A2CLoss(
        actor,
        critic,
        entropy_bonus=True,
        entropy_coeff=float(cfg["entropy_coef"]),
        critic_coeff=float(cfg["critic_coef"]),
    )


def _a2c_consume(cfg, data, loss_module, optim):
    """One gradient step on the whole short rollout. No trust region."""
    loss_td = loss_module(data.reshape(-1))
    total, grad = _step_once(
        cfg, loss_td, loss_module, optim,
        ("loss_objective", "loss_critic", "loss_entropy"),
    )
    return {
        "loss_total": total,
        "loss_policy": float(loss_td["loss_objective"].detach()),
        "loss_critic": (
            float(loss_td["loss_critic"].detach())
            if "loss_critic" in loss_td.keys() else ""
        ),
        "entropy": (
            float(loss_td["entropy"].detach().mean())
            if "entropy" in loss_td.keys() else ""
        ),
        "grad_norm": grad,
    }


# --------------------------------------------------------------------------- #
# REINFORCE
# --------------------------------------------------------------------------- #

def _reinforce_loss(cfg, actor, critic):
    return ReinforceLoss(actor, critic_network=critic)


def _reinforce_consume(cfg, data, loss_module, optim):
    """One Monte Carlo gradient step per batch of complete episodes."""
    flat = data.reshape(-1)
    loss_td = loss_module(flat)

    # entropy is not part of the ReinforceLoss objective, so it is added here
    # explicitly -- the report needs an entropy curve for every PG method, and
    # several grid entries vary the entropy coefficient.
    keys = ["loss_actor"]
    entropy = None
    coef = float(cfg.get("entropy_coef", 0.0))
    dist = loss_module.actor_network.get_dist(flat)
    entropy = dist.entropy().mean()
    if coef > 0:
        loss_td["loss_entropy"] = -coef * entropy
        keys.append("loss_entropy")
    if "loss_value" in loss_td.keys():
        keys.append("loss_value")

    total, grad = _step_once(cfg, loss_td, loss_module, optim, tuple(keys))
    return {
        "loss_total": total,
        "loss_policy": float(loss_td["loss_actor"].detach()),
        "loss_critic": (
            float(loss_td["loss_value"].detach())
            if "loss_value" in loss_td.keys() else ""
        ),
        "entropy": float(entropy.detach()),
        "grad_norm": grad,
    }


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #

def train_ppo(cfg, seed=0, logger=None, progress=True):
    return _run("ppo", cfg, seed, logger, progress, _ppo_loss, _ppo_consume)


def train_a2c(cfg, seed=0, logger=None, progress=True):
    return _run("a2c", cfg, seed, logger, progress, _a2c_loss, _a2c_consume)


def train_reinforce(cfg, seed=0, logger=None, progress=True):
    return _run(
        "reinforce", cfg, seed, logger, progress,
        _reinforce_loss, _reinforce_consume,
        use_baseline=bool(cfg.get("use_baseline", True)),
    )
