"""Cross-border liquidity desk environment package."""

from gymnasium.envs.registration import register

from environment.custom_env import (
    ACTION_NAMES,
    CURRENCIES,
    HUB,
    N_ACTIONS,
    N_FIAT,
    N_POOLS,
    OBS_DIM,
    REBALANCE_MAP,
    DeskConfig,
    LiquidityDeskEnv,
)

ENV_ID = "LiquidityDesk-v0"

# Registered so TorchRL can build it with GymEnv(ENV_ID).
# max_episode_steps stays None: the environment owns its own truncation, and a
# TimeLimit wrapper on top would double-truncate.
register(
    id=ENV_ID,
    entry_point="environment.custom_env:LiquidityDeskEnv",
    max_episode_steps=None,
)

__all__ = [
    "ACTION_NAMES",
    "CURRENCIES",
    "ENV_ID",
    "HUB",
    "N_ACTIONS",
    "N_FIAT",
    "N_POOLS",
    "OBS_DIM",
    "REBALANCE_MAP",
    "DeskConfig",
    "LiquidityDeskEnv",
]
