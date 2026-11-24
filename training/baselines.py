"""
Non-learning baseline policies.

These exist for two reasons. They make `uv run main.py` produce a working 3D
demo before any network has been trained, and they give the report an honest
comparison floor: an RL result only means something next to the rule-based desk
operator it is supposed to beat.

Every baseline reads the same 35-dim observation the agents receive, so no
baseline has access to information the learners lack.

Observation layout (see LiquidityDeskEnv._observe)
--------------------------------------------------
     0:4   pool value as a fraction of equity
     4:8   pool value relative to a balanced book, minus 1
     8:11  log FX drift since the start of the episode, x10
    11:14  EWMA FX log-return, x500
    14:17  GARCH volatility regime, minus 1
    17     quoted spread, relative to its default
    18:21  per-corridor directional order pressure
    21     pending order size / equity, x20
    22     pending order direction (+1 fiat in, -1 fiat out)
    23:26  pending order corridor, one-hot
    26:29  corridor paused flags
    29     equity ratio, minus 1
    30     drawdown from high-water mark
    31     fraction of the episode remaining
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from environment import HUB, N_ACTIONS, N_FIAT  # noqa: F401

# observation slice names, so the policies below read as intent not arithmetic
IDX_BALANCE_DEV = slice(4, 8)
IDX_MOMENTUM = slice(11, 14)
IDX_VOL_REGIME = slice(14, 17)
IDX_SPREAD = 17
IDX_PAUSED = slice(26, 29)
IDX_DRAWDOWN = 30
IDX_PRESSURE = slice(32, 35)


def fiat_to_hub_action(fiat: int) -> int:
    """Action id that moves the given fiat pool into the USDT rail."""
    return 1 + 2 * fiat


def hub_to_fiat_action(fiat: int) -> int:
    """Action id that funds the given fiat pool from the USDT rail."""
    return 2 + 2 * fiat


def pause_action(fiat: int) -> int:
    """Action id that stops quoting the given corridor."""
    return 9 + fiat


class Policy(Protocol):
    name: str

    def __call__(self, obs: np.ndarray) -> int: ...


class RandomPolicy:
    """Uniform random action. The floor any agent must clear."""

    name = "Random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        """
        Re-key the action stream for a new episode.

        Evaluation seeds the environment per episode; without this the random
        baseline would draw from OS entropy and its score would move between
        runs while every other policy's stayed fixed. The offset keeps the
        policy's stream distinct from the environment's for the same seed.
        """
        self._rng = np.random.default_rng(seed + 1_000_003)

    def __call__(self, obs: np.ndarray) -> int:
        return int(self._rng.integers(N_ACTIONS))


class HoldPolicy:
    """Never act. Measures how much the FX drift alone costs the desk."""

    name = "Hold"

    def __call__(self, obs: np.ndarray) -> int:
        return 0


class HeuristicPolicy:
    """
    A reasonable human desk operator, expressed as rules.

    Priorities, in order:
      1. refill a pool that is drifting toward a stockout,
      2. sell down a fiat pool that is both over-weight and depreciating,
      3. pause a corridor only in a genuine volatility blowout,
      4. price volatility into the spread,
      5. otherwise stand still and collect the spread.

    Thresholds are set from measured behaviour, not taste. Pausing is the
    expensive action -- a closed corridor declines every order on it, and
    rotating the inventory out into the rail is almost always the better answer
    -- so `vol_panic` is set high enough that only an extreme regime triggers
    it. The volatility observation is log1p-scaled, so 2.5 corresponds to
    variance around twelve times its base level.
    """

    name = "Heuristic"

    def __init__(
        self,
        deficit_threshold: float = -0.20,
        surplus_threshold: float = 0.05,
        vol_panic: float = 2.5,
        derisk_floor: float = -0.45,
        derisk_pressure: float = 0.55,
    ) -> None:
        self.deficit_threshold = deficit_threshold
        self.surplus_threshold = surplus_threshold
        self.vol_panic = vol_panic
        self.derisk_floor = derisk_floor
        self.derisk_pressure = derisk_pressure

    def __call__(self, obs: np.ndarray) -> int:
        dev = obs[IDX_BALANCE_DEV]          # how far each pool is from balanced
        momentum = obs[IDX_MOMENTUM]        # >0 means the fiat is depreciating
        vol = obs[IDX_VOL_REGIME]
        pressure = obs[IDX_PRESSURE]        # devaluation pressure, the signal
        paused = obs[IDX_PAUSED] > 0.5
        spread = obs[IDX_SPREAD]

        fiat_dev = dev[:N_FIAT]
        neediest = int(np.argmin(dev))

        # 1. inventory first: corrections are now gap-sized, so acting early is
        #    cheap and a stockout costs several trades' revenue
        if dev[neediest] < self.deficit_threshold:
            if neediest == HUB:
                donor = int(np.argmax(fiat_dev))
                if fiat_dev[donor] > self.surplus_threshold:
                    return fiat_to_hub_action(donor)
            elif dev[HUB] > self.surplus_threshold:
                return hub_to_fiat_action(neediest)

        # 2. carry less of a currency that is under devaluation pressure, or
        #    already falling.
        #
        #    Pressure is the leading indicator: it builds under sustained
        #    dollar demand and depreciation, and the devaluation hazard rises
        #    exponentially with it. Acting on pressure is insurance bought
        #    before the event; acting on momentum alone is closing the stable
        #    door. Rebalances move a small fixed share of the source pool, so
        #    this fires repeatedly while the signal persists and accumulates
        #    into a real reduction in exposure. The floor stops it draining a
        #    pool to the point where fiat payouts start stocking out.
        risk = pressure + np.maximum(momentum, 0.0) * 0.3
        eligible = (risk > self.derisk_pressure) & (fiat_dev > self.derisk_floor)
        if eligible.any():
            scored = np.where(eligible, risk, -np.inf)
            return fiat_to_hub_action(int(np.argmax(scored)))

        # 3. step away from a corridor only when volatility has truly blown out.
        #    PAUSE_<corridor> names the corridor explicitly, so checking
        #    `paused[riskiest]` genuinely prevents re-pausing the same one --
        #    which an aggregate "pause the riskiest" action could not.
        riskiest = int(np.argmax(vol))
        if vol[riskiest] > self.vol_panic and not paused[riskiest]:
            return pause_action(riskiest)

        # 4. park the spread near its revenue-maximising level.
        #    Revenue per step is proportional to s * (1.35 - 30s), which peaks
        #    at s = 2.25%. Quoting the 3% maximum earns the same as 1.5% while
        #    filling half as many orders, so the spread is a band to sit in,
        #    not a dial to crank. obs[17] is (spread - 1.5%) / 1.5%, so the
        #    2.0-2.5% target band is 0.33 to 0.67.
        if spread < 0.30:
            return 7  # WIDEN_SPREAD
        if spread > 0.70:
            return 8  # TIGHTEN_SPREAD

        return 0  # HOLD


BASELINES: dict[str, type] = {
    "random": RandomPolicy,
    "hold": HoldPolicy,
    "heuristic": HeuristicPolicy,
}


def make_baseline(name: str, **kwargs) -> Policy:
    key = name.lower()
    if key not in BASELINES:
        raise KeyError(
            f"unknown baseline {name!r}; choose from {sorted(BASELINES)}"
        )
    return BASELINES[key](**kwargs)
