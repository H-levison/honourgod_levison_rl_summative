"""
Correctness tests for LiquidityDeskEnv.

The accounting invariants matter more than they look: if equity is not
conserved under a no-op in a frozen market, every reward the agents learn from
is quietly wrong, and the whole four-algorithm comparison is meaningless.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment import (
    ACTION_NAMES, HUB, N_ACTIONS, N_FIAT, OBS_DIM, DeskConfig, LiquidityDeskEnv,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def frozen_market_config(**overrides) -> DeskConfig:
    """A market with no FX movement and no client flow, for exact accounting."""
    base = dict(
        fx_drift=(0.0, 0.0, 0.0),
        fx_vol=(0.0, 0.0, 0.0),
        jump_prob=0.0,
        order_prob=0.0,
        flow_ou_sigma=0.0,
        drawdown_coef=0.0,
    )
    base.update(overrides)
    return DeskConfig(**base)


def remark_book(e: LiquidityDeskEnv) -> None:
    """
    Re-baseline the book after editing balances by hand.

    A test that reaches into `e.balances` changes equity behind the
    environment's back. Without this, the next step charges the edit itself to
    the agent as a mark-to-market loss, and can trip the insolvency check.
    """
    equity = e._equity()
    e._equity_prev = equity
    e._high_water = equity
    e._initial_equity = equity


@pytest.fixture
def env() -> LiquidityDeskEnv:
    return LiquidityDeskEnv(DeskConfig())


# --------------------------------------------------------------------------- #
# Gymnasium API compliance
# --------------------------------------------------------------------------- #

def test_passes_gymnasium_env_checker():
    check_env(LiquidityDeskEnv(), skip_render_check=True)


def test_spaces_are_as_documented(env):
    assert env.action_space.n == N_ACTIONS == 13
    assert env.observation_space.shape == (OBS_DIM,) == (35,)
    assert len(ACTION_NAMES) == N_ACTIONS


def test_observation_always_inside_declared_space(env):
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs), "reset obs outside space"
    for _ in range(600):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs), "step obs outside space"
        assert np.all(np.isfinite(obs)), "non-finite observation"
        if term or trunc:
            obs, _ = env.reset()


def test_no_nan_in_reward_or_info(env):
    env.reset(seed=3)
    for _ in range(400):
        _, reward, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(reward)
        assert np.isfinite(info["equity_usd"])
        if term or trunc:
            break


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

def test_same_seed_gives_identical_trajectory():
    actions = [i % N_ACTIONS for i in range(300)]

    def rollout(seed: int):
        e = LiquidityDeskEnv()
        obs, _ = e.reset(seed=seed)
        trace = [obs.copy()]
        rewards = []
        for a in actions:
            obs, r, term, trunc, _ = e.step(a)
            trace.append(obs.copy())
            rewards.append(r)
            if term or trunc:
                break
        return np.array(trace), np.array(rewards)

    o1, r1 = rollout(42)
    o2, r2 = rollout(42)
    o3, r3 = rollout(43)

    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(r1, r2)
    assert not np.array_equal(r1, r3), "different seeds produced identical rewards"


# --------------------------------------------------------------------------- #
# Accounting invariants
# --------------------------------------------------------------------------- #

def test_hold_in_frozen_market_conserves_equity_exactly():
    e = LiquidityDeskEnv(frozen_market_config())
    e.reset(seed=1)
    start = e._equity()
    for _ in range(200):
        _, reward, _, _, info = e.step(0)  # HOLD
        assert info["equity_usd"] == pytest.approx(start, rel=1e-12)
        assert reward == pytest.approx(0.0, abs=1e-12)
    assert e._equity() == pytest.approx(start, rel=1e-12)


def test_rebalance_costs_exactly_fee_plus_slippage_plus_gas():
    """A rebalance moves rebalance_frac of the source pool, at a known cost."""
    cfg = frozen_market_config()
    e = LiquidityDeskEnv(cfg)
    e.reset(seed=2)
    e.balances[0] = 40_000.0 * e.rates[0]   # naira pool rich
    e.balances[HUB] = 10_000.0              # rail poor
    remark_book(e)

    before = e._pool_usd().copy()
    equity_before = e._equity()

    _, _, _, _, info = e.step(1)  # NGN -> USDT

    amount = cfg.rebalance_frac * before[0]
    assert amount == pytest.approx(3_200.0)

    slip = cfg.slippage_coef * cfg.rebalance_frac
    expected = amount * (cfg.rebalance_fee_bps / 1e4 + slip) + cfg.gas_cost_usd

    assert info["cost_usd"] == pytest.approx(expected, rel=1e-9)
    assert e._equity() == pytest.approx(equity_before - expected, rel=1e-9)
    assert info["cost_usd"] < 10.0, "a single correction must stay cheap"


def test_repeated_rebalances_build_a_directional_position():
    """
    The agent must be able to choose an allocation, not just restore balance.

    Without this, FX exposure is pinned near 75% fiat whatever the policy does,
    devaluations become unavoidable noise, and there is nothing to learn.
    """
    e = LiquidityDeskEnv(frozen_market_config())
    e.reset(seed=5)

    start_fiat_share = e._pool_usd()[:N_FIAT].sum() / e._equity()
    for _ in range(60):
        for action in (1, 3, 5):      # rotate all three fiat pools into the rail
            e.step(action)

    end_fiat_share = e._pool_usd()[:N_FIAT].sum() / e._equity()
    assert end_fiat_share < 0.15, (
        f"could not de-risk: fiat share only fell "
        f"{start_fiat_share:.2f} -> {end_fiat_share:.2f}"
    )


def test_rebalance_moves_value_in_the_named_direction():
    e = LiquidityDeskEnv(frozen_market_config())
    e.reset(seed=4)
    e.balances[0] = 40_000.0 * e.rates[0]
    e.balances[HUB] = 10_000.0
    remark_book(e)

    before = e._pool_usd().copy()
    e.step(1)  # NGN -> USDT
    after = e._pool_usd()
    assert after[0] < before[0], "source pool did not fall"
    assert after[HUB] > before[HUB], "hub pool did not rise"


def test_rebalance_from_an_empty_pool_is_a_no_op():
    cfg = frozen_market_config()
    e = LiquidityDeskEnv(cfg)
    e.reset(seed=3)
    e.balances[0] = 0.0
    remark_book(e)

    before = e._pool_usd().copy()
    _, _, _, _, info = e.step(1)  # NGN -> USDT, but there is no naira

    assert info["cost_usd"] == 0.0
    np.testing.assert_allclose(e._pool_usd(), before, rtol=1e-12)


def test_spread_actions_respect_configured_bounds():
    cfg = DeskConfig()
    e = LiquidityDeskEnv(cfg)
    e.reset(seed=5)
    for _ in range(30):
        e.step(7)  # WIDEN
    assert e.spread == pytest.approx(cfg.spread_max)
    for _ in range(30):
        e.step(8)  # TIGHTEN
    assert e.spread == pytest.approx(cfg.spread_min)


def test_filled_order_earns_exactly_the_spread():
    cfg = frozen_market_config(order_prob=1.0, spread_init=0.01)
    e = LiquidityDeskEnv(cfg)
    e.reset(seed=6)

    fills = 0
    for _ in range(300):
        size = e._pending["size_usd"] if e._pending else 0.0
        equity_before = e._equity()
        _, _, term, trunc, info = e.step(0)  # HOLD, just quote
        if info["outcome"] == "filled":
            gain = e._equity() - equity_before
            assert gain == pytest.approx(size * cfg.spread_init, rel=1e-9)
            fills += 1
        if term or trunc:
            break
    assert fills > 50, f"expected plenty of fills in a frozen market, got {fills}"


# --------------------------------------------------------------------------- #
# Failure modes the agent must learn to avoid
# --------------------------------------------------------------------------- #

def test_empty_hub_causes_stockout_not_negative_balance():
    cfg = frozen_market_config(order_prob=1.0, spread_init=0.005)
    e = LiquidityDeskEnv(cfg)
    e.reset(seed=7)
    e.balances[HUB] = 0.0  # rail completely drained
    remark_book(e)

    saw_stockout = False
    for _ in range(300):
        # force a fiat-in order, which must be paid out in USDT
        e._pending = {"corridor": 0, "direction": 1, "size_usd": 1000.0}
        _, _, term, trunc, info = e.step(0)
        assert np.all(e.balances >= -1e-9), "balances went negative"
        if info["outcome"] == "stockout":
            saw_stockout = True
            break
        if term or trunc:
            break
    assert saw_stockout, "drained hub never produced a stockout"


def test_stockout_is_penalised_relative_to_a_quiet_step():
    cfg = frozen_market_config(order_prob=1.0, spread_init=0.005)
    e = LiquidityDeskEnv(cfg)
    e.reset(seed=8)
    e.balances[HUB] = 0.0
    remark_book(e)

    for _ in range(300):
        e._pending = {"corridor": 0, "direction": 1, "size_usd": 2000.0}
        _, reward, term, trunc, info = e.step(0)
        if info["outcome"] == "stockout":
            expected = -(cfg.stockout_coef * 2000.0) / cfg.reward_scale
            assert reward == pytest.approx(expected, rel=1e-9)
            return
        if term or trunc:
            break
    pytest.fail("no stockout observed")


def test_paused_corridor_declines_without_stockout_penalty():
    cfg = frozen_market_config(order_prob=1.0)
    e = LiquidityDeskEnv(cfg)
    e.reset(seed=9)

    # the pause actions name their corridor, so this is deterministic
    e.step(9)  # PAUSE_NGN
    assert e._paused[0] > 0
    assert e._paused[1] == 0 and e._paused[2] == 0

    e._pending = {"corridor": 0, "direction": 1, "size_usd": 1500.0}
    _, _, _, _, info = e.step(0)
    assert info["outcome"] == "declined"
    assert info["cum_unfilled_usd"] == 0.0


def test_each_pause_action_targets_its_own_corridor():
    e = LiquidityDeskEnv(DeskConfig())
    for corridor, action in enumerate((9, 10, 11)):
        e.reset(seed=100 + corridor)
        e.step(action)
        assert e._paused[corridor] > 0, f"action {action} did not pause {corridor}"
        assert e._paused.astype(bool).sum() == 1, "paused more than one corridor"


def test_resume_all_clears_every_pause():
    e = LiquidityDeskEnv(DeskConfig())
    e.reset(seed=10)
    e.step(9)
    assert e._paused.sum() > 0
    e.step(12)  # RESUME_ALL
    assert e._paused.sum() == 0


# --------------------------------------------------------------------------- #
# Terminal conditions
# --------------------------------------------------------------------------- #

def test_truncates_at_episode_length_without_terminating():
    cfg = frozen_market_config(episode_len=120)
    e = LiquidityDeskEnv(cfg)
    e.reset(seed=11)
    for step in range(1, 200):
        _, _, term, trunc, _ = e.step(0)
        if step < cfg.episode_len:
            assert not (term or trunc), f"ended early at step {step}"
        else:
            assert trunc and not term, "end of week should truncate, not terminate"
            assert e.t == cfg.episode_len
            break


def test_insolvency_terminates_and_is_penalised():
    # a brutal market: heavy drift against every fiat pool, no client revenue
    cfg = DeskConfig(
        episode_len=5000,
        fx_drift=(0.02, 0.02, 0.02),
        fx_vol=(1e-12, 1e-12, 1e-12),
        jump_prob=0.0,
        order_prob=0.0,
        insolvency_frac=0.90,
    )
    e = LiquidityDeskEnv(cfg)
    e.reset(seed=12)
    for _ in range(5000):
        _, reward, term, trunc, info = e.step(0)
        if term:
            assert info["equity_ratio"] < cfg.insolvency_frac
            assert reward < -cfg.insolvency_penalty / 2, "insolvency barely penalised"
            return
        assert not trunc, "should have gone insolvent before the week ended"
    pytest.fail("never reached insolvency under a brutal drift")


def test_terminated_and_truncated_are_never_both_true():
    e = LiquidityDeskEnv(DeskConfig(episode_len=200))
    for seed in range(8):
        e.reset(seed=seed)
        for _ in range(250):
            _, _, term, trunc, _ = e.step(e.action_space.sample())
            assert not (term and trunc)
            if term or trunc:
                break


# --------------------------------------------------------------------------- #
# Dynamics sanity
# --------------------------------------------------------------------------- #

def test_fx_rates_move_and_volatility_clusters():
    e = LiquidityDeskEnv(DeskConfig())
    e.reset(seed=13)
    rates = []
    for _ in range(400):
        e.step(0)
        rates.append(e.rates[:3].copy())
    rates = np.array(rates)
    assert np.std(rates[:, 0]) > 0, "NGN rate never moved"

    rets = np.diff(np.log(rates[:, 0]))
    sq = rets ** 2
    # GARCH implies squared returns are positively autocorrelated
    autocorr = np.corrcoef(sq[:-1], sq[1:])[0, 1]
    assert autocorr > 0.0, f"no volatility clustering (autocorr={autocorr:.3f})"


def test_render_state_is_json_serialisable():
    import json

    e = LiquidityDeskEnv(DeskConfig(), render_mode="state")
    e.reset(seed=14)
    for _ in range(40):
        e.step(e.action_space.sample())
    blob = json.dumps(e.render_state())
    assert len(blob) > 200
    state = json.loads(blob)
    assert state["currencies"] == ["NGN", "RWF", "KES", "USDT"]
    assert len(state["pool_fill"]) == 4
    assert 0.0 <= min(state["pool_fill"]) <= max(state["pool_fill"]) <= 1.0


def test_pressure_builds_then_releases_on_devaluation():
    """
    The devaluation hazard must be foreseeable in probability, never in timing.

    A memoryless Poisson jump is unpredictable by construction, which leaves
    de-risking with no edge and the four-algorithm comparison with nothing to
    separate. Pressure is the leading indicator that fixes that.
    """
    cfg = DeskConfig()
    e = LiquidityDeskEnv(cfg)
    e.reset(seed=77)

    saw_release = False
    peak_before_release = 0.0
    prev = e._pressure.copy()

    for _ in range(4000):
        e.step(0)
        hazard = e.devaluation_hazard()
        assert np.all(hazard >= cfg.devaluation_base_prob - 1e-12)
        assert np.all(np.isfinite(hazard))

        broke = e._devaluation_flash
        if broke.any():
            i = int(np.argmax(broke))
            peak_before_release = max(peak_before_release, float(prev[i]))
            assert e._pressure[i] == 0.0, "pressure did not release on the break"
            saw_release = True
        prev = e._pressure.copy()

        if e.t >= cfg.episode_len:
            e.reset()

    assert saw_release, "no devaluation observed in 4000 steps"
    assert peak_before_release > 0.0


def test_hazard_rises_with_pressure():
    e = LiquidityDeskEnv(DeskConfig())
    e.reset(seed=78)
    e._pressure[:] = 0.0
    low = e.devaluation_hazard().copy()
    e._pressure[:] = 1.2
    high = e.devaluation_hazard()
    assert np.all(high > low * 5.0), "pressure barely moves the hazard"


def test_pressure_is_observable():
    """The signal is useless to the agent unless it reaches the observation."""
    e = LiquidityDeskEnv(DeskConfig())
    obs, _ = e.reset(seed=79)
    e._pressure[:] = [0.4, 1.1, 0.2]
    obs = e._observe()
    np.testing.assert_allclose(obs[32:35], [0.4, 1.1, 0.2], rtol=1e-5)


def test_config_rejects_nonstationary_garch():
    with pytest.raises(AssertionError):
        DeskConfig(vol_persistence=0.95, vol_shock=0.10)


# --------------------------------------------------------------------------- #
# Performance: 48 hyperparameter runs have to be tractable
# --------------------------------------------------------------------------- #

# This is a regression guard, not a benchmark. An idle machine reaches several
# thousand steps/sec; the bar sits far below that so the suite stays green on a
# loaded or slower host, while still failing loudly if someone introduces an
# accidental O(n^2) in the step function. The measured figure is printed either
# way, and the README quotes a measurement rather than this threshold.
MIN_STEPS_PER_SECOND = 1_000


def test_steps_fast_enough_for_the_full_study():
    e = LiquidityDeskEnv(DeskConfig())
    e.reset(seed=15)
    actions = np.random.default_rng(0).integers(0, N_ACTIONS, size=20_000)

    start = time.perf_counter()
    for a in actions:
        _, _, term, trunc, _ = e.step(int(a))
        if term or trunc:
            e.reset()
    elapsed = time.perf_counter() - start

    fps = len(actions) / elapsed
    print(f"\nenvironment throughput: {fps:,.0f} steps/sec")
    assert fps > MIN_STEPS_PER_SECOND, (
        f"environment throughput collapsed to {fps:,.0f} steps/sec "
        f"(floor {MIN_STEPS_PER_SECOND:,}). This is far below anything a "
        f"busy machine explains -- suspect an algorithmic regression in step()."
    )

# --------------------------------------------------------------------------- #
# Reproducibility: a reported score must not move between runs
# --------------------------------------------------------------------------- #

def _score_episode(policy, seed: int) -> float:
    """Mirror of main.rollout, reduced to the return."""
    e = LiquidityDeskEnv(DeskConfig())
    reseed = getattr(policy, "reseed", None)
    if callable(reseed):
        reseed(seed)
    obs, _ = e.reset(seed=seed)
    total = 0.0
    while True:
        obs, reward, term, trunc, _ = e.step(int(policy(obs)))
        total += reward
        if term or trunc:
            return total


def test_stochastic_baseline_is_reproducible_across_fresh_instances():
    """
    Two independently constructed random policies must score identically on
    the same seed. Without the reseed hook each would draw from OS entropy and
    the scoreboard's random row would move between runs -- which is how the
    inconsistency this test guards against was originally found.
    """
    from training.baselines import make_baseline

    for seed in (9000, 9001, 9137):
        a = _score_episode(make_baseline("random"), seed)
        b = _score_episode(make_baseline("random"), seed)
        assert a == pytest.approx(b, abs=1e-9), (
            f"random baseline scored {a} then {b} on seed {seed}"
        )


def test_deterministic_baselines_are_reproducible():
    from training.baselines import make_baseline

    for name in ("hold", "heuristic"):
        a = _score_episode(make_baseline(name), 9000)
        b = _score_episode(make_baseline(name), 9000)
        assert a == pytest.approx(b, abs=1e-9), f"{name} is not deterministic"


def test_different_seeds_give_different_episodes():
    """Guards the opposite failure: a reseed that ignores its argument."""
    from training.baselines import make_baseline

    scores = {_score_episode(make_baseline("random"), s) for s in range(9000, 9006)}
    assert len(scores) > 1, "every seed produced an identical episode"
