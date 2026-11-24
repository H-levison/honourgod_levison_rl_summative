# Cross-Border Liquidity Desk — RL Summative

A custom reinforcement learning environment and a four-algorithm comparison
study, built with [Gymnasium](https://gymnasium.farama.org/),
[TorchRL](https://pytorch.org/rl/) and [Three.js](https://threejs.org/).

An RL agent runs the treasury of a cross-border crypto-fiat desk. It holds
inventory in three African currencies (NGN, RWF, KES) plus a USDT settlement
rail, and must keep every corridor fillable through a week of client order flow
while FX moves against the book.

The tension that defines the problem: **earning money requires holding fiat
inventory, and holding fiat inventory is exactly what loses money when a
currency devalues.**

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). No manual virtual environment, no
`pip install`.

```bash
git clone <this-repo>
cd honourgod_levison_rl_summative

uv sync                 # resolves and installs everything, including Python 3.12
uv run play.py          # 3D scene, best available agent
```

`uv run play.py` works immediately after cloning. The best checkpoint per
algorithm ships with the repository, so this shows a **trained** agent, not a
placeholder. If the checkpoints are removed it falls back to the rule-based
desk operator rather than failing, so the scene is always demonstrable.

Three.js is **vendored** under `assets/vendor/` (pinned to r169), so the scene
loads with no network access and no Node.js toolchain.

### Commands

```bash
uv run play.py                            watch the best agent
uv run play.py --policy ppo --fps 2.5     a specific algorithm, slower
uv run main.py evaluate --actions         headless scoreboard + action histogram
uv run main.py train --algo dqn           one training run
uv run main.py sweep --algo ppo           that algorithm's 12-run grid
uv run python -m training.plots           build the report figures
uv run python -m training.tables          build the report tables
uv run python training/hparams.py         print the full hyperparameter grid
uv run pytest                             test suite
```

---

## Why reinforcement learning

The criterion is not "is this hard" but "does this need RL". Four properties
say it does.

**The decision is sequential and state is persistent.** Inventory carried into
step *t* is the inventory the desk settles with at *t+1*. A rebalance is not a
prediction, it is a change to the state that every later step inherits.

**Payoffs are delayed and their sign depends on the future.** A rebalance costs
a fee now to avoid a stockout tens of steps later, or to cut exposure before a
devaluation that may not come. There is no label for "the correct action here",
so there is nothing for a supervised model to be trained against — the quality
of an action is only revealed by what follows it.

**The optimal action depends on a latent regime.** Devaluation pressure is not
directly observable; the agent sees a noisy proxy and must infer how much
insurance to buy. That is a decision under partial observability, not a
classification.

**A fixed rule leaves measurable value on the table.** The repository ships
three reference policies so this is an empirical claim rather than an assertion.
Over 200 held-out seeds, paired by seed:

| Policy | Mean return | vs. `hold` (paired) |
|---|---|---|
| `random` | −155.9 | −41.85 ± 15.79, t = −2.65 |
| `hold` | −114.0 | — |
| `heuristic` | −80.5 | **+33.48 ± 12.27, t = +2.73** |

The comparison is paired because the dominant variance term — the FX path — is
common to every policy on a given seed and controllable by none of them. The
unpaired standard deviations (±137 to ±173) are almost entirely that shared
market noise and badly understate the real separation between policies.

And the learners clear that bar. On the same 200 seeds, three of the four beat
the hand-written operator, and the best of them is the only policy that ends
the week with more equity than it started:

| Policy | Mean return | Equity | Max drawdown | Fees $ | vs. `hold` (paired) |
|---|---:|---:|---:|---:|---|
| **PPO** `p06-lambda-low` | **−5.6** | **+0.64%** | **2.47%** | 365 | **+108.44 ± 13.81, t = +7.85** |
| REINFORCE `r01-baseline` | −46.3 | −1.72% | 5.03% | 145 | +67.68 ± 12.38, t = +5.47 |
| A2C `a06-critic-weak` | −58.1 | −2.30% | 5.80% | 160 | +55.92 ± 11.02, t = +5.08 |
| `heuristic` | −80.5 | −3.79% | 6.07% | 2,085 | +33.48 ± 12.27, t = +2.73 |
| DQN `d09-batch-large` | −92.3 | −3.48% | 5.80% | 715 | +21.71 ± 14.00, t = +1.55 (n.s.) |

The fee column is where the learned policies separate from the written one. The
heuristic prevents every stockout and pays $2,085 to do it; PPO tolerates 16.5
stockouts a week, pays $365, and finishes 75 points ahead. Nothing in the reward
function names that trade-off — it is the trade-off the agent had to discover,
and it is exactly what a desk risk manager is paid to judge. PPO also runs at
**half the drawdown** of every other policy, so it wins on risk control rather
than on a lucky run of seeds.

Full tables, including all 48 configurations, are generated into
`report/tables.md`.

---

## The environment

`environment/custom_env.py` — `LiquidityDeskEnv`, registered as
`LiquidityDesk-v0`.

### Action space — `Discrete(13)`

Every action maps to something a real desk operator does.

| # | Action | Real-world meaning |
|---|--------|--------------------|
| 0 | `HOLD` | Keep quoting, move no inventory |
| 1 | `NGN_TO_USDT` | Sell down naira inventory into the stablecoin rail |
| 2 | `USDT_TO_NGN` | Fund the naira pool from the rail |
| 3 | `RWF_TO_USDT` | Sell down Rwandan franc inventory |
| 4 | `USDT_TO_RWF` | Fund the franc pool |
| 5 | `KES_TO_USDT` | Sell down Kenyan shilling inventory |
| 6 | `USDT_TO_KES` | Fund the shilling pool |
| 7 | `WIDEN_SPREAD` | Charge more per fill, lose price-sensitive clients |
| 8 | `TIGHTEN_SPREAD` | Charge less, win more flow |
| 9 | `PAUSE_NGN` | Stop quoting the naira corridor |
| 10 | `PAUSE_RWF` | Stop quoting the franc corridor |
| 11 | `PAUSE_KES` | Stop quoting the shilling corridor |
| 12 | `RESUME_ALL` | Reopen every paused corridor |

All rebalances route through USDT, which is how the corridor actually settles.
A rebalance moves **8%** of the source pool and costs a 15 bps fee, slippage
proportional to trade size, and a fixed gas cost. The fraction is deliberately
small: a single correction stays cheap, while repeated moves accumulate into a
real directional position, so the agent can de-risk into the rail ahead of a
devaluation at the cost of the fiat it needs to settle payouts.

The pause actions **name their corridor** rather than offering one "pause the
riskiest" action. An aggregate would be resolved by a formula inside the
environment that the agent cannot observe, so it could request a pause and
silently get a different corridor than the one it was reacting to.

One consequence worth stating plainly, because it changes how the results read:
`RESUME_ALL` clears every pause flag, so **when nothing is paused it is a
second no-op**, identical in effect to `HOLD`. Three of the four trained agents
settled on it rather than on `HOLD`, which inflates that column of the action
histogram. The evaluation therefore reports a conservative *idle* floor,
`HOLD + max(0, resume - pause)`, since a resume can only act if a pause
preceded it. Without that correction REINFORCE looks like it spends its week
managing corridors when at least 60% of its steps change nothing.

### Observation space — `Box(-10, 10, (35,))`, `float32`

Every dimension, with its encoding, where the value comes from, how often it
updates, and how it reaches the agent. All observations are assembled by
`LiquidityDeskEnv._observe()` once per `step()` and returned as a single
`float32` vector; the agent consumes it directly as network input, with no
external feed, buffering or frame stacking.

| Idx | Meaning | Encoding / range | Source or simulation rule | Cadence |
|---|---|---|---|---|
| 0–3 | Pool value share (NGN, RWF, KES, USDT) | fraction of equity, `[0,1]`, sums to 1 | `balance_i / rate_i / equity` | every step |
| 4–7 | Pool value vs. a balanced book | `pool/target − 1`, clipped `[−5,5]` | target = equity / 4 | every step |
| 8–10 | FX drift since the week opened | `log(rate/rate₀) × 10` | GARCH path vs. episode-open spot | every step |
| 11–13 | FX momentum | EWMA(λ=0.1) of log-return × 500 | smoothed per-step FX return | every step |
| 14–16 | Volatility regime | `log1p(var/var_base − 1)`, `≈[0,4]` | GARCH(1,1), α=0.07, β=0.90 | every step |
| 17 | Quoted spread | `(spread − 1.5%)/1.5%`, `[−0.67,1.0]` | desk control, set by actions 7/8 | on action |
| 18–20 | Directional order pressure | OU process, clipped `[−2.5,2.5]` | θ=0.05, σ=0.30 per corridor | every step |
| 21 | Pending order size | `size/equity × 20`, clipped `[0,10]` | `N(900, 420)` USD, clipped | on arrival |
| 22 | Pending order direction | `+1` fiat in, `−1` fiat out, `0` none | logit of flow, momentum, pressure | on arrival |
| 23–25 | Pending order corridor | one-hot over NGN/RWF/KES | uniform over corridors | on arrival |
| 26–28 | Corridor paused flags | `{0,1}` | pause timer, 20-step auto-resume | every step |
| 29 | Equity ratio | `equity/equity₀ − 1` | mark-to-market of all four pools | every step |
| 30 | Drawdown | `(high_water − equity)/high_water`, `≥0` | running high-water mark | every step |
| 31 | Episode remaining | `1 − t/500`, `[0,1]` | step counter | every step |
| **32–34** | **Devaluation pressure** | `[0, 2.5]` | **latent hazard driver — see below** | every step |

The desk sees the incoming quote request (21–25) **before** it acts, as a real
desk sees an RFQ before responding.

Indices 8–20 and 32–34 are per-corridor triples ordered **NGN, RWF, KES**.

### Devaluation pressure — the leading indicator

Indices 32–34 are what make this an RL problem rather than a fee-management
exercise.

Managed African currencies do not drift smoothly; they hold a level and then
gap. The naira moved roughly 40% overnight in June 2023 and again in February
2024. Modelling that as a memoryless Poisson jump makes it **unpredictable by
construction** — and if the event cannot be anticipated, de-risking has no edge:
the agent pays fees to cut exposure and earns nothing in expectation.

So each corridor instead carries a latent pressure that builds under sustained
one-way dollar demand and persistent depreciation, with the hazard rising
exponentially in it:

```
pressure <- 0.985*pressure + 0.012*max(flow,0) + 0.035*max(momentum,0) + noise
hazard    = 6e-5 * exp(3.2 * pressure)
```

On a break the peg releases and a 30-step refractory window stops the jump's
own depreciation from instantly rebuilding the pressure behind it.

This makes a devaluation **foreseeable in probability, never in timing** — a
genuine insurance decision. Adverse selection scales with pressure too, so the
desk is handed the most naira precisely when naira is most likely to gap.
Measured: **1.14 devaluations per episode, in 67% of weeks.**

### Reward

```
r_t  =  d(equity) / 50
      - 0.05 * unfilled_USD / 50          reputational cost of a stockout
      - 2.0  * drawdown^2                 risk penalty, quadratic
      - 20.0                              on insolvency
      + 5.0 * (final equity ratio - 1)    on surviving the week
```

`d(equity)` already contains spread revenue, rebalance fees and FX
mark-to-market, so none of those are double counted. Four forces pull against
each other: fee drag, stockout risk, FX exposure and drawdown control.

The stockout coefficient sits **above** the spread deliberately: a P2P desk
that cannot settle loses the counterparty, not just the fee.

### Start state

Pools hold equal USD value (25% each of $100,000), perturbed ±15%. Spot rates
are seeded near real levels (₦1600, 1300 RWF, 130 KES per USD) and perturbed
±2%. Spread opens at 1.5%, no corridor paused, order-flow pressure randomised —
so a policy must generalise rather than memorise one week.

### Terminal conditions

- **`terminated`** — insolvency: equity falls below 85% of opening equity.
- **`truncated`** — the trading week ends at step 500.

Kept distinct, so bootstrapping is correct at the time limit.

---

## Visualisation and integration

`environment/rendering.py` + `assets/scene.html` + `assets/js/scene.js`

A dark treasury vault in Three.js. Four glass columns hold the inventory pools
and **liquid level is the pool balance**, so the policy is legible without
reading a number. An **operator drone** embodies the agent: it flies to the
column it has decided to act on and fires a working beam, climbs to the spread
gauge to re-quote, and drops to a corridor gate to seal it. A pipe manifold
pulses on each rebalance, an FX wall plots each currency against USD, three
gauges show devaluation pressure, and a klaxon fires on a regime break.

### Serialisation contract

Training never imports the renderer. The environment is pure NumPy and the
visualisation attaches only at demo time, receiving one JSON state frame per
step over a websocket:

```
LiquidityDeskEnv.render_state()   ->  DeskRenderer.push()
                                           |  websocket, port 8766
                                      assets/scene.html
                                           ^  http, port 8765
```

`render_state()` is the public interface — a flat JSON document, no Python
types, consumable by any frontend. Its fields are pinned by
`tests/test_rendering.py::REQUIRED_KEYS`, so renaming one fails a test rather
than silently blanking the scene.

| Field | Type | Meaning |
|---|---|---|
| `pools_usd`, `pool_fill` | `float[4]` | pool value; `pool_fill` normalised for column height |
| `rates`, `rate_change` | `float[3]` | spot and change since open |
| `pressure`, `hazard` | `float[3]` | devaluation pressure and per-step probability |
| `vol_regime`, `flow` | `float[3]` | volatility multiple, directional pressure |
| `paused`, `devaluation` | `bool[3]` | corridor state, regime-break flags |
| `action`, `action_name`, `reward` | `int`, `str`, `float` | what the agent just did |
| `rebalance` | `object?` | `{src, dst, amount_usd}` when one fired |
| `pending_order` | `object?` | `{corridor, direction, size_usd}` |
| `equity_usd`, `drawdown`, `stats` | `float`, `object` | book state and running counters |

Frames are dropped rather than queued if the browser cannot keep up, so a slow
viewer can never stall the simulation. The browser also reports runtime errors
back over the same socket, so a WebGL failure appears in the terminal instead
of silently rendering a black canvas.

---

## Algorithms

All four interact with the identical environment, action space and reward, and
share an actor, critic, optimiser and evaluation loop — so a difference in
results is attributable to the objective, not to incidental tuning.

| Family | Algorithm | Why it is interesting here |
|--------|-----------|----------------------------|
| Value based | **DQN** | Bootstrapping propagates delayed stockout cost well, but uncontrollable FX noise lands straight in the TD target |
| Policy gradient | **REINFORCE** | Monte Carlo returns over 500 steps give the highest gradient variance in the study |
| Policy gradient | **PPO** | The clipped trust region should be most stable on a noisy, non-stationary reward |
| Actor critic | **A2C** | Short bootstrapped rollouts, no trust region — the midpoint between the two above |

Reference policies (`training/baselines.py`): `random`, `hold` (what FX drift
alone costs) and `heuristic` (a reasonable human operator). Each reads the same
35-dim observation, so no baseline has information the learners lack.

---

## Hyperparameter study

`training/hparams.py` defines **12 configurations per algorithm, 48 runs
total**, exceeding the required 10 per algorithm.

The grids are not random jitter. This environment's defining difficulty is
delayed credit assignment, so the parameters controlling how far reward
propagates backwards are varied deliberately across every family: discount
factor, bootstrap length, GAE lambda, and baseline choice. Each configuration
records the hypothesis it tests.

Two are controls rather than candidates. `r02-no-baseline` keeps the identical
critic but zeroes and freezes it, so GAE with λ=1 returns the raw discounted
return — textbook unbaselined REINFORCE with every other moving part held
constant. `p07-lambda-one` pushes PPO's GAE to the full Monte Carlo advantage,
bridging directly to the REINFORCE family.

### Compute

Runs execute as separate processes. Measured on this project:

| | CPU | GPU (RTX 3060) |
|---|---|---|
| Gradient steps/sec | 398 | 555 (1.40×) |
| Per DQN iteration | 10.1 ms gradients | 7.2 ms gradients |
| Environment stepping | 41.2 ms | 41.2 ms |

A 35→256→256→13 MLP is too small to saturate a GPU, and the NumPy environment
costs four times what the gradients do — so the accelerator is worth only ~6%
end to end, while six single-threaded worker processes on 16 cores is close to
a six-fold speedup. **The sweep therefore runs on CPU with parallelism across
runs.** A `cu124` extra is available (`uv sync --extra cu124`) but is not the
default, which also keeps a fresh clone small.

---

## Project layout

```
pyproject.toml            uv project definition
uv.lock                   locked dependency set
.python-version           pinned to 3.12 for wheel availability
play.py                   watch the best trained agent
main.py                   demo, evaluate, train, sweep

environment/
  custom_env.py           LiquidityDeskEnv  (pure NumPy, no render imports)
  rendering.py            http + websocket bridge to the Three.js scene
  __init__.py             Gym registration as LiquidityDesk-v0

training/
  hparams.py              48 documented hyperparameter configurations
  baselines.py            random / hold / heuristic reference policies
  common.py               networks, logging schema, evaluation, checkpoints
  dqn_training.py         value-based trainer
  pg_training.py          REINFORCE, PPO and A2C trainers
  runner.py               single-run and parallel sweep orchestration
  plots.py                the report figures
  tables.py               the report tables

models/dqn/, models/pg/   best checkpoint per algorithm (ships with the repo)
logs/runs/                per-run metrics CSV + metadata, 48 runs
logs/sweeps/              per-algorithm grid summaries
logs/final_evaluation.csv held-out scoreboard, 200 shared seeds
report/figures/           the six generated figures
report/tables.md          the generated tables
assets/                   Three.js scene; vendor/ is vendored r169 (offline)
tests/                    environment and rendering test suites
```

---

## Tests

```bash
uv run pytest -v
```

Environment: Gymnasium API compliance, seeded determinism, exact accounting
invariants (equity conserved to floating-point error under a no-op in a frozen
market; a rebalance costs exactly fee + slippage + gas), the failure modes the
agent must learn to avoid, `terminated` vs `truncated` correctness, volatility
clustering, pressure building and releasing, and throughput.

Reproducibility: every policy, including the stochastic baseline, is re-keyed
from the episode seed, so a reported score does not move between runs. The
paired comparison depends on this.

Rendering: static assets served, vendored Three.js complete, no remote
dependencies, websocket frames carrying every field the scene reads, frame
ordering, and port handling.

---

## Author

HonourGod Levison — African Leadership University
