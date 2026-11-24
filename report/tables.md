# Generated tables

Produced by `uv run python -m training.tables` from the CSVs in `logs/`. Do not edit by hand -- regenerate instead, so the report and its evidence cannot drift apart.


## Headline: best configuration per algorithm

| Algorithm | Best configuration | Sweep score (40 ep) | Held-out score (200 ep) | vs `hold` |
|---|---|---:|---:|---:|
| DQN | `d09-batch-large` | -54.97 | -85.51 | +31.30 (t = +2.20) |
| REINFORCE | `r01-baseline` | -13.76 | -35.40 | +81.41 (t = +6.83) |
| PPO | `p06-lambda-low` | +27.39 | +0.33 | +117.14 (t = +9.69) |
| A2C | `a06-critic-weak` | -30.36 | -43.89 | +72.92 (t = +6.69) |

The two score columns differ because the sweep ranks configurations on 40 episodes while the held-out column re-scores the winner on 200. The gap is a measurement of evaluation noise, not a discrepancy: 40 episodes is enough to rank, not enough to report.


## Held-out evaluation

All policies on the identical seed set 20000..20199 (n = 200).

| Policy | Mean return | Equity | Max DD | Stockouts | Fees $ | Revenue $ | Idle | vs `hold` (paired) | t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | +0.33 | +0.86% | 2.47% | 14.9 | 357 | 4,870 | 41% | +117.14 ± 12.09 | **+9.69** |
| REINFORCE | -35.40 | -1.24% | 4.43% | 7.3 | 172 | 4,973 | 58% | +81.41 ± 11.92 | **+6.83** |
| A2C | -43.89 | -1.74% | 4.92% | 5.8 | 153 | 5,008 | 44% | +72.92 ± 10.90 | **+6.69** |
| Heuristic | -69.10 | -3.28% | 5.73% | 0.0 | 2,085 | 5,664 | 24% | +47.71 ± 12.33 | **+3.87** |
| DQN | -85.51 | -3.16% | 5.56% | 18.3 | 722 | 4,112 | 10% | +31.30 ± 14.25 | **+2.20** |
| Hold | -116.81 | -4.14% | 7.46% | 27.2 | 0 | 4,560 | 100% | — | — |
| Random | -132.18 | -6.27% | 7.73% | 0.0 | 1,203 | 2,340 | 8% | -15.37 ± 14.94 | -1.03 |

Paired because the FX path is common to every policy on a given seed and controllable by none of them; differencing removes it. Bold *t* exceeds |2|, the significance bar used throughout.

*Idle* is a conservative floor on how often a policy does nothing. `RESUME_ALL` clears every pause flag, so with nothing paused it is a second no-op, identical in effect to `HOLD` -- and the policy-gradient agents settled on it rather than on `HOLD`. A resume can only act if a pause preceded it, so `idle >= HOLD + max(0, resume - pause)`. Read without this correction the action histogram suggests REINFORCE spends its week managing corridors, when most of its steps change nothing.

## Hyperparameter study


### DQN — 12 configurations

| Rank | Configuration | Score | Equity | Max DD | Stockouts | Insolvency | Hypothesis tested |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | **`d09-batch-large`** | **-54.97** | -1.59% | 4.58% | 19.5 | 10% | Lower-variance gradients per update, fewer updates. Tests the batch-size / update-count trade-off at fixed compute. |
| 2 | `d08-no-double` | -79.87 | -3.00% | 5.32% | 16.9 | 10% | Isolates Double-DQN's contribution. Expect optimistic value bias, visible as overconfident holding during devaluations. |
| 3 | `d11-explore-short` | -81.32 | -3.55% | 5.98% | 5.4 | 18% | Commits to greedy behaviour early. Risks locking in a do-nothing policy before the cost of stockouts is learned. |
| 4 | `d12-net-wide` | -83.69 | -3.80% | 5.72% | 4.8 | 10% | More capacity for the 35-dim observation, with the learning rate reduced to keep the larger network stable. |
| 5 | `d02-lr-high` | -91.98 | -3.74% | 6.14% | 13.1 | 18% | Faster value updates. Expect quicker early progress but noisier Q-estimates once FX jumps start appearing. |
| 6 | `d04-gamma-short` | -110.43 | -5.03% | 7.34% | 4.4 | 20% | Shortens the effective horizon to ~20 steps. Should under-value rebalances whose payoff is a stockout avoided much later. |
| 7 | `d05-gamma-long` | -140.71 | -6.17% | 9.25% | 8.2 | 32% | Horizon ~200 steps, close to the episode length. Better credit assignment, but higher variance in the bootstrap target. |
| 8 | `d06-buffer-small` | -146.79 | -6.04% | 9.07% | 17.8 | 32% | Replay holds roughly 40 episodes. Tests sensitivity to stale transitions drawn from a non-stationary FX process. |
| 9 | `d01-baseline` | -165.11 | -7.49% | 9.10% | 6.6 | 35% | Reference settings for the family. |
| 10 | `d03-lr-low` | -172.55 | -7.57% | 9.14% | 13.3 | 30% | Slow, stable bootstrapping. Tests whether d02's instability is really a learning-rate effect. |
| 11 | `d10-explore-long` | -211.60 | -7.23% | 9.35% | 58.0 | 30% | Exploration for 80% of training. Stockouts are rare events, so sustained exploration may be needed to ever see them. |
| 12 | `d07-target-hard` | -229.74 | -8.56% | 10.13% | 48.1 | 38% | Hard periodic target copy instead of Polyak averaging, the original DQN scheme. |

Spread between best and worst configuration: **174.8 points** (`d09-batch-large` -54.97 vs `d07-target-hard` -229.74).


### REINFORCE — 12 configurations

| Rank | Configuration | Score | Equity | Max DD | Stockouts | Insolvency | Hypothesis tested |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | **`r01-baseline`** | **-13.76** | -0.14% | 4.33% | 7.7 | 10% | Reference settings for the family. |
| 2 | `r09-entropy-zero` | -22.90 | -0.55% | 4.55% | 8.2 | 12% | No entropy bonus. Watch for premature collapse onto HOLD, the action with the lowest immediate cost. |
| 3 | `r07-batch-small` | -47.04 | -1.24% | 4.11% | 20.2 | 8% | Frequent but very noisy updates. Expect erratic learning curves. |
| 4 | `r04-lr-high` | -50.54 | -1.31% | 5.15% | 20.9 | 10% | Large steps on a high-variance gradient estimate. Tests whether the policy can survive its own update noise. |
| 5 | `r10-gamma-short` | -50.65 | -1.99% | 4.69% | 9.1 | 2% | Short horizon on a Monte Carlo estimator, which already sees the whole episode. Tests how much the discount alone matters. |
| 6 | `r06-batch-large` | -51.49 | -1.27% | 4.72% | 22.9 | 10% | Triples the episodes per gradient step. The cleanest available lever on Monte Carlo variance. |
| 7 | `r08-entropy-high` | -64.19 | -2.17% | 5.35% | 17.8 | 12% | Strong exploration pressure. Should keep the agent trying rebalances it would otherwise abandon early. |
| 8 | `r05-lr-low` | -65.56 | -2.87% | 5.87% | 3.8 | 15% | Small steps to average out variance, at the cost of sample efficiency over a fixed frame budget. |
| 9 | `r11-gamma-undiscounted` | -97.14 | -4.11% | 6.19% | 9.4 | 20% | No discounting at all. Correct for a fixed-length episode, but maximal return variance. |
| 10 | `r02-no-baseline` | -99.28 | -3.22% | 5.74% | 29.2 | 18% | Textbook REINFORCE. Expect the highest gradient variance of any run in the study -- this is the control that shows why baselines exist. |
| 11 | `r03-no-normalize` | -109.66 | -4.04% | 7.88% | 22.1 | 25% | Raw returns. Reward scale varies with FX regime, so this should destabilise the policy when volatility clusters. |
| 12 | `r12-net-small` | -165.64 | -4.66% | 6.82% | 67.1 | 15% | A smaller policy generalises more and overfits the FX noise less; paired with a higher learning rate. |

Spread between best and worst configuration: **151.9 points** (`r01-baseline` -13.76 vs `r12-net-small` -165.64).


### PPO — 12 configurations

| Rank | Configuration | Score | Equity | Max DD | Stockouts | Insolvency | Hypothesis tested |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | **`p06-lambda-low`** | **+27.39** | +2.09% | 1.39% | 14.0 | 0% | GAE closer to one-step TD: lower variance, more bias. Should blur credit for rebalances that pay off slowly. |
| 2 | `p04-epochs-few` | +3.09 | +0.93% | 3.61% | 12.9 | 10% | Less reuse per batch. Lower off-policy drift, worse sample efficiency. |
| 3 | `p09-batch-small` | +2.43 | +0.84% | 2.66% | 12.3 | 5% | Frequent updates on short rollouts that rarely span a full devaluation episode. |
| 4 | `p01-baseline` | -13.18 | +0.35% | 2.74% | 18.6 | 2% | Reference settings for the family. |
| 5 | `p03-clip-loose` | -14.00 | -0.23% | 3.18% | 7.3 | 5% | Weak constraint, approaching unclipped policy gradient. Tests whether clipping is what keeps PPO stable here. |
| 6 | `p10-entropy-high` | -40.35 | -0.99% | 4.13% | 17.3 | 10% | Keeps the action distribution broad for longer, counteracting early collapse onto HOLD. |
| 7 | `p02-clip-tight` | -44.04 | -0.45% | 3.31% | 32.6 | 8% | Very conservative trust region. Slow but should be the most monotonic curve in the study. |
| 8 | `p07-lambda-one` | -45.27 | -1.20% | 3.90% | 17.9 | 12% | GAE becomes the full Monte Carlo advantage. Unbiased, high variance -- the direct bridge to the REINFORCE family. |
| 9 | `p08-batch-large` | -50.08 | -1.70% | 4.58% | 13.4 | 5% | Four times the data per policy update. Better advantage estimates, fewer updates within the frame budget. |
| 10 | `p05-epochs-many` | -53.89 | -2.26% | 3.82% | 5.5 | 10% | Heavy reuse of each batch. Expect the ratio to leave the trust region and clipping to bind on most minibatches. |
| 11 | `p12-gamma-long` | -56.05 | -1.73% | 5.38% | 16.7 | 18% | Long horizon with long GAE traces, matched to the 500-step episode. The configuration theory favours for this task. |
| 12 | `p11-lr-high` | -56.79 | -2.33% | 5.32% | 5.8 | 15% | Tests whether PPO's clipping absorbs an aggressive learning rate that would break REINFORCE outright. |

Spread between best and worst configuration: **84.2 points** (`p06-lambda-low` +27.39 vs `p11-lr-high` -56.79).


### A2C — 12 configurations

| Rank | Configuration | Score | Equity | Max DD | Stockouts | Insolvency | Hypothesis tested |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | **`a06-critic-weak`** | **-30.36** | -0.94% | 5.08% | 7.9 | 10% | Under-trained critic means a poor baseline, pushing A2C back toward REINFORCE-like variance. |
| 2 | `a08-entropy-high` | -34.89 | -1.19% | 4.60% | 7.2 | 12% | Sustained exploration across the 13 discrete actions. |
| 3 | `a07-critic-strong` | -35.41 | -1.27% | 4.66% | 6.3 | 15% | Critic dominates the shared gradient. Tests whether value accuracy or policy improvement is the binding constraint. |
| 4 | `a01-baseline` | -35.73 | -1.20% | 4.33% | 8.9 | 10% | Reference settings for the family. |
| 5 | `a12-net-wide` | -41.87 | -1.68% | 4.12% | 5.0 | 12% | More capacity in the shared trunk, with a reduced learning rate. |
| 6 | `a09-entropy-zero` | -49.69 | -1.97% | 5.48% | 7.6 | 5% | Expect the fastest policy collapse of any actor-critic run. |
| 7 | `a03-nstep-long` | -64.15 | -2.82% | 6.47% | 3.2 | 15% | Long rollouts approach PPO's batch regime without the clipped objective -- a clean isolation of what clipping buys. |
| 8 | `a02-nstep-short` | -65.68 | -2.85% | 6.20% | 4.4 | 12% | Very short bootstrap. Fast, biased, and likely blind to delayed stockout costs. |
| 9 | `a11-gamma-long` | -67.30 | -2.41% | 6.15% | 14.0 | 18% | Extends the horizon while keeping rollouts short, stressing the critic's bootstrap accuracy. |
| 10 | `a04-lr-high` | -80.92 | -2.37% | 5.84% | 29.7 | 18% | A2C has no trust region, so this should be visibly less stable than PPO at a comparable learning rate. |
| 11 | `a05-lr-low` | -95.25 | -3.38% | 6.63% | 23.0 | 15% | Conservative steps to compensate for the missing trust region. |
| 12 | `a10-lambda-zero` | -101.69 | -4.38% | 8.07% | 7.6 | 20% | Pure one-step TD advantage, classical A2C. Maximum bias, minimum variance. |

Spread between best and worst configuration: **71.3 points** (`a06-critic-weak` -30.36 vs `a10-lambda-zero` -101.69).
