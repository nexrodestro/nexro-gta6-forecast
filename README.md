# nexro-gta6-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nexrodestro/nexro-gta6-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nexrodestro/nexro-gta6-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's launch-day Twitch concurrent viewers hour by hour: for each of hours 2–24 of 2026-11-19 (UTC), five quantiles (5%, 25%, 50%, 75%, 95%) of viewers in the "Grand Theft Auto VI" category, anchored on the case's hour-1 print.

## Model

The shape of a launch day depends on how big hour 1 already is. Tiny hour-1 launches ramp 10–1000×; mega launches (hour-1 ≥ 300k) dip to ~half of hour 1 by hours 9–10, then rise to ~1.1× in the evening (hours 17–19) and ease off. Each hour's median log-ratio to hour 1 comes from a quadratic log-linear regression of log(v_h / v_1) on log10(v_1) across the case's comparables — captures the size-dependent ramp and saturates at the top end (a GTA VI hour 1 above the biggest comparable, 875k, uses the top-end shape instead of extrapolating down). The spread is max(quad residual spread, size-kNN neighbour spread) × 0.55, floored at 0.10 log units: the raw residual spread was far too wide (practice 90% coverage was 4/4 at every hour), and 0.55× is the value that won on practice, leave-one-out overall, big launches and console-midnight launches alike.

Quantiles: q_tau(h) = p1 · exp(median_h + z_tau · spread_h), z = −1.645/−0.674/0/0.674/1.645. Only the case's hour-1 print and its comparables are used; the program runs offline, deterministically, stdlib only.

## Backtest (leave-one-out on the kit's 37 comparable launches)

| model | mean pinball loss (all) | big launches (hour-1 ≥ 20k) |
|---|---|---|
| persistence baseline (house starter) | 0.930 | 0.321 |
| **this entry (v2)** | **0.383** | **0.259** |

Server practice score (4 past launches): **0.189** (v1: 0.237; house baseline: 0.387).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=kp9z8kfp&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
