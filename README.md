# nexro-gta6-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nexrodestro/nexro-gta6-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nexrodestro/nexro-gta6-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's launch-day Twitch concurrent viewers hour by hour: for each of hours 2–24 of 2026-11-19 (UTC), five quantiles (5%, 25%, 50%, 75%, 95%) of viewers in the "Grand Theft Auto VI" category, anchored on the case's hour-1 print.

## Model

The shape of a launch day depends on how big hour 1 already is. Tiny hour-1 launches ramp 10–1000×; mega launches (hour-1 ≥ 300k) dip to ~half of hour 1 by hours 9–10, then rise to ~1.1× in the evening (hours 17–19) and ease off. Two estimators of each hour's log-ratio to hour 1 are blended (mean of log-medians, wider of the two spreads):

1. a quadratic log-linear regression of log(v_h / v_1) on log10(v_1) across the case's comparables — captures the size-dependent ramp and saturates at the top end;
2. size-kNN: the median log-ratio of the 8 comparables whose hour-1 is closest in log terms — captures the mega-launch U-shape directly.

Quantiles: q_tau(h) = p1 · exp(median_h + z_tau · spread_h), z = −1.645/−0.674/0/0.674/1.645, spread floored at 0.10 log units. Only the case's hour-1 print and its comparables are used; the program runs offline, deterministically, stdlib only.

## Backtest (leave-one-out on the kit's 37 comparable launches)

| model | mean pinball loss (all) | big launches (hour-1 ≥ 20k) |
|---|---|---|
| persistence baseline (house starter) | 0.930 | 0.321 |
| **this entry** | **0.399** | **0.280** |

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=kp9z8kfp&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
