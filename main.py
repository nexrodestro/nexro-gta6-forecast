"""gta6-twitch-ccu-forecast-01 entry v2: launch-day Twitch viewer forecaster.

Forecasts, for each of hours 2..24 of a game's UTC release day, five
quantiles (5/25/50/75/95%) of Twitch concurrent viewers in the game's
category, anchored on the case's hour-1 print and calibrated on the
case file's own comparables: past big launches' hourly curves.

Model v2 (revised 2026-11-17; supersedes the v1 quad+kNN blend):

  Per hour, a quadratic log-linear regression of log(v_h / v_1) on
  log10(hour-1) across the comparables. The launch-day shape depends
  on how big hour 1 already is: tiny hour-1 launches ramp 10-1000x;
  mega launches (hour-1 >= 300k) dip to ~half of hour 1 by hour 9-10,
  then rise to ~1.1x in the evening (hours 17-19) and ease off. The
  fit is evaluated at log10(p1) clamped to the calibration range, so
  a GTA VI hour 1 above the biggest comparable (875k) saturates at
  the top-end shape instead of extrapolating down.

  Quantiles: q_tau(h) = p1 * exp(median_h + z_tau * spread_h), with
  z = -1.645/-0.674/0/0.674/1.645 and spread_h the fit's residual
  spread scaled by SPREAD_SCALE = 0.55, floored at 0.10 log units.

Why v2 beats v1 (validated on the server's practice metric, reproduced
locally from the competition's own source data, and on leave-one-out
over the kit's 37 comparables):

  * v1 blended the quad fit 50/50 with a size-kNN median and took the
    wider of the two spreads. The blend's median was worse than the
    quad fit alone on every validation set, so v2's median is the pure
    quad fit. The wider spread, though, earns its keep: taking
    max(quad residual spread, kNN neighbour spread) x 0.55 is
    identical to quad-only spread on practice (0.1885 vs 0.1888),
    big launches (0.2594 vs 0.2597) and mega launches (0.1458 vs
    0.1458), but much better on small launches (0.7165 vs 0.8301) —
    insurance if GTA VI's hour 1, or its shape, turns out unlike the
    mega comparables.
  * Narrowing the spread to 0.55x was the single biggest win, and it
    held on every set (not a practice artefact): practice 0.2367 ->
    0.1885, LOO all 0.3987 -> 0.3829, LOO big launches 0.2800 ->
    0.2594, LOO console-midnight 0.2245 -> 0.1809. Coverage on
    practice stays honest at 0.978.
  * Tried and rejected (all validated the same way): blend weights
    0.25/0.75 (worse), spread min/mean (worse on LOO), cubic fit
    (worse), empirical-quantile kNN (much worse on LOO), a per-hour
    median-bias correction fitted on LOO residuals (practice gain
    +0.008 but nested-LOO loss -0.012: does not generalise), and
    similarity weighting (console/midnight) — helps LOO slightly but
    costs practice; the quad fit already conditions on size.
  * Server practice score: v1 0.237, v2 0.189 (house baseline 0.387,
    rival entry 0.197). Every practice case improved: p00 0.290 ->
    0.185, p01 0.229 -> 0.225, p02 0.210 -> 0.166, p03 0.218 -> 0.179.

Only the case's `known` hour-1 print and its `comparables` are used.
Everything is deterministic: no randomness, no clock, no network,
stdlib only.

Run by the grader as `python3 -P -s /bot/main.py` (Python 3.13),
reading /input/<id>.in and writing /output/<id>.out. Locally:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a
# module shipped beside this file imports. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Floor on the per-hour log spread: even the most predictable hour
# keeps an honest minimum width.
SPREAD_FLOOR = 0.10
# Scale on the fit's residual spread. The raw residual spread is far
# too wide (practice coverage was 4/4 at every hour); 0.55x is the
# value that won on practice, LOO-all, LOO-big and LOO console-midnight
# alike (see module docstring). Do not "fix" this back to 1.0.
SPREAD_SCALE = 0.55
# Size-kNN neighbourhood for the spread hedge (median stays quad-only).
KNN_K = 8


def round2(x):
    """x rounded to 2 decimals, half up (as the starter rounds it).

    A whole number comes back as an int, so json.dumps writes 1240
    where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def parse_comparables(case):
    """Release-day curves from the case's comparables.

    Returns a list of {"h1": viewers in hour 1, "hours": {h: viewers}}
    for every comparable whose release-day hour 1 exists. A comparable
    with no hour-1 print (e.g. GTA VI's own pre-release history in
    the final case) cannot anchor a ratio curve and is left out.
    """
    curves = []
    for comp in case.get("comparables") or []:
        try:
            day_start = datetime.fromisoformat(
                comp["day_start"].replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError):
            continue
        hours = {}
        for point in comp.get("hourly") or []:
            try:
                t = datetime.fromisoformat(
                    point["from"].replace("Z", "+00:00"))
                viewers = point["viewers"]
            except (KeyError, TypeError, ValueError):
                continue
            h = int((t - day_start).total_seconds() // 3600) + 1
            if 1 <= h <= 24 and viewers is not None and viewers > 0:
                hours[h] = float(viewers)
        if hours.get(1) and len(hours) >= 9:
            curves.append({"h1": hours[1], "hours": hours})
    return curves


def hour_log_ratios(curves, h):
    """log(v_h / v_1) for every curve that has hour h."""
    out = []
    for c in curves:
        v = c["hours"].get(h)
        if v is not None and v > 0:
            out.append(math.log(v / c["h1"]))
    return out


def solve_linear(matrix, vector):
    """Small dense solve by Gaussian elimination; None if singular."""
    n = len(vector)
    m = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col:
                fac = m[r][col] / m[col][col]
                for cc in range(col, n + 1):
                    m[r][cc] -= fac * m[col][cc]
    return [m[i][n] / m[i][i] for i in range(n)]


def quad_loglin(curves, h):
    """Quadratic fit of log-ratio on log10(hour-1) for hour h.

    Returns (coef, mean_x, feats, ys), or None when too few curves
    have hour h.
    """
    xs, ys = [], []
    for c in curves:
        v = c["hours"].get(h)
        if v is not None and v > 0:
            xs.append(math.log10(c["h1"]))
            ys.append(math.log(v / c["h1"]))
    if len(ys) < 10:
        return None
    mean_x = sum(xs) / len(xs)
    # Design [1, u, u^2] with u centred, so the fit is stable.
    feats = []
    for x in xs:
        u = x - mean_x
        feats.append((1.0, u, u * u))
    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3
    for f, y in zip(feats, ys):
        for i in range(3):
            atb[i] += f[i] * y
            for j in range(3):
                ata[i][j] += f[i] * f[j]
    coef = solve_linear(ata, atb)
    if coef is None:
        return None
    return coef, mean_x, feats, ys


def quad_loglin_forecast(p1, curves, spread_scale=SPREAD_SCALE):
    """Per-hour (median log-ratio, spread) from the quadratic fit.

    The fit is evaluated at log10(p1) clamped to the calibration
    range: past the biggest comparable launch there is no evidence
    for a steeper fall, so the top-end launches' own shape is used
    instead of extrapolating the quadratic further down. The spread
    is the fit's residual spread scaled by spread_scale (default
    SPREAD_SCALE; pass 1.0 for the raw spread).
    """
    out = {}
    logs = [math.log10(c["h1"]) for c in curves]
    log_p1 = min(max(math.log10(p1), min(logs)), max(logs))
    for h in range(2, 25):
        fit = quad_loglin(curves, h)
        if fit is None:
            ratios = hour_log_ratios(curves, h)
            if not ratios:
                out[h] = (0.0, SPREAD_FLOOR)
                continue
            med = statistics.median(ratios)
            spread = statistics.pstdev(ratios) if len(ratios) > 1 else 0.3
            out[h] = (med, max(SPREAD_FLOOR, spread * spread_scale))
            continue
        coef, mean_x, feats, ys = fit
        u = log_p1 - mean_x
        median = coef[0] + coef[1] * u + coef[2] * u * u
        resid = [y - (coef[0] + coef[1] * f[1] + coef[2] * f[2])
                 for f, y in zip(feats, ys)]
        spread = statistics.pstdev(resid) if len(resid) > 1 else 0.3
        out[h] = (median, max(SPREAD_FLOOR, spread * spread_scale))
    return out


def knn_spread(p1, curves):
    """Per-hour raw log-spread from the KNN_K comparables whose hour-1
    is closest to p1 in log terms. Used only as a hedge inside the
    spread (see forecast_case): small launches have wildly different
    shapes, and their neighbour spread is the honest width there,
    where the quad fit's residual spread is too narrow.
    """
    log_p1 = math.log(p1)
    near = sorted(curves, key=lambda c: abs(math.log(c["h1"]) - log_p1))
    near = near[:KNN_K] if len(near) > KNN_K else near
    out = {}
    for h in range(2, 25):
        ratios = hour_log_ratios(near, h)
        if not ratios:
            ratios = hour_log_ratios(curves, h)
        spread = statistics.pstdev(ratios) if len(ratios) > 1 else 0.3
        out[h] = spread
    return out


def persistence_forecast(p1):
    """House-baseline fallback: every hour stays at hour 1, spread
    widening like a random walk's. Used only when the case carries
    too few comparables to calibrate on."""
    sigma = 0.15
    return {h: (0.0, sigma * math.sqrt(h - 1)) for h in range(2, 25)}


def hour_quantiles(p1, median, spread):
    """Five quantiles for one hour, rounded half-up to 2 decimals,
    non-decreasing, above 0 and at most 1e9."""
    qs = []
    for z in Z:
        q = p1 * math.exp(median + z * spread)
        q = max(0.01, min(1e9, q))
        qs.append(round2(q))
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None or not first.get("viewers"):
        # No hour-1 print: the case has no answer, but still write a
        # valid forecast anchored on the last history point, if any.
        history = case.get("history") or []
        p1 = 1.0
        for point in history:
            if point.get("viewers"):
                p1 = max(1.0, float(point["viewers"]))
    else:
        p1 = max(1.0, float(first["viewers"]))
    curves = parse_comparables(case)
    if len(curves) >= 10:
        quad = quad_loglin_forecast(p1, curves, spread_scale=1.0)
        knn_sp = knn_spread(p1, curves)
        blended = {}
        for h in range(2, 25):
            spread = max(quad[h][1], knn_sp[h]) * SPREAD_SCALE
            blended[h] = (quad[h][0], max(SPREAD_FLOOR, spread))
    else:
        blended = persistence_forecast(p1)
    return {"hours": [{"hour": row["hour"],
                       "q": hour_quantiles(p1, *blended[row["hour"]])}
                      for row in case["forecast"]]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
