"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01), account @huaminyo.

Forecasts every GTA VI PS5 trophy's Exophase earn rate at the case's target as
five quantiles, from the house's recorded reads of the trophy page.

Model (stdlib only, deterministic, no network)
----------------------------------------------
A trophy's rate is earners / players_tracked, so the two are modelled
separately and recombined -- a rate can fall while its earners grow, whenever
fresh players flood in faster than anyone finishes anything (exactly what the
launch-night refresh did: 638 -> 3,868 players, every rate but the 100% one
down).

Two branches, chosen by what the case's history actually shows:

A. Flat history (every read identical -- no refresh observed yet, the
   practice cases' shape). A refresh is then the modal next event, and the
   kit's own account of the launch-night refresh pins its size: players
   638 -> 3,868, trophies earned 943 -> 4,276. The new cohort averaged 1.03
   trophies each, so the refresh is a fresh flood: every newcomer holds the
   one 100% trophy (Gone Fishing) and almost nothing else, and every other
   rate is diluted by ~x6. The house baseline's own stated practice score
   discriminates this from uniform scaling: persistence scores a loss of
   about 1.51 with a typical miss of 0.68 points, which the flood predicts
   (1.40-1.69 / 0.68) and uniform scaling does not (0.20 / 0.22). So the
   median freezes each trophy's earners and divides by the refreshed player
   count, plus half of the small leftover trophy budget (the refresh's new
   trophies minus one Gone Fishing per newcomer: 103), spread over trophies
   by sqrt of their public easiness (the live Exophase page's rates: early
   trophies collect the newcomers' second trophies first). The spread
   (sigma 0.20) prices the one thing the counts cannot say: exactly which
   trophies those leftover trophies landed on.

B. History with refreshes (the final case's shape). Distinct refresh states
   are extracted (consecutive identical reads collapse to one state) and, per
   trophy, log(earners) is fit by least squares against log(1 + hours since
   release) -- a power law in game age -- with the slope shrunk toward the
   tier median (barely, when the fit is clean on many states; hard, when it
   is thin or noisy) and re-anchored at the last observed earners, so the
   level is always the freshest read and only the growth rate comes from the
   fit. Players tracked and total trophies earned get the same treatment,
   with each slope blended 30/70 between all states and the last 24 hours,
   because launch growth decelerates. The
   per-trophy earner forecasts are then rescaled so their sum matches the
   forecast total trophies earned (count consistency), capped by tier
   ceilings, and blended 75/25 with persistence in logit space. The spread
   combines a horizon term, the fit's residual, and a Poisson term for small
   earner counts, so rare trophies honestly get wider bands.

Quantiles use the standard normal z at each level, rounded half-up to 4
decimals like the house baseline. Two runs on the same case write the same
bytes: no randomness, no wall clock, no dict-order dependence.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Release of GTA VI (UTC): the clock every growth curve is measured against.
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Launch-night refresh, as the kit documents it (competition.yaml / RULES.txt):
# players 638 -> 3,868, trophies earned 943 -> 4,276. A flat history's next
# refresh is expected to scale players and trophies by the same ratios.
REFRESH_PLAYERS_RATIO = 3868 / 638
REFRESH_EARNED_RATIO = 4276 / 943

# Trophy easiness prior: each trophy's earn rate on the PUBLIC Exophase page
# (https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/) as read on
# 2026-11-20 ~13:53 UTC, in trophy-id order (1..55). Early trophies a new
# player reaches within hours (Chapter 1, the fishing challenge, Breakfast
# Run) sit at 3-13%; late and grind trophies at 0.00-0.05%. Only the SHAPE
# is used (sqrt-weighted, see forecast_flat): which trophies a fresh
# player's second trophy is likely to be. Not the answers to anything --
# the page shows today's rates, and the practice answers are 19 Nov 04:30.
LIVE_EASINESS = [
    0.00, 100.0, 11.10, 1.44, 0.03, 0.02, 0.02, 3.77, 0.03, 0.02, 0.02,
    0.02, 0.02, 0.00, 0.00, 0.01, 0.01, 0.00, 0.01, 12.76, 1.31, 0.01,
    0.01, 0.01, 0.00, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.27,
    0.03, 1.42, 0.07, 0.22, 0.19, 0.04, 0.01, 0.01, 0.04, 0.01, 0.15,
    0.01, 1.09, 0.01, 0.01, 0.01, 0.64, 0.05, 0.01, 0.03, 0.01, 0.01,
]

# Tier ceilings on a forecast rate (percent): loose anti-explosion caps only.
# A trophy already above its cap keeps its own level (caps never force a fall).
TIER_CAP = {"platinum": 4.0, "gold": 12.0, "silver": 20.0, "bronze": 55.0}


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    c = clamp(pct)
    return math.log(c / (100 - c))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100 / (1 + math.exp(-lg))
    e = math.exp(lg)
    return 100 * e / (1 + e)


def round4(x: float):
    """Round to 4 decimals, half up (the house baseline's rounding)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(t: datetime) -> float:
    return max(0.0, (t - RELEASE).total_seconds() / 3600.0)


def log_time(t: datetime) -> float:
    """log(1 + hours since release): the axis growth curves are linear on."""
    return math.log1p(hours_since_release(t))


def ols(xs, ys):
    """Least-squares line y = a + b x. Returns (slope, intercept, resid_sd, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    if n == 1:
        return 0.0, ys[0], 0.0, 1
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my, 0.0, n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n) if n > 2 else 0.0
    return b, a, resid, n


def median(values):
    s = sorted(values)
    m = len(s)
    if m == 0:
        return 0.0
    return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])


def distinct_states(history):
    """Collapse consecutive identical reads to one state each (refresh states).

    Returns a list of dicts with keys t (datetime), x (log-time), players,
    earned, rates (list). Only the first read of each run is kept.
    """
    states = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), read.get("trophies_earned"),
               tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        try:
            t = parse_time(read["at"])
        except (KeyError, ValueError):
            continue
        states.append({
            "t": t,
            "x": log_time(t),
            "players": read.get("players_tracked"),
            "earned": read.get("trophies_earned"),
            "rates": rates,
        })
    return states


def quantiles_from_logit(center: float, sigmas) -> list:
    """Five quantiles from a logit centre with per-level spreads (sigmas list
    of (z, sigma) pairs, in level order). Returns rounded, non-decreasing
    values clipped to [0, 100]."""
    qs = []
    for z, s in sigmas:
        qs.append(round4(min(100.0, max(0.0, inv_logit(center + z * s)))))
    # Rounding can only preserve order, but enforce it defensively.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def symmetric_quantiles(center: float, sigma: float) -> list:
    return quantiles_from_logit(center, [(z, sigma) for z in Z])


def forecast_flat(last_state, trophies, horizon_days: float) -> list:
    """Branch A: history never refreshed. Fresh-flood refresh median (see
    module docstring): earners frozen, players scaled by the documented
    refresh ratio, plus half the leftover trophy budget spread by easiness.
    """
    last_rates = last_state["rates"]
    n_last = last_state["players"] or 1
    t_last = last_state["earned"] or 0
    n_next = n_last * REFRESH_PLAYERS_RATIO
    t_next = t_last * REFRESH_EARNED_RATIO
    # New players each add one Gone Fishing trophy; the rest is the leftover
    # budget for every other trophy (103 at launch-night levels).
    leftover = max(0.0, (t_next - t_last) - (n_next - n_last))
    weights = []
    for i, trophy in enumerate(trophies):
        last = last_rates[i] if i < len(last_rates) else 0
        if last is not None and last >= 100:
            weights.append(0.0)
        else:
            ease = LIVE_EASINESS[i] if i < len(LIVE_EASINESS) else 0.0
            weights.append(math.sqrt(max(ease, 0.0)))
    w_sum = sum(weights)
    sigma = 0.20 + 0.10 * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(trophies):
        last = last_rates[i] if i < len(last_rates) else 0
        if last is None:
            last = 0.0
        if last >= 100:
            # Everyone holds it; newcomers hold it too.
            rows.append({"id": trophy["id"], "q": [99.9, 99.98, 100, 100, 100]})
            continue
        earners = last * n_last / 100.0
        # Half the leftover budget, easiness-weighted; the other half stays
        # unassigned (shrinkage toward frozen earners).
        extra = 0.5 * leftover * weights[i] / w_sum if w_sum > 0 else 0.0
        med = 100 * (earners + extra) / n_next
        med = min(99.99, max(0.01, med))
        rows.append({"id": trophy["id"],
                     "q": symmetric_quantiles(logit(med), sigma)})
    return rows


def forecast_growth(case, states, trophies, horizon_days: float) -> list:
    """Branch B: history holds refreshes. Power-law earners + players,
    count-consistent rescale, persistence blend, calibrated spread."""
    n_troph = len(trophies)
    last = states[-1]
    x_last = last["x"]
    target_t = parse_time(case["target"])
    x_target = log_time(target_t)
    dx = max(0.0, x_target - x_last)

    xs = [s["x"] for s in states]

    def blended_slope(values):
        """Slope of log(values) on log-time: all states, blended 30/70 with
        the last 24 hours' slope when that window holds >= 4 states. Launch
        growth decelerates, so the recent slope is the better guide to the
        next days, while the full-history slope steadies it."""
        pairs = [(s["x"], math.log(max(1, v)))
                 for s, v in zip(states, values)
                 if isinstance(v, (int, float)) and v > 0]
        if not pairs:
            return 0.0
        slope_all, _, _, _ = ols([p[0] for p in pairs], [p[1] for p in pairs])
        t_last = states[-1]["t"]
        recent = [(s["x"], math.log(max(1, v)))
                  for s, v in zip(states, values)
                  if isinstance(v, (int, float)) and v > 0
                  and (t_last - s["t"]).total_seconds() <= 24 * 3600]
        if len(recent) >= 4:
            slope_recent, _, _, _ = ols([p[0] for p in recent],
                                        [p[1] for p in recent])
            return 0.3 * slope_all + 0.7 * slope_recent
        return slope_all

    # --- Players tracked. ---
    slope_n = blended_slope([s["players"] for s in states])
    slope_n = max(0.0, slope_n)  # tracked players do not shrink
    n_last = last["players"] if isinstance(last["players"], (int, float)) else 1
    n_target = n_last * math.exp(slope_n * dx)
    # Anti-explosion cap: at most x5 a day.
    n_target = min(n_target, n_last * (5.0 ** max(horizon_days, 0.0)))
    n_target = max(n_target, float(n_last))

    # --- Total trophies earned, same treatment (for count consistency). ---
    slope_t = blended_slope([s["earned"] for s in states])
    slope_t = max(0.0, slope_t)
    t_last = last["earned"] if isinstance(last["earned"], (int, float)) else 0
    t_target = t_last * math.exp(slope_t * dx) if t_last > 0 else 0.0

    # --- Per-trophy earner slopes (log earners on log-time). ---
    raw_slopes = []
    tier_slopes = {}
    per_trophy = []
    for i, trophy in enumerate(trophies):
        pts_x, pts_y = [], []
        for s in states:
            r = s["rates"][i] if i < len(s["rates"]) else None
            p = s["players"]
            if r is None or not isinstance(p, (int, float)) or p <= 0:
                continue
            earners = r * p / 100.0
            pts_x.append(s["x"])
            pts_y.append(math.log(earners + 0.5))
        slope, _, resid, cnt = ols(pts_x, pts_y)
        raw_slopes.append(slope)
        tier_slopes.setdefault(trophy.get("tier", "bronze"), []).append(slope)
        per_trophy.append({"slope": slope, "resid": resid, "n": cnt})
    tier_med = {t: median(v) for t, v in tier_slopes.items()}
    global_med = median(raw_slopes)

    # --- Last earners, anchored forecasts, ceilings. ---
    sigma_base = 0.12 + 0.11 * math.sqrt(horizon_days)
    forecasts = []
    for i, trophy in enumerate(trophies):
        last_rate = last["rates"][i] if i < len(last["rates"]) else None
        if last_rate is None:
            # Fall back to the last non-null rate in the history.
            last_rate = 0.0
            for s in reversed(states):
                r = s["rates"][i] if i < len(s["rates"]) else None
                if r is not None:
                    last_rate = r
                    break
        e_last = last_rate * n_last / 100.0
        info = per_trophy[i]
        tier = trophy.get("tier", "bronze")
        # Shrink the slope toward the tier median. A clean fit on many states
        # (low residual) has earned the right to speak for itself: only half
        # a pseudo-observation of shrinkage. A thin or noisy fit gets three.
        prior = tier_med.get(tier, global_med)
        k = 0.5 if (info["n"] >= 5 and info["resid"] <= 0.25) else 3.0
        slope = (info["n"] * info["slope"] + k * prior) / (info["n"] + k)
        e_hat = max(e_last, 0.5) * math.exp(slope * dx)
        forecasts.append({"e": e_hat, "e_last": e_last, "last_rate": last_rate,
                          "resid": info["resid"], "tier": tier,
                          "id": trophy["id"]})
    # Count consistency: rescale earners to the forecast total (clipped).
    total_e = sum(f["e"] for f in forecasts)
    if total_e > 0 and t_target > 0:
        factor = t_target / total_e
        factor = min(2.0, max(0.5, factor))
        for f in forecasts:
            f["e"] *= factor

    rows = []
    for f in forecasts:
        if f["last_rate"] >= 100:
            rows.append({"id": f["id"], "q": [99.9, 99.98, 100, 100, 100]})
            continue
        rate_model = 100 * f["e"] / n_target
        cap = TIER_CAP.get(f["tier"], 55.0)
        cap = max(cap, f["last_rate"])
        rate_model = min(rate_model, cap)
        rate_model = max(rate_model, 0.01)
        # Persistence blend in logit space: 75% model, 25% last rate.
        center = 0.75 * logit(rate_model) + 0.25 * logit(max(f["last_rate"], 0.01))
        sigma = math.sqrt(sigma_base ** 2 + min(f["resid"], 0.5) ** 2
                          + 1.0 / max(f["e_last"], 1.0))
        sigma = min(1.2, max(0.15, sigma))
        rows.append({"id": f["id"], "q": symmetric_quantiles(center, sigma)})
    return rows


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case.get("history"):
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    states = distinct_states(case["history"])
    if not states:
        raise ValueError(f"{name} has no parsable history")
    if len(states) == 1:
        rows = forecast_flat(states[0], trophies, horizon_days)
    else:
        rows = forecast_growth(case, states, trophies, horizon_days)
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(out.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name} forecast exceeds 65,536 bytes")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
