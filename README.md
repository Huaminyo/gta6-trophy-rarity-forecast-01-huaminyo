# GTA VI trophy rarity forecast — @huaminyo

Forecasts every Grand Theft Auto VI (PS5) trophy's Exophase earn rate at the
case target as five quantiles, for
[`gta6-trophy-rarity-forecast-01`](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
($250, top 20). One file, Python 3.13 standard library only, deterministic.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input dir, writes `<id>.out` per case:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Model

A rate is earners / players tracked, so the two are modelled separately and
recombined — a rate can fall while its earners grow, whenever fresh players
flood in faster than anyone finishes anything.

**Flat history (no refresh observed yet — the practice cases).** A refresh is
the modal next event. The kit documents the launch-night refresh (players
638 → 3,868, trophies earned 943 → 4,276); the new cohort averaged 1.03
trophies each, so the refresh is a fresh flood: every newcomer holds the one
100% trophy and almost nothing else. The house baseline's own stated practice
score discriminates this from uniform scaling — persistence scores ~1.51 loss
/ 0.68 pp miss, which the flood predicts (1.40–1.69 / 0.68) and uniform
scaling does not (0.20 / 0.22). Median: earners frozen, players scaled by the
documented refresh ratio, plus half the leftover trophy budget (103 at
launch-night levels) spread by sqrt of each trophy's public easiness (live
Exophase rates, 20 Nov 2026). Spread sigma 0.20.

**History with refreshes (the final case).** Consecutive identical reads
collapse to refresh states; per trophy, log(earners) is fit by least squares
on log(1 + hours since release), slope shrunk toward the tier median (barely
for clean fits, hard for thin/noisy ones), re-anchored at the last observed
earners. Players tracked and total trophies earned use the same fit with
slopes blended 30/70 between all states and the last 24 hours, because launch
growth decelerates. Earner forecasts are rescaled to match the forecast total
trophies earned (count consistency), capped by tier ceilings, and blended
75/25 with persistence in logit space. Spread combines a horizon term, the
fit residual, and a Poisson term for small earner counts.

## Local validation

- Practice cases: valid output, byte-identical across runs, ~0.1 s/case.
  Simulated practice loss under the plausible refresh outcomes (frozen /
  pro-rata / concentrated extras): 0.09–0.14, vs 1.37–1.69 for the house
  persistence baseline under the same outcomes.
- Synthetic final case (60 reads, Nov 18–21, dynamics calibrated on the live
  Exophase page): loss 0.117 vs 0.189 for the baseline; Ch1 15.96% → 23.98%
  median (truth 42.53% in the aggressive-growth simulation).
- Edge cases (2-state history, null rates mid-history): valid output.

## Entry

- Competition: `gta6-trophy-rarity-forecast-01`
- Account: @huaminyo (GitHub: Huaminyo)
- License: MIT
