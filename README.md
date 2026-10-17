# NBA Clutch Decision Engine

A sports analytics and machine learning project that models **shot quality in NBA clutch time** and turns it into **Expected Points (xP)** to compare late-game offensive options.

> **Status:** 🚧 Work in progress. Phase 1 (data ingestion) is working. Everything else is planned. See the [Roadmap](#roadmap).

---

## Motivation

Modern analytics tends to penalize mid-range jumpers in favor of layups and threes. But in the last five minutes of a close game, the context changes:

- Pace slows down and transition opportunities disappear.
- Defenses tighten, switch, and pack the paint.
- Stars (Brunson, Shai Gilgeous-Alexander, Jokić) take a disproportionate share of self-created shots.

Season-long averages and box scores don't capture that context: score margin, time left, court location, shot type. This project asks:

> Given the game state, how does shot efficiency shift under pressure, and which shot type and location carry the highest expected value?

## Approach

1. **Ingest** shot-level data from the NBA stats API (`nba_api`) and cache it locally as Parquet.
2. **Enrich** shots with play-by-play data to attach the exact score margin at the time of each attempt.
3. **Model** `P(make | shot context)` with gradient-boosted trees (XGBoost), with an emphasis on **probability calibration**, not just accuracy.
4. **Convert** probabilities to expected points:

   ```
   xP = P(make | X) × shot value (2 or 3)
   ```

5. **Decide**: in an interactive Streamlit app, compare options for a given game state. In tight situations (e.g. down 3 late), raw xP isn't the right objective, so the simulator will also report the probability of tying or taking the lead.

Everything runs locally on a laptop CPU. No GPU and no cloud services.

## Roadmap

| Phase | Deliverable | Status |
| :--- | :--- | :--- |
| 1. Foundation | Fetch league-wide shot data, cache to Parquet, half-court plotting | 🟡 Fetching works, plotting pending |
| 2. Context enrichment | Merge play-by-play to add score margin; define clutch as last 5 min with margin ≤ 5 | ⬜ Planned |
| 3. ML engine | Feature engineering, XGBoost classifier, calibration (Brier score, reliability curves), player residuals | ⬜ Planned |
| 4. Simulator | Streamlit app with game-state inputs and an xP heatmap on the court | ⬜ Planned |

## Setup

Requires Python 3.10+.

```bash
git clone <your-repo-url>
cd <repo-folder>

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Fetch shot data (cached per team, so reruns skip anything already downloaded):

```bash
python fetch_clutch_shots.py
```

This downloads every field-goal attempt per team and season from `stats.nba.com`, stores the raw data under `data/`, and writes a filtered table of late-game shots (period ≥ 4, ≤ 300 seconds remaining).

The first run takes a few minutes because requests are spaced out to avoid rate limiting. After that, loading from cache is instant.

## Data

| Source | Used for |
| :--- | :--- |
| `ShotChartDetail` (`nba_api`) | Shot location (`LOC_X`, `LOC_Y`), distance, zone, action type, clock, result |
| `PlayByPlay` (`nba_api`) | Score margin and game context *(Phase 2)* |

Data is **not** committed to the repo. It is reproducible by running the fetch script.

## Planned features

- `TIME_REMAINING_SEC`: seconds left in the period
- `SCORE_MARGIN`: lead or deficit from the shooting team's perspective
- `SHOT_DISTANCE`, `SHOT_ANGLE`: polar coordinates derived from `LOC_X`, `LOC_Y`
- `SHOT_VALUE`: 2PT or 3PT attempt
- Shot type / action type, season, regular season vs. playoffs

## Known limitations

Being upfront about what this model can and can't claim:

- **Selection bias.** The model learns from shots that were *taken*, not from alternatives that weren't. It estimates shot quality, not the causal effect of choosing a different shot.
- **No defender information.** The shot chart data has no contest distance or defender context, which is a major driver of shot quality.
- **Small samples.** Clutch attempts per player are limited, so player-level effects will be shrunk toward the league average rather than reported raw.
- **Assisted vs. unassisted shots** aren't equivalent options. A kick-out three requires a pass first.
- **No free throws or turnovers** in the shot chart, so drawing fouls isn't captured in xP.

## Tech stack

Python · pandas · PyArrow/Parquet · nba_api · XGBoost · scikit-learn · Matplotlib · Streamlit

## Acknowledgments

Data from [`nba_api`](https://github.com/swar/nba_api), a client for the public NBA stats endpoints. This project is not affiliated with or endorsed by the NBA.