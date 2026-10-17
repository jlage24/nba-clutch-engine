import time
from pathlib import Path

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import shotchartdetail
import pandas as pd

SEASON = ["2025-26", "2024-25", "2023-24", "2022-23"][0]
SEASON_TYPE = ["Regular Season", "Playoffs"][0]
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def fetch_team_shots(team_id: int, retries: int = 3) -> pd.DataFrame:
    # All FGA for one team
    for attempt in range(1, retries + 1):
        try: 
            response = shotchartdetail.ShotChartDetail(
                team_id=team_id,
                player_id=0,
                season_nullable=SEASON,
                season_type_all_star=SEASON_TYPE,
                context_measure_simple="FGA",
                timeout=60,
            )
            return response.get_data_frames()[0]
        except Exception as e:
            wait = 5 * attempt
            print(f"  attempt {attempt} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Could not fetch team {team_id}")

frames = []
for team in teams.get_teams():
    cache = DATA_DIR / f"raw_shots_{SEASON}_{team['abbreviation']}.parquet"
    if cache.exists():
        print(f"Loading cached data for {team['full_name']}...")
        df = pd.read_parquet(cache)
    else:
        print(f"Fetching data for {team['full_name']}...")
        df = fetch_team_shots(team["id"])
        df.to_parquet(cache, index=False)
        time.sleep(1.5)  # be nice to the API
    frames.append(df)

shots = pd.concat(frames, ignore_index=True)
shots["TIME_REMAINING_SEC"] = shots["MINUTES_REMAINING"] * 60 + shots["SECONDS_REMAINING"]

late_game = shots[(shots["PERIOD"] >= 4) & (shots["TIME_REMAINING_SEC"] <= 300)].copy()

print(f"All shots: {len(shots):,} | late-game shots: {len(late_game):,}")
late_game.to_parquet(DATA_DIR / f"late_game_shots_{SEASON}.parquet", index=False)