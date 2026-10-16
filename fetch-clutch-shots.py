from nba_api.stats.static import players
from nba_api.stats.endpoints import shotchartdetail
import pandas as pd

# Find Jalen Brunson id automatically
nba_players = players.find_players_by_full_name("Jalen Brunson")
brunson_id = nba_players[0]["id"]

print(f"Extracting data for: {nba_players[0]['full_name']} (ID: {brunson_id})")

# Fetch all season shots
response = shotchartdetail.ShotChartDetail(
    team_id=0,
    player_id=brunson_id,
    season_nullable="2025-26",
    season_type_all_star="Regular Season",
    context_measure_simple="FGA"
)

shots_df = response.get_data_frames()[0]

# Filter just the final moments of the game
clutch_shots = shots_df[
    (shots_df["PERIOD"] >= 4) &
    (shots_df["MINUTES_REMAINING"] < 5)
].copy()

print(f"Total number of shots in the season: {len(shots_df)}")
print(f"Shots in the last 5 minutes of the 4th quarter: {len(clutch_shots)}")

cols = [
    "GAME_DATE",
    "PERIOD",
    "MINUTES_REMAINING",
    "ACTION_TYPE",
    "SHOT_ZONE_BASIC",
    "SHOT_DISTANCE",
    "LOC_X",
    "LOC_Y",
    "SHOT_MADE_FLAG"
]

print(clutch_shots[cols].head(10))

clutch_shots.to_parquet("brunson_clutch_shots.parquet", index=False)