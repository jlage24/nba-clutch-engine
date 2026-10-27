from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

late = pd.concat(
    [pd.read_parquet(p) for p in sorted(DATA.glob("late_game_shots_*.parquet"))],
    ignore_index=True,
)
clutch = pd.read_parquet(DATA / "processed" / "clutch_shots.parquet")
n_pbp = len(list((DATA / "raw" / "pbp").glob("*.parquet")))

print("== Coverage ==")
print(f"games with late-game shots: {late['GAME_ID'].nunique()} | PBP files: {n_pbp}")
print(f"late-game shots: {len(late):,} | clutch shots: {len(clutch):,} ({len(clutch)/len(late):.1%})")
print(f"games with clutch shots: {clutch['GAME_ID'].nunique()}")

print("\n== Made vs missed margin (leakage smell test) ==")
made = clutch.loc[clutch["SHOT_MADE_FLAG"] == 1, "SCORE_MARGIN"]
miss = clutch.loc[clutch["SHOT_MADE_FLAG"] == 0, "SCORE_MARGIN"]
diff = made.mean() - miss.mean()
se = np.sqrt(made.var(ddof=1) / len(made) + miss.var(ddof=1) / len(miss))
print(f"made {made.mean():.2f} | missed {miss.mean():.2f} | diff {diff:.2f} | z = {diff/se:.1f}")

print("\n== FG% by shot type ==")
print(clutch.groupby("SHOT_TYPE")["SHOT_MADE_FLAG"].agg(["mean", "count"]).round(3))

print("\n== FG% by zone ==")
print(clutch.groupby("SHOT_ZONE_BASIC")["SHOT_MADE_FLAG"].agg(["mean", "count"]).round(3))

print("\n== Shots per player (small-sample check) ==")
per_player = clutch.groupby("PLAYER_NAME").size()
print(per_player.describe().round(1))
print("players with >= 30 clutch shots:", int((per_player >= 30).sum()))
print("top 5:", per_player.sort_values(ascending=False).head(5).to_dict())