from pathlib import Path

import pandas as pd
from build_clutch_dataset import add_pre_event_score

PBP_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "pbp"

total, bad = 0, 0
for path in PBP_DIR.glob("*.parquet"):
    pbp = add_pre_event_score(pd.read_parquet(path))
    made = pbp[(pbp["isFieldGoal"] == 1) & (pbp["shotResult"] == "Made")]
    gained = (made["scoreHome"] + made["scoreAway"]) - (made["pre_home"] + made["pre_away"])
    total += len(made)
    bad += int((gained != made["shotValue"]).sum())

    # Show the offending rows (usually only a handful across the whole season)
    off = made[gained != made["shotValue"]]
    if len(off):
        print(off[["gameId", "actionNumber", "clock", "description",
                   "pre_home", "pre_away", "scoreHome", "scoreAway"]].to_string(header=False))

print(f"made FGs checked: {total:,} | mismatches: {bad}")