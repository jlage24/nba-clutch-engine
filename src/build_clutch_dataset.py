from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PBP_DIR = DATA / "raw" / "pbp"
OUT_DIR = DATA / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_pre_event_score(pbp: pd.DataFrame) -> pd.DataFrame:
    """Score BEFORE each event. Using the row's own score would leak the shot result."""
    pbp = pbp.sort_values("actionNumber").copy()
    for c in ("scoreHome", "scoreAway"):
        pbp[c] = pd.to_numeric(pbp[c], errors="coerce")  # '' -> NaN
    pbp["pre_home"] = pbp["scoreHome"].ffill().fillna(0).shift(1, fill_value=0)
    pbp["pre_away"] = pbp["scoreAway"].ffill().fillna(0).shift(1, fill_value=0)
    return pbp


def load_shot_events() -> pd.DataFrame:
    frames = []
    for path in sorted(PBP_DIR.glob("*.parquet")):
        pbp = add_pre_event_score(pd.read_parquet(path))
        fg = pbp[pbp["isFieldGoal"] == 1]
        frames.append(
            fg[["gameId", "actionNumber", "location", "pre_home", "pre_away", "xLegacy", "yLegacy"]]
        )
    return pd.concat(frames, ignore_index=True)


def main():
    shots = pd.concat(
        [pd.read_parquet(p) for p in sorted(DATA.glob("late_game_shots_*.parquet"))],
        ignore_index=True,
    )
    shots["GAME_ID"] = shots["GAME_ID"].astype(str)
    events = load_shot_events()

    # Only keep shots from games we actually downloaded (matters when using --limit)
    shots = shots[shots["GAME_ID"].isin(events["gameId"].unique())]

    merged = shots.merge(
        events,
        left_on=["GAME_ID", "GAME_EVENT_ID"],
        right_on=["gameId", "actionNumber"],
        how="left",
        indicator=True,
    )

    # Validation: if these fail, stop and debug before modelling
    match_rate = (merged["_merge"] == "both").mean()
    both = merged[merged["_merge"] == "both"]
    coord_match = ((both["LOC_X"] == both["xLegacy"]) & (both["LOC_Y"] == both["yLegacy"])).mean()
    print(f"shots: {len(merged):,} | matched to PBP: {match_rate:.1%} | coords agree: {coord_match:.1%}")

    merged = both.copy()
    is_home = merged["location"] == "h"
    merged["SCORE_MARGIN"] = np.where(
        is_home, merged["pre_home"] - merged["pre_away"], merged["pre_away"] - merged["pre_home"]
    )

    clutch = merged[merged["SCORE_MARGIN"].abs() <= 5].copy()
    print(f"clutch shots (last 5 min, margin <= 5): {len(clutch):,} of {len(merged):,} late-game shots")

    clutch.to_parquet(OUT_DIR / "clutch_shots.parquet", index=False)


if __name__ == "__main__":
    main()