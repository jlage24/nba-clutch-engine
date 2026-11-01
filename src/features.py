import sys
from pathlib import Path
from typing import Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "clutch_shots.parquet"
OUT = ROOT / "data" / "processed" / "features.parquet"

TARGET = "SHOT_MADE_FLAG"
ID_COLS = ["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "SEASON"]
NUMERIC = [
    "TIME_REMAINING_SEC", "SCORE_MARGIN", "SHOT_DISTANCE_FT", "SHOT_ANGLE",
    "LOC_X", "LOC_Y", "SHOT_VALUE", "IS_OT", "IS_HOME", "IS_PLAYOFFS",
    "LOC_PLACEHOLDER", "IS_PUTBACK",
]
CATEGORICAL = ["ACTION_GROUP"]
FEATURES = NUMERIC + CATEGORICAL

ACTION_CATEGORIES = [
    "dunk",
    "layup",
    "hook",
    "floater",
    "stepback",
    "pullup",
    "fadeaway",
    "jump_shot",
    "other",
]
ACTION_DTYPE = pd.CategoricalDtype(categories=ACTION_CATEGORIES)


def action_group(action_type: pd.Series) -> pd.Series:
    """Collapse raw action types into shot families (order matters).

    Uses a fixed CategoricalDtype so that categorical levels and encodings
    remain identical between training sets, test sets, and single-row inference.
    """
    s = action_type.astype(str).str.lower()
    conditions = [
        s.str.contains("dunk"),
        s.str.contains("layup|finger roll"),
        s.str.contains("hook"),
        s.str.contains("floating|floater"),
        s.str.contains("step back"),
        s.str.contains("pullup|pull-up"),
        s.str.contains("fadeaway|turnaround"),
        s.str.contains(r"jump.*shot|jumper|bank"),
    ]
    choices = [
        "dunk",
        "layup",
        "hook",
        "floater",
        "stepback",
        "pullup",
        "fadeaway",
        "jump_shot",
    ]
    raw = np.select(conditions, choices, default="other")
    return pd.Series(pd.Categorical(raw, dtype=ACTION_DTYPE), index=action_type.index)


def build_features(df: Union[pd.DataFrame, dict, pd.Series]) -> pd.DataFrame:
    """Transform raw shot records into model-ready features.

    Accepts a full DataFrame, a single-row DataFrame, a Series, or a dict
    (for single-shot inference in live mode).
    """
    if isinstance(df, dict):
        df = pd.DataFrame([df])
    elif isinstance(df, pd.Series):
        df = df.to_frame().T
    else:
        df = df.copy()

    # Season and playoff flags derived from GAME_ID (e.g. '0022500001' -> 2025-26 regular season)
    if "GAME_ID" in df.columns:
        gid = df["GAME_ID"].astype(str)
        if "SEASON" not in df.columns:
            df["SEASON"] = 2000 + gid.str[3:5].astype(int)  # 2025 means 2025-26
        if "IS_PLAYOFFS" not in df.columns:
            df["IS_PLAYOFFS"] = gid.str[:3].eq("004").astype(int)
    else:
        if "SEASON" not in df.columns:
            df["SEASON"] = 2025
        if "IS_PLAYOFFS" not in df.columns:
            df["IS_PLAYOFFS"] = 0

    # Game clock in seconds
    if "TIME_REMAINING_SEC" not in df.columns:
        if "MINUTES_REMAINING" in df.columns and "SECONDS_REMAINING" in df.columns:
            df["TIME_REMAINING_SEC"] = (
                df["MINUTES_REMAINING"].astype(float) * 60 + df["SECONDS_REMAINING"].astype(float)
            )
        else:
            raise KeyError(
                "Either 'TIME_REMAINING_SEC' or both 'MINUTES_REMAINING' and 'SECONDS_REMAINING' must be provided."
            )

    # Overtime flag (period > 4)
    if "IS_OT" not in df.columns:
        if "PERIOD" in df.columns:
            df["IS_OT"] = (df["PERIOD"].astype(int) > 4).astype(int)
        else:
            df["IS_OT"] = 0

    # Home/away flag
    if "IS_HOME" not in df.columns:
        if "location" in df.columns:
            df["IS_HOME"] = (df["location"].astype(str).str.lower() == "h").astype(int)
        else:
            df["IS_HOME"] = 1

    # Polar coordinates: LOC_X / LOC_Y are tenths of a foot, hoop at origin (0, 0).
    # Distance in feet; angle in degrees (0 = straight on, positive = shooter's right).
    if "LOC_X" in df.columns and "LOC_Y" in df.columns:
        x = df["LOC_X"].astype(float)
        y = df["LOC_Y"].astype(float)
        df["LOC_X"] = x
        df["LOC_Y"] = y
        if "SHOT_DISTANCE_FT" not in df.columns:
            df["SHOT_DISTANCE_FT"] = np.hypot(x, y) / 10.0
        if "SHOT_ANGLE" not in df.columns:
            df["SHOT_ANGLE"] = np.degrees(np.arctan2(x, y))
    elif "SHOT_DISTANCE_FT" not in df.columns or "SHOT_ANGLE" not in df.columns:
        raise KeyError(
            "Either 'LOC_X' and 'LOC_Y' or 'SHOT_DISTANCE_FT' and 'SHOT_ANGLE' must be provided."
        )

    # Shot point value (2 or 3)
    if "SHOT_VALUE" not in df.columns:
        if "SHOT_TYPE" in df.columns:
            df["SHOT_VALUE"] = np.where(df["SHOT_TYPE"].astype(str).str.startswith("3"), 3, 2)
        else:
            df["SHOT_VALUE"] = 2

    # Categorical shot family
    if "ACTION_GROUP" not in df.columns:
        if "ACTION_TYPE" in df.columns:
            df["ACTION_GROUP"] = action_group(df["ACTION_TYPE"])
        else:
            df["ACTION_GROUP"] = pd.Series(
                pd.Categorical(["other"] * len(df), dtype=ACTION_DTYPE),
                index=df.index,
            )
    else:
        df["ACTION_GROUP"] = pd.Categorical(df["ACTION_GROUP"], dtype=ACTION_DTYPE)

    # Indicator for exact (0, 0) placeholder coordinate
    if "LOC_PLACEHOLDER" not in df.columns:
        if "LOC_X" in df.columns and "LOC_Y" in df.columns:
            df["LOC_PLACEHOLDER"] = ((df["LOC_X"] == 0) & (df["LOC_Y"] == 0)).astype(int)
        else:
            df["LOC_PLACEHOLDER"] = 0
    else:
        df["LOC_PLACEHOLDER"] = df["LOC_PLACEHOLDER"].astype(int)

    # Putback / tip attempt indicator
    if "IS_PUTBACK" not in df.columns:
        if "ACTION_TYPE" in df.columns:
            df["IS_PUTBACK"] = df["ACTION_TYPE"].astype(str).str.contains("tip|putback", case=False, na=False).astype(int)
        else:
            df["IS_PUTBACK"] = 0
    else:
        df["IS_PUTBACK"] = df["IS_PUTBACK"].astype(int)

    # Ensure required numeric columns are cast
    for col in NUMERIC:
        if col not in df.columns:
            raise KeyError(f"Missing required numeric feature: '{col}'")

    out_cols = [c for c in ID_COLS if c in df.columns] + FEATURES + ([TARGET] if TARGET in df.columns else [])
    return df[out_cols]


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on earlier seasons, test on the latest. One season: split by game."""
    seasons = sorted(df["SEASON"].unique())
    if len(seasons) >= 2:
        test = df["SEASON"] == seasons[-1]
        return df[~test], df[test]
    print("Only one season available: using a grouped-by-game 80/20 split instead.")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["GAME_ID"]))
    return df.iloc[train_idx], df.iloc[test_idx]


if __name__ == "__main__":
    feats = build_features(pd.read_parquet(SRC))
    feats.to_parquet(OUT, index=False)

    seasons = [int(s) for s in sorted(feats["SEASON"].unique())]
    print(f"rows: {len(feats):,} | seasons: {seasons}")
    print(f"FG%: {feats[TARGET].mean():.3f}")
    print("NaNs per feature:", int(feats[FEATURES].isna().sum().sum()))

    print("\nAction group breakdown (count & FG%):")
    ag_summary = feats.groupby("ACTION_GROUP", observed=False)[TARGET].agg(
        count="count",
        fg_pct="mean",
    )
    ag_summary["fg_pct"] = ag_summary["fg_pct"].round(3)
    print(ag_summary.sort_values(by="count", ascending=False))

    print("\nNumeric feature summary:")
    print(feats[NUMERIC].describe().round(2).T[["min", "mean", "max"]])

    train, test = split(feats)
    print(f"\ntrain: {len(train):,} | test: {len(test):,}")
