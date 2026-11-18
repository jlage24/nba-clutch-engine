"""Build the all-shots dataset for Phase 3b.

Loads every per-team raw shot file, attaches pre-shot score margin and shooter
side from the play-by-play (reusing ``add_pre_event_score`` from
``build_clutch_dataset``), labels each shot as REGULAR / LATE_BLOWOUT / CLUTCH,
builds baseline features, and writes ``data/processed/all_shots.parquet``.

Run from the repo root::

    python src/build_all_shots_dataset.py

Definitions (from Phase 3b brief §5):
  TIME_REMAINING_SEC = MINUTES_REMAINING * 60 + SECONDS_REMAINING
  LATE_WINDOW        = (PERIOD >= 4) and (TIME_REMAINING_SEC <= 300)
  CLUTCH             = LATE_WINDOW and abs(SCORE_MARGIN) <= 5
  LATE_BLOWOUT       = LATE_WINDOW and abs(SCORE_MARGIN) > 5
  REGULAR            = not LATE_WINDOW

Expected totals: 219,160 total | 196,198 REGULAR | 15,858 LATE_BLOWOUT | 7,104 CLUTCH

Unmatched-shot policy
---------------------
One shot (GAME_ID=0022500792, GAME_EVENT_ID=403, Jabari Smith Jr., Period 3,
4:46 remaining, Running Layup, missed) has no matching field-goal event in the
play-by-play.  The PBP jumps from actionNumber 401 to 405 in that game,
possibly because the event was corrected or deleted after the fact by the NBA
stats provider and the gap was left in the shot-chart feed.

Because the shot is in Period 3 it is REGULAR by clock alone (LATE_WINDOW
requires PERIOD >= 4); SCORE_MARGIN is only needed to split LATE_WINDOW shots
into CLUTCH vs LATE_BLOWOUT.  We therefore retain the shot as REGULAR with
SCORE_MARGIN = NaN, preserving the 219,160 total.  An assertion in
``assert_unmatched_are_regular`` enforces that any future unmatched shot is
also REGULAR; if an unmatched shot were ever inside LATE_WINDOW (where margin
matters for labelling), the pipeline raises an error rather than silently
assigning a wrong label.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.build_clutch_dataset import add_pre_event_score  # noqa: E402
from src.features import build_features  # noqa: E402

DATA = ROOT / "data"
PBP_DIR = DATA / "raw" / "pbp"
OUT_DIR = DATA / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Expected counts from the brief (§2 and §5)
EXPECTED_TOTAL = 219_160
EXPECTED_CLUTCH = 7_104
EXPECTED_REGULAR = 196_198
EXPECTED_LATE_BLOWOUT = 15_858


# ---------------------------------------------------------------------------
# Step 1.1 — Load all per-team raw shot files
# ---------------------------------------------------------------------------

def load_all_shots() -> pd.DataFrame:
    """Concatenate all per-team raw shot files into one DataFrame.

    Files are named ``raw_shots_<season>_<TEAM>.parquet`` under ``data/``.
    GAME_ID is kept as a string throughout (brief §3.8).
    TIME_REMAINING_SEC is derived from MINUTES_REMAINING / SECONDS_REMAINING
    if not already present.
    """
    files = sorted(DATA.glob("raw_shots_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No raw_shots_*.parquet files found under {DATA}. "
            "Run the fetch scripts first."
        )
    frames: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_parquet(path)
        df["GAME_ID"] = df["GAME_ID"].astype(str)
        frames.append(df)
    shots = pd.concat(frames, ignore_index=True)

    # Derive TIME_REMAINING_SEC if not already present
    if "TIME_REMAINING_SEC" not in shots.columns:
        shots["TIME_REMAINING_SEC"] = (
            shots["MINUTES_REMAINING"].astype(float) * 60
            + shots["SECONDS_REMAINING"].astype(float)
        )

    # Derive SEASON from GAME_ID (e.g. '0022500014' -> 2025)
    shots["SEASON"] = 2000 + shots["GAME_ID"].str[3:5].astype(int)

    return shots


# ---------------------------------------------------------------------------
# Step 1.3 — Attach pre-shot SCORE_MARGIN and shooter side from PBP
# ---------------------------------------------------------------------------

def load_pbp_events() -> pd.DataFrame:
    """Load all PBP field-goal events and compute pre-shot scores.

    Reuses ``add_pre_event_score`` exactly as in ``build_clutch_dataset``.
    Only field-goal rows are returned; non-FG rows are dropped after scoring.
    """
    frames: list[pd.DataFrame] = []
    pbp_files = sorted(PBP_DIR.glob("*.parquet"))
    if not pbp_files:
        raise FileNotFoundError(f"No PBP parquet files found under {PBP_DIR}.")
    for path in pbp_files:
        pbp = add_pre_event_score(pd.read_parquet(path))
        fg = pbp[pbp["isFieldGoal"] == 1]
        frames.append(
            fg[["gameId", "actionNumber", "location",
                "pre_home", "pre_away", "xLegacy", "yLegacy"]]
        )
    return pd.concat(frames, ignore_index=True)


def merge_with_pbp(shots: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Left-join shots to PBP events; compute SCORE_MARGIN on matched rows.

    Unmatched shots (those with no corresponding PBP field-goal entry) receive
    SCORE_MARGIN = NaN and location = NaN.  The caller must subsequently call
    ``assert_unmatched_are_regular`` to guarantee no unmatched shot falls
    inside LATE_WINDOW (where margin is needed for CLUTCH/LATE_BLOWOUT
    classification).

    Margin is from the shooter's perspective (positive = shooter's team ahead).
    """
    merged = shots.merge(
        events,
        left_on=["GAME_ID", "GAME_EVENT_ID"],
        right_on=["gameId", "actionNumber"],
        how="left",
        indicator=True,
    )

    is_matched = merged["_merge"] == "both"
    is_home = merged["location"] == "h"

    # Compute SCORE_MARGIN only for matched rows; NaN elsewhere
    merged["SCORE_MARGIN"] = np.where(
        is_matched & is_home,
        merged["pre_home"] - merged["pre_away"],
        np.where(
            is_matched & ~is_home,
            merged["pre_away"] - merged["pre_home"],
            np.nan,
        ),
    )
    return merged


# ---------------------------------------------------------------------------
# Step 1.4 — Validation helpers
# ---------------------------------------------------------------------------

def validate_merge_stats(merged: pd.DataFrame) -> dict[str, object]:
    """Compute PBP match-rate and coordinate-agreement statistics.

    Returns a dict with counts and the DataFrame of coordinate-disagreeing
    shots (empty if none) for further inspection.
    """
    n_total = len(merged)
    n_matched = (merged["_merge"] == "both").sum()
    n_unmatched = n_total - n_matched
    match_rate = n_matched / n_total

    both = merged[merged["_merge"] == "both"]
    coord_ok = (both["LOC_X"] == both["xLegacy"]) & (both["LOC_Y"] == both["yLegacy"])
    coord_match_rate = coord_ok.mean()
    coord_disagree = both[~coord_ok].copy()

    unmatched_rows = merged[merged["_merge"] != "both"][
        ["GAME_ID", "GAME_EVENT_ID", "PLAYER_NAME", "PERIOD",
         "MINUTES_REMAINING", "SECONDS_REMAINING", "ACTION_TYPE",
         "SHOT_MADE_FLAG", "TIME_REMAINING_SEC"]
    ].copy()

    return {
        "n_total": n_total,
        "n_matched": int(n_matched),
        "n_unmatched": int(n_unmatched),
        "match_rate": float(match_rate),
        "coord_match_rate": float(coord_match_rate),
        "n_coord_disagree": int(len(coord_disagree)),
        "coord_disagree_df": coord_disagree,
        "unmatched_df": unmatched_rows,
    }


def assert_unmatched_are_regular(merged: pd.DataFrame) -> None:
    """Fail loudly if any unmatched shot falls inside LATE_WINDOW.

    An unmatched shot has SCORE_MARGIN = NaN, so it cannot be classified as
    CLUTCH or LATE_BLOWOUT.  If such a shot were inside LATE_WINDOW it would
    be silently dropped from both sets, corrupting the counts.  This function
    raises ValueError before that can happen.

    Callers must invoke this *before* label_shots() so the assertion fires
    while the _merge indicator column is still present.
    """
    unmatched = merged[merged["_merge"] != "both"]
    if unmatched.empty:
        return
    # LATE_WINDOW: PERIOD >= 4 AND TIME_REMAINING_SEC <= 300
    bad = unmatched[
        (unmatched["PERIOD"].astype(int) >= 4) &
        (unmatched["TIME_REMAINING_SEC"] <= 300)
    ]
    if not bad.empty:
        details = bad[["GAME_ID", "GAME_EVENT_ID", "PLAYER_NAME",
                        "PERIOD", "TIME_REMAINING_SEC"]].to_string(index=False)
        raise ValueError(
            f"STOP: {len(bad)} unmatched shot(s) fall inside LATE_WINDOW where "
            f"SCORE_MARGIN is required for CLUTCH/LATE_BLOWOUT classification.\n"
            f"These shots cannot be correctly labelled:\n{details}\n"
            "Investigate the PBP data before continuing."
        )


def inspect_coord_disagreements(merged: pd.DataFrame, coord_disagree_df: pd.DataFrame) -> None:
    """Print full details of shots where LOC_X/LOC_Y differs from xLegacy/yLegacy.

    For each disagreeing shot prints: GAME_ID, GAME_EVENT_ID, set label
    (requires LATE_WINDOW to already be labelled on merged, so we compute it
    inline from PERIOD and TIME_REMAINING_SEC), PERIOD, clock, both coordinate
    pairs, and the delta.  Also classifies each as a small correction or a
    probable wrong pairing (|delta| > 50 tenths-of-foot ≈ 5 ft).
    """
    if coord_disagree_df.empty:
        print("    No coordinate disagreements.")
        return

    df = coord_disagree_df.copy()

    # Compute set label inline (SCORE_MARGIN already present from merge_with_pbp)
    df["_LATE_WINDOW"] = (df["PERIOD"].astype(int) >= 4) & (df["TIME_REMAINING_SEC"] <= 300)
    df["_CLUTCH"] = df["_LATE_WINDOW"] & (df["SCORE_MARGIN"].abs() <= 5)
    df["_LATE_BLOWOUT"] = df["_LATE_WINDOW"] & (df["SCORE_MARGIN"].abs() > 5)

    def _set_label(row: pd.Series) -> str:
        if row["_CLUTCH"]:
            return "CLUTCH"
        if row["_LATE_BLOWOUT"]:
            return "LATE_BLOWOUT"
        if row["_LATE_WINDOW"]:
            return "LATE_WINDOW(margin=NaN?)"
        return "REGULAR"

    df["SET"] = df.apply(_set_label, axis=1)

    df["delta_x"] = df["LOC_X"] - df["xLegacy"]
    df["delta_y"] = df["LOC_Y"] - df["yLegacy"]
    df["delta_dist"] = np.hypot(df["delta_x"], df["delta_y"])
    # delta_dist is in tenths of a foot; > 50 (= 5 ft) suggests wrong pairing
    df["classification"] = np.where(
        df["delta_dist"] > 50,
        "PROBABLE_MISPAIRING",
        "small_correction",
    )

    print(f"\n    Coordinate disagreements: {len(df)} shots")
    cols = ["GAME_ID", "GAME_EVENT_ID", "SET", "PERIOD", "TIME_REMAINING_SEC",
            "LOC_X", "LOC_Y", "xLegacy", "yLegacy", "delta_x", "delta_y",
            "delta_dist", "classification"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))

    n_late = int(df["_LATE_WINDOW"].sum())
    n_clutch = int(df["_CLUTCH"].sum())
    n_blowout = int(df["_LATE_BLOWOUT"].sum())
    n_mispairing = int((df["classification"] == "PROBABLE_MISPAIRING").sum())
    print(f"\n    Summary: {n_late} in LATE_WINDOW  |  {n_clutch} CLUTCH  |  "
          f"{n_blowout} LATE_BLOWOUT  |  {n_mispairing} probable mispairings")

    bad_clutch = df[(df["_CLUTCH"] | df["_LATE_BLOWOUT"]) &
                    (df["classification"] == "PROBABLE_MISPAIRING")]
    if not bad_clutch.empty:
        raise ValueError(
            f"STOP: {len(bad_clutch)} probable coordinate mispairing(s) in "
            f"CLUTCH or LATE_BLOWOUT shots. Investigate before continuing.\n"
            f"{bad_clutch[['GAME_ID','GAME_EVENT_ID','SET','delta_dist']].to_string(index=False)}"
        )


def validate_score_changes(shots: pd.DataFrame) -> dict[str, object]:
    """Check that made FGs produce the expected score change in PBP.

    Loads all PBP files with full score columns, joins on (GAME_ID,
    GAME_EVENT_ID), and counts how many made FGs have a score-change
    mismatch vs the shot's SHOT_VALUE.

    The ``shots`` DataFrame passed here should already have SHOT_DISTANCE_FT
    (renamed from SHOT_DISTANCE) and SCORE_MARGIN attached; it is the full
    219,160-row table (including the 1 unmatched shot with NaN SCORE_MARGIN).
    """
    frames: list[pd.DataFrame] = []
    for path in sorted(PBP_DIR.glob("*.parquet")):
        pbp = pd.read_parquet(path)
        # Sort by actionId (chronological) to get correct post-event scores
        pbp = pbp.sort_values("actionId").copy()
        for col in ("scoreHome", "scoreAway"):
            pbp[col] = pd.to_numeric(pbp[col], errors="coerce")
        pbp["post_home"] = pbp["scoreHome"].ffill().fillna(0)
        pbp["post_away"] = pbp["scoreAway"].ffill().fillna(0)
        pbp["pre_home_sc"] = pbp["post_home"].shift(1, fill_value=0)
        pbp["pre_away_sc"] = pbp["post_away"].shift(1, fill_value=0)
        fg = pbp[pbp["isFieldGoal"] == 1].copy()
        fg["score_delta"] = np.where(
            fg["location"] == "h",
            fg["post_home"] - fg["pre_home_sc"],
            fg["post_away"] - fg["pre_away_sc"],
        )
        frames.append(
            fg[["gameId", "actionNumber", "shotValue", "score_delta", "shotResult"]]
        )
    pbp_fg = pd.concat(frames, ignore_index=True)

    # Check only made field goals (SHOT_MADE_FLAG == 1)
    made = shots[shots["SHOT_MADE_FLAG"] == 1].copy()
    check = made.merge(
        pbp_fg,
        left_on=["GAME_ID", "GAME_EVENT_ID"],
        right_on=["gameId", "actionNumber"],
        how="left",
    )
    # Mismatch: score_delta != SHOT_VALUE.
    # Note: and-1s can legitimately produce delta == SHOT_VALUE + 1 at the
    # free-throw event, not the FG event; we still flag delta != SHOT_VALUE.
    checked = check[check["score_delta"].notna()]
    mismatch = checked[checked["score_delta"] != checked["SHOT_VALUE"]]

    mismatch_details: list[dict] = []
    if not mismatch.empty:
        for _, row in mismatch.iterrows():
            mismatch_details.append({
                "GAME_ID": row["GAME_ID"],
                "GAME_EVENT_ID": int(row["GAME_EVENT_ID"]),
                "SHOT_VALUE": int(row["SHOT_VALUE"]),
                "score_delta": row["score_delta"],
            })

    return {
        "n_made_fg": int(len(made)),
        "n_checked": int(len(checked)),
        "n_unmatched_made": int(len(check) - len(checked)),
        "n_mismatches": int(len(mismatch)),
        "mismatch_game_ids": sorted(mismatch["GAME_ID"].unique().tolist()),
        "mismatch_details": mismatch_details[:20],
    }


# ---------------------------------------------------------------------------
# Step 1.5 — Label shots and build baseline features
# ---------------------------------------------------------------------------

def label_shots(shots: pd.DataFrame) -> pd.DataFrame:
    """Add LATE_WINDOW, CLUTCH, LATE_BLOWOUT, REGULAR boolean columns.

    SCORE_MARGIN may be NaN for shots that had no PBP match.  NaN comparisons
    evaluate to False in pandas, so:
    - A NaN-margin shot inside LATE_WINDOW is neither CLUTCH nor LATE_BLOWOUT
      (both False), and LATE_WINDOW is True but REGULAR is False.
    - ``assert_unmatched_are_regular`` must be called before this function to
      guarantee no LATE_WINDOW shot has a NaN margin; after that check passes,
      NaN margins only occur on REGULAR shots.
    """
    shots = shots.copy()
    shots["LATE_WINDOW"] = (shots["PERIOD"].astype(int) >= 4) & (
        shots["TIME_REMAINING_SEC"] <= 300
    )
    # NaN margin propagates as False in both comparisons (correct for REGULAR)
    shots["CLUTCH"] = shots["LATE_WINDOW"] & (shots["SCORE_MARGIN"].abs() <= 5)
    shots["LATE_BLOWOUT"] = shots["LATE_WINDOW"] & (shots["SCORE_MARGIN"].abs() > 5)
    shots["REGULAR"] = ~shots["LATE_WINDOW"]
    return shots


def add_baseline_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Build baseline features using existing ``build_features`` logic.

    The baseline feature set (brief §5) is shot-context only:
      LOC_X, LOC_Y, SHOT_DISTANCE_FT, SHOT_ANGLE, SHOT_VALUE,
      ACTION_GROUP, IS_PUTBACK

    Game-state features (TIME_REMAINING_SEC, SCORE_MARGIN, IS_OT, IS_HOME,
    period, player/team IDs) are deliberately excluded from the model inputs
    but kept as metadata columns.  build_features preserves them as ID_COLS.

    NaN values in SCORE_MARGIN are metadata-only and not used as model features,
    so they do not affect build_features output.
    """
    return build_features(shots)


# ---------------------------------------------------------------------------
# Step 1.6 — Critical validation against existing clutch_shots.parquet
# ---------------------------------------------------------------------------

def validate_against_existing_clutch(
    all_shots: pd.DataFrame,
    clutch_path: Path,
) -> dict[str, object]:
    """Compare the CLUTCH subset of all_shots with the existing clutch table.

    Requirements (brief §7 step 6):
    - Exactly 7,104 rows
    - Same (GAME_ID, GAME_EVENT_ID) set
    - Identical SCORE_MARGIN values
    """
    existing = pd.read_parquet(clutch_path)
    existing["GAME_ID"] = existing["GAME_ID"].astype(str)

    new_clutch = all_shots[all_shots["CLUTCH"]].copy()

    n_new = int(len(new_clutch))
    n_existing = int(len(existing))

    new_keys = set(zip(new_clutch["GAME_ID"], new_clutch["GAME_EVENT_ID"]))
    existing_keys = set(zip(existing["GAME_ID"], existing["GAME_EVENT_ID"]))

    keys_match = new_keys == existing_keys
    only_in_new = new_keys - existing_keys
    only_in_existing = existing_keys - new_keys

    # SCORE_MARGIN agreement on common keys
    common = new_clutch.merge(
        existing[["GAME_ID", "GAME_EVENT_ID", "SCORE_MARGIN"]],
        on=["GAME_ID", "GAME_EVENT_ID"],
        suffixes=("_new", "_existing"),
        how="inner",
    )
    margin_matches = bool(
        (common["SCORE_MARGIN_new"] == common["SCORE_MARGIN_existing"]).all()
    )
    margin_mismatches = int(
        (common["SCORE_MARGIN_new"] != common["SCORE_MARGIN_existing"]).sum()
    )

    return {
        "n_new_clutch": n_new,
        "n_existing_clutch": n_existing,
        "counts_match": n_new == EXPECTED_CLUTCH and n_existing == EXPECTED_CLUTCH,
        "keys_match": keys_match,
        "only_in_new": int(len(only_in_new)),
        "only_in_existing": int(len(only_in_existing)),
        "margin_matches": margin_matches,
        "margin_mismatches": margin_mismatches,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    print("=" * 60)
    print("Phase 3b — Step 1: build_all_shots_dataset")
    print("=" * 60)

    # 1.1 Load all shots
    print("\n[1] Loading per-team shot files …")
    shots = load_all_shots()
    print(f"    Loaded {len(shots):,} shots from {len(sorted(DATA.glob('raw_shots_*.parquet')))} files")

    # 1.2 Row-count and duplicate check
    print("\n[2] Checking row count and duplicates …")
    n_total = len(shots)
    n_dupes = shots.duplicated(["GAME_ID", "GAME_EVENT_ID"]).sum()
    print(f"    Total rows   : {n_total:,}  (expected {EXPECTED_TOTAL:,})")
    print(f"    Duplicates   : {n_dupes:,}  (expected 0)")
    if n_total != EXPECTED_TOTAL:
        raise ValueError(
            f"STOP: total shot count {n_total:,} != expected {EXPECTED_TOTAL:,}."
        )
    if n_dupes != 0:
        raise ValueError(
            f"STOP: {n_dupes:,} duplicate (GAME_ID, GAME_EVENT_ID) pairs found."
        )

    # Rename SHOT_DISTANCE -> SHOT_DISTANCE_FT before merge so it is consistent
    if "SHOT_DISTANCE" in shots.columns and "SHOT_DISTANCE_FT" not in shots.columns:
        shots = shots.rename(columns={"SHOT_DISTANCE": "SHOT_DISTANCE_FT"})

    # 1.3 Attach pre-shot SCORE_MARGIN from PBP
    print("\n[3] Loading play-by-play and computing pre-shot scores …")
    events = load_pbp_events()
    print(f"    PBP field-goal events loaded: {len(events):,}")
    merged = merge_with_pbp(shots, events)

    # 1.4 Validate merge quality
    print("\n[4] Validating PBP merge …")
    val = validate_merge_stats(merged)
    print(f"    Total shots  : {val['n_total']:,}")
    print(f"    Matched      : {val['n_matched']:,}  ({val['match_rate']:.6%})")
    print(f"    Unmatched    : {val['n_unmatched']:,}  (kept as REGULAR with SCORE_MARGIN=NaN)")
    print(f"    Coord agree  : {val['coord_match_rate']:.6%}  ({val['n_coord_disagree']} disagreements)")

    if val["n_unmatched"] > 0:
        print("\n    Unmatched shot(s) — possibly a gap from a corrected/deleted PBP event:")
        print(val["unmatched_df"].to_string(index=False))

    # Safety: assert all unmatched shots are REGULAR (PERIOD < 4 or time > 300)
    # Must be called before label_shots so we still have _merge column
    assert_unmatched_are_regular(merged)
    print("\n    ✓ All unmatched shots are REGULAR (no LATE_WINDOW shot is missing a margin).")

    # 1.4b Coordinate-disagreement inspection
    print("\n[4b] Inspecting coordinate disagreements …")
    inspect_coord_disagreements(merged, val["coord_disagree_df"])

    # 1.5 Label shots
    print("\n[5] Labelling shots (LATE_WINDOW / CLUTCH / LATE_BLOWOUT / REGULAR) …")
    labelled = label_shots(merged)
    n_regular = int(labelled["REGULAR"].sum())
    n_clutch = int(labelled["CLUTCH"].sum())
    n_late_blowout = int(labelled["LATE_BLOWOUT"].sum())
    n_late_window = int(labelled["LATE_WINDOW"].sum())
    print(f"    REGULAR      : {n_regular:,}  (expected {EXPECTED_REGULAR:,})")
    print(f"    CLUTCH       : {n_clutch:,}  (expected {EXPECTED_CLUTCH:,})")
    print(f"    LATE_BLOWOUT : {n_late_blowout:,}  (expected {EXPECTED_LATE_BLOWOUT:,})")
    print(f"    LATE_WINDOW  : {n_late_window:,}  (= CLUTCH + LATE_BLOWOUT = {n_clutch + n_late_blowout:,})")
    print(f"    Sum check    : {n_regular + n_clutch + n_late_blowout:,}  (expected {EXPECTED_TOTAL:,})")

    if n_regular != EXPECTED_REGULAR:
        raise ValueError(
            f"STOP: REGULAR count {n_regular:,} != expected {EXPECTED_REGULAR:,}."
        )
    if n_clutch != EXPECTED_CLUTCH:
        raise ValueError(
            f"STOP: CLUTCH count {n_clutch:,} != expected {EXPECTED_CLUTCH:,}."
        )
    if n_late_blowout != EXPECTED_LATE_BLOWOUT:
        raise ValueError(
            f"STOP: LATE_BLOWOUT count {n_late_blowout:,} != expected {EXPECTED_LATE_BLOWOUT:,}."
        )

    # Disjoint check
    assert not (labelled["CLUTCH"] & labelled["LATE_BLOWOUT"]).any(), "CLUTCH and LATE_BLOWOUT overlap!"
    assert not (labelled["CLUTCH"] & labelled["REGULAR"]).any(), "CLUTCH and REGULAR overlap!"
    assert not (labelled["LATE_BLOWOUT"] & labelled["REGULAR"]).any(), "LATE_BLOWOUT and REGULAR overlap!"
    print("    ✓ Sets are disjoint and exhaustive.")

    # 1.5b Build baseline features
    print("\n[5b] Building baseline features …")
    featured = add_baseline_features(labelled)
    # Re-attach set labels and metadata that build_features does not preserve
    for col in ["LATE_WINDOW", "CLUTCH", "LATE_BLOWOUT", "REGULAR", "SCORE_MARGIN",
                "PERIOD", "TIME_REMAINING_SEC", "_merge"]:
        if col in labelled.columns and col not in featured.columns:
            featured = featured.copy()
            featured[col] = labelled[col].values
    # Drop the internal merge indicator before saving
    featured = featured.drop(columns=["_merge"], errors="ignore")
    print(f"    Columns in output: {list(featured.columns)}")

    # 1.4c Score-change validation (needs SHOT_DISTANCE_FT and SCORE_MARGIN)
    print("\n[4c] Validating made-FG score changes …")
    sc = validate_score_changes(featured)
    print(f"    Made FGs          : {sc['n_made_fg']:,}")
    print(f"    Matched to PBP    : {sc['n_checked']:,}  (unmatched: {sc['n_unmatched_made']})")
    print(f"    Score mismatches  : {sc['n_mismatches']:,}  (earlier Phase 2 found 12 in 103,227)")
    if sc["n_mismatches"] > 0:
        print(f"    Mismatch game IDs : {sc['mismatch_game_ids']}")
        print("    Mismatch details  :")
        for d in sc["mismatch_details"]:
            print(f"      {d}")

    # 1.6 Critical validation against existing clutch_shots.parquet
    print("\n[6] Critical validation against existing clutch_shots.parquet …")
    clutch_path = OUT_DIR / "clutch_shots.parquet"
    cv = validate_against_existing_clutch(featured, clutch_path)
    print(f"    New CLUTCH rows      : {cv['n_new_clutch']:,}")
    print(f"    Existing CLUTCH rows : {cv['n_existing_clutch']:,}")
    print(f"    Counts match         : {cv['counts_match']}")
    print(f"    Keys match exactly   : {cv['keys_match']}")
    print(f"    Only in new          : {cv['only_in_new']}")
    print(f"    Only in existing     : {cv['only_in_existing']}")
    print(f"    SCORE_MARGIN agrees  : {cv['margin_matches']}  (mismatches: {cv['margin_mismatches']})")

    if not cv["counts_match"]:
        raise ValueError(
            f"STOP: CLUTCH count — new {cv['n_new_clutch']:,} vs existing {cv['n_existing_clutch']:,}."
        )
    if not cv["keys_match"]:
        raise ValueError(
            f"STOP: CLUTCH key sets differ — "
            f"{cv['only_in_new']} only in new, {cv['only_in_existing']} only in existing."
        )
    if not cv["margin_matches"]:
        raise ValueError(
            f"STOP: SCORE_MARGIN disagreement on {cv['margin_mismatches']} clutch shots."
        )
    print("    ✓ CLUTCH subset matches existing clutch_shots.parquet exactly.")

    # 1.7 Write output
    print("\n[7] Writing data/processed/all_shots.parquet …")
    out_path = OUT_DIR / "all_shots.parquet"
    featured.to_parquet(out_path, index=False)
    print(f"    Written {len(featured):,} rows → {out_path}")
    print("\n✓ Step 1 complete.")


if __name__ == "__main__":
    main()
