import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playbyplayv3

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PBP_DIR = DATA / "raw" / "pbp"
PBP_DIR.mkdir(parents=True, exist_ok=True)


def fetch_game(game_id: str, retries: int = 3) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        try:
            df = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=60).get_data_frames()[0]
            if df.empty:
                raise ValueError("empty play-by-play")
            return df
        except Exception as e:
            wait = 5 * attempt
            print(f"  {game_id} attempt {attempt} failed ({e!r}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Could not fetch {game_id}")


def main(limit: int | None):
    shots = pd.concat(
        [pd.read_parquet(p) for p in sorted(DATA.glob("late_game_shots_*.parquet"))],
        ignore_index=True,
    )
    game_ids = sorted(shots["GAME_ID"].astype(str).unique())
    if limit:
        game_ids = game_ids[:limit]
    print(f"{len(game_ids)} games to process")

    for i, gid in enumerate(game_ids, 1):
        out = PBP_DIR / f"{gid}.parquet"
        if out.exists():
            continue
        print(f"[{i}/{len(game_ids)}] {gid}")
        fetch_game(gid).to_parquet(out, index=False)
        time.sleep(1.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only fetch the first N games")
    main(ap.parse_args().limit)