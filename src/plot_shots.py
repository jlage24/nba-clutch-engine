from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from court import draw_court

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data" / "processed" / "clutch_shots.parquet")

fig, ax = plt.subplots(figsize=(7.5, 6.5))
hb = ax.hexbin(
    df["LOC_X"], df["LOC_Y"], C=df["SHOT_MADE_FLAG"],
    reduce_C_function=np.mean,   # mean of made flags = FG% per hexagon
    gridsize=26, extent=(-250, 250, -50, 350),
    mincnt=8,                    # hide hexagons with too few shots to trust
    cmap="RdYlGn", vmin=0.25, vmax=0.70, linewidths=0.2, edgecolors="white",
)
draw_court(ax)
cbar = fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("FG%")
ax.set_title(f"Clutch FG% by location (n={len(df):,}, hexes with ≥ 8 shots)")

out = ROOT / "docs"
out.mkdir(exist_ok=True)
fig.savefig(out / "clutch_shot_map.png", dpi=200, bbox_inches="tight")
plt.show()