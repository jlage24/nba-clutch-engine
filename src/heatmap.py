"""Phase 4c — Step 3: Pure-Python half-court heatmap grid and evaluator.

Builds a 1 ft (10 tenths of a foot) grid over the NBA half court, applies the
exported 2 ft historical density mask (and optional shot-type threshold from
``export/court_defaults.json``), and evaluates ``p_make``, ``xp``, and ``v``
exclusively through ``src/decision_layer.py:evaluate_options`` (which uses
``src/simulator_model.py:predict_make_probability``).

No Streamlit imports and no dependency on ``data/`` or ``models/``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Literal, Optional, Union

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.court import draw_court  # noqa: E402
from src.decision_layer import GameState, evaluate_options  # noqa: E402
from src.export_court_defaults import assign_2ft_grid_indices  # noqa: E402
from src.shot_geometry import compute_shot_distance_and_angle, compute_shot_value  # noqa: E402
from src.simulator_model import DEFAULT_EXPORT_DIR, ExportedModel, load_export  # noqa: E402

DEFAULT_COURT_DEFAULTS_PATH = DEFAULT_EXPORT_DIR / "court_defaults.json"
TYPICAL_MIX_KEY = "typical_mix"
MIN_GROUP_WEIGHT_PER_1000 = 10
WEIGHT_SCALE = 1_000

HEATMAP_X_MIN = -250.0
HEATMAP_X_MAX = 250.0
HEATMAP_Y_MIN = -50.0
HEATMAP_Y_MAX = 420.0
HEATMAP_CELL_TENTHS = 10.0

ACTION_GROUP_LABELS: dict[str, str] = {
    "typical_mix": "Typical mix",
    "dunk": "Dunk",
    "layup": "Layup",
    "hook": "Hook",
    "floater": "Floater",
    "stepback": "Stepback",
    "pullup": "Pullup",
    "fadeaway": "Fadeaway",
    "jump_shot": "Jump shot",
}


@dataclass(frozen=True)
class CourtGrid:
    """1 ft resolution half-court coordinate grid and derived baseline geometry."""

    x_centers: np.ndarray
    y_centers: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    xx: np.ndarray
    yy: np.ndarray
    shot_distance_ft: np.ndarray
    shot_angle: np.ndarray
    shot_value: np.ndarray


@dataclass(frozen=True)
class HeatmapResult:
    """Evaluated half-court grids for make probability, xP, and end-state value V."""

    x_centers: np.ndarray
    y_centers: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    shot_value: np.ndarray
    mask: np.ndarray
    p_make: np.ndarray
    xp: np.ndarray
    v: np.ndarray
    shot_type: str
    smooth: bool
    gate_open: bool
    meta: dict[str, Any]


def load_court_defaults(
    path: Union[str, Path] = DEFAULT_COURT_DEFAULTS_PATH,
) -> dict[str, Any]:
    """Load exported typical-mix weights and 2 ft density mask from JSON."""
    defaults_path = Path(path)
    if not defaults_path.exists():
        raise FileNotFoundError(f"Court defaults file not found: {defaults_path}")
    return json.loads(defaults_path.read_text(encoding="utf-8"))


def get_active_shot_types(court_defaults: dict[str, Any]) -> list[str]:
    """Return active ACTION_GROUP names (non-zero share in REGULAR shots; excludes 'other')."""
    return list(court_defaults["meta"]["active_action_groups"])


def get_distance_weights(
    court_defaults: dict[str, Any],
    distance_ft: Union[int, float],
) -> dict[str, int]:
    """Look up integer weights (sum 1,000) for an integer distance in feet."""
    mix = court_defaults["action_group_mix"]
    key = str(int(distance_ft))
    if key in mix["by_distance_ft"]:
        return {k: int(v) for k, v in mix["by_distance_ft"][key]["weights_per_1000"].items()}
    return {k: int(v) for k, v in mix["pooled_ge_22ft"]["weights_per_1000"].items()}


def _validate_shot_type(shot_type: str, court_defaults: dict[str, Any]) -> str:
    active = get_active_shot_types(court_defaults)
    if shot_type == TYPICAL_MIX_KEY or shot_type in active:
        return shot_type
    allowed = [TYPICAL_MIX_KEY, *active]
    raise ValueError(f"Unknown shot_type {shot_type!r}. Expected one of {allowed}.")


def _normalize_zero(val: float) -> float:
    return 0.0 if val == 0.0 else float(val)


def build_court_grid() -> CourtGrid:
    """Build the 50 x 47 grid of 1 ft cells covering [-250, 250] x [-50, 420] tenths."""
    x_edges = np.arange(HEATMAP_X_MIN, HEATMAP_X_MAX + 1.0, HEATMAP_CELL_TENTHS)
    y_edges = np.arange(HEATMAP_Y_MIN, HEATMAP_Y_MAX + 1.0, HEATMAP_CELL_TENTHS)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
    dist, angle = compute_shot_distance_and_angle(xx, yy, is_putback=0, clamp_distance=True)
    shot_val = compute_shot_value(xx, yy)
    return CourtGrid(
        x_centers=x_centers,
        y_centers=y_centers,
        x_edges=x_edges,
        y_edges=y_edges,
        xx=xx,
        yy=yy,
        shot_distance_ft=dist,
        shot_angle=angle,
        shot_value=shot_val,
    )


def build_active_mask(
    grid: CourtGrid,
    court_defaults: dict[str, Any],
    shot_type: str = TYPICAL_MIX_KEY,
    smooth: bool = True,
) -> np.ndarray:
    """Build boolean mask (50, 47) combining 2 ft shot density and shot-type validity."""
    shot_type = _validate_shot_type(shot_type, court_defaults)
    dm = court_defaults["density_mask"]
    mask_2ft = np.asarray(dm["mask"], dtype=bool)
    x_edges_2ft = np.asarray(dm["x_edges_tenths"], dtype=float)
    y_edges_2ft = np.asarray(dm["y_edges_tenths"], dtype=float)

    ix2, iy2, in_box = assign_2ft_grid_indices(
        grid.xx.ravel(),
        grid.yy.ravel(),
        x_edges_2ft,
        y_edges_2ft,
    )
    nx, ny = grid.xx.shape
    base_flat = np.zeros(nx * ny, dtype=bool)
    base_flat[in_box] = mask_2ft[ix2[in_box], iy2[in_box]]
    base_mask = base_flat.reshape(nx, ny)

    if shot_type == TYPICAL_MIX_KEY:
        raw_mask = base_mask
    else:
        dists = grid.shot_distance_ft.astype(int)
        dist_cache: dict[int, bool] = {}
        type_mask = np.zeros((nx, ny), dtype=bool)
        for d in np.unique(dists):
            d_int = int(d)
            weights = get_distance_weights(court_defaults, d_int)
            dist_cache[d_int] = int(weights.get(shot_type, 0)) >= MIN_GROUP_WEIGHT_PER_1000
        for ix in range(nx):
            for iy in range(ny):
                if base_mask[ix, iy]:
                    type_mask[ix, iy] = dist_cache[int(dists[ix, iy])]
        raw_mask = type_mask

    if smooth:
        return raw_mask & raw_mask[::-1, :]
    return raw_mask


def build_custom_shot_preset(
    loc_x: float,
    loc_y: float,
    shot_type: str = TYPICAL_MIX_KEY,
    smooth: bool = True,
    court_defaults: Optional[dict[str, Any]] = None,
    label: str = "Custom Shot",
) -> dict[str, Any]:
    """Build a preset entry for ``evaluate_options`` at ``(loc_x, loc_y)``.

    For ``shot_type == 'typical_mix'``, rows repeat each active ``ACTION_GROUP``
    in proportion to its integer weight out of 1,000 at that distance (1,000 rows
    when ``smooth=False``; 2,000 rows across the point and its mirror image when
    ``smooth=True``). For a specific group, builds 1 row (or 2 mirrored rows).
    """
    defaults = court_defaults if court_defaults is not None else load_court_defaults()
    shot_type = _validate_shot_type(shot_type, defaults)
    active_groups = get_active_shot_types(defaults)

    x = _normalize_zero(float(loc_x))
    y = _normalize_zero(float(loc_y))
    dist, angle = compute_shot_distance_and_angle(x, y, is_putback=0, clamp_distance=True)
    shot_val = int(compute_shot_value(x, y))

    coords: list[tuple[float, float, float]] = [(x, float(dist), _normalize_zero(float(angle)))]
    if smooth:
        mx = _normalize_zero(-x)
        mdist, mang = compute_shot_distance_and_angle(mx, y, is_putback=0, clamp_distance=True)
        coords.append((mx, float(mdist), _normalize_zero(float(mang))))

    if shot_type == TYPICAL_MIX_KEY:
        weights = get_distance_weights(defaults, int(dist))
        group_repeats = [(ag, int(weights[ag])) for ag in active_groups if int(weights[ag]) > 0]
    else:
        group_repeats = [(shot_type, 1)]

    shots: list[dict[str, Any]] = []
    for cx, cdist, cang in coords:
        for ag, rep in group_repeats:
            row = {
                "LOC_X": cx,
                "LOC_Y": y,
                "SHOT_DISTANCE_FT": cdist,
                "SHOT_ANGLE": cang,
                "SHOT_VALUE": shot_val,
                "ACTION_GROUP": ag,
                "IS_PUTBACK": 0,
            }
            shots.extend([row] * rep)

    return {
        "label": label,
        "shot_value": shot_val,
        "shots": shots,
    }


def compute_heatmap(
    state: Union[GameState, dict[str, Any]],
    player_id: Union[int, str, None] = None,
    shift_mode: Literal["overall", "by_shot_type", "none"] = "overall",
    shot_type: str = TYPICAL_MIX_KEY,
    smooth: bool = True,
    time_gate_sec: float = 24.0,
    tie_value: float = 0.5,
    model: Optional[ExportedModel] = None,
    court_defaults: Optional[dict[str, Any]] = None,
) -> HeatmapResult:
    """Compute masked half-court heatmaps of make probability, xP, and V."""
    loaded_model = model if model is not None else load_export(DEFAULT_EXPORT_DIR)
    defaults = court_defaults if court_defaults is not None else load_court_defaults()
    shot_type = _validate_shot_type(shot_type, defaults)
    active_groups = get_active_shot_types(defaults)

    grid = build_court_grid()
    mask = build_active_mask(grid, defaults, shot_type=shot_type, smooth=smooth)
    nx, ny = grid.xx.shape

    p_make_grid = np.full((nx, ny), np.nan, dtype=float)
    xp_grid = np.full((nx, ny), np.nan, dtype=float)
    v_grid = np.full((nx, ny), np.nan, dtype=float)

    active_indices = np.argwhere(mask)
    if len(active_indices) == 0:
        dummy = {
            "label": "_dummy",
            "LOC_X": 0.0,
            "LOC_Y": 250.0,
            "SHOT_DISTANCE_FT": 25.0,
            "SHOT_ANGLE": 0.0,
            "SHOT_VALUE": 3,
            "ACTION_GROUP": "jump_shot",
            "IS_PUTBACK": 0,
        }
        dummy_res = evaluate_options(
            state=state,
            options=[dummy],
            player_id=player_id,
            shift_mode=shift_mode,
            time_gate_sec=time_gate_sec,
            tie_value=tie_value,
            model=loaded_model,
            presets={},
        )
        return HeatmapResult(
            x_centers=grid.x_centers,
            y_centers=grid.y_centers,
            x_edges=grid.x_edges,
            y_edges=grid.y_edges,
            shot_value=grid.shot_value,
            mask=mask,
            p_make=p_make_grid,
            xp=xp_grid,
            v=v_grid,
            shot_type=shot_type,
            smooth=smooth,
            gate_open=bool(dummy_res.meta["gate_open"]),
            meta=dummy_res.meta,
        )

    presets: dict[str, dict[str, Any]] = {}
    options: list[str] = []
    # Record cell -> [(option_idx, weight)] so weighted sums recover typical mix cleanly.
    cell_terms: list[tuple[int, int, list[tuple[int, int]]]] = []
    weight_cache: dict[int, dict[str, int]] = {}

    for ix, iy in active_indices:
        ix_int = int(ix)
        iy_int = int(iy)
        cx = _normalize_zero(float(grid.xx[ix_int, iy_int]))
        cy = _normalize_zero(float(grid.yy[ix_int, iy_int]))
        dist = float(grid.shot_distance_ft[ix_int, iy_int])
        ang = _normalize_zero(float(grid.shot_angle[ix_int, iy_int]))
        sv = int(grid.shot_value[ix_int, iy_int])

        if shot_type == TYPICAL_MIX_KEY:
            d_int = int(dist)
            if d_int not in weight_cache:
                weight_cache[d_int] = get_distance_weights(defaults, d_int)
            weights = weight_cache[d_int]
            terms_spec = [(ag, int(weights[ag])) for ag in active_groups if int(weights[ag]) > 0]
        else:
            terms_spec = [(shot_type, WEIGHT_SCALE)]

        terms: list[tuple[int, int]] = []
        for ag, w in terms_spec:
            opt_idx = len(options)
            key = f"{ix_int}_{iy_int}_{ag}"
            presets[key] = {
                "label": key,
                "shot_value": sv,
                "shots": [
                    {
                        "LOC_X": cx,
                        "LOC_Y": cy,
                        "SHOT_DISTANCE_FT": dist,
                        "SHOT_ANGLE": ang,
                        "SHOT_VALUE": sv,
                        "ACTION_GROUP": ag,
                        "IS_PUTBACK": 0,
                    }
                ],
            }
            options.append(key)
            terms.append((opt_idx, w))
        cell_terms.append((ix_int, iy_int, terms))

    res = evaluate_options(
        state=state,
        options=options,
        player_id=player_id,
        shift_mode=shift_mode,
        time_gate_sec=time_gate_sec,
        tie_value=tie_value,
        model=loaded_model,
        presets=presets,
    )

    p_vals = res.table["p_make"].to_numpy(dtype=float)
    xp_vals = res.table["xp"].to_numpy(dtype=float)
    v_vals = res.table["v"].to_numpy(dtype=float)
    gate_open = bool(res.meta["gate_open"])

    raw_p = np.full((nx, ny), np.nan, dtype=float)
    raw_xp = np.full((nx, ny), np.nan, dtype=float)
    raw_v = np.full((nx, ny), np.nan, dtype=float)

    for ix_int, iy_int, terms in cell_terms:
        p_sum = 0.0
        xp_sum = 0.0
        v_sum = 0.0
        for opt_idx, w in terms:
            p_sum += w * p_vals[opt_idx]
            xp_sum += w * xp_vals[opt_idx]
            if gate_open:
                v_sum += w * v_vals[opt_idx]
        raw_p[ix_int, iy_int] = p_sum / float(WEIGHT_SCALE)
        raw_xp[ix_int, iy_int] = xp_sum / float(WEIGHT_SCALE)
        if gate_open:
            raw_v[ix_int, iy_int] = v_sum / float(WEIGHT_SCALE)

    if smooth:
        p_make_grid = 0.5 * (raw_p + raw_p[::-1, :])
        xp_grid = 0.5 * (raw_xp + raw_xp[::-1, :])
        if gate_open:
            v_grid = 0.5 * (raw_v + raw_v[::-1, :])
    else:
        p_make_grid = raw_p
        xp_grid = raw_xp
        v_grid = raw_v

    return HeatmapResult(
        x_centers=grid.x_centers,
        y_centers=grid.y_centers,
        x_edges=grid.x_edges,
        y_edges=grid.y_edges,
        shot_value=grid.shot_value,
        mask=mask,
        p_make=p_make_grid,
        xp=xp_grid,
        v=v_grid,
        shot_type=shot_type,
        smooth=smooth,
        gate_open=gate_open,
        meta=res.meta,
    )


def lookup_cell(
    heatmap: HeatmapResult,
    loc_x: float,
    loc_y: float,
) -> dict[str, Any]:
    """Return nearest 1 ft cell coordinates, mask status, and evaluated metrics."""
    ix = int(np.argmin(np.abs(heatmap.x_centers - float(loc_x))))
    iy = int(np.argmin(np.abs(heatmap.y_centers - float(loc_y))))
    cx = float(heatmap.x_centers[ix])
    cy = float(heatmap.y_centers[iy])
    dist, angle = compute_shot_distance_and_angle(cx, cy, is_putback=0, clamp_distance=True)
    in_mask = bool(heatmap.mask[ix, iy])
    return {
        "ix": ix,
        "iy": iy,
        "cell_x": cx,
        "cell_y": cy,
        "shot_distance_ft": float(dist),
        "shot_angle": float(angle),
        "shot_value": int(heatmap.shot_value[ix, iy]),
        "in_mask": in_mask,
        "p_make": float(heatmap.p_make[ix, iy]) if in_mask else float("nan"),
        "xp": float(heatmap.xp[ix, iy]) if in_mask else float("nan"),
        "v": float(heatmap.v[ix, iy]) if (in_mask and heatmap.gate_open) else float("nan"),
    }


_METRIC_SPEC = {
    "p_make": ("Make probability", "0.0%", 0.15, 0.85),
    "xp": ("Expected points (xP)", "0.00", 0.50, 1.75),
    "v": ("Outcome score", "0.00", 0.00, 1.00),
}


def render_court_heatmap(
    heatmap: HeatmapResult,
    metric: Literal["p_make", "xp", "v"] = "v",
    highlight_xy: Optional[tuple[float, float]] = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (6.8, 5.8),
    bg_color: str = "#090f1a",
    court_color: str = "#cbd5e1",
    mask_color: str = "#162033",
    text_color: str = "#f1f5f9",
    highlight_color: str = "#cd3a29",
) -> matplotlib.figure.Figure:
    """Render a half-court heatmap figure using ``src/court.py:draw_court``."""
    if metric not in _METRIC_SPEC:
        raise ValueError(f"Unknown metric {metric!r}. Expected one of {tuple(_METRIC_SPEC)}.")
    if metric == "v" and not heatmap.gate_open:
        raise ValueError("Outcome score is unavailable when the decision gate is closed.")

    values = getattr(heatmap, metric)
    label, _, default_vmin, default_vmax = _METRIC_SPEC[metric]

    finite = values[np.isfinite(values)]
    if finite.size > 0:
        vmin = min(default_vmin, float(np.nanmin(finite)))
        vmax = max(default_vmax, float(np.nanmax(finite)))
        if metric == "v":
            vmin = max(0.0, float(np.nanmin(finite)))
            vmax = min(1.0, max(float(np.nanmax(finite)), vmin + 1e-3))
    else:
        vmin, vmax = default_vmin, default_vmax

    fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    ax.set_facecolor(bg_color)
    cmap_obj = plt.get_cmap(cmap).with_extremes(bad=mask_color)

    img = ax.imshow(
        values.T,
        origin="lower",
        extent=(
            float(heatmap.x_edges[0]),
            float(heatmap.x_edges[-1]),
            float(heatmap.y_edges[0]),
            float(heatmap.y_edges[-1]),
        ),
        interpolation="nearest",
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        zorder=0,
    )
    draw_court(ax=ax, color=court_color, lw=1.6)

    if highlight_xy is not None:
        hx, hy = highlight_xy
        ax.scatter(
            [hx],
            [hy],
            s=140,
            facecolors="none",
            edgecolors=text_color,
            linewidths=2.4,
            zorder=5,
        )
        ax.scatter(
            [hx],
            [hy],
            s=75,
            facecolors="none",
            edgecolors=highlight_color,
            linewidths=2.0,
            zorder=6,
        )
        ax.scatter([hx], [hy], s=18, color=highlight_color, zorder=7)

    cbar = fig.colorbar(img, ax=ax, fraction=0.042, pad=0.02)
    cbar.set_label(label, color=text_color)
    cbar.ax.yaxis.set_tick_params(color=text_color)
    plt.setp(cbar.ax.get_yticklabels(), color=text_color)
    cbar.outline.set_edgecolor(court_color)
    fig.tight_layout()
    return fig
