"""Geometry helpers turning NBA court coordinates into model baseline features.

NBA shot-chart coordinates ``(LOC_X, LOC_Y)`` are measured in tenths of a foot
with the hoop at the origin ``(0, 0)``, positive ``LOC_X`` to the shooter's
right, and positive ``LOC_Y`` toward half-court.

Conventions verified against all 219,160 shots in ``data/processed/all_shots.parquet``:
- ``SHOT_DISTANCE_FT`` in the API feed equals ``floor(hypot(LOC_X, LOC_Y) / 10)``
  (100.0% exact agreement). Non-putback shots below 1 ft are clamped to 1.0 ft
  to match ``src/simulator_model.py`` and ``src/predict.py``.
- ``SHOT_ANGLE`` equals ``degrees(arctan2(LOC_X, LOC_Y))`` (100.0% exact agreement).
- ``SHOT_VALUE`` follows the NBA three-point boundary: ``22.0`` ft (220 tenths)
  in the corners up to the corner-arc intersection ``y = sqrt(237.5^2 - 220^2)``,
  and ``23.75`` ft (237.5 tenths) around the arc.
"""

from __future__ import annotations

import math
from typing import Union, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

CORNER_THREE_X_TENTHS = 220.0
ARC_THREE_RADIUS_TENTHS = 237.5
CORNER_ARC_INTERSECT_Y_TENTHS = math.sqrt(
    ARC_THREE_RADIUS_TENTHS**2 - CORNER_THREE_X_TENTHS**2
)

Scalar = Union[int, float, np.number]


def _is_scalar(value: object) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)


@overload
def compute_shot_distance_and_angle(
    loc_x: Scalar,
    loc_y: Scalar,
    is_putback: Union[int, bool, np.integer] = 0,
    clamp_distance: bool = True,
) -> tuple[float, float]: ...


@overload
def compute_shot_distance_and_angle(
    loc_x: ArrayLike,
    loc_y: ArrayLike,
    is_putback: Union[int, bool, ArrayLike] = 0,
    clamp_distance: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


def compute_shot_distance_and_angle(
    loc_x: Union[Scalar, ArrayLike],
    loc_y: Union[Scalar, ArrayLike],
    is_putback: Union[int, bool, ArrayLike] = 0,
    clamp_distance: bool = True,
) -> Union[tuple[float, float], tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Compute ``(SHOT_DISTANCE_FT, SHOT_ANGLE)`` from ``(LOC_X, LOC_Y)``.

    Uses truncated integer feet ``floor(hypot(LOC_X, LOC_Y) / 10.0)`` to match
    the NBA API ``SHOT_DISTANCE`` stored in ``all_shots.parquet``, and clamps
    non-putback shots with distance ``< 1.0`` ft to ``1.0`` ft when
    ``clamp_distance=True``.
    """
    scalar_input = _is_scalar(loc_x) and _is_scalar(loc_y) and _is_scalar(is_putback)
    x = np.asarray(loc_x, dtype=float)
    y = np.asarray(loc_y, dtype=float)
    putback = np.asarray(is_putback, dtype=int)

    dist = np.floor(np.hypot(x, y) / 10.0).astype(float)
    if clamp_distance:
        dist = np.where((putback == 0) & (dist < 1.0), 1.0, dist)

    angle = np.degrees(np.arctan2(x, y)).astype(float)
    angle = np.where((x == 0.0) & (y == 0.0), 0.0, angle)

    if scalar_input:
        return float(dist.item()), float(angle.item())
    return dist, angle


@overload
def compute_shot_value(loc_x: Scalar, loc_y: Scalar) -> int: ...


@overload
def compute_shot_value(loc_x: ArrayLike, loc_y: ArrayLike) -> NDArray[np.int64]: ...


def compute_shot_value(
    loc_x: Union[Scalar, ArrayLike],
    loc_y: Union[Scalar, ArrayLike],
) -> Union[int, NDArray[np.int64]]:
    """Return ``2`` or ``3`` from court coordinates ``(LOC_X, LOC_Y)`` in tenths of a foot."""
    scalar_input = _is_scalar(loc_x) and _is_scalar(loc_y)
    x = np.asarray(loc_x, dtype=float)
    y = np.asarray(loc_y, dtype=float)

    is_three = np.where(
        y <= CORNER_ARC_INTERSECT_Y_TENTHS,
        np.abs(x) >= CORNER_THREE_X_TENTHS,
        np.hypot(x, y) >= ARC_THREE_RADIUS_TENTHS,
    )
    values = np.where(is_three, 3, 2).astype(np.int64)

    if scalar_input:
        return int(values.item())
    return values
