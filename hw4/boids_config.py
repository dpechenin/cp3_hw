from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class BoidsParams:
    """Container for boids interaction and integration parameters."""

    alignment_weight: float = 0.9
    cohesion_weight: float = 1.5
    separation_weight: float = 1.35
    wall_weight: float = 1.25
    noise_weight: float = 0.1
    repulsive_obstacle_weight: float = 300.0
    attractive_obstacle_weight: float = 700.0
    max_force: float = 50.0
    max_speed: float = 250.0
    perception_radius: float = 38.0
    separation_radius: float = 16.0
    wall_margin: float = 45.0
    obstacle_padding: float = 70.0
    dt: float = 1.0 / 60.0


def default_obstacles(width: float, height: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a default set of static circular obstacles for the scene.

    The function creates two obstacle classes:
    - class `0`: repulsive obstacles (push boids away),
    - class `1`: attractive obstacles (pull boids toward center).

    Parameters
    ----------
    width : float
        Simulation world width.
    height : float
        Simulation world height.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A 4-tuple `(cx, cy, radii, kinds)` where:
        - `cx`: x-coordinates of obstacle centers, `float32`, shape `(M,)`;
        - `cy`: y-coordinates of obstacle centers, `float32`, shape `(M,)`;
        - `radii`: obstacle radii, `float32`, shape `(M,)`;
        - `kinds`: obstacle classes (`0` repulsive, `1` attractive), `int8`, shape `(M,)`.
    """
    cx = np.array(
        [
            0.22 * width,
            0.38 * width,
            0.73 * width,
            0.84 * width,
            0.56 * width,
            0.17 * width,
        ],
        dtype=np.float32,
    )
    cy = np.array(
        [
            0.25 * height,
            0.72 * height,
            0.31 * height,
            0.78 * height,
            0.55 * height,
            0.52 * height,
        ],
        dtype=np.float32,
    )
    radii = np.array([24.0, 34.0, 30.0, 26.0, 42.0, 20.0], dtype=np.float32)
    kinds = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    return cx, cy, radii, kinds
