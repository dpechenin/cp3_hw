from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from boids_config import BoidsParams


@njit(cache=True, inline="always")
def _limit_vector(x: float, y: float, max_value: float) -> tuple[float, float]:
    """Clamp a 2D vector length to the specified upper bound.

    Parameters
    ----------
    x, y:
        Input vector components.
    max_value:
        Maximum allowed vector magnitude.

    Returns
    -------
    tuple[float, float]
        Original vector if `||v|| <= max_value`, otherwise scaled vector with
        the same direction and exact length `max_value`.
    """
    mag_sq = x * x + y * y
    max_sq = max_value * max_value
    if mag_sq > max_sq and mag_sq > 1e-12:
        inv_mag = 1.0 / math.sqrt(mag_sq)
        scale = max_value * inv_mag
        return x * scale, y * scale
    return x, y


@njit(cache=True, inline="always")
def _set_mag_and_steer(
    vx: float,
    vy: float,
    target_mag: float,
    cur_vx: float,
    cur_vy: float,
    max_force: float,
) -> tuple[float, float]:
    """Build a steering vector from desired direction and current velocity.

    This functions implements the classic boids steering pattern:
    1) normalize desired direction,
    2) scale it to target speed,
    3) subtract current velocity,
    4) limit by max steering force.

    Parameters
    ----------
    vx, vy:
        Desired direction vector components.
    target_mag:
        Desired speed magnitude for the resulting velocity target.
    cur_vx, cur_vy:
        Current boid velocity components.
    max_force:
        Maximum steering force magnitude.

    Returns
    -------
    tuple[float, float]
        Clamped steering force `(fx, fy)`.
    """
    mag_sq = vx * vx + vy * vy
    if mag_sq <= 1e-12:
        return 0.0, 0.0
    inv_mag = 1.0 / math.sqrt(mag_sq)
    desired_x = vx * inv_mag * target_mag
    desired_y = vy * inv_mag * target_mag
    steer_x = desired_x - cur_vx
    steer_y = desired_y - cur_vy
    return _limit_vector(steer_x, steer_y, max_force)


@njit(cache=True)
def build_spatial_hash(
    pos: np.ndarray,
    width: float,
    height: float,
    cell_size: float,
    heads: np.ndarray,
    next_idx: np.ndarray,
) -> tuple[int, int]:
    """Build a linked-list uniform grid used for neighbor queries.

    Each cell stores the head index of a singly linked list of boids located
    in that cell. Links are held in `next_idx`.

    Parameters
    ----------
    pos:
        Boid positions, shape `(N, 2)`.
    width, height:
        Simulation domain dimensions.
    cell_size:
        Uniform grid cell size.
    heads:
        Preallocated array of shape `(nx*ny,)` with cell list heads.
    next_idx:
        Preallocated array of shape `(N,)` with linked-list pointers.

    Returns
    -------
    tuple[int, int]
        Grid resolution `(nx, ny)` for subsequent neighbor scanning.
    """
    n = pos.shape[0]
    nx = int(math.ceil(width / cell_size))
    ny = int(math.ceil(height / cell_size))
    total_cells = nx * ny
    for c in range(total_cells):
        heads[c] = -1
    for i in range(n):
        x = pos[i, 0]
        y = pos[i, 1]
        cx = int(x / cell_size)
        cy = int(y / cell_size)
        if cx < 0:
            cx = 0
        elif cx >= nx:
            cx = nx - 1
        if cy < 0:
            cy = 0
        elif cy >= ny:
            cy = ny - 1
        cell = cy * nx + cx
        next_idx[i] = heads[cell]
        heads[cell] = i
    return nx, ny


@njit(parallel=True, cache=True)
def compute_acceleration(
    pos: np.ndarray,
    vel: np.ndarray,
    noise: np.ndarray,
    heads: np.ndarray,
    next_idx: np.ndarray,
    nx: int,
    ny: int,
    cell_size: float,
    width: float,
    height: float,
    obstacle_x: np.ndarray,
    obstacle_y: np.ndarray,
    obstacle_r: np.ndarray,
    obstacle_kind: np.ndarray,
    alignment_weight: float,
    cohesion_weight: float,
    separation_weight: float,
    wall_weight: float,
    noise_weight: float,
    repulsive_obstacle_weight: float,
    attractive_obstacle_weight: float,
    max_force: float,
    max_speed: float,
    perception_radius_sq: float,
    separation_radius_sq: float,
    wall_margin: float,
    obstacle_padding: float,
    out_acc: np.ndarray,
) -> None:
    """Compute total acceleration for every boid in parallel.

    The function combines these interaction terms:
    - alignment: match average neighbor velocity,
    - cohesion: steer to average neighbor position,
    - separation: push away from close neighbors,
    - walls: soft force from boundaries,
    - noise: normalized random direction per boid,
    - obstacles: repulsive and attractive circular fields.

    Parameters
    ----------
    pos, vel:
        Boid positions and velocities, shape `(N, 2)`.
    noise:
        Random vectors for current step, shape `(N, 2)`.
    heads, next_idx, nx, ny, cell_size:
        Spatial hash data and geometry.
    width, height:
        Domain dimensions.
    obstacle_x, obstacle_y, obstacle_r, obstacle_kind:
        Obstacle arrays, where `obstacle_kind` is 0 for repulsion and 1 for
        attraction.
    *_weight:
        Scalar coefficients for each interaction term.
    max_force, max_speed:
        Dynamic constraints for steering.
    perception_radius_sq, separation_radius_sq:
        Squared neighbor radii for fast distance checks.
    wall_margin:
        Thickness of wall influence zone.
    obstacle_padding:
        Extra obstacle influence range added to obstacle radius.
    out_acc:
        Output accelerations, shape `(N, 2)`.
    """
    n = pos.shape[0]
    n_obs = obstacle_x.shape[0]
    for i in prange(n):
        x = pos[i, 0]
        y = pos[i, 1]
        vx = vel[i, 0]
        vy = vel[i, 1]

        sum_align_x = 0.0
        sum_align_y = 0.0
        sum_coh_x = 0.0
        sum_coh_y = 0.0
        sum_sep_x = 0.0
        sum_sep_y = 0.0
        align_count = 0
        sep_count = 0

        cx = int(x / cell_size)
        cy = int(y / cell_size)
        if cx < 0:
            cx = 0
        elif cx >= nx:
            cx = nx - 1
        if cy < 0:
            cy = 0
        elif cy >= ny:
            cy = ny - 1

        for oy in range(-1, 2):
            gy = cy + oy
            if gy < 0 or gy >= ny:
                continue
            for ox in range(-1, 2):
                gx = cx + ox
                if gx < 0 or gx >= nx:
                    continue
                cell = gy * nx + gx
                j = heads[cell]
                while j != -1:
                    if j != i:
                        dx = pos[j, 0] - x
                        dy = pos[j, 1] - y
                        d2 = dx * dx + dy * dy
                        if d2 <= perception_radius_sq:
                            sum_align_x += vel[j, 0]
                            sum_align_y += vel[j, 1]
                            sum_coh_x += pos[j, 0]
                            sum_coh_y += pos[j, 1]
                            align_count += 1
                        if separation_radius_sq >= d2 > 1e-12:
                            inv_d2 = 1.0 / d2
                            sum_sep_x -= dx * inv_d2
                            sum_sep_y -= dy * inv_d2
                            sep_count += 1
                    j = next_idx[j]

        align_x = 0.0
        align_y = 0.0
        coh_x = 0.0
        coh_y = 0.0
        sep_x = 0.0
        sep_y = 0.0

        if align_count > 0:
            inv_count = 1.0 / align_count
            ax = sum_align_x * inv_count
            ay = sum_align_y * inv_count
            align_x, align_y = _set_mag_and_steer(ax, ay, max_speed, vx, vy, max_force)
            cx_target = sum_coh_x * inv_count - x
            cy_target = sum_coh_y * inv_count - y
            coh_x, coh_y = _set_mag_and_steer(cx_target, cy_target, max_speed, vx, vy, max_force)

        if sep_count > 0:
            inv_sep = 1.0 / sep_count
            sx = sum_sep_x * inv_sep
            sy = sum_sep_y * inv_sep
            sep_x, sep_y = _set_mag_and_steer(sx, sy, max_speed, vx, vy, max_force)

        wall_x = 0.0
        wall_y = 0.0
        if x < wall_margin:
            wall_x += (wall_margin - x) / wall_margin
        elif x > width - wall_margin:
            wall_x -= (x - (width - wall_margin)) / wall_margin
        if y < wall_margin:
            wall_y += (wall_margin - y) / wall_margin
        elif y > height - wall_margin:
            wall_y -= (y - (height - wall_margin)) / wall_margin

        noise_x = noise[i, 0]
        noise_y = noise[i, 1]
        noise_mag_sq = noise_x * noise_x + noise_y * noise_y
        if noise_mag_sq > 1e-12:
            inv_noise_mag = 1.0 / math.sqrt(noise_mag_sq)
            noise_x *= inv_noise_mag
            noise_y *= inv_noise_mag
        else:
            noise_x = 0.0
            noise_y = 0.0

        obs_rep_x = 0.0
        obs_rep_y = 0.0
        obs_att_x = 0.0
        obs_att_y = 0.0
        for k in range(n_obs):
            odx = obstacle_x[k] - x
            ody = obstacle_y[k] - y
            d2 = odx * odx + ody * ody
            if d2 <= 1e-12:
                continue
            dist = math.sqrt(d2)
            influence = obstacle_r[k] + obstacle_padding
            if dist > influence:
                continue
            strength = (influence - dist) / influence
            dir_x = odx / dist
            dir_y = ody / dist
            if obstacle_kind[k] == 0:
                obs_rep_x -= dir_x * strength
                obs_rep_y -= dir_y * strength
            else:
                obs_att_x += dir_x * strength
                obs_att_y += dir_y * strength

        total_x = (
            alignment_weight * align_x
            + cohesion_weight * coh_x
            + separation_weight * sep_x
            + wall_weight * wall_x
            + noise_weight * noise_x
            + repulsive_obstacle_weight * obs_rep_x
            + attractive_obstacle_weight * obs_att_x
        )
        total_y = (
            alignment_weight * align_y
            + cohesion_weight * coh_y
            + separation_weight * sep_y
            + wall_weight * wall_y
            + noise_weight * noise_y
            + repulsive_obstacle_weight * obs_rep_y
            + attractive_obstacle_weight * obs_att_y
        )

        total_x, total_y = _limit_vector(total_x, total_y, max_force * 2.0)
        out_acc[i, 0] = total_x
        out_acc[i, 1] = total_y


@njit(parallel=True, cache=True)
def integrate(
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    dt: float,
    max_speed: float,
    width: float,
    height: float,
) -> None:
    """Integrate velocity/position one time step with speed and bounds limits.

    Integration scheme:
    - `v <- v + a * dt`
    - clamp `||v|| <= max_speed`
    - `x <- x + v * dt`
    - apply inelastic boundary response to keep boids inside domain.

    Parameters
    ----------
    pos, vel:
        In/out arrays with positions and velocities, shape `(N, 2)`.
    acc:
        Input acceleration array, shape `(N, 2)`.
    dt:
        Time step.
    max_speed:
        Maximum allowed speed magnitude.
    width, height:
        Domain dimensions used for boundary handling.
    """
    n = pos.shape[0]
    for i in prange(n):
        vx = vel[i, 0] + acc[i, 0] * dt
        vy = vel[i, 1] + acc[i, 1] * dt
        speed_sq = vx * vx + vy * vy
        max_sq = max_speed * max_speed
        if speed_sq > max_sq and speed_sq > 1e-12:
            inv_speed = 1.0 / math.sqrt(speed_sq)
            scale = max_speed * inv_speed
            vx *= scale
            vy *= scale

        x = pos[i, 0] + vx * dt
        y = pos[i, 1] + vy * dt

        if x < 0.0:
            x = 0.0
            vx = abs(vx) * 0.5
        elif x > width:
            x = width
            vx = -abs(vx) * 0.5

        if y < 0.0:
            y = 0.0
            vy = abs(vy) * 0.5
        elif y > height:
            y = height
            vy = -abs(vy) * 0.5

        pos[i, 0] = x
        pos[i, 1] = y
        vel[i, 0] = vx
        vel[i, 1] = vy


class BoidsSimulation:
    """Control class of Boids simulation."""

    def __init__(
        self,
        n_agents: int,
        width: float,
        height: float,
        params: BoidsParams,
        obstacle_x: np.ndarray,
        obstacle_y: np.ndarray,
        obstacle_r: np.ndarray,
        obstacle_kind: np.ndarray,
        seed: int = 42,
    ) -> None:
        """Initialize boid state, obstacle data and working buffers.

        Parameters
        ----------
        n_agents:
            Number of boids.
        width, height:
            Domain dimensions in simulation units.
        params:
            Simulation coefficients and limits.
        obstacle_x, obstacle_y, obstacle_r, obstacle_kind:
            Obstacle definitions; arrays must have equal length.
        seed:
            RNG seed for reproducible initialization/noise.
        """
        self.n_agents = int(n_agents)
        self.width = float(width)
        self.height = float(height)
        self.params = params
        self.rng = np.random.default_rng(seed)

        self.pos = np.empty((self.n_agents, 2), dtype=np.float32)
        self.pos[:, 0] = self.rng.uniform(0.0, self.width, size=self.n_agents).astype(np.float32)
        self.pos[:, 1] = self.rng.uniform(0.0, self.height, size=self.n_agents).astype(np.float32)

        angle = self.rng.uniform(0.0, 2.0 * np.pi, size=self.n_agents).astype(np.float32)
        speed = self.rng.uniform(0.2 * params.max_speed, 0.8 * params.max_speed, size=self.n_agents).astype(
            np.float32
        )
        self.vel = np.empty((self.n_agents, 2), dtype=np.float32)
        self.vel[:, 0] = np.cos(angle) * speed
        self.vel[:, 1] = np.sin(angle) * speed

        self.noise = np.empty((self.n_agents, 2), dtype=np.float32)
        self.acc = np.empty((self.n_agents, 2), dtype=np.float32)

        self.cell_size = params.perception_radius
        nx = int(math.ceil(width / self.cell_size))
        ny = int(math.ceil(height / self.cell_size))
        self.heads = np.empty(nx * ny, dtype=np.int32)
        self.next_idx = np.empty(self.n_agents, dtype=np.int32)

        self.obstacle_x = obstacle_x.astype(np.float32, copy=False)
        self.obstacle_y = obstacle_y.astype(np.float32, copy=False)
        self.obstacle_r = obstacle_r.astype(np.float32, copy=False)
        self.obstacle_kind = obstacle_kind.astype(np.int8, copy=False)

        self.perception_radius_sq = np.float32(params.perception_radius * params.perception_radius)
        self.separation_radius_sq = np.float32(params.separation_radius * params.separation_radius)

    def step(self) -> None:
        """Do one time step for Boids simulation.

        Work done per step:
        1. sample random noise vectors,
        2. rebuild spatial hash,
        3. compute interaction accelerations,
        4. integrate kinematics.
        """
        self.noise[:] = self.rng.normal(0.0, 1.0, size=self.noise.shape).astype(np.float32)
        nx, ny = build_spatial_hash(self.pos, self.width, self.height, self.cell_size, self.heads, self.next_idx)
        compute_acceleration(self.pos, self.vel, self.noise, self.heads, self.next_idx, nx, ny, self.cell_size, self.width, self.height,
            self.obstacle_x, self.obstacle_y, self.obstacle_r, self.obstacle_kind, self.params.alignment_weight, self.params.cohesion_weight,
            self.params.separation_weight, self.params.wall_weight, self.params.noise_weight, self.params.repulsive_obstacle_weight, self.params.attractive_obstacle_weight,
            self.params.max_force, self.params.max_speed, self.perception_radius_sq, self.separation_radius_sq, self.params.wall_margin, self.params.obstacle_padding,
            self.acc)
        integrate(self.pos, self.vel, self.acc, self.params.dt, self.params.max_speed, self.width, self.height,)

    def warmup(self, steps: int = 3) -> None:
        """Run a few steps to trigger Numba compilation before render loop.

        Parameters
        ----------
        steps:
            Number of warmup iterations.
        """
        for _ in range(steps):
            self.step()
