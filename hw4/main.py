from __future__ import annotations

import argparse
from pathlib import Path

from boids_config import BoidsParams, default_obstacles
from boids_core import BoidsSimulation
from boids_runtime import run_visualization


def parse_args() -> argparse.Namespace:
    """Build CLI parser and parse arguments."""
    parser = argparse.ArgumentParser(description="Boids HW4")
    parser.add_argument("--n", type=int, default=1000, help="Number of boids")
    parser.add_argument("--width", type=int, default=1280, help="Window width in pixels")
    parser.add_argument("--height", type=int, default=720, help="Window height in pixels")
    parser.add_argument("--world-width", type=float, default=1280.0, help="Simulation world width")
    parser.add_argument("--world-height", type=float, default=720.0, help="Simulation world height")
    parser.add_argument("--fps", type=int, default=60, help="Target render FPS")
    parser.add_argument("--steps", type=int, default=0, help="Limit number of rendered steps (0 means infinite)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--record", type=Path, default=None, help="Optional output MP4 path")
    parser.add_argument("--backend", type=str, default=None, help="VisPy backend (pyqt6/pyqt5/pyglet)")
    return parser.parse_args()


def build_simulation(args: argparse.Namespace) -> BoidsSimulation:
    """Create simulation object from parsed CLI arguments."""
    params = BoidsParams()
    obstacles_x, obstacles_y, obstacles_rad, obstacles_kind = default_obstacles(args.world_width, args.world_height)
    return BoidsSimulation(
        n_agents=args.n,
        width=args.world_width,
        height=args.world_height,
        params=params,
        obstacle_x=obstacles_x,
        obstacle_y=obstacles_y,
        obstacle_r=obstacles_rad,
        obstacle_kind=obstacles_kind,
        seed=args.seed,
    )


def main() -> None:
    """Entry point: configure simulation, launch GUI, optionally run benchmark."""
    args = parse_args()
    sim = build_simulation(args)

    max_frames = args.steps if args.steps > 0 else None
    run_visualization(
        simulation=sim,
        width_px=args.width,
        height_px=args.height,
        fps=args.fps,
        max_frames=max_frames,
        output_mp4=args.record,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
