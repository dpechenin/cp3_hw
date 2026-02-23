import platform
import time
from pathlib import Path
from typing import Optional

from ffmpeg_recorder import FFmpegRecorder
from boids_core import BoidsSimulation

def select_vispy_backend(user_backend: Optional[str]) -> Optional[str]:
    """Select a VisPy GUI backend.

    Selection order:
    1. explicit user-provided backend,
    2. `pyqt6` for macOS,
    3. `pyglet` for Windows,
    4. default backend resolution by VisPy on other systems.
    """
    if user_backend:
        return user_backend
    system = platform.system().lower()
    if system == "darwin":
        return "pyqt6"
    if system == "windows":
        return "pyglet"
    return None


def run_visualization(
    simulation: BoidsSimulation,
    width_px: int,
    height_px: int,
    fps: int,
    max_frames: Optional[int],
    output_mp4: Optional[Path],
    backend: Optional[str],
) -> None:
    """Run realtime rendering loop, HUD updates and optional MP4 recording.

    Parameters
    ----------
    simulation:
        Initialized simulation object.
    width_px, height_px:
        Canvas size in pixels.
    fps:
        Target timer frequency.
    max_frames:
        Optional frame cap; `None` means run until window close.
    output_mp4:
        Optional path for MP4 recording. If `None`, recording is disabled.
    backend:
        Optional explicit VisPy backend name.
    """
    from vispy import app, scene

    backend_name = select_vispy_backend(backend)
    if backend_name is not None:
        app.use_app(backend_name)

    canvas = scene.SceneCanvas(
        keys="interactive",
        size=(width_px, height_px),
        bgcolor="#0A1325",
        show=True,
        title="Boids HW4",
    )
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.PanZoomCamera(aspect=1.0)
    view.camera.set_range(x=(0.0, simulation.width), y=(0.0, simulation.height))

    boids_visual = scene.visuals.Markers(parent=view.scene)
    boids_visual.set_data(
        simulation.pos.copy(),
        face_color=(0.95, 0.96, 1.0, 0.95),
        edge_color=(0.40, 0.75, 1.0, 1.0),
        size=5.0,
    )

    for i in range(simulation.obstacle_x.shape[0]):
        c = (1.0, 0.35, 0.35, 1.0) if simulation.obstacle_kind[i] == 0 else (0.35, 1.0, 0.72, 1.0)
        scene.visuals.Ellipse(
            center=(float(simulation.obstacle_x[i]), float(simulation.obstacle_y[i])),
            radius=(float(simulation.obstacle_r[i]), float(simulation.obstacle_r[i])),
            color=(0.0, 0.0, 0.0, 0.0),
            border_color=c,
            border_width=2.5,
            parent=view.scene,
        )

    top_hud = scene.Text(
        "",
        parent=canvas.scene,
        color="white",
        font_size=8,
        anchor_x="left",
        anchor_y="top",
        pos=(10, 20),
    )
    bottom_hud = scene.Text(
        "",
        parent=canvas.scene,
        color="white",
        font_size=8,
        anchor_x="left",
        anchor_y="bottom",
        pos=(10, height_px - 20),
    )

    fb_w, fb_h = canvas.physical_size
    recorder = FFmpegRecorder(output_mp4, int(fb_w), int(fb_h), fps) if output_mp4 else None
    frame_limit = max_frames if max_frames is not None else -1
    frame_count = 0
    fps_value = 0.0
    fps_alpha = 0.90
    last_t = time.perf_counter()

    simulation.warmup(steps=2)

    params = simulation.params

    def update_hud() -> None:
        """Refresh overlay text with FPS and current simulation parameters."""
        top_hud.text = (
            f"N={simulation.n_agents} | fps={fps_value:6.2f} | "
            f"alignment={params.alignment_weight:.2f} | cohesion={params.cohesion_weight:.2f} | "
            f"separation={params.separation_weight:.2f} | wall={params.wall_weight:.2f} | "
            f"noise={params.noise_weight:.2f}"
        )
        bottom_hud.text = (
            f"repulsive obstacle weight={params.repulsive_obstacle_weight:.2f} | "
            f"attractive obstacle weight={params.attractive_obstacle_weight:.2f} | "
            f"perception radius={params.perception_radius:.1f} |  separation radius={params.separation_radius:.1f} | "
            f"recording={'on' if recorder is not None else 'off'} | frames={frame_count}"
        )

    def on_timer(_event) -> None:
        """Timer callback: advance simulation, redraw and optionally encode frame."""
        nonlocal frame_count, fps_value, last_t

        simulation.step()
        boids_visual.set_data(
            simulation.pos,
            face_color=(0.95, 0.96, 1.0, 0.95),
            edge_color=(0.40, 0.75, 1.0, 1.0),
            size=5.0,
        )

        now = time.perf_counter()
        dt = now - last_t
        last_t = now
        instant_fps = 1.0 / dt if dt > 1e-9 else 0.0
        fps_value = instant_fps if fps_value == 0.0 else fps_alpha * fps_value + (1.0 - fps_alpha) * instant_fps
        if frame_count % 6 == 0:
            update_hud()

        canvas.update()

        if recorder is not None:
            frame = canvas.render(alpha=True)
            recorder.write(frame)

        frame_count += 1
        if frame_limit > 0 and frame_count >= frame_limit:
            timer.stop()
            if recorder is not None:
                recorder.close()
            canvas.close()
            app.quit()

    @canvas.events.close.connect
    def on_close(_event) -> None:
        """Close recording pipeline when canvas is closed by user or program."""
        if recorder is not None:
            recorder.close()

    timer = app.Timer(interval=1.0 / fps, connect=on_timer, start=True)
    update_hud()
    app.run()
