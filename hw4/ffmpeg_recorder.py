from pathlib import Path

import numpy as np
import ffmpeg

class FFmpegRecorder:
    """Asynchronous frame writer."""

    def __init__(self, out_path: Path, width: int, height: int, fps: int) -> None:
        """Start ffmpeg process configured for H.264 MP4 output.

        Parameters
        ----------
        out_path:
            Destination video file path.
        width, height:
            Frame resolution in pixels.
        fps:
            Output frame rate.
        """
        self.out_path = out_path
        self.closed = False
        self.process = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgba",
                s=f"{width}x{height}",
                framerate=fps,
            )
            .output(
                str(out_path),
                vcodec="libx264",
                pix_fmt="yuv420p",
                r=fps,
                movflags="+faststart",
            )
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True)
        )

    def write(self, rgba_frame: np.ndarray) -> None:
        """Write one rendered frame to ffmpeg stdin.

        Parameters
        ----------
        rgba_frame:
            Frame array with shape `(H, W, 4)` and RGBA channels.
        """
        self.process.stdin.write(rgba_frame.tobytes())

    def close(self) -> None:
        """Close ffmpeg stdin and wait for encoder process completion safely."""
        if self.closed:
            return
        self.closed = True
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.wait()
