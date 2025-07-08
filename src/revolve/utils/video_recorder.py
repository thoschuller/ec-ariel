"""TODO(jmdm): description of script.

Author:     jmdm
Date:       2025-05-02
Py Ver:     3.12
OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro

Status:     In progress ⚙️
Status:     Paused ⏸️
Status:     Completed ✅
Status:     Incomplete ❌
Status:     Broken ⚠️
Status:     To Improve ⬆️

Sources
-----
    1.

Notes
-----
    *

Todo
-----
    * [ ]

"""

# Standard library
import datetime
from pathlib import Path

# Third-party libraries
import cv2
import numpy as np
from numpy import typing as npt


class VideoRecorder:
    """Simple video recorder."""

    # Encoding: 'mp4v' or 'avc1' for H.264
    _video_encoding: str = "mp4v"
    _add_timestamp_to_file_name: bool = True

    def __init__(
        self,
        video_name: str = "video",
        output_folder: str = "./",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        """Create a video recording."""
        # Ensure output folder exists
        Path(output_folder).mkdir(exist_ok=True, parents=True)

        # Generate video name
        output_folder = output_folder.rstrip("/")
        if self._add_timestamp_to_file_name:
            timestamp = datetime.datetime.now(tz=datetime.UTC).strftime(
                "%Y%m%d%H%M%S",
            )
            video_name += f"_{timestamp}"
        output_file = f"{output_folder}/{video_name}.mp4"

        # Create recorder object
        fourcc = cv2.VideoWriter_fourcc(*self._video_encoding)
        video_writer = cv2.VideoWriter(
            output_file,
            fourcc,
            fps,
            (width, height),
        )

        # Class attributes
        self.frame_count = 0
        self.video_writer = video_writer

    def write(self, frame: npt.ArrayLike) -> None:
        """Write MuJoCo frame to video."""
        # Convert PIL Image to numpy array (OpenCV uses BGR format)
        opencv_image = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        # Save frame
        self.video_writer.write(opencv_image)

        # Increment frame counter
        self.frame_count += 1

    def release(self) -> None:
        """Close video writer and save video locally."""
        self.video_writer.release()
