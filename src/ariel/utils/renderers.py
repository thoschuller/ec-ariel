"""TODO(jmdm): description of script.

Todo:
----
    [ ] ".rotate" as superclass method?
    [ ] Better documentation
"""

# Standard library
import datetime
import math

# Third-party libraries
import mujoco
from PIL import Image
from rich.console import Console

# Local libraries
from ariel.utils.video_recorder import VideoRecorder

# Global functions
console = Console()


def single_frame_renderer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    steps: int = 10,
    *,
    save: bool = False,
    save_path: str | None = None,
    append_date: bool = True,
) -> None:
    """
    Render a single frame of the simulation using MuJoCo's rendering engine.

    Parameters
    ----------
    model : mujoco.MjModel
        The MuJoCo model to render.
    data : mujoco.MjData
        The MuJoCo data to render.
    steps : int, optional
        The number of simulation steps to take before rendering, by default 10
    """
    # Enable joint visualization option:
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # Call rendering engine
    msg = f"Rendering single frame with [bold blue] {steps} [/bold blue] steps."
    console.log(f"[bold yellow] --> {msg} [/bold yellow]")
    with mujoco.Renderer(model) as renderer:
        # Move simulation forward one iteration/step
        mujoco.mj_step(model, data, nstep=steps)

        # Update rendering engine
        renderer.update_scene(
            data,
            scene_option=scene_option,
        )

        # Generate frame using rendering engine
        frame = renderer.render()

        # Convert frame into an image which can be shown
        img = Image.fromarray(frame)

        # Save or show
        if save is True:
            # No save path given (use default)
            if save_path is None:
                save_path = "./frame.png"

            # Add date to name
            if append_date is True:
                now = datetime.datetime.now(tz=datetime.UTC)
                date = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_format = save_path.split(".")[-1]

                # Update file name
                save_path = save_path[: -(len(file_format) + 1)]
                save_path += f"_{date}"
                save_path += f".{file_format}"

            # Save image locally
            img.save(save_path, format="png")
        else:
            img.show()

    console.log("[bold green] --> Rendering done![/bold green]")


def video_renderer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float = 10.0,
    video_recorder: VideoRecorder | None = None,
) -> None:
    """
    Render a video of the simulation using MuJoCo's rendering engine.

    Parameters
    ----------
    model : mujoco.MjModel
        The MuJoCo model to render.
    data : mujoco.MjData
        The MuJoCo data to render.
    duration : float, optional
        The duration of the video in seconds, by default 10.0
    video_recorder : VideoRecorder | None, optional
        The video recorder to use, by default None
    """
    # Get video recorder
    if video_recorder is None:
        video_recorder = VideoRecorder()

    # Enable joint visualization option:
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # Calculate steps per frame to avoid single iterations (see 'Notes'.)
    options = mujoco.MjOption()
    steps_per_frame = duration / (
        options.timestep * duration * video_recorder.fps
    )

    # Call rendering engine
    with mujoco.Renderer(
        model,
        width=video_recorder.width,
        height=video_recorder.height,
    ) as renderer:
        while data.time < duration:
            # Move simulation forward one iteration/step
            mujoco.mj_step(model, data, nstep=math.floor(steps_per_frame))

            # Update rendering engine
            renderer.update_scene(data, scene_option=scene_option)

            # Save frame
            video_recorder.write(frame=renderer.render())

    # Exit (and save locally) the generated video
    console.log(video_recorder.frame_count)
    video_recorder.release()


def tracking_video_renderer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float = 10.0,
    video_recorder: VideoRecorder | None = None,
    tracking_distance: float = 1.5,
    tracking_angle: float = 135,
) -> None:
    """
    Render a video of the simulation with camera tracking the "core" module.

    Parameters
    ----------
    model : mujoco.MjModel
        The MuJoCo model to render.
    data : mujoco.MjData
        The MuJoCo data to render.
    duration : float, optional
        The duration of the video in seconds, by default 10.0
    video_recorder : VideoRecorder | None, optional
        The video recorder to use, by default None
    tracking_distance : float, optional
        Distance from the core module for camera positioning, by default 1.5
    tracking_angle : float, optional
        Angle relative to the robot that the camera will be recording from.
        By default 135, meaning the robot will walk to the left. If set to 0
        the robot will walk towards the camera.
    """
    # Get video recorder
    if video_recorder is None:
        video_recorder = VideoRecorder()

    # Enable joint visualization option:
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # Find the core body ID for tracking
    try:
        core_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "robot-core",
        )
        console.log(f"Tracking core body ID: {core_body_id}")
    except ValueError:
        # Fallback: try to find any body with "core" in the name
        core_body_id = None
        for i in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and "core" in body_name:
                core_body_id = i
                break

    if core_body_id is None:
        console.log(
            "[bold red] --> Warning: No core body found for tracking. Using default camera.[/bold red]",
        )
    else:
        console.log(
            f"[bold blue] --> Tracking core body with ID: {core_body_id}[/bold blue]",
        )

    # Calculate steps per frame to avoid single iterations
    options = mujoco.MjOption()
    steps_per_frame = duration / (
        options.timestep * duration * video_recorder.fps
    )

    # Call rendering engine
    with mujoco.Renderer(
        model,
        width=video_recorder.width,
        height=video_recorder.height,
    ) as renderer:
        # Set up tracking camera if core body found
        if core_body_id is not None:
            # Create a tracking camera
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            camera.trackbodyid = core_body_id
            camera.distance = tracking_distance
            camera.azimuth = tracking_angle  # Angle around the target
            camera.elevation = -30.0  # Angle above/below the target

            # Update the renderer's camera
            renderer.update_scene(
                data,
                scene_option=scene_option,
                camera=camera,
            )
        else:
            # Use default camera
            renderer.update_scene(data, scene_option=scene_option)

        while data.time < duration:
            # Move simulation forward one iteration/step
            mujoco.mj_step(model, data, nstep=math.floor(steps_per_frame))

            # Update rendering engine with tracking camera
            if core_body_id is not None:
                renderer.update_scene(
                    data,
                    scene_option=scene_option,
                    camera=camera,
                )
            else:
                renderer.update_scene(data, scene_option=scene_option)

            # Save frame
            video_recorder.write(frame=renderer.render())

    # Exit (and save locally) the generated video
    console.log(
        f"[bold green] --> Tracking video rendered with {video_recorder.frame_count} frames[/bold green]"
    )
    video_recorder.release()
