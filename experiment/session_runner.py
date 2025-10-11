from ariel.utils.tracker import Tracker
from ariel.simulation.controllers.controller import Controller
import mujoco

# from typing import Any
from examples.z_ec_course.A3_template_jack import Vector
import experiment.constants as constants
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.renderers import tracking_video_renderer
import numpy as np
from pathlib import Path
from ariel.simulation.environments import BaseWorld
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from experiment.terminal import console
from mujoco import viewer
import copy
import numpy.typing as npt

RNG = np.random.default_rng(constants.SEED)

def initialize_world_and_robot(
    gecko_body: CoreModule,
    spawn_pos: list[float],
    world: type[BaseWorld] = constants.SIM_WORLD,
) -> tuple[object, mujoco.MjData, BaseWorld, Tracker]:
    mujoco.set_mjcb_control(None)

    world_instance = world()

    usable_gecko = copy.deepcopy(gecko_body)

    world_instance.spawn(usable_gecko.spec, position=spawn_pos, rotation=[90, 0, 0])
    model = world_instance.spec.compile()
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)

    # Create tracker for position and orientation data
    tracker = Tracker(
        mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM,
        name_to_bind="core",
        observable_attributes=["xpos", "xmat"],
    )

    return model, data, world_instance, tracker


def _find_track_body_name(model: mujoco.MjModel) -> str:
    """
    Find a valid body name to track for video recording.
    Preference order:
    1) Any body with 'core' in its name.
    2) The first non-world body (id 1) if available.
    This avoids passing an invalid name to tracking_video_renderer.
    """
    # Prefer names containing 'core'
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and "core" in name:
            return name
    # Fallback: first non-world body if exists
    console.log(
        f"Warning: No body with 'core' in name found, using first non-world body if available."
    )
    if model.nbody > 1:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 1)
        if name:
            console.log(f"Using first non-world body: {name}")
            return name
    # Last resort: return an empty string (renderer may use default camera)
    console.log("No valid body found for tracking.")
    return ""


def run_bot_session(
    weights: np.ndarray,
    method: str,
    gecko_body: CoreModule,
    duration: float,
    spawn_pos: list[float],
    options: dict[str, str | float] | None = None,
) -> Tracker:
    """
    Run a single simulation session with given weights and method.
    """

    # Clear any existing MuJoCo callbacks for process isolation
    mujoco.set_mjcb_control(None)

    model, data, world, tracker = initialize_world_and_robot(gecko_body, spawn_pos)

    # Define controller callback
    def _controller_callback(_: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
        outputs = _controller(
            data=d,
            weights=weights,
            input_size=len(d.qpos),
            hidden_size=constants.HIDDEN_SIZE,
            output_size=model.nu,
        )
        return outputs

    # Instantiate Controller with tracker
    ctrl = Controller(
        controller_callback_function=_controller_callback,
        tracker=tracker,
        # alpha=CONFIG["OUTPUT_DELTA"],
    )

    # Setup tracker before simulation
    tracker.setup(world.spec, data)

    def _control_callback(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        """Control callback for mujoco simulation."""
        ctrl.set_control(m, d)

    mujoco.set_mjcb_control(_control_callback)

    run_duration = duration

    match method:
        case "record":
            video_path = Path(__file__).parent / "output" / "videos"
            video_path.mkdir(exist_ok=True)
            video_file = ""
            if options:
                video_file += f"{options.get('filename','recording')} mode {options.get('mode','unknown')}_fit {options.get('fitness',0.0):.4f}"

            video_recorder = VideoRecorder(
                output_folder=video_path,
                file_name=video_file,
                width=1200,
                height=960,
                fps=30,
            )
            # Choose a safe body to track for the camera
            body_name_to_track = _find_track_body_name(model)
            tracking_video_renderer(
                model,
                data,
                duration=10 + run_duration,
                video_recorder=video_recorder,
                geom_to_track=body_name_to_track,
            )
            mujoco.set_mjcb_control(None)
            console.log(f"Recorded episode saved to {video_path}/{video_file}")
        case "viewer":
            viewer.launch(model, data)
        case "headless":  # for evaluation-only sessions
            # Use the project's simple_runner helper instead of calling mj_step directly.
            simple_runner(model, data, duration=run_duration)
        case _:
            raise ValueError(f"Unknown method: {method}")

    mujoco.set_mjcb_control(None)

    # Convert tracker history to expected format
    return tracker

class RandomNN:
    def __init__(self, robot: CoreModule) -> None:
        _, data, _, _ = initialize_world_and_robot(gecko_body=robot, spawn_pos=[0, 0, 0])

        # Get relevant info
        self.input_size = len(data.qpos.copy())
        self.hidden_size = 8
        self.output_size = len(data.ctrl)

        # Clear cache
        del data

    def random_controller(
        self,
    ) -> None:
        # Initialize the networks weights randomly
        # Normally, you would use the genes of an individual as the weights,
        # Here we set them randomly for simplicity.
        w1 = RNG.normal(
            loc=0.0138,
            scale=0.5,
            size=(self.input_size, self.hidden_size),
        )
        w2 = RNG.normal(
            loc=0.0138,
            scale=0.5,
            size=(self.hidden_size, self.hidden_size),
        )
        w3 = RNG.normal(
            loc=0.0138,
            scale=0.5,
            size=(self.hidden_size, self.output_size),
        )
        self.weights = (w1, w2, w3)

    def set_controller_weights(
        self,
        weights: tuple[Vector, Vector, Vector],
    ) -> None:
        self.weights = weights

    def forward(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> npt.NDArray[np.float64]:
        # Get inputs, in this case the positions of the actuator motors (hinges)
        inputs = data.qpos

        # Run the inputs through the lays of the network.
        layer1 = np.tanh(np.dot(inputs, self.weights[0]))
        layer2 = np.tanh(np.dot(layer1, self.weights[1]))
        outputs = np.tanh(np.dot(layer2, self.weights[2]))

        # Scale the outputs
        return outputs * np.pi

def _controller(
    data: mujoco.MjData,
    weights: np.ndarray,
    input_size: int,
    hidden_size: int,
    output_size: int,
) -> np.ndarray:
    # `weights` is expected to be a flat numpy array of parameters (float)

    # Dynamically unpack weights for multiple hidden layers
    layer_sizes = (
        [input_size] + [hidden_size] * constants.NUM_HIDDEN_LAYERS + [output_size]
    )
    weight_shapes = [
        (layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
    ]
    weight_sizes = [a * b for a, b in weight_shapes]
    indices = np.cumsum([0] + weight_sizes)
    ws = [
        weights[indices[i] : indices[i + 1]].reshape(weight_shapes[i])
        for i in range(len(weight_shapes))
    ]

    # Forward pass
    inputs = data.qpos
    x = inputs
    for i in range(constants.NUM_HIDDEN_LAYERS):
        x = np.tanh(np.dot(x, ws[i]))
    outputs = np.tanh(np.dot(x, ws[-1]))
    outputs = outputs * np.pi

    return outputs
