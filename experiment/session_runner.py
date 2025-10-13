from networkx import DiGraph
from ariel.utils.tracker import Tracker
from ariel.simulation.controllers.controller import Controller
import mujoco

# from typing import Any

import constants as constants
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.renderers import video_renderer
import numpy as np
from pathlib import Path
from terminal import console
from mujoco import viewer
import numpy.typing as npt
from ariel.simulation.environments import BaseWorld
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from typing import cast
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph

type Vector = npt.NDArray[np.float64]

RNG = np.random.default_rng(constants.SEED)


def quick_spawn(
    gecko_body: CoreModule,
    spawn_pos: list[float],
) -> tuple[mujoco.MjModel, mujoco.MjData, BaseWorld]:
    mujoco.set_mjcb_control(None)
    world = constants.SIM_WORLD()
    temp_robot_xml = gecko_body.spec.to_xml()
    temp_robot = mujoco.MjSpec.from_string(temp_robot_xml)
    world.spawn(
        temp_robot,
        position=spawn_pos,
    )
    model = world.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    return (cast("mujoco.MjModel", model), data, world)


def run_bot_session(
    weights: np.ndarray,
    method: str,
    gecko_body: DiGraph,  # pyright: ignore
    duration: float,
    spawn_pos: list[float],
    options: dict[str, str | float] | None = None,
) -> Tracker:
    """
    Run a single simulation session with given weights and method.
    """

    # Clear any existing MuJoCo callbacks for process isolation
    mujoco.set_mjcb_control(None)

    model, data, world = quick_spawn(construct_mjspec_from_graph(gecko_body), spawn_pos)

    tracker = Tracker(
        mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM,
        name_to_bind="core",
        observable_attributes=["xpos", "xmat"],
    )

    # Define controller callback
    def _controller_callback(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
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
                width=320,
                height=960,
                fps=30,
            )
            # Choose a safe body to track for the camera
            # DEBUG: Causing issues right now, switched from tracking to non-tracking
            video_renderer(
                model,
                data,
                duration=10 + run_duration,
                video_recorder=video_recorder,
                cam_fovy=8,
                cam_pos=[2.1, 0, 50],
                cam_quat=[-0.7071, 0, 0, 0.7071],
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
    def __init__(self, robot: DiGraph) -> None:  # pyright: ignore
        _, data, _ = quick_spawn(
            gecko_body=construct_mjspec_from_graph(robot), spawn_pos=[0, 0, 0]
        )

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

    try:

        ws = []
        for i in range(len(weight_shapes)):
            ws.append(weights[indices[i] : indices[i + 1]].reshape(weight_shapes[i]))

    except ValueError as e:
        console.log(f"[red] [ERROR] Weight reshaping error: {e}")
        console.log(
            f"[red] [ERROR] Weights length: {len(weights)}, Expected total params: {sum(weight_sizes)}"
        )
        console.log(f"[red] [ERROR] Layer sizes: {layer_sizes}")
        console.log(f"[red] [ERROR] Weight shapes: {weight_shapes}")
        console.log(f"[red] [ERROR] Weight sizes: {weight_sizes}")
        raise ValueError("Weight reshaping failed.")

    # Forward pass
    inputs = data.qpos
    x = inputs
    for i in range(constants.NUM_HIDDEN_LAYERS):
        x = np.tanh(np.dot(x, ws[i]))
    outputs = np.tanh(np.dot(x, ws[-1]))
    outputs = outputs * np.pi

    return outputs
