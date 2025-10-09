#type: ignore

"""
This is an evolutionary algorithm experiment using the Ariel framework.
It evolves a population of numpy neural network weights to control a gecko robot.
The evolution is based on CMA-ES instead of a self-built EA.
The evolution is based on two different fitness calculations, for which both an experiment and baseline evaluation are executed in threefold to compare their effectiveness.

Written by;
- Thomas Schuller
- Maélis Chaulot-Talmon
- Madalena Barceló
- Hang Tran
"""

# Third-party libraries
import csv
from typing import Any

import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
import time
from pathlib import Path
import multiprocessing
# if you get errors here, you may need to install torch and torchvision
# uv pip install torch torchvision --torch-backend=auto
from rich.console import Console
from rich.traceback import install
from rich.progress import Progress
from rich.prompt import Prompt
import math
import gc
import sys
import warnings

from typing import cast

# CMA-ES library
import cma
from functools import partial

# Local libraries
from ariel.utils.renderers import tracking_video_renderer, single_frame_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.simulation.environments.olympic_arena import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.ec.a001 import Individual, JSONIterable
# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph 
from ariel.simulation.controllers.controller import Controller
from networkx import DiGraph  # Add this import

import random

# --- Configurable global settings --- #
NEURALNET_EVO_CONFIG = {
    "SIM_WORLD": OlympicArena,
    "SEED": 42,
    "SEGMENT_LENGTH": 250,
    "POP_SIZE": 30,
    "MAX_GENERATIONS": 125,
    "TIME_LIMIT": 60*10, 
    "HIDDEN_SIZE": 8,
    "DURATION": 15,
    "OUTPUT_DELTA": 0.05,
    "NUM_HIDDEN_LAYERS": 1,
    "FITNESS_MODE": "lateral_adjusted",
    "UNIFORM_CROSSOVER": True,
    "LATERAL_PENALTY_FACTOR": 0.1,
    "INTERACTIVE_MODE": False,
    "PARALLEL": True,
    "RECORD_LAST": False,
    "BATCH_SIZE": 25,
    "RECORD_BATCH": False,
    "DETAILED_LOGGING": False,
    "DEVICE": "cpu",
    "PARALLEL_CORES": multiprocessing.cpu_count()-1 if multiprocessing.cpu_count() > 1 else 1,
    "MULTI_RUN_OPTIONS": {},
    "MULTI_EVAL_RUNS": 1,
    "RNG": np.random.default_rng(),
    "FITNESS_FUNCTION": None,
    "GECKO_BODY": None,
    "CONSOLE": None,
    "SAVE_PLOTS": False,
    "SECTIONED_MODE": False,
    "SPAWN_POSITION": None,
}

def set_config(overrides: dict):
    if overrides:
        NEURALNET_EVO_CONFIG.update(overrides)

def set_fitness_function(func):
    NEURALNET_EVO_CONFIG["FITNESS_FUNCTION"] = func

def set_gecko_body(body):
    NEURALNET_EVO_CONFIG["GECKO_BODY"] = body

def get_rng():
    return NEURALNET_EVO_CONFIG["RNG"]

def get_stats_csv_path():
    path = Path(__file__).parent / "output" / "logs" / f"gen_stats_{NEURALNET_EVO_CONFIG['FITNESS_MODE']}_run {time.strftime('%Y%m%d-%H%M%S')}.csv"
    path.parent.mkdir(exist_ok=True)
    return path

def get_stats_output_path():
    path = Path(__file__).parent / "output"
    path.mkdir(exist_ok=True)
    return path



# Double-log to file and terminal
class DualWriter:
    def __init__(self, *files: Any) -> None:
        self.files: tuple[Any, ...] = files
    def write(self, data: str) -> None:
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self) -> None:
        for f in self.files:
            f.flush()

log_file_path = Path(__file__).parent / "output" / "logs" / f"log-{time.strftime('%Y%m%d-%H%M%S')}.txt"
log_file_path.parent.mkdir(exist_ok=True)
log_file = open(log_file_path, "w", encoding="utf-8", buffering=1)  # line-buffered for immediate flush

dual_writer = DualWriter(sys.stdout, log_file)

# Fancy console messages and progress bars
if NEURALNET_EVO_CONFIG["CONSOLE"] is None:
    install()
    console = Console(file=dual_writer, emoji=False, markup=False)
    #console = Console()
    NEURALNET_EVO_CONFIG["CONSOLE"] = console
else:
    console = cast(Console, NEURALNET_EVO_CONFIG["CONSOLE"])
console.log(f"Experiment started with SEED={NEURALNET_EVO_CONFIG['SEED']}, DEVICE={NEURALNET_EVO_CONFIG['DEVICE']}, PARALLEL={NEURALNET_EVO_CONFIG['PARALLEL']}, PARALLEL_CORES={NEURALNET_EVO_CONFIG['PARALLEL_CORES']}")

plt.ioff()  # Turn off interactive mode for plotting to avoid blocking when running non-interactively

def log_generation_stats(filename, pop_mean, pop_std, pop_max):
    """
    Log generation population statistics to a CSV file.
    """
    file_exists = Path(filename).exists()
    with open(filename, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow([
                "mean_fitness", "stdev_fitness", "max_fitness"
            ])
        writer.writerow([
           pop_mean, pop_std, pop_max
        ])

def yaw_from_xmat(xmat_flat: np.ndarray) -> float:
    R = xmat_flat.reshape(3, 3)
    return math.atan2(R[1, 0], R[0, 0])  # Z-up convention


def sample_glorot_flat(weight_shapes: list[tuple[int, int]], rng: np.random.Generator) -> np.ndarray:
    """
    Sample weights using Glorot/Xavier initialization for tanh networks.
    For each weight matrix with shape (fan_in, fan_out):
    range = ±sqrt(6 / (fan_in + fan_out))
    """
    parts = []
    for fan_in, fan_out in weight_shapes:
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        W = rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)
        parts.append(W.reshape(-1))
    return np.concatenate(parts, dtype=np.float32)


def numpy_nn_controller_move_with_weights(model, data: mujoco.MjData, weights: np.ndarray, input_size, hidden_size, output_size) -> np.ndarray:
    # `weights` is expected to be a flat numpy array of parameters (float)

    # Dynamically unpack weights for multiple hidden layers
    layer_sizes = [input_size] + [hidden_size] * NEURALNET_EVO_CONFIG["NUM_HIDDEN_LAYERS"] + [output_size]
    weight_shapes = [(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
    weight_sizes = [a * b for a, b in weight_shapes]
    indices = np.cumsum([0] + weight_sizes)
    ws = [weights[indices[i]:indices[i + 1]].reshape(weight_shapes[i]) for i in range(len(weight_shapes))]

    # Forward pass
    inputs = data.qpos
    x = inputs
    for i in range(NEURALNET_EVO_CONFIG["NUM_HIDDEN_LAYERS"]):
        x = np.tanh(np.dot(x, ws[i]))
    outputs = np.tanh(np.dot(x, ws[-1]))
    outputs = outputs * np.pi
    
    return outputs


def initialize_world_and_robot(gecko_body=None):
    mujoco.set_mjcb_control(None)
    world = NEURALNET_EVO_CONFIG["SIM_WORLD"]()
    gecko_core = gecko_body if gecko_body is not None else NEURALNET_EVO_CONFIG["GECKO_BODY"] if NEURALNET_EVO_CONFIG["GECKO_BODY"] else gecko()
    
    # If gecko_core is a DiGraph (robot_graph), reconstruct the spec from it
    if isinstance(gecko_core, DiGraph):
        import copy
        gecko_core = construct_mjspec_from_graph(copy.deepcopy(gecko_core))
    
    # Determine spawn position
    if NEURALNET_EVO_CONFIG["SPAWN_POSITION"] is not None:
        spawn_pos = NEURALNET_EVO_CONFIG["SPAWN_POSITION"]
    elif NEURALNET_EVO_CONFIG["SIM_WORLD"] == SimpleFlatWorld:
        spawn_pos = [0, 0, 0]
    elif NEURALNET_EVO_CONFIG["SIM_WORLD"] == OlympicArena:
        spawn_pos = [-0.8, 0, 0.1]
    else:
        spawn_pos = [0, 0, 1]
    
    world.spawn(gecko_core.spec, spawn_position=spawn_pos, spawn_orientation=[90, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    
    # Create tracker for position and orientation data
    tracker = Tracker(
        mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM,
        name_to_bind="core",
        observable_attributes=["xpos", "xmat"],
    )
    
    return model, data, world, tracker

def convert_tracker_to_history(tracker: Tracker) -> list:
    """
    Convert tracker history to the format expected by fitness functions.
    Returns a list of [x, y, z, yaw] arrays.
    """
    if not tracker.history or "xpos" not in tracker.history or "xmat" not in tracker.history:
        return []
    
    xpos_history = tracker.history["xpos"][0]  # First tracked object
    xmat_history = tracker.history["xmat"][0]  # First tracked object
    
    history = []
    for pos, xmat in zip(xpos_history, xmat_history):
        yaw = yaw_from_xmat(xmat)
        history.append(np.array([pos[0], pos[1], pos[2], yaw], dtype=np.float32))
    
    return history

def run_bot_session(weights: np.ndarray, method: str, options: dict = None, gecko_body=None) -> list:
    """
    Run a single simulation session with given weights and method.
    """

    # Clear any existing MuJoCo callbacks for process isolation
    mujoco.set_mjcb_control(None)

    model, data, world, tracker = initialize_world_and_robot(gecko_body)

    # Define controller callback
    def nn_controller_callback(m, d):
        outputs = numpy_nn_controller_move_with_weights(m, d, weights, len(d.qpos), NEURALNET_EVO_CONFIG["HIDDEN_SIZE"], model.nu)
        return outputs

    # Instantiate Controller with tracker
    ctrl = Controller(
        controller_callback_function=nn_controller_callback,
        tracker=tracker,
        # alpha=CONFIG["OUTPUT_DELTA"],
    )

    # Setup tracker before simulation
    tracker.setup(world.spec, data)

    mujoco.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    match method:
        case "record":
            video_path = Path(__file__).parent / "output" / "videos"
            video_path.mkdir(exist_ok=True)
            video_file = ""
            if options:
                video_file += f"{options.get('filename','recording')} mode {options.get('mode','unknown')}_fit {options.get('fitness',0.0):.4f}"

            video_recorder = VideoRecorder(output_folder=video_path, file_name=video_file, width=1200, height=960, fps=30)
            tracking_video_renderer(
                model,
                data,
                duration=10 + NEURALNET_EVO_CONFIG['DURATION'],
                video_recorder=video_recorder,
            )
            mujoco.set_mjcb_control(None)
            console.log(f"Recorded episode saved to {video_path}/{video_file}")
        case "viewer":
            viewer.launch(model, data)
        case "headless": # for evaluation-only sessions
            # Use the project's simple_runner helper instead of calling mj_step directly.
            duration_seconds = NEURALNET_EVO_CONFIG["DURATION"]
            simple_runner(model, data, duration=duration_seconds)
        case _:
            raise ValueError(f"Unknown method: {method}")

    mujoco.set_mjcb_control(None)

    # Convert tracker history to expected format
    return tracker

def calc_origin_distance(history: list) -> float:
    """
    Calculate the straight-line distance from the start to the end position.
    """
    start = np.array(history[0][:2])
    end = np.array(history[-1][:2])
    return np.linalg.norm(end - start)


def calc_lateral_distance(history: list) -> float:
    """
    Calculate the lateral distance traveled by the robot perpendicular to its initial heading.
    Projects the displacement vector onto the direction perpendicular to the initial heading.
    """
    if not history or len(history) < 2:
        return 0.0
    arr = np.asarray(history, dtype=np.float32)
    x0, y0, _, yaw0 = arr[0]
    xT, yT, _, _ = arr[-1]
    dxy = np.array([xT - x0, yT - y0])
    h0 = np.array([np.cos(yaw0), np.sin(yaw0)])
    lateral = -float(np.dot(dxy, h0))
    return lateral

def calc_median_segment_lateral_distance(history: list) -> float:
    """
    Calculates the median absolute lateral distance per segment, projecting each segment's displacement onto the direction perpendicular to the initial heading.
    """
    if not history or len(history) < 2:
        return 0.0
    arr = np.asarray(history, dtype=np.float32)
    x0, y0, _, yaw0 = arr[0]
    h0 = np.array([np.cos(yaw0), np.sin(yaw0)])
    segment_laterals = []
    for i in range(0, len(arr), NEURALNET_EVO_CONFIG['SEGMENT_LENGTH']):
        segment = arr[i:i + NEURALNET_EVO_CONFIG['SEGMENT_LENGTH']]
        if len(segment) < 2:
            continue
        sx0, sy0, _, _ = segment[0]
        sxT, syT, _, _ = segment[-1]
        sdxy = np.array([sxT - sx0, syT - sy0])
        lateral = -float(np.dot(sdxy, h0))
        segment_laterals.append(abs(lateral))
    return float(np.median(segment_laterals)) if segment_laterals else 0.0


def calc_forward_distance(history: list) -> float:
    """
    Calculate the forward distance traveled by the robot along its initial heading.
    Projects the displacement vector onto the initial heading.
    """
    if not history or len(history) < 2:
        return 0.0
    arr = np.asarray(history, dtype=np.float32)
    x0, y0, _, yaw0 = arr[0]
    xT, yT, _, _ = arr[-1]
    dxy = np.array([xT - x0, yT - y0])
    h0_perp = np.array([-np.sin(yaw0), np.cos(yaw0)])
    forward = float(np.dot(dxy, h0_perp))
    return -forward

def calc_median_segment_forward_distance(history: list) -> float:
    """
    Calculates the median forward distance per segment, projecting each segment's displacement onto the initial heading.
    """
    if not history or len(history) < 2:
        return 0.0
    arr = np.asarray(history, dtype=np.float32)
    _, _, _, yaw0 = arr[0]
    h0_perp = np.array([-np.sin(yaw0), np.cos(yaw0)])
    segment_forwards = []
    for i in range(0, len(arr), NEURALNET_EVO_CONFIG['SEGMENT_LENGTH']):
        segment = arr[i:i + NEURALNET_EVO_CONFIG['SEGMENT_LENGTH']]
        if len(segment) < 2:
            continue
        sx0, sy0, _, _ = segment[0]
        sxT, syT, _, _ = segment[-1]
        sdxy = np.array([sxT - sx0, syT - sy0])
        forward = float(np.dot(sdxy, h0_perp))
        segment_forwards.append(-forward)
    return float(np.median(segment_forwards)) if segment_forwards else 0.0


def _target_unit_direction_from_start(start_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    """
    Return the 2D unit direction vector from start_xy to target_xy. If the
    distance is zero return None.
    """
    vec = np.array(target_xy[:2], dtype=np.float32) - np.array(start_xy[:2], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return None
    return vec / norm


def calc_forward_towards_target(history: list, target: list | np.ndarray) -> float:
    """
    Calculate scalar forward progress from the trajectory start towards the
    provided target (3D). Returns the projection of the overall displacement
    onto the start->target direction (positive if moving toward the target).
    """
    if not history or len(history) < 2:
        return 0.0
    arr = np.asarray(history, dtype=np.float32)
    start = arr[0][:2]
    end = arr[-1][:2]
    target = np.array(target, dtype=np.float32)[:2]
    target_dir = _target_unit_direction_from_start(start, target)
    if target_dir is None:
        return 0.0
    dxy = end - start
    return float(np.dot(dxy, target_dir))


def calc_median_segment_forward_towards_target(history: list, target: list | np.ndarray) -> float:
    """
    For each segment, project the segment displacement onto the start->target
    direction and return the median of non-negative projections.
    """
    if not history or len(history) < 2:
        return 0.0
    arr = np.asarray(history, dtype=np.float32)
    start = arr[0][:2]
    target = np.array(target, dtype=np.float32)[:2]
    target_dir = _target_unit_direction_from_start(start, target)
    if target_dir is None:
        return 0.0
    segment_forwards = []
    for i in range(0, len(arr), NEURALNET_EVO_CONFIG['SEGMENT_LENGTH']):
        segment = arr[i:i + NEURALNET_EVO_CONFIG['SEGMENT_LENGTH']]
        if len(segment) < 2:
            continue
        sx0, sy0 = segment[0][:2]
        sxT, syT = segment[-1][:2]
        sdxy = np.array([sxT - sx0, syT - sy0], dtype=np.float32)
        proj = float(np.dot(sdxy, target_dir))
        # clip to zero so only positive progress counts
        segment_forwards.append(max(proj, 0.0))
    return float(np.median(segment_forwards)) if segment_forwards else 0.0


def time_to_reach_target_seconds(history: list, target: list | np.ndarray) -> float | None:
    """
    Return the time in seconds when the trajectory first reaches or exceeds
    the target along the start->target direction. If the target is never
    reached return None. Uses CONFIG['DURATION'] to map history indices to
    seconds.
    """
    if not history or len(history) < 2:
        return None
    arr = np.asarray(history, dtype=np.float32)
    start = arr[0][:2]
    target_xy = np.array(target, dtype=np.float32)[:2]
    target_dir = _target_unit_direction_from_start(start, target_xy)
    if target_dir is None:
        return None
    target_distance = float(np.linalg.norm(target_xy - start))
    if target_distance == 0.0:
        return 0.0
    # iterate through history and find first index where projection >= target_distance
    for i in range(len(arr)):
        pos = arr[i][:2]
        proj = float(np.dot(pos - start, target_dir))
        if proj >= target_distance:
            # map index to seconds using CONFIG['DURATION'] and number of samples
            # use (len(arr)-1) as denominator so last index maps to full duration
            denom = max(1, len(arr) - 1)
            t = (i / denom) * float(NEURALNET_EVO_CONFIG.get('DURATION', 0.0))
            return float(t)
    return None


def calc_lateral_relative_to_target(history: list, target: list | np.ndarray) -> float:
    """
    Compute lateral deviation (perpendicular distance) of the final position
    relative to the line from start -> target.
    """
    if not history or len(history) < 2:
        return 0.0
    arr = np.asarray(history, dtype=np.float32)
    start = arr[0][:2]
    end = arr[-1][:2]
    target = np.array(target, dtype=np.float32)[:2]
    target_dir = _target_unit_direction_from_start(start, target)
    if target_dir is None:
        return 0.0
    dxy = end - start
    # perpendicular component = dxy - projection_along_target
    proj = float(np.dot(dxy, target_dir))
    perp = dxy - proj * target_dir
    return float(np.linalg.norm(perp))


def calc_median_segment_lateral_relative_to_target(history: list, target: list | np.ndarray) -> float:
    """
    For each segment compute the perpendicular distance to the start->target
    direction and return the median of absolute values.
    """
    if not history or len(history) < 2:
        return 0.0
    arr = np.asarray(history, dtype=np.float32)
    start = arr[0][:2]
    target = np.array(target, dtype=np.float32)[:2]
    target_dir = _target_unit_direction_from_start(start, target)
    if target_dir is None:
        return 0.0
    segment_laterals = []
    for i in range(0, len(arr), NEURALNET_EVO_CONFIG['SEGMENT_LENGTH']):
        segment = arr[i:i + NEURALNET_EVO_CONFIG['SEGMENT_LENGTH']]
        if len(segment) < 2:
            continue
        sx0, sy0 = segment[0][:2]
        sxT, syT = segment[-1][:2]
        sdxy = np.array([sxT - sx0, syT - sy0], dtype=np.float32)
        proj = float(np.dot(sdxy, target_dir))
        perp = sdxy - proj * target_dir
        segment_laterals.append(float(np.linalg.norm(perp)))
    return float(np.median(segment_laterals)) if segment_laterals else 0.0

def calc_median_segment_distance(history: list) -> float:
    # Project each segment displacement onto the global displacement direction
    # so we only reward movement in the same direction as the overall travel.
    # This prevents backward or sideways movement from increasing the score.
    total_start = np.array(history[0][:2])
    total_end = np.array(history[-1][:2])
    total_disp = total_end - total_start
    total_norm = np.linalg.norm(total_disp)
    if total_norm == 0:
        return 0.0
    total_dir = total_disp / total_norm

    projected_segment_fits = []
    for i in range(0, len(history), NEURALNET_EVO_CONFIG['SEGMENT_LENGTH']):
        segment = history[i:i + NEURALNET_EVO_CONFIG['SEGMENT_LENGTH']]
        if len(segment) < NEURALNET_EVO_CONFIG['SEGMENT_LENGTH']:
            continue
        start_pos = np.array(segment[0][:2])
        end_pos = np.array(segment[-1][:2])
        seg_disp = end_pos - start_pos
        # projection scalar of segment displacement onto total direction
        proj = float(np.dot(seg_disp, total_dir))
        # do not punish negative projections; clip to zero so only forward
        # movement increases the score
        projected_segment_fits.append(max(proj, 0.0))

    median_segment_fit = np.median(projected_segment_fits) if projected_segment_fits else 0.0
    return median_segment_fit

def fitness(history: list) -> float:


    segment_count = len(history) // NEURALNET_EVO_CONFIG['SEGMENT_LENGTH']
    origin_distance = calc_origin_distance(history)
    normalized_origin_distance = origin_distance / segment_count if segment_count > 0 else 0.0
    # Allow override
    if NEURALNET_EVO_CONFIG["FITNESS_FUNCTION"]:
        return NEURALNET_EVO_CONFIG["FITNESS_FUNCTION"](history)
    match NEURALNET_EVO_CONFIG["FITNESS_MODE"]:
        case "modern":
            arr = np.asarray(history, dtype=np.float32)
            x0, y0, z0, yaw0 = arr[0]
            xT, yT, zT, yawT = arr[-1]
            dxy = np.array([xT - x0, yT - y0])
            h0 = np.array([np.cos(yaw0), np.sin(yaw0)])
            forward = max(0.0, float(np.dot(dxy, h0)))
            lateral = float(np.max(arr[:, 1]) - np.min(arr[:, 1]))
            yaw_change = float(abs(np.unwrap(arr[:, 3].astype(float))[-1] - np.unwrap(arr[:, 3].astype(float))[0]))
            z_drop = max(0.0, float(z0 - np.min(arr[:, 2])))
            max_height = float(np.max(arr[:, 2]))
            height_penalty = max(0.0, max_height - 0.3)
            score = (
                forward
                - 0.2 * lateral
                - 0.1 * yaw_change
                - 0.5 * z_drop
                - 0.3 * height_penalty
            )
            return max(0.0, score)
        case "simple":
            fit = normalized_origin_distance
        case "segment_median":
            median_segment_fit = calc_median_segment_distance(history)
            fit = (normalized_origin_distance + median_segment_fit) / 2
        case "lateral_adjusted":
            # For OlympicArena prefer movement towards a fixed target point instead
            # of the general 'forward' direction. The chosen target is [5, 0, 0.5].
            if NEURALNET_EVO_CONFIG["SIM_WORLD"] == OlympicArena:
                target = np.array([5.0, 0.0, 0.5], dtype=np.float32)
                # compute forward progress towards the fixed target and scale to a
                # dimensionless fraction in [0, 1] where 0 == start and 1 == finish
                forward_towards_target = calc_forward_towards_target(history, target)
                # safe start position fallback if history is empty
                start_xy = np.array(history[0][:2]) if history else np.array([0.0, 0.0])
                target_distance = float(np.linalg.norm(target[:2] - start_xy))
                fraction_towards_target = (
                    forward_towards_target / target_distance if target_distance > 0.0 else 0.0
                )
                # clamp to [0, 1] so reaching or passing the goal saturates at 1.0
                fraction_towards_target = max(0.0, min(fraction_towards_target, 1.0))

                # lateral deviation normalized by the same start->target distance
                lateral_rel = calc_lateral_relative_to_target(history, target)
                normalized_lateral_distance = (
                    abs(lateral_rel) / target_distance if target_distance > 0.0 else 0.0
                )

                # final fitness: fraction toward goal minus scaled lateral penalty
                fit = max(0.0, fraction_towards_target - normalized_lateral_distance * NEURALNET_EVO_CONFIG["LATERAL_PENALTY_FACTOR"])
                # time bonus if target reached: bonus = max(0, 1 - (1/120) * t)
                t_reach = time_to_reach_target_seconds(history, target)
                if t_reach is not None:
                    bonus = max(0.0, 1.0 - (1.0 / 120.0) * float(t_reach))
                    fit = fit + bonus
            else:
                forward_distance = calc_forward_distance(history)
                normalized_forward_distance = forward_distance / segment_count if segment_count > 0 else 0.0
                lateral_distance = calc_lateral_distance(history)
                normalized_lateral_distance = abs(lateral_distance) / segment_count if segment_count > 0 else 0.0
                fit = max(0.0, normalized_forward_distance - normalized_lateral_distance * NEURALNET_EVO_CONFIG["LATERAL_PENALTY_FACTOR"])
        case "lateral_median":
            if NEURALNET_EVO_CONFIG["SIM_WORLD"] == OlympicArena:
                target = np.array([5.0, 0.0, 0.5], dtype=np.float32)
                median_forward_distance = calc_median_segment_forward_towards_target(history, target)
                median_lateral_distance = abs(calc_median_segment_lateral_relative_to_target(history, target))
                # normalize by start->target distance to produce a fraction in [0,1]
                start_xy = np.array(history[0][:2]) if history else np.array([0.0, 0.0])
                target_distance = float(np.linalg.norm(target[:2] - start_xy))
                median_fraction = (
                    median_forward_distance / target_distance if target_distance > 0.0 else 0.0
                )
                median_fraction = max(0.0, min(median_fraction, 1.0))
                normalized_median_lateral = (
                    median_lateral_distance / target_distance if target_distance > 0.0 else 0.0
                )
                fit = max(0.0, (median_fraction - normalized_median_lateral * NEURALNET_EVO_CONFIG["LATERAL_PENALTY_FACTOR"]))
                t_reach = time_to_reach_target_seconds(history, target)
                if t_reach is not None:
                    bonus = max(0.0, 1.0 - (1.0 / 120.0) * float(t_reach))
                    fit = fit + bonus
            else:
                median_forward_distance = calc_median_segment_forward_distance(history)
                median_lateral_distance = abs(calc_median_segment_lateral_distance(history))
                fit = max(0.0, (median_forward_distance - median_lateral_distance * NEURALNET_EVO_CONFIG["LATERAL_PENALTY_FACTOR"]))
        case _:
            raise ValueError(f"Unknown FITNESS_MODE: {NEURALNET_EVO_CONFIG['FITNESS_MODE']}")
    return fit


def fitness_sectioned(weights: np.ndarray, gecko_body=None) -> float:
    """
    Evaluate fitness across three sections of the Olympic Arena independently.
    Returns the sum of normalized fitness across all sections (scaled to be < 1.0).
    
    Sections:
    1. Flat: spawn=[-1.2, 0, 0.1], goal=[0.5, 0, 0.1]
    2. Rugged: spawn=[0.5, 0, 0.1], goal=[2.5, 0, 0.1] (averaged over 3 runs)
    3. Inclined: spawn=[2.5, 0, 0.1], goal=[4.5, 0, 0.1]
    """
    sections = [
        # Section 1: Flat
        {"spawn": [-1.2, 0, 0.1], "goal": [0.5, 0, 0.1], "runs": 1},
        # Section 2: Rugged (random generation, needs averaging)
        {"spawn": [0.5, 0, 0.1], "goal": [2.5, 0, 0.1], "runs": 1},
        # Section 3: Inclined
        {"spawn": [2.5, 0, 0.1], "goal": [4.5, 0, 0.1], "runs": 1},
    ]
    
    section_fitnesses = []
    
    for idx, section in enumerate(sections):
        spawn_pos = section["spawn"]
        goal_pos = section["goal"]
        num_runs = section["runs"]
        
        run_fitnesses = []
        for _ in range(num_runs):
            # Temporarily override spawn position
            original_spawn = NEURALNET_EVO_CONFIG["SPAWN_POSITION"]
            NEURALNET_EVO_CONFIG["SPAWN_POSITION"] = spawn_pos
            
            try:
                # Run simulation
                tracker = run_bot_session(weights, method="headless", gecko_body=gecko_body)
                history = convert_tracker_to_history(tracker)
                
                # Calculate fitness for this section towards its goal
                if not history or len(history) < 2:
                    run_fitnesses.append(0.0)
                    continue
                
                # Calculate progress toward section goal
                forward_towards_goal = calc_forward_towards_target(history, goal_pos)
                start_xy = np.array(history[0][:2])
                goal_xy = np.array(goal_pos[:2])
                section_distance = float(np.linalg.norm(goal_xy - start_xy))
                
                # Fraction of section completed (clamped to [0, 1])
                fraction_completed = (
                    forward_towards_goal / section_distance if section_distance > 0.0 else 0.0
                )
                fraction_completed = max(0.0, min(fraction_completed, 1.0))
                
                # Lateral penalty
                lateral_rel = calc_lateral_relative_to_target(history, goal_pos)
                normalized_lateral = (
                    abs(lateral_rel) / section_distance if section_distance > 0.0 else 0.0
                )
                
                section_fit = max(0.0, fraction_completed - normalized_lateral * NEURALNET_EVO_CONFIG["LATERAL_PENALTY_FACTOR"])
                run_fitnesses.append(section_fit)
                
            finally:
                # Restore original spawn position
                NEURALNET_EVO_CONFIG["SPAWN_POSITION"] = original_spawn
        
        # Average over runs for this section
        avg_section_fitness = float(np.mean(run_fitnesses)) if run_fitnesses else 0.0
        section_fitnesses.append(avg_section_fitness)
    
    # Sum of all sections (each is in [0, 1], so total is in [0, 3])
    # Divide by 3 to get average in [0, 1], then map to [-1, 0] range
    # This ensures complete separation: sectioned in [-1, 0], full arena in [0, 2+]
    # Formula: -1 + avg maps [0, 1] to [-1, 0]
    avg_section_fitness = sum(section_fitnesses) / 3.0
    total_fitness = -1.0 + avg_section_fitness
    
    return total_fitness

def evaluate_ind(ind: Individual) -> float:
    weights = np.array(ind.genotype, dtype=np.float32)
    runs = NEURALNET_EVO_CONFIG["MULTI_EVAL_RUNS"]
    if runs > 1:
        fits = []
        for _ in range(runs):
            tracker = run_bot_session(weights, method="headless")
            history = convert_tracker_to_history(tracker)
            fits.append(fitness(history))
        fit = float(np.mean(fits))
    else:
        tracker = run_bot_session(weights, method="headless")
        history = convert_tracker_to_history(tracker)
        fit = fitness(history)
    return fit

def evaluate_individual_isolated(genotype_list: list, gecko_body=None) -> float:
    mujoco.set_mjcb_control(None)
    weights = np.array(genotype_list, dtype=np.float32)
    try:
        # Use sectioned fitness if enabled
        if NEURALNET_EVO_CONFIG["SECTIONED_MODE"]:
            fit = fitness_sectioned(weights, gecko_body=gecko_body)
            return fit
        
        # Otherwise use normal fitness evaluation
        runs = NEURALNET_EVO_CONFIG["MULTI_EVAL_RUNS"]
        if runs > 1:
            fits = []
            for _ in range(runs):
                tracker = run_bot_session(weights, method="headless", gecko_body=gecko_body)
                history = convert_tracker_to_history(tracker)
                fits.append(fitness(history))
            fit = float(np.mean(fits))
        else:
            tracker = run_bot_session(weights, method="headless", gecko_body=gecko_body)
            history = convert_tracker_to_history(tracker)
            fit = fitness(history)
        return fit
    except Exception as e:
        console.log(f"Evaluation failed for individual: {e}")
        return -1000.0
    finally:
        mujoco.set_mjcb_control(None)


def cma_evaluate_individual(genotype_array: np.ndarray, gecko_body=None) -> float:
    fit = evaluate_individual_isolated(genotype_array.tolist(), gecko_body=gecko_body)
    if fit == -1000.0:
        return 1000.0  # penalty for minimization
    return -fit  # negate for CMA-ES minimization


def show_qpos_history(history: dict, save: bool = False) -> None:
    # Calculate fitness metrics
    fit = fitness(history)
    origin_distance = calc_origin_distance(history)
    median_segment_distance = calc_median_segment_distance(history)

    # Single plot with background
    fig, background_axis = plt.subplots(figsize=(12, 8))

    # Create background rendering
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # Initialize world to get the background
    mujoco.set_mjcb_control(None)
    world = NEURALNET_EVO_CONFIG["SIM_WORLD"]()
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Create temporary background image
    output_path = Path(__file__).parent / "output" / "plots"
    output_path.mkdir(exist_ok=True)
    background_path = output_path / "temp_background.png"

    single_frame_renderer(
        model,
        data,
        camera=camera,
        save_path=str(background_path),
        save=True,
    )

    # Setup background image
    img = plt.imread(str(background_path))
    background_axis.imshow(img)
    w, h, _ = img.shape

    # Convert list of [x,y,z,yaw] positions to numpy array (only use x,y,z)
    pos_data = np.array(history)[:, :3]  # Only take x, y, z coordinates

    # Get spawn position for coordinate conversion
    if NEURALNET_EVO_CONFIG["SIM_WORLD"] == SimpleFlatWorld:
        spawn_pos = [0, 0, 0]
    elif NEURALNET_EVO_CONFIG["SIM_WORLD"] == OlympicArena:
        spawn_pos = [-0.8, 0, 0.1]
    else:
        spawn_pos = [0, 0, 1]

    # Calculate initial position mapping (based on A3_body.py approach)
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, spawn_pos[0]

    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    # Plot trajectory on background
    background_axis.plot(x0, y0, "kx", markersize=10, label="[0, 0, 0]")
    background_axis.plot(pos_data_pixel[:, 1], pos_data_pixel[:, 0], "b-", linewidth=2, label="Path")
    background_axis.plot(yc, xc, "go", markersize=8, label="Start")  # Note: xc and yc swapped here too
    background_axis.plot(pos_data_pixel[-1, 1], pos_data_pixel[-1, 0], "ro", markersize=8, label="End")

    # Add labels and title with fitness information
    background_axis.set_xlabel("X Position (pixels)")
    background_axis.set_ylabel("Y Position (pixels)")
    bg_title = f'Robot Trajectory - Fitness ({NEURALNET_EVO_CONFIG["FITNESS_MODE"]}): {fit:.5f}'
    bg_title += f'\nOrigin Distance: {origin_distance:.3f}, Median Segment Distance: {median_segment_distance:.5f}'
    background_axis.set_title(bg_title, fontsize=12)
    background_axis.legend()

    plt.tight_layout()

    if NEURALNET_EVO_CONFIG["INTERACTIVE_MODE"]:
        plt.show(block=False)
    plt.pause(0.1)

    if save:
        timestamp = int(time.time())
        filename = output_path / f"trajectory_fit_{fit:.5f}_{timestamp}.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        console.log(f"Plot saved to {filename}")

        # Clean up temporary background file
        if background_path.exists():
            background_path.unlink()
    else:
        # Clean up temporary background file
        if background_path.exists():
            background_path.unlink()

def get_controller_from_weights(weights: np.ndarray) -> Controller:
    """
    Return a controller that uses the given weights.
    """
    def nn_controller_callback(m, d):
        outputs = numpy_nn_controller_move_with_weights(m, d, weights, len(d.qpos), CONFIG["HIDDEN_SIZE"], m.nu)
        return outputs

    ctrl = Controller(
        controller_callback_function=nn_controller_callback,
        alpha=CONFIG["OUTPUT_DELTA"],
    )
    return ctrl

def create_individual(weight_shapes: list[tuple[int, int]]) -> Individual:
    """Create a random individual with Glorot/Xavier weight initialization."""
    rng = NEURALNET_EVO_CONFIG["RNG"]
    genotype = sample_glorot_flat(weight_shapes, rng)
    ind = Individual()
    ind.genotype = genotype.tolist()  # Store as list to avoid numpy ambiguity
    ind.requires_eval = True
    return ind


def analyze_trajectory_with_plots(weights: np.ndarray, save_plots: bool = True) -> dict:
    """
    Utility function to analyze a genotype and create detailed plots.
    Returns a dictionary with analysis results.
    """
    console.log("Running trajectory analysis...")
    
    # Run simulation
    tracker = run_bot_session(weights, method="headless")
    history = convert_tracker_to_history(tracker)
    
    # Calculate metrics
    fit = fitness(history)
    origin_distance = calc_origin_distance(history)
    forward_distance = calc_forward_distance(history)
    lateral_distance = calc_lateral_distance(history)
    median_segment_distance = calc_median_segment_distance(history)
    
    # Create plots
    if save_plots:
        show_qpos_history(history, save=True)
    
    # Return analysis results
    results = {
        'fitness': fit,
        'origin_distance': origin_distance,
        'forward_distance': forward_distance,
        'lateral_distance': lateral_distance,
        'median_segment_distance': median_segment_distance,
        'history_length': len(history),
        'start_position': history[0][:3] if history else None,
        'end_position': history[-1][:3] if history else None,
    }
    
    console.log(f"Analysis complete - Fitness: {fit:.5f}")
    return results

def evolve_using_cma_es(
    config_overrides: dict = None,
    pool=None,
    fitness_function=None,
    gecko_body=None,
) -> dict[str, float]:
    """
    Main evolutionary loop using CMA-ES. Returns (best_individual.genotype, best_fitness, best_tracker).
    Allows overriding config, fitness function, and robot body.
    """
    set_config(config_overrides)
    if fitness_function:
        set_fitness_function(fitness_function)
    if gecko_body:
        set_gecko_body(gecko_body)
    if NEURALNET_EVO_CONFIG["PARALLEL"] and NEURALNET_EVO_CONFIG["PARALLEL_CORES"] > 1:
        if pool is None:
            pool = multiprocessing.Pool(NEURALNET_EVO_CONFIG["PARALLEL_CORES"])
        console.log(f"Using multiprocessing pool with {NEURALNET_EVO_CONFIG['PARALLEL_CORES']} cores")
    else:
        pool = None
        console.log("Running in single-threaded mode")
    console.rule("[green]Starting CMA-ES Run")
    model, data, world, tracker = initialize_world_and_robot()
    input_size = model.nq
    output_size = model.nu
    hidden_size = NEURALNET_EVO_CONFIG["HIDDEN_SIZE"]
    num_hidden_layers = NEURALNET_EVO_CONFIG["NUM_HIDDEN_LAYERS"]
    layer_sizes = [input_size] + [hidden_size] * num_hidden_layers + [output_size]
    weight_shapes = [(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
    total_params = sum(a * b for a, b in weight_shapes)
    
    console.log(f"Fitness Mode: {NEURALNET_EVO_CONFIG['FITNESS_MODE']}, Population Size: {NEURALNET_EVO_CONFIG['POP_SIZE']}, Total Params: {total_params}")
    
    # Initialize CMA-ES
    initial_solution = sample_glorot_flat(weight_shapes, NEURALNET_EVO_CONFIG["RNG"])
    sigma = 0.1  # Initial step size
    options = {
        'popsize': NEURALNET_EVO_CONFIG["POP_SIZE"],
        'maxiter': NEURALNET_EVO_CONFIG["MAX_GENERATIONS"],
        'verb_log': 0,  # Reduce logging
        'verb_disp': 1 if NEURALNET_EVO_CONFIG["DETAILED_LOGGING"] else 0,
    }
    es = cma.CMAEvolutionStrategy(initial_solution, sigma, options)
    
    evolution_start_time = time.time()
    
    best_fitness = -np.inf
    best_solution = None
    
    try:
        interactive_mode = NEURALNET_EVO_CONFIG["INTERACTIVE_MODE"]
        multi_run_options = NEURALNET_EVO_CONFIG["MULTI_RUN_OPTIONS"]
        if NEURALNET_EVO_CONFIG["PROGRESS"] is not None:
            progress = NEURALNET_EVO_CONFIG["PROGRESS"]
        elif multi_run_options and multi_run_options.get('progress') is not None:
            progress = multi_run_options['progress']
            multi_task = multi_run_options['task']
        else:
            progress = Progress(console=console, transient=True)
            progress.start()
        
        gen = 0
        try:
            outer_loop = progress.add_task("CMA-ES Evolution Progress", total=NEURALNET_EVO_CONFIG["MAX_GENERATIONS"])
            if interactive_mode or NEURALNET_EVO_CONFIG["DETAILED_LOGGING"]:
                inner_loop = progress.add_task(f"Generation {gen+1}", total=1)
            
            while not es.stop():
                if NEURALNET_EVO_CONFIG["TIME_LIMIT"] > 0 and (time.time() - evolution_start_time) > NEURALNET_EVO_CONFIG["TIME_LIMIT"]:
                    console.log("Time limit reached, terminating CMA-ES.")
                    break
                
                solutions = es.ask()
                fitnesses = []
                if NEURALNET_EVO_CONFIG["PARALLEL"] and pool:
                    fitnesses = pool.map(partial(cma_evaluate_individual, gecko_body=gecko_body), solutions)
                else:
                    fitnesses = [cma_evaluate_individual(x, gecko_body) for x in solutions]
                
                es.tell(solutions, fitnesses)
                
                gen += 1
                
                # Update best
                current_best_fitness = -es.result.fbest
                if current_best_fitness > best_fitness:
                    best_fitness = current_best_fitness
                    best_solution = es.result.xbest
                
                progress.update(outer_loop, completed=gen if NEURALNET_EVO_CONFIG["MAX_GENERATIONS"] > 0 else (time.time() - evolution_start_time) // 60 if NEURALNET_EVO_CONFIG["TIME_LIMIT"] > 0 else None)

                if interactive_mode or NEURALNET_EVO_CONFIG["DETAILED_LOGGING"]:
                    progress.update(inner_loop, description=f"Generation {gen} - Best Fitness: {best_fitness:.5f}")
                    if multi_run_options and multi_run_options.get('progress') is not None:
                        progress.update(multi_task, advance=1)
                
                if NEURALNET_EVO_CONFIG["RECORD_BATCH"] and gen % NEURALNET_EVO_CONFIG["BATCH_SIZE"] == 0:
                    console.log(f"Recording best individual at generation {gen} with fitness {best_fitness:.5f}")
                    best_weights = np.array(best_solution, dtype=np.float32)
                    run_bot_session(best_weights, method="record", options={"filename": f"cma_batch_{gen}", "mode": NEURALNET_EVO_CONFIG["FITNESS_MODE"], "fitness": best_fitness})
                    save_genotype(best_weights, best_fitness)
                    tracker = run_bot_session(best_weights, method="headless")
                    history = convert_tracker_to_history(tracker)
                    if NEURALNET_EVO_CONFIG["INTERACTIVE_MODE"] or NEURALNET_EVO_CONFIG["SAVE_PLOTS"]:
                        show_qpos_history(history, save=NEURALNET_EVO_CONFIG["SAVE_PLOTS"])
                
                if interactive_mode and gen % 10 == 0:  # Check every 10 generations for interactivity
                    progress.stop()
                    console.rule(f"Generation {gen} - Best Fitness: {best_fitness:.5f}")
                    console.log("Running best individual in viewer...")
                    console.log(f"Current runtime: {(time.time() - evolution_start_time)/60:.2f} minutes")
                    best_weights = np.array(best_solution, dtype=np.float32)
                    history = run_bot_session(best_weights, method="headless")
                    show_qpos_history(history)
                    console.log(f"total distance walked: {calc_origin_distance(history):.2f}")
                    console.log(f"total forward distance: {calc_forward_distance(history):.2f}")
                    console.log(f"total lateral distance: {calc_lateral_distance(history):.2f}")
                    console.log(f"Make sure to close the viewer window to continue evolution.")
                    run_bot_session(best_weights, method="viewer")
                    user_input = Prompt.ask("Continue CMA-ES? (y)es, (n)o", choices=["y", "n"], default="y")
                    if user_input == 'n':
                        console.log("CMA-ES terminated by user.")
                        break
                    progress.start()

                if gen % NEURALNET_EVO_CONFIG["BATCH_SIZE"] == 0:
                    console.log(f"CMA-ES Evolution Generation {gen} - Best Fitness so far: {best_fitness:.5f}")
                
        finally:
            if multi_run_options is not None and multi_run_options.get('progress') is not None:
                pass
            else:
                progress.stop()
    finally:
        if pool:
            pool.close()
            pool.join()
    
    best_weights = np.array(best_solution, dtype=np.float32)
    if NEURALNET_EVO_CONFIG["RECORD_LAST"]:
        run_bot_session(best_weights, method="record", options={"filename": "cma_final_recording", "mode": NEURALNET_EVO_CONFIG["FITNESS_MODE"], "fitness": best_fitness})

    tracker = run_bot_session(best_weights, method="headless")
    history = convert_tracker_to_history(tracker)
    save_genotype(best_weights, best_fitness)
    if NEURALNET_EVO_CONFIG["INTERACTIVE_MODE"] or NEURALNET_EVO_CONFIG["SAVE_PLOTS"]:
        show_qpos_history(history, save=NEURALNET_EVO_CONFIG["SAVE_PLOTS"])

    console.rule(f"CMA-ES complete in {(time.time() - evolution_start_time)/60:.2f} minutes. Best fitness: {best_fitness:.5f}")
    console.log(f"Best fitness: {best_fitness:.5f}")
    console.log(f"Total distance walked: {calc_origin_distance(history):.2f}")
    console.log(f"Total forward distance: {calc_forward_distance(history):.2f}")
    console.log(f"Total lateral distance: {calc_lateral_distance(history):.2f}")
    console.log(f"Median forward distance: {calc_median_segment_forward_distance(history):.2f}")
    console.log(f"Median lateral distance: {calc_median_segment_lateral_distance(history):.2f}")
    console.log(f"Sanity check - re-evaluated fitness: {fitness(history):.5f}")
    
    if interactive_mode:
        console.log("Running best individual in viewer...")
        run_bot_session(best_weights, method="viewer")
    
    # Return as Individual for compatibility
    best_ind = Individual()
    best_ind.genotype = best_solution.tolist()
    best_ind.fitness = best_fitness
    return {"genotype": best_ind.genotype, "fitness": best_fitness, "tracker": tracker}

def save_genotype(weights: np.ndarray, fitness: float = 0.0) -> None:
    """
    Saves a genotype (numpy network weights) to a .npy file.
    """
    output_path = Path(__file__).parent / "output" / "genotypes"
    output_path.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = output_path / f"genotype_cma_{NEURALNET_EVO_CONFIG['FITNESS_MODE']}_fit {fitness:.4f}_{timestamp}"
    np.save(filename, weights)
    console.log(f"Saved best genotype to {filename}.npy")


def load_genotype(file_path: str) -> np.ndarray:
    """
    Loads a genotype (numpy network weights) from a .npy file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Genotype file not found: {file_path}")
    weights = np.load(path)
    console.log(f"Loaded genotype from {file_path}")
    return weights

def test_loaded_genotype(file_path: str) -> None:
    """
    Loads a genotype from file, evaluates its fitness, and runs it in viewer mode.
    """
    weights = load_genotype(file_path)
    tracker = run_bot_session(weights, method="headless")
    history = convert_tracker_to_history(tracker)
    fit = fitness(history)
    console.log(f"Tested loaded genotype fitness: {fit:.5f}")
    if NEURALNET_EVO_CONFIG["INTERACTIVE_MODE"]:
        run_bot_session(weights, method="viewer")
        show_qpos_history(history)

def run_weights_only(weights: np.ndarray, method: str = "viewer", options: dict = None, gecko_body: DiGraph = None) -> None:
    """
    Runs a provided set of weights in the specified method (viewer, headless, record).
    """
    tracker = run_bot_session(weights, method=method, options=options, gecko_body=gecko_body)
    history = convert_tracker_to_history(tracker)
    fit = fitness(history)
    console.log(f"Ran provided weights with fitness: {fit:.5f}")
    if method == "headless" and NEURALNET_EVO_CONFIG["INTERACTIVE_MODE"]:
        show_qpos_history(history)
    if method == "viewer":
        console.log("Viewer session complete.")

def plot_weights_only(weights: np.ndarray, save: bool = True) -> None:
    """
    Analyzes and plots the trajectory of the provided weights without running evolution.
    """
    results = analyze_trajectory_with_plots(weights, save_plots=save)
    console.log("Trajectory analysis results:")
    for key, value in results.items():
        console.log(f"  {key}: {value}")

def fitness_of_weights(weights: np.ndarray) -> float:
    """
    Evaluates and returns the fitness of the provided weights.
    """
    tracker = run_bot_session(weights, method="headless")
    history = convert_tracker_to_history(tracker)
    fit = fitness(history)
    console.log(f"Evaluated fitness of provided weights: {fit:.5f}")
    return fit

def main():
    evolve_using_cma_es()

if __name__ == "__main__":
    main()
