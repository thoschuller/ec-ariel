#type: ignore

"""
This is an evolutionary algorithm experiment using the Ariel framework.
It evolves a population of numpy neural network weights to control a gecko robot.
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

from ariel.simulation.controllers.controller import Controller

from typing import cast

# Local libraries
from ariel.utils.renderers import tracking_video_renderer, single_frame_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.simulation.environments.olympic_arena import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.ec.a001 import Individual, JSONIterable
from ariel.ec.a004 import EAStep, EA, Population
# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

import random


# --- Configurable global settings --- #
CONFIG = {
    "SIM_WORLD": OlympicArena,
    "SEED": 42,
    "SEGMENT_LENGTH": 250,
    "POP_SIZE": 15,
    "MAX_GENERATIONS": 75,
    "TIME_LIMIT": 60*15, # expected duration is 7-8 minutes
    "HIDDEN_SIZE": 8,
    "DURATION": 15,
    "OUTPUT_DELTA": 0.05,
    "NUM_HIDDEN_LAYERS": 1,
    "FITNESS_MODE": "lateral_adjusted",
    "UNIFORM_CROSSOVER": True,
    "LATERAL_PENALTY_FACTOR": 0.1,
    "MULTI_EVAL": True,
    "INTERACTIVE_MODE": False,
    "PARALLEL": True,
    "RECORD_LAST": True,
    "BATCH_SIZE": 20,
    "RECORD_BATCH": True,
    "DETAILED_LOGGING": True,
    "DEVICE": "cpu",
    "PARALLEL_CORES": multiprocessing.cpu_count()-1 if multiprocessing.cpu_count() > 1 else 1,
    "MULTI_RUN_OPTIONS": {},
    "MULTI_EVAL_RUNS": 5,
    "RNG": np.random.default_rng(42),
    "FITNESS_FUNCTION": None,  # Allow override
    "GECKO_BODY": None,        # Allow override
    "CONSOLE": None,
    "MUTATION_PROBABILITY": 0.5,
    "MUTATION_STDDEV": 0.1,
}

# --- Pool management --- #
GLOBAL_POOL = None
def get_pool():
    global GLOBAL_POOL
    if GLOBAL_POOL is None and CONFIG["PARALLEL"] and CONFIG["PARALLEL_CORES"] > 1:
        GLOBAL_POOL = multiprocessing.Pool(processes=CONFIG["PARALLEL_CORES"])
    return GLOBAL_POOL
def close_pool():
    global GLOBAL_POOL
    if GLOBAL_POOL is not None:
        GLOBAL_POOL.close()
        GLOBAL_POOL.join()
        GLOBAL_POOL = None

def set_config(overrides: dict):
    if overrides:
        CONFIG.update(overrides)

def set_fitness_function(func):
    CONFIG["FITNESS_FUNCTION"] = func

def set_gecko_body(func):
    CONFIG["GECKO_BODY"] = func

def get_rng():
    return CONFIG["RNG"]

def get_stats_csv_path():
    path = Path(__file__).parent / "output" / "logs" / f"gen_stats_{CONFIG['FITNESS_MODE']}_run {time.strftime('%Y%m%d-%H%M%S')}.csv"
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
if CONFIG["CONSOLE"] is None:
    install()
    # console = Console(file=dual_writer, emoji=False, markup=False)
    console = Console()
    CONFIG["CONSOLE"] = console
else:
    console = cast(Console, CONFIG["CONSOLE"])
console.log(f"Experiment started with SEED={CONFIG['SEED']}, DEVICE={CONFIG['DEVICE']}, PARALLEL={CONFIG['PARALLEL']}, PARALLEL_CORES={CONFIG['PARALLEL_CORES']}")

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


def numpy_nn_controller_move_with_weights(model, data: mujoco.MjData, weights: np.ndarray, input_size, hidden_size, output_size) -> np.ndarray:
    # `weights` is expected to be a flat numpy array of parameters (float)

    # Dynamically unpack weights for multiple hidden layers
    layer_sizes = [input_size] + [hidden_size] * CONFIG["NUM_HIDDEN_LAYERS"] + [output_size]
    weight_shapes = [(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
    weight_sizes = [a * b for a, b in weight_shapes]
    indices = np.cumsum([0] + weight_sizes)
    ws = [weights[indices[i]:indices[i + 1]].reshape(weight_shapes[i]) for i in range(len(weight_shapes))]

    # Forward pass
    inputs = data.qpos
    x = inputs
    for i in range(CONFIG["NUM_HIDDEN_LAYERS"]):
        x = np.tanh(np.dot(x, ws[i]))
    outputs = np.tanh(np.dot(x, ws[-1]))
    outputs = outputs * (np.pi / 2)  # Scale to [-pi/2, pi/2]
    
    return outputs


def initialize_world_and_robot():
    mujoco.set_mjcb_control(None)
    world = CONFIG["SIM_WORLD"]()
    gecko_core = CONFIG["GECKO_BODY"]() if CONFIG["GECKO_BODY"] else gecko()
    if CONFIG["SIM_WORLD"] == SimpleFlatWorld:
        spawn_pos = [0, 0, 0]
    elif CONFIG["SIM_WORLD"] == OlympicArena:
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

def run_bot_session(weights: np.ndarray, method: str, options: dict = None) -> list:
    """
    Run a single simulation session with given weights and method.
    """

    # Clear any existing MuJoCo callbacks for process isolation
    mujoco.set_mjcb_control(None)

    model, data, world, tracker = initialize_world_and_robot()
    # Define controller callback
    def nn_controller_callback(m, d):
        outputs = numpy_nn_controller_move_with_weights(m, d, weights, model.nq, CONFIG["HIDDEN_SIZE"], model.nu)
        return outputs

    # Instantiate Controller with tracker
    ctrl = Controller(
        controller_callback_function=nn_controller_callback,
        tracker=tracker,
        alpha=CONFIG["OUTPUT_DELTA"],
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
                duration=10 + CONFIG['DURATION'],
                video_recorder=video_recorder,
            )
            mujoco.set_mjcb_control(None)
            console.log(f"Recorded episode saved to {video_path}/{video_file}")
        case "viewer":
            viewer.launch(model, data)
        case "headless": # for evaluation-only sessions
            # Use the project's simple_runner helper instead of calling mj_step directly.
            # Convert the number of simulation steps to seconds using the model timestep.
            try:
                timestep = float(model.opt.timestep)
            except Exception:
                # Fall back to a reasonable default timestep if unavailable
                timestep = 0.002
            duration_seconds = CONFIG["DURATION"]
            simple_runner(model, data, duration=duration_seconds)
        case _:
            raise ValueError(f"Unknown method: {method}")

    mujoco.set_mjcb_control(None)

    # Convert tracker history to expected format
    return convert_tracker_to_history(tracker)

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
    for i in range(0, len(arr), CONFIG['SEGMENT_LENGTH']):
        segment = arr[i:i + CONFIG['SEGMENT_LENGTH']]
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
    for i in range(0, len(arr), CONFIG['SEGMENT_LENGTH']):
        segment = arr[i:i + CONFIG['SEGMENT_LENGTH']]
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
    for i in range(0, len(arr), CONFIG['SEGMENT_LENGTH']):
        segment = arr[i:i + CONFIG['SEGMENT_LENGTH']]
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
            t = (i / denom) * float(CONFIG.get('DURATION', 0.0))
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
    for i in range(0, len(arr), CONFIG['SEGMENT_LENGTH']):
        segment = arr[i:i + CONFIG['SEGMENT_LENGTH']]
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
    for i in range(0, len(history), CONFIG['SEGMENT_LENGTH']):
        segment = history[i:i + CONFIG['SEGMENT_LENGTH']]
        if len(segment) < CONFIG['SEGMENT_LENGTH']:
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


    segment_count = len(history) // CONFIG['SEGMENT_LENGTH']
    origin_distance = calc_origin_distance(history)
    normalized_origin_distance = origin_distance / segment_count if segment_count > 0 else 0.0
    # Allow override
    if CONFIG["FITNESS_FUNCTION"]:
        return CONFIG["FITNESS_FUNCTION"](history)
    match CONFIG["FITNESS_MODE"]:
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
            if CONFIG["SIM_WORLD"] == OlympicArena:
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
                fit = max(0.0, fraction_towards_target - normalized_lateral_distance * CONFIG["LATERAL_PENALTY_FACTOR"])
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
                fit = max(0.0, normalized_forward_distance - normalized_lateral_distance * CONFIG["LATERAL_PENALTY_FACTOR"])
        case "lateral_median":
            if CONFIG["SIM_WORLD"] == OlympicArena:
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
                fit = max(0.0, (median_fraction - normalized_median_lateral * CONFIG["LATERAL_PENALTY_FACTOR"]))
                t_reach = time_to_reach_target_seconds(history, target)
                if t_reach is not None:
                    bonus = max(0.0, 1.0 - (1.0 / 120.0) * float(t_reach))
                    fit = fit + bonus
            else:
                median_forward_distance = calc_median_segment_forward_distance(history)
                median_lateral_distance = abs(calc_median_segment_lateral_distance(history))
                fit = max(0.0, (median_forward_distance - median_lateral_distance * CONFIG["LATERAL_PENALTY_FACTOR"]))
        case _:
            raise ValueError(f"Unknown FITNESS_MODE: {CONFIG['FITNESS_MODE']}")
    return fit

def evaluate_ind(ind: Individual) -> float:
    weights = np.array(ind.genotype, dtype=np.float32)
    runs = CONFIG["MULTI_EVAL_RUNS"]
    if runs > 1:
        fits = []
        for _ in range(runs):
            history = run_bot_session(weights, method="headless")
            fits.append(fitness(history))
        fit = float(np.mean(fits))
    else:
        history = run_bot_session(weights, method="headless")
        fit = fitness(history)
    return fit

def evaluate_pop(pop: Population, pool=None) -> Population:
    if CONFIG["PARALLEL"] and CONFIG["PARALLEL_CORES"] > 1 and pool is not None:
        to_eval = [ind for ind in pop if ind.requires_eval]
        if to_eval:
            genotypes = [ind.genotype for ind in to_eval]
            fitness_values = pool.map(evaluate_individual_isolated, genotypes)
            for ind, fitness_val in zip(to_eval, fitness_values):
                ind.fitness = fitness_val
                ind.requires_eval = False
    else:
        for ind in pop:
            if ind.requires_eval:
                ind.fitness = evaluate_ind(ind)
                ind.requires_eval = False
    return pop

def evaluate_individual_isolated(genotype_list: list) -> float:
    mujoco.set_mjcb_control(None)
    weights = np.array(genotype_list, dtype=np.float32)
    try:
        runs = CONFIG["MULTI_EVAL_RUNS"]
        if runs > 1:
            fits = []
            for _ in range(runs):
                history = run_bot_session(weights, method="headless")
                fits.append(fitness(history))
            fit = float(np.mean(fits))
        else:
            history = run_bot_session(weights, method="headless")
            fit = fitness(history)
        return fit
    except Exception as e:
        console.log(f"Evaluation failed for individual: {e}")
        return -1000.0
    finally:
        mujoco.set_mjcb_control(None)


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
    world = CONFIG["SIM_WORLD"]()
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
    if CONFIG["SIM_WORLD"] == SimpleFlatWorld:
        spawn_pos = [0, 0, 0]
    elif CONFIG["SIM_WORLD"] == OlympicArena:
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
    background_axis.plot(xc, yc, "go", markersize=8, label="Start")
    background_axis.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", linewidth=2, label="Path")
    background_axis.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", markersize=8, label="End")

    # Add labels and title with fitness information
    background_axis.set_xlabel("X Position (pixels)")
    background_axis.set_ylabel("Y Position (pixels)")
    bg_title = f'Robot Trajectory - Fitness ({CONFIG["FITNESS_MODE"]}): {fit:.5f}'
    bg_title += f'\nOrigin Distance: {origin_distance:.3f}, Median Segment Distance: {median_segment_distance:.5f}'
    background_axis.set_title(bg_title, fontsize=12)
    background_axis.legend()

    plt.tight_layout()

    if CONFIG["INTERACTIVE_MODE"]:
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

def create_individual(weight_shapes: list[tuple[int, int]]) -> Individual:
    """Create a random individual with Glorot/Xavier weight initialization."""
    rng = CONFIG["RNG"]
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
    history = run_bot_session(weights, method="headless")
    
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

def create_population(weight_shapes: list[tuple[int, int]], pop_size: int, pool=None) -> Population:
    """Create population using Glorot/Xavier weight initialization."""
    console.log(f"WORLD: {CONFIG['SIM_WORLD'].__name__}, POP_SIZE: {pop_size}, WEIGHT_SHAPES: {weight_shapes}, PARALLEL: {CONFIG['PARALLEL']}, PARALLEL_CORES: {CONFIG['PARALLEL_CORES']}")
    if pool:
        return pool.map(create_individual, [weight_shapes] * pop_size)
    else:
        return [create_individual(weight_shapes) for _ in range(pop_size)]

def parent_selection(population: Population) -> Population:
    """Tournament selection"""

    # Shuffle population to avoid bias
    random.shuffle(population)

    # Tournament selection
    for idx in range(0, len(population) - 1, 2):
        ind_i = population[idx]
        ind_j = population[idx + 1]

        # Compare fitness values and update tags
        if ind_i.fitness > ind_j.fitness:
            ind_i.tags['ps'] = True
            ind_j.tags['ps'] = False
        else:
            ind_i.tags['ps'] = False
            ind_j.tags['ps'] = True
    return population


class Crossover:
    @staticmethod
    def one_point(
        parent_i: JSONIterable,
        parent_j: JSONIterable,
        crossover_point: int = None,
    ) -> tuple[JSONIterable, JSONIterable]:
        parent_i_arr_shape = np.array(parent_i).shape
        parent_j_arr_shape = np.array(parent_j).shape
        parent_i_arr = np.array(parent_i).flatten().copy()
        parent_j_arr = np.array(parent_j).flatten().copy()
        if parent_i_arr_shape != parent_j_arr_shape:
            msg = "Parents must have the same length"
            raise ValueError(msg)
        # Use CONFIG['RNG'] for multiprocessing compatibility
        if crossover_point is None:
            crossover_point = CONFIG['RNG'].integers(0, len(parent_i_arr))
        child1 = parent_i_arr.copy()
        child2 = parent_j_arr.copy()
        child1[crossover_point:] = parent_j_arr[crossover_point:]
        child2[crossover_point:] = parent_i_arr[crossover_point:]
        child1 = child1.reshape(parent_i_arr_shape).astype(float).tolist()
        child2 = child2.reshape(parent_j_arr_shape).astype(float).tolist()
        return child1, child2
    
    @staticmethod
    def uniform(
        parent_i: JSONIterable,
        parent_j: JSONIterable,
    ) -> tuple[JSONIterable, JSONIterable]:
        parent_i_arr_shape = np.array(parent_i).shape
        parent_j_arr_shape = np.array(parent_j).shape
        parent_i_arr = np.array(parent_i).flatten().copy()
        parent_j_arr = np.array(parent_j).flatten().copy()
        if parent_i_arr_shape != parent_j_arr_shape:
            msg = "Parents must have the same length"
            raise ValueError(msg)
        # Use CONFIG['RNG'] for multiprocessing compatibility
        mask = CONFIG['RNG'].integers(0, 2, size=len(parent_i_arr)).astype(bool)
        child1 = parent_i_arr.copy()
        child2 = parent_j_arr.copy()
        child1[mask] = parent_j_arr[mask]
        child2[mask] = parent_i_arr[mask]
        child1 = child1.reshape(parent_i_arr_shape).astype(float).tolist()
        child2 = child2.reshape(parent_j_arr_shape).astype(float).tolist()
        return child1, child2

def crossover_individuals(ind1 : Individual, ind2: Individual, uniform_crossover: bool = False) -> tuple[Individual, Individual]:

    # Perform one-point crossover

    parent_i = ind1.model_copy()
    parent_j = ind2.model_copy()

    # Decide which to crossover and which to clone directly

    if random.random() < 0.25:
        genotype_i = parent_i.genotype
        genotype_j = parent_j.genotype
    
    else:
        if uniform_crossover:
            genotype_i, genotype_j = Crossover.uniform(
                cast("list[float]", parent_i.genotype),
                cast("list[float]", parent_j.genotype),
            )
        else:
            genotype_i, genotype_j = Crossover.one_point(
                cast("list[float]", parent_i.genotype),
                cast("list[float]", parent_j.genotype),
            )

    # First child   
    child_i = Individual()
    child_i.genotype = genotype_i
    child_i.tags['mut'] = True
    child_i.requires_eval = True

    # Second child
    child_j = Individual()
    child_j.genotype = genotype_j
    child_j.tags['mut'] = True
    child_j.requires_eval = True

    parent_i.tags['ps'] = False
    parent_j.tags['ps'] = False

    return child_i, child_j


def crossover_parallel(population: Population, pool) -> Population:
    parents = [ind for ind in population if ind.tags.get('ps', False)]
    #shuffle parents to avoid bias
    random.shuffle(parents)
    if pool and len(parents) >= 2:
        children = []
        parent_pairs = [(parents[i], parents[i + 1]) for i in range(0, len(parents) - 1, 2)]
        children_pairs = pool.starmap(crossover_individuals, [(p1, p2, CONFIG['UNIFORM_CROSSOVER']) for p1, p2 in parent_pairs])
        for child_i, child_j in children_pairs:
            children.extend([child_i, child_j])
        population.extend(children)
    elif not pool:
        for idx in range(0, len(parents) - 1, 2):
            parent_i = parents[idx]
            parent_j = parents[idx+1]
            child_i, child_j = crossover_individuals(parent_i, parent_j, CONFIG['UNIFORM_CROSSOVER'])
            population.extend([child_i, child_j])
        else:
            pass
    return population

def mutate_float(
    individual: list[float],
    input_size: int,
    hidden_size: int,
    output_size: int,
    num_hidden_layers: int,
    mutation_probability: float = 0.5,
    stddev: float = 0.1,
) -> list[float]:
    """
    Mutate weights using per-layer bounds based on Glorot initialization.
    Each layer gets its own mutation scale and clamping bounds.
    """
    rng = CONFIG['RNG']
    arr = np.array(individual, dtype=np.float32, copy=True)

    # Reconstruct per-layer views (same logic as in controller)
    layer_sizes = [input_size] + [hidden_size] * num_hidden_layers + [output_size]
    shapes = [(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
    sizes = [a*b for a, b in shapes]
    idx = np.cumsum([0] + sizes)

    for li, (fan_in, fan_out) in enumerate(shapes):
        start, end = idx[li], idx[li+1]
        view = arr[start:end].reshape(fan_in, fan_out)
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        sigma = stddev * limit  # Scale mutation by the layer's initialization range
        mask = rng.random(view.shape) < mutation_probability
        view[mask] = view[mask] + rng.normal(0.0, sigma, size=mask.sum()).astype(np.float32)
        # Clamp per layer to 3x the initialization range
        view[...] = np.clip(view, -3*limit, 3*limit)

    return arr.astype(float).tolist()

def mutate_individual_with_shape(ind: Individual, input_size, hidden_size, output_size, num_hidden_layers, mutation_probability, stddev) -> Individual:
    mutated = mutate_float(
        individual=cast("list[float]", ind.genotype),
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_hidden_layers=num_hidden_layers,
        mutation_probability=mutation_probability,
        stddev=stddev,
    )
    ind.genotype = mutated
    ind.tags = {'mut': False}
    ind.requires_eval = True
    return ind

def mutation(population: Population, pool) -> Population:
    to_mutate = [ind for ind in population if ind.tags.get('mut', False)]
    # Get network shape info from CONFIG (should be set in evolve_using_ariel_ec)
    input_size = CONFIG.get("INPUT_SIZE")
    hidden_size = CONFIG["HIDDEN_SIZE"]
    output_size = CONFIG.get("OUTPUT_SIZE")
    num_hidden_layers = CONFIG["NUM_HIDDEN_LAYERS"]
    mutation_probability = CONFIG['MUTATION_PROBABILITY']
    stddev = CONFIG['MUTATION_STDDEV']
    if pool and to_mutate:
        # Use a wrapper to pass shape info
        import functools
        mutate_fn = functools.partial(
            mutate_individual_with_shape,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_hidden_layers=num_hidden_layers,
            mutation_probability=mutation_probability,
            stddev=stddev,
        )
        mutated_inds = pool.map(mutate_fn, to_mutate)
        # Replace mutated individuals in population
        mutate_idx = [i for i, ind in enumerate(population) if ind.tags.get('mut', False)]
        for idx, mutated in zip(mutate_idx, mutated_inds):
            population[idx] = mutated
    elif not pool:
        for i, ind in enumerate(population):
            if ind.tags.get('mut', False):
                population[i] = mutate_individual_with_shape(
                    ind,
                    input_size,
                    hidden_size,
                    output_size,
                    num_hidden_layers,
                    mutation_probability,
                    stddev,
                )
    return population

def survivor_selection(population: Population) -> Population:

    # Shuffle population to avoid bias
    random.shuffle(population)
    current_pop_size = len(population)

    for idx in range(len(population)):
        ind_i = population[idx]
        ind_j = population[idx + 1]

        # Kill worse individual
        if ind_i.fitness > ind_j.fitness:
            ind_j.alive = False
        else:
            ind_i.alive = False

        # Termination condition
        current_pop_size -= 1
        if current_pop_size <= CONFIG["POP_SIZE"]:
            break
    return population


def log_stats(population: Population) -> Population:
    pop_fitness = [ind.fitness for ind in population]
    pop_mean = float(np.mean(pop_fitness)) if pop_fitness else 0.0
    pop_std = float(np.std(pop_fitness)) if pop_fitness else 0.0
    pop_max = float(np.max(pop_fitness)) if pop_fitness else 0.0

    log_generation_stats(
        filename=STATS_CSV_PATH,
        pop_mean=pop_mean,
        pop_std=pop_std,
        pop_max=pop_max,
    )
    # console.log(f"Generation mean fitness = {pop_mean:.5f}, std = {pop_std:.5f}, max = {pop_max:.5f}")
    return population



def evolve_using_ariel_ec(
    config_overrides: dict = None,
    pool=None,
    fitness_function=None,
    gecko_body=None,
) -> tuple[Any, float]:
    """
    Main evolutionary loop. Returns (best_individual, best_fitness).
    Allows overriding config, fitness function, and robot body.
    """
    set_config(config_overrides)
    if fitness_function:
        set_fitness_function(fitness_function)
    if gecko_body:
        set_gecko_body(gecko_body)
    pool = pool if pool is not None else get_pool()
    console.rule("[green]Starting Evolutionary Run")
    model, data, world, tracker = initialize_world_and_robot()
    input_size = model.nq
    output_size = model.nu
    hidden_size = CONFIG["HIDDEN_SIZE"]
    num_hidden_layers = CONFIG["NUM_HIDDEN_LAYERS"]
    layer_sizes = [input_size] + [hidden_size] * num_hidden_layers + [output_size]
    weight_shapes = [(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
    total_params = sum(a * b for a, b in weight_shapes)
    
    # Set CONFIG values needed for per-layer mutation
    CONFIG["INPUT_SIZE"] = input_size
    CONFIG["OUTPUT_SIZE"] = output_size
    class EAPoolStep(EAStep):
        def __init__(self, name: str, operation, pool=None):
            super().__init__(name, operation)
            self.pool = pool
        def __call__(self, *args, **kwargs):
            # Always pass the pool parameter to functions that expect it
            kwargs['pool'] = self.pool
            return self.operation(*args, **kwargs)
    evolution_start_time = time.time()
    try:
        pop: Population = create_population(weight_shapes=weight_shapes, pop_size=CONFIG["POP_SIZE"], pool=pool)
        pop = evaluate_pop(pop, pool=pool)
        console.log(f"Fitness Mode: {CONFIG['FITNESS_MODE']}, Population Size: {len(pop)}, Total Params: {total_params}")
        ops = [
            EAPoolStep("evaluation", evaluate_pop, pool=pool),
            EAStep("parent_selection", parent_selection),
            EAPoolStep("crossover", crossover_parallel, pool=pool),
            EAPoolStep("mutation",  mutation, pool=pool),
            EAPoolStep("evaluation", evaluate_pop, pool=pool),
            EAStep("survivor_selection", survivor_selection),
            # EAStep("log_stats", log_stats),
        ]
        ea = EA(
            population=pop,
            operations=ops,
            num_of_generations=CONFIG["MAX_GENERATIONS"],
            quiet=False,
        )
        def terminate() -> bool:
            if CONFIG["TIME_LIMIT"] > 0 and (time.time() - evolution_start_time) > CONFIG["TIME_LIMIT"]:
                console.log("Time limit reached, terminating evolution.")
                return True
            if CONFIG["MAX_GENERATIONS"] > 0 and gen >= CONFIG["MAX_GENERATIONS"]:
                console.log("Max generations reached, terminating evolution.")
                return True
            return False
        interactive_mode = CONFIG["INTERACTIVE_MODE"]
        multi_run_options = CONFIG["MULTI_RUN_OPTIONS"]
        if multi_run_options and multi_run_options.get('progress') is not None:
            progress = multi_run_options['progress']
            multi_task = multi_run_options['task']
        else:
            progress = Progress(console=console, transient=True)
            progress.start()
        gen = 0
        try:
            outer_loop = progress.add_task("Evolution Progress", total=CONFIG["MAX_GENERATIONS"])
            if interactive_mode or CONFIG["DETAILED_LOGGING"]:
                inner_loop = progress.add_task(f"Generation {gen+1} to {gen+CONFIG['BATCH_SIZE']}", total=CONFIG["BATCH_SIZE"])
            while not terminate():
                batch_size = CONFIG["BATCH_SIZE"] if (CONFIG["MAX_GENERATIONS"] <= 0 or gen + CONFIG["BATCH_SIZE"] <= CONFIG["MAX_GENERATIONS"]) else (CONFIG["MAX_GENERATIONS"] - gen)
                if interactive_mode or CONFIG["DETAILED_LOGGING"]:
                    progress.reset(inner_loop, description=f"Batch {gen // CONFIG['BATCH_SIZE'] + 1}" if CONFIG["MAX_GENERATIONS"] > 0 else f"Generation {gen+1} to {gen+CONFIG['BATCH_SIZE']}", total=batch_size, completed=0)
                    console.log(f"Starting batch {gen // CONFIG['BATCH_SIZE'] + 1}" if CONFIG["MAX_GENERATIONS"] > 0 else f"Starting generation {gen+1} to {gen+CONFIG['BATCH_SIZE']}")
                for _ in range(batch_size):
                    ea.step()
                    gen += 1
                    if interactive_mode or CONFIG["DETAILED_LOGGING"]:
                        progress.update(inner_loop, advance=1)
                        progress.update(outer_loop, completed=gen if CONFIG["MAX_GENERATIONS"] > 0 else (time.time() - evolution_start_time) // 60 if CONFIG["TIME_LIMIT"] > 0 else None, description=f"Evolution Progress - current best {ea.get_solution('best', only_alive=False).fitness:.5f}")
                        if multi_run_options and multi_run_options.get('progress') is not None:
                            progress.update(multi_task, advance=1)
                best_individual: Individual = ea.get_solution('best', only_alive=False)
                best_weights = np.array(best_individual.genotype, dtype=np.float32)
                if CONFIG["RECORD_BATCH"] or interactive_mode:
                    best_history = run_bot_session(best_weights, method="headless")
                if CONFIG["RECORD_BATCH"]:
                    console.log(f"Recording best individual of generation {gen} with fitness {best_individual.fitness:.5f}")
                    run_bot_session(best_weights, method="record", options={"filename": "auto_recording", "mode": CONFIG["FITNESS_MODE"], "fitness": best_individual.fitness})
                    save_genotype(best_weights, best_individual.fitness)
                    show_qpos_history(best_history, save=True)
                if interactive_mode:
                    progress.stop()
                    console.rule(f"Generation {gen} - Best Fitness: {best_individual.fitness:.5f}")
                    console.log("Running best individual in viewer...")
                    console.log(f"Current runtime: {(time.time() - evolution_start_time)/60:.2f} minutes")
                    show_qpos_history(best_history)
                    console.log(f"total distance walked: {calc_origin_distance(best_history):.2f}")
                    console.log(f"total forward distance: {calc_forward_distance(best_history):.2f}")
                    console.log(f"total lateral distance: {calc_lateral_distance(best_history):.2f}")
                    console.log(f"Make sure to close the viewer window to continue evolution.")
                    run_bot_session(best_weights, method="viewer")
                    user_input = Prompt.ask("Continue evolution? (y)es, (n)o, (s)kip interactive", choices=["y", "n", "s"], default="y")
                    if user_input == 'n':
                        console.log("Evolution terminated by user.")
                        break
                    elif user_input == 's':
                        interactive_mode = False
                        progress.remove_task(inner_loop)
                        console.log("Skipping further interactive prompts.")
                    else:
                        progress.start()
                else:
                    pass
                if terminate():
                    break
                progress.update(outer_loop, description=f"Evolution Progress - current best {best_individual.fitness:.5f}", completed=gen if CONFIG["MAX_GENERATIONS"] > 0 else (time.time() - evolution_start_time) // 60 if CONFIG["TIME_LIMIT"] > 0 else None)
        finally:
            if multi_run_options is not None and multi_run_options.get('progress') is not None:
                pass
            else:
                progress.stop()
    finally:
        pass  # Pool is managed externally
    best = ea.get_solution("best", only_alive=False)
    median = ea.get_solution("median", only_alive=False)
    worst = ea.get_solution("worst", only_alive=False)
    best_weights = np.array(best.genotype, dtype=np.float32)
    if CONFIG["RECORD_LAST"]:
        run_bot_session(best_weights, method="record", options={"filename": "final_recording", "mode": CONFIG["FITNESS_MODE"], "fitness": best.fitness})
    history = run_bot_session(best_weights, method="headless")
    median_history = run_bot_session(np.array(median.genotype, dtype=np.float32), method="headless")
    worst_history = run_bot_session(np.array(worst.genotype, dtype=np.float32), method="headless")
    show_qpos_history(worst_history, save=True)
    show_qpos_history(median_history, save=True)
    save_genotype(best_weights, best.fitness)
    show_qpos_history(history, save=True)
    console.rule(f"Evolution complete in {(time.time() - evolution_start_time)/60:.2f} minutes.   Best fitness: {best.fitness:.5f}")
    console.log(f"Best fitness: {best.fitness:.5f}")
    console.log(f"Median fitness: {median.fitness:.5f}")
    console.log(f"Worst fitness: {worst.fitness:.5f}")
    console.rule("Final Best Individual Analysis")
    console.log(f"Best fitness: {best.fitness:.5f}")
    console.log(f"Total distance walked: {calc_origin_distance(history):.2f}")
    console.log(f"Total forward distance: {calc_forward_distance(history):.2f}")
    console.log(f"Total lateral distance: {calc_lateral_distance(history):.2f}")
    console.log(f"Median forward distance: {calc_median_segment_forward_distance(history):.2f}")
    console.log(f"Median lateral distance: {calc_median_segment_lateral_distance(history):.2f}")
    console.log(f"Sanity check - re-evaluated fitness: {fitness(history):.5f}")
    if interactive_mode:
        console.log("Running best individual in viewer...")
        run_bot_session(best_weights, method="viewer")
    return best, best.fitness

def save_genotype(weights: np.ndarray, fitness: float = 0.0) -> None:
    """
    Saves a genotype (numpy network weights) to a .npy file.
    """
    output_path = Path(__file__).parent / "output" / "genotypes"
    output_path.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = output_path / f"genotype_{CONFIG['FITNESS_MODE']}_fit {fitness:.4f}_{timestamp}"
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
    history = run_bot_session(weights, method="headless")
    fit = fitness(history)
    console.log(f"Tested loaded genotype fitness: {fit:.5f}")
    if CONFIG["INTERACTIVE_MODE"]:
        run_bot_session(weights, method="viewer")
        show_qpos_history(history)

def main():
    evolve_using_ariel_ec()

if __name__ == "__main__":
    main()
