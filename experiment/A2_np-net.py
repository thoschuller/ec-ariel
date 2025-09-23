# Third-party libraries
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
from numpy import floating
from rich.console import Console
from rich.traceback import install
from rich.progress import track, Progress
from rich.prompt import Prompt
import math

from typing import cast, Literal

# Local libraries
from ariel.utils.renderers import tracking_video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.simulation.environments.rugged_heightmap import RuggedTerrainWorld
import ariel.ec as ec
from ariel.ec.a000 import IntegerMutator
from ariel.ec.a001 import Individual
from ariel.ec.a005 import Crossover
from ariel.ec.a004 import EASettings, EAStep, EA, Population
# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

import random


# === experiment constants/settings ===
SIM_WORLD = SimpleFlatWorld
SEGMENT_LENGTH = 250
POP_SIZE = 50
MAX_GENERATIONS = 250
TIME_LIMIT = 60 * 60 * 10000  # max run time in seconds
HIDDEN_SIZE = 8
SIM_STEPS = 7500  # running at 500 steps per second
OUTPUT_DELTA = 0.05 # change in output per step, to smooth out controls
NUM_HIDDEN_LAYERS = 1
FITNESS_MODE = "lateral_adjusted"  # Options: "segment_median", "simple", "modern", "lateral_adjusted", "lateral_median"

INTERACTIVE_MODE = False  # If True, show and ask every X generations; if False, run to max
# NOTE: keep PARALLEL disabled for now, seems to cause an interaction bug with mujoco
PARALLEL = True  # If True, evaluate individuals in parallel using multiple CPU cores
# IMPORTANT NOTE: in interactive mode, it is required to close the viewer window to continue running
RECORD_LAST = True  # If True, record a video of the last individual
BATCH_SIZE = 50
RECORD_BATCH = True  # If True, record a video of the best individual every BATCH_SIZE generations
DETAILED_LOGGING = True  # If True, log detailed fitness components each generation


UPRIGHT_FACTOR = 0  # Set to 0 to ignore vertical orientation
LATERAL_PENALTY_FACTOR = 0.1 


# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
PARALLEL_CORES = 1 if DEVICE == "cuda" or PARALLEL == False else multiprocessing.cpu_count() - 1  # leave one core free for the system itself

global SEED
SEED = 42
global RNG
RNG = np.random.default_rng(SEED)
np.random.seed(SEED)
random.seed(SEED)


install()
console = Console()

config = EASettings()
config.is_maximisation = True
config.db_handling = "delete"

plt.ioff()  # Turn off interactive mode for plotting to avoid blocking

# def sigmoid(x):
#         return 1.0 / (1.0 + np.exp(-x))

# def random_controller_move(model, data: mujoco.MjData, to_track, weights: np.ndarray, input_size, hidden_size, output_size, history: dict) -> None:
#     num_joints = model.nu 
#     hinge_range = np.pi/2
#     rand_moves = np.random.uniform(low= -hinge_range, high=hinge_range, size=num_joints) 
#     delta = 0.05
#     data.ctrl += rand_moves * delta 
#     data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)
#     history['xpos'].append(to_track[0].xpos.copy())
#     history['xmat'].append(to_track[0].xmat.copy())

def yaw_from_xmat(xmat_flat: np.ndarray) -> float:
    R = xmat_flat.reshape(3, 3)
    return math.atan2(R[1, 0], R[0, 0])  # Z-up convention


def numpy_nn_controller_move_with_weights(model, data: mujoco.MjData, to_track, weights: np.ndarray, input_size, hidden_size, output_size, history: list) -> None:
    # `weights` is expected to be a flat numpy array of parameters (float)

    # Dynamically unpack weights for multiple hidden layers
    layer_sizes = [input_size] + [hidden_size] * NUM_HIDDEN_LAYERS + [output_size]
    weight_shapes = [(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
    weight_sizes = [a * b for a, b in weight_shapes]
    indices = np.cumsum([0] + weight_sizes)
    ws = [weights[indices[i]:indices[i + 1]].reshape(weight_shapes[i]) for i in range(len(weight_shapes))]

    # Forward pass
    inputs = data.qpos
    x = inputs
    for i in range(NUM_HIDDEN_LAYERS):
        x = np.tanh(np.dot(x, ws[i]))
    outputs = np.tanh(np.dot(x, ws[-1]))
    outputs = outputs * (np.pi / 2)  # Scale to [-pi/2, pi/2]
    outputs += outputs * OUTPUT_DELTA
    data.ctrl = np.clip(outputs, -np.pi / 2, np.pi / 2)
    pos = to_track[0].xpos.copy()
    yaw = yaw_from_xmat(to_track[0].xmat.copy())
    history.append(np.array([pos[0], pos[1], pos[2], yaw], dtype=np.float32))   


def initialize_world_and_robot() -> Any:
    mujoco.set_mjcb_control(None)

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = SIM_WORLD()

    # Initialise robot body
    # YOU MUST USE THE GECKO BODY
    gecko_core = gecko()     # DO NOT CHANGE

    # Spawn robot in the world
    # Check docstring for spawn conditions
    if SIM_WORLD == RuggedTerrainWorld:
        spawn_pos = [0, 0, 0.25]
    else:
        spawn_pos = [0, 0, 0]

    world.spawn(gecko_core.spec, spawn_position=spawn_pos)

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Initialise data tracking
    # to_track is automatically updated every time step
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    return model, data, to_track

def run_bot_session(weights: np.ndarray, method: str, options: dict = None) -> list:
    np.random.seed(SEED)
    random.seed(SEED)
    # Clear any existing MuJoCo callbacks for process isolation
    mujoco.set_mjcb_control(None)
    
    model, data, to_track = initialize_world_and_robot()

    # Initialise history tracking
    history = []
    
    mujoco.set_mjcb_control(lambda m, d: numpy_nn_controller_move_with_weights(m, d, to_track, weights, model.nq, HIDDEN_SIZE, model.nu, history))


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
                duration=10 + SIM_STEPS / 500,
                video_recorder=video_recorder,
            )
            mujoco.set_mjcb_control(None)
            console.log(f"Recorded episode saved to {video_path}/{video_file}")
        case "viewer":
            viewer.launch(model, data)
        case "headless":
            mujoco.mj_step(model, data, nstep=SIM_STEPS)

    mujoco.set_mjcb_control(None)

    return history

def calc_origin_distance(history: list) -> float:
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
    for i in range(0, len(arr), SEGMENT_LENGTH):
        segment = arr[i:i + SEGMENT_LENGTH]
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
    x0, y0, _, yaw0 = arr[0]
    h0_perp = np.array([-np.sin(yaw0), np.cos(yaw0)])
    segment_forwards = []
    for i in range(0, len(arr), SEGMENT_LENGTH):
        segment = arr[i:i + SEGMENT_LENGTH]
        if len(segment) < 2:
            continue
        sx0, sy0, _, _ = segment[0]
        sxT, syT, _, _ = segment[-1]
        sdxy = np.array([sxT - sx0, syT - sy0])
        forward = float(np.dot(sdxy, h0_perp))
        segment_forwards.append(-forward)
    return float(np.median(segment_forwards)) if segment_forwards else 0.0

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
    for i in range(0, len(history), SEGMENT_LENGTH):
        segment = history[i:i + SEGMENT_LENGTH]
        if len(segment) < SEGMENT_LENGTH:
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


    segment_count = SIM_STEPS / SEGMENT_LENGTH

    # calculate distance from origin
    origin_distance = calc_origin_distance(history)
    normalized_origin_distance = origin_distance / segment_count if segment_count > 0 else 0.0

    match FITNESS_MODE:
        case "modern":
            arr = np.asarray(history, dtype=np.float32)
            x0, y0, z0, yaw0 = arr[0]
            xT, yT, zT, yawT = arr[-1]

            # forward progress along initial heading
            dxy = np.array([xT - x0, yT - y0])
            h0 = np.array([np.cos(yaw0), np.sin(yaw0)])
            forward = max(0.0, float(np.dot(dxy, h0)))

            # side drift
            lateral = float(np.max(arr[:, 1]) - np.min(arr[:, 1]))

            # yaw change
            yaw_change = float(abs(np.unwrap(arr[:, 3].astype(float))[-1] -
                                np.unwrap(arr[:, 3].astype(float))[0]))

            # crouching penalty (z drop)
            z_drop = max(0.0, float(z0 - np.min(arr[:, 2])))

            # high jump penalty
            max_height = float(np.max(arr[:, 2]))
            height_penalty = max(0.0, max_height - 0.3)  # threshold adjustable

            # combine
            score = (
                forward
                - 0.2 * lateral
                - 0.1 * yaw_change
                - 0.5 * z_drop
                - 0.3 * height_penalty     
            )
            return max(0.0, score)   # optional clamp to avoid negatives

        case "simple":
            fit = normalized_origin_distance

        case "segment_median":
            # calculate median distance per segment to reward steady movement
            median_segment_fit = calc_median_segment_distance(history)
            fit = (normalized_origin_distance + median_segment_fit) / 2
        case "lateral_adjusted":
            forward_distance = calc_forward_distance(history)
            normalized_forward_distance = forward_distance / segment_count if segment_count > 0 else 0.0
            lateral_distance = calc_lateral_distance(history)
            normalized_lateral_distance = abs(lateral_distance)
            fit = max(0.0, normalized_forward_distance - normalized_lateral_distance * LATERAL_PENALTY_FACTOR)
        case "lateral_median":
            median_forward_distance = calc_median_segment_forward_distance(history)
            median_lateral_distance = abs(calc_median_segment_lateral_distance(history))
            fit = max(0.0, (median_forward_distance - median_lateral_distance * LATERAL_PENALTY_FACTOR))



        case _:
            raise ValueError(f"Unknown FITNESS_MODE: {FITNESS_MODE}")

        # combine with origin distance to promote outward movement
    return fit

def evaluate_ind(ind: Individual) -> float:
    # Convert genotype back to numpy array for evaluation
    weights = np.array(ind.genotype, dtype=np.float32)
    history = run_bot_session(weights, method="headless")
    fit = fitness(history)
    return fit

def evaluate_pop(pop: Population, pool=None) -> Population:
    if PARALLEL and PARALLEL_CORES > 1 and pool is not None:
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
    """
    Isolated evaluation function for multiprocessing.
    This function ensures each process has its own MuJoCo instance and state.
    """
    # Clear any existing MuJoCo callbacks to ensure isolation
    mujoco.set_mjcb_control(None)
    
    # Convert genotype back to numpy array
    weights = np.array(genotype_list, dtype=np.float32)
    
    try:
        # Each process gets its own fresh MuJoCo environment
        history = run_bot_session(weights, method="headless")
        fit = fitness(history)
        return fit
    except Exception as e:
        # Return a very low fitness if evaluation fails
        console.log(f"Evaluation failed for individual: {e}")
        return -1000.0
    finally:
        # Clean up MuJoCo callbacks to prevent interference
        mujoco.set_mjcb_control(None)


def show_qpos_history(history: dict, save: bool = False) -> None:
    fit = fitness(history)
    origin_distance = calc_origin_distance(history)
    median_segment_distance = calc_median_segment_distance(history)

    # Convert list of [x, y, z, yaw] arrays to numpy array
    pos_data = np.array(history)

    plt.figure(figsize=(10, 6))

    # Plot x, y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Robot Trajectory - Fitness ({FITNESS_MODE}): {fit:.5f}, Origin Distance: {origin_distance:.2f}, Median Segment Distance: {median_segment_distance:.5f}')
    plt.legend()
    plt.grid(True)

    plt.axis('equal')
    max_range = max(abs(pos_data[:, :2]).max(), 0.3) * 1.5
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)

    plt.show(block=False)
    plt.draw()
    plt.pause(0.1)

    if save:
        output_path = Path(__file__).parent / "output" / "plots"
        output_path.mkdir(exist_ok=True)
        timestamp = int(time.time())
        filename = output_path / f"trajectory_fit_{fit:.5f}_{timestamp}.png"
        plt.savefig(filename)
        console.log(f"Plot saved to {filename}")

def create_individual(total_params: int) -> Individual:
    # Create a random individual with weights in [-1, 1]
    genotype = np.random.uniform(-1.0, 1.0, size=total_params).astype(np.float32)
    ind = Individual()
    ind.genotype = genotype.tolist()  # Store as list to avoid numpy ambiguity
    return ind

def create_population(total_params: int, pop_size: int, pool = None) -> Population:
    if pool:
        return pool.map(create_individual, [total_params] * pop_size)
    else:
        return [create_individual(total_params) for _ in range(pop_size)]

def parent_selection(population: Population) -> Population:
    """Tournament selection"""
    safe_population = population.copy()

    # Shuffle population to avoid bias
    random.shuffle(safe_population)

    # Tournament selection
    for idx in range(0, len(safe_population) - 1, 2):
        ind_i = safe_population[idx]
        ind_j = safe_population[idx + 1]

        # Compare fitness values and update tags
        if ind_i.fitness > ind_j.fitness and config.is_maximisation:
            ind_i.tags = {"ps": True}
            ind_j.tags = {"ps": False}
        else:
            ind_i.tags = {"ps": False}
            ind_j.tags = {"ps": True}
    return population

def crossover(population: Population) -> Population:
    """One point crossover"""

    parents = [ind for ind in population if ind.tags.get("ps", False)]
    for idx in range(0, len(parents) - 1, 2):
        parent_i = parents[idx].model_copy()
        parent_j = parents[idx+1].model_copy()
        genotype_i, genotype_j = Crossover.one_point(
            cast("list[float]", parent_i.genotype),
            cast("list[float]", parent_j.genotype),
        )

        # First child   
        child_i = Individual()
        child_i.genotype = genotype_i
        child_i.tags = {"mut": True}
        child_i.requires_eval = True

        # Second child
        child_j = Individual()
        child_j.genotype = genotype_j
        child_j.tags = {"mut": True}
        child_j.requires_eval = True

        population.extend([child_i, child_j])
    return population

def crossover_individuals(ind1 : Individual, ind2: Individual) -> tuple[Individual, Individual]:
    parent_i = ind1.model_copy()
    parent_j = ind2.model_copy()
    genotype_i, genotype_j = Crossover.one_point(
        cast("list[float]", parent_i.genotype),
        cast("list[float]", parent_j.genotype),
    )

    # First child   
    child_i = Individual()
    child_i.genotype = genotype_i
    child_i.tags = {"mut": True}
    child_i.requires_eval = True

    # Second child
    child_j = Individual()
    child_j.genotype = genotype_j
    child_j.tags = {"mut": True}
    child_j.requires_eval = True

    return child_i, child_j


def crossover_parallel(population: Population, pool) -> Population:
    parents = [ind for ind in population if ind.tags.get("ps", False)]
    children = []
    if pool and len(parents) >= 2:
        parent_pairs = [(parents[i], parents[i + 1]) for i in range(0, len(parents) - 1, 2)]
        children_pairs = pool.starmap(crossover_individuals, parent_pairs)
        for child_i, child_j in children_pairs:
            children.extend([child_i, child_j])
    elif not pool:
        children = crossover(population)
    population.extend(children)
    return population

def mutation(population: Population) -> Population:
    for ind in population:
        if ind.tags.get("mut", False):
            genes = cast("list[float]", ind.genotype)
            mutated = IntegerMutator.float_creep(
                individual=genes,
                span=5,
                mutation_probability=0.5,
            )
            ind.genotype = mutated
            ind.tags = {"mut": False}
            ind.requires_eval = True

    return population

def mutate_individual(ind: Individual) -> Individual:

    mutated = IntegerMutator.float_creep(
        individual=cast("list[float]", ind.genotype),
        span=5,
        mutation_probability=0.5,
    )
    ind.genotype = mutated
    ind.tags = {"mut": False}
    ind.requires_eval = True
    return ind

def mutation_parallel(population: Population, pool) -> Population:
    to_mutate = [ind for ind in population if ind.tags.get("mut", False)]
    if pool and to_mutate:
        mutated_inds = pool.map(mutate_individual, to_mutate)
        # Replace mutated individuals in population
        mutate_idx = [i for i, ind in enumerate(population) if ind.tags.get("mut", False)]
        for idx, mutated in zip(mutate_idx, mutated_inds):
            population[idx] = mutated
    elif not pool:
        population = mutation(population)
    return population

def survivor_selection(population: Population) -> Population:

    # Shuffle population to avoid bias
    random.shuffle(population)
    current_pop_size = len(population)

    for idx in range(len(population)):
        ind_i = population[idx]
        ind_j = population[idx + 1]

        # Kill worse individual
        if ind_i.fitness > ind_j.fitness and config.is_maximisation:
            ind_j.alive = False
        else:
            ind_i.alive = False

        # Termination condition
        current_pop_size -= 1
        if current_pop_size <= config.target_population_size:
            break
    return population


def evolve_using_ariel_ec():    

    console.rule("[green]Starting Evolutionary Run")

    # input size is the number of position sensors (qpos)
    model, data, to_track = initialize_world_and_robot()
    input_size = model.nq
    output_size = model.nu  # number of actuators
    hidden_size = HIDDEN_SIZE
    num_hidden_layers = NUM_HIDDEN_LAYERS

    # compute total number of scalar weights used by the controller (all dense layers)
    layer_sizes = [input_size] + [hidden_size] * num_hidden_layers + [output_size]
    weight_shapes = [(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
    total_params = sum(a * b for a, b in weight_shapes)

    # remove all variables that are no longer necessary

    class EAPoolStep(EAStep):
        def __init__(self, name: str, operation, pool=None):
            super().__init__(name, operation)
            self.pool = pool

        def __call__(self, *args, **kwargs):
            # Only pass pool if not already present in kwargs
            if self.pool is not None and 'pool' not in kwargs:
                return self.operation(*args, pool=self.pool, **kwargs)
            else:
                return self.operation(*args, **kwargs)

    pool = None
    if PARALLEL and PARALLEL_CORES > 1:
        pool = multiprocessing.Pool(processes=PARALLEL_CORES)

    evolution_start_time = time.time()
    

    try:
        pop: Population = create_population(total_params=total_params, pop_size=POP_SIZE, pool=pool)
        pop = evaluate_pop(pop, pool=pool)

        ops = [
            EAPoolStep("evaluation", evaluate_pop, pool=pool),
            EAStep("parent_selection", parent_selection),
            EAPoolStep("crossover", crossover_parallel, pool=pool),
            EAPoolStep("mutation",  mutation_parallel, pool=pool),
            EAPoolStep("evaluation", evaluate_pop, pool=pool),
            EAStep("survivor_selection", survivor_selection)
        ]

        # initialize EA
        ea = EA(
            population=pop,
            operations=ops,
            num_of_generations=MAX_GENERATIONS,
            quiet=False,
        )
        def terminate() -> bool:
            if TIME_LIMIT > 0 and (time.time() - evolution_start_time) > TIME_LIMIT:
                console.log("Time limit reached, terminating evolution.")
                return True
            if MAX_GENERATIONS > 0 and gen >= MAX_GENERATIONS:
                console.log("Max generations reached, terminating evolution.")
                return True
            return False
        
        interactive_mode = INTERACTIVE_MODE


        progress = Progress()
        progress.start()
        gen = 0

        try:

            outer_loop = progress.add_task("Evolution Progress", total=MAX_GENERATIONS if MAX_GENERATIONS > 0 else TIME_LIMIT // 60 if TIME_LIMIT > 0 else None)
            if interactive_mode or DETAILED_LOGGING:
                inner_loop = progress.add_task(f"Generation {gen+1} to {gen+BATCH_SIZE}", total=BATCH_SIZE)

            # Main evolutionary loop

            while not terminate():
                batch_size = BATCH_SIZE if (MAX_GENERATIONS <= 0 or gen + BATCH_SIZE <= MAX_GENERATIONS) else (MAX_GENERATIONS - gen)
                if interactive_mode or DETAILED_LOGGING:
                    progress.reset(inner_loop, description=f"Batch {gen // BATCH_SIZE + 1}" if MAX_GENERATIONS > 0 else f"Generation {gen+1} to {gen+BATCH_SIZE}", total=batch_size, completed=0)
                for _ in range(batch_size):
                    ea.step()
                    gen += 1
                    if interactive_mode or DETAILED_LOGGING:
                        progress.update(inner_loop, advance=1)
                        progress.update(outer_loop, completed=gen if MAX_GENERATIONS > 0 else (time.time() - evolution_start_time) // 60 if TIME_LIMIT > 0 else None, description=f"Evolution Progress - current best {ea.get_solution('best', only_alive=False).fitness:.5f}")
                best_individual: Individual = ea.get_solution('best', only_alive=False)
                best_weights = np.array(best_individual.genotype, dtype=np.float32)

                if interactive_mode:
                    progress.stop()
                    console.rule(f"Generation {gen} - Best Fitness: {best_individual.fitness:.5f}")
                    console.log("Running best individual in viewer...")
                    console.log(f"Current runtime: {(time.time() - evolution_start_time)/60:.2f} minutes")

                    history = run_bot_session(best_weights, method="headless")
                    show_qpos_history(history)
                    console.log(f"total distance walked: {calc_origin_distance(history):.2f}")
                    console.log(f"total forward distance: {calc_forward_distance(history):.2f}")
                    console.log(f"total lateral distance: {calc_lateral_distance(history):.2f}")

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
                if RECORD_BATCH:
                    run_bot_session(best_weights, method="record", options={"filename": "auto_recording", "mode": FITNESS_MODE, "fitness": best_individual.fitness})
                    save_genotype(best_weights, best_individual.fitness)
                if terminate():
                    break
                progress.update(outer_loop, description=f"Evolution Progress - current best {best_individual.fitness:.5f}", completed=gen if MAX_GENERATIONS > 0 else (time.time() - evolution_start_time) // 60 if TIME_LIMIT > 0 else None)


        finally:
            progress.stop()
    finally:
        if pool:
            pool.close()
            pool.join()        

    best = ea.get_solution("best", only_alive=False)
    median = ea.get_solution("median", only_alive=False)
    worst = ea.get_solution("worst", only_alive=False)

    
    best_weights = np.array(best.genotype, dtype=np.float32)

    if RECORD_LAST:
        run_bot_session(best_weights, method="record", options={"filename": "final_recording", "mode": FITNESS_MODE, "fitness": best.fitness})

    history = run_bot_session(best_weights, method="headless")

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

    if interactive_mode:
        console.log("Running best individual in viewer...")
        run_bot_session(best_weights, method="viewer")

    return ea

def save_genotype(weights: np.ndarray, fitness: float) -> None:
    output_path = Path(__file__).parent / "output" / "genotypes"
    output_path.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = output_path / f"genotype_{FITNESS_MODE}_fit {fitness:.4f}_{timestamp}"
    np.save(filename, weights)
    console.log(f"Saved best genotype to {filename}.npy")


def load_genotype(file_path: str) -> np.ndarray:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Genotype file not found: {file_path}")
    weights = np.load(path)
    console.log(f"Loaded genotype from {file_path}")
    return weights

def test_loaded_genotype(file_path: str) -> None:
    weights = load_genotype(file_path)
    history = run_bot_session(weights, method="headless")
    fit = fitness(history)
    console.log(f"Tested loaded genotype fitness: {fit:.5f}")
    run_bot_session(weights, method="viewer")
    show_qpos_history(history)

def main():
    ea: EA = evolve_using_ariel_ec()

if __name__ == "__main__":
    main()
