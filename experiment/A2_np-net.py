# Third-party libraries
from typing import Any

import evotorch.logging
import matplotlib
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
from evotorch import Problem, Solution
from evotorch.algorithms import *
import time
import os
from pathlib import Path
import multiprocessing
from evotorch.neuroevolution import NEProblem
import evotorch.neuroevolution.net

# if you get errors here, you may need to install torch and torchvision
# uv pip install torch torchvision --torch-backend=auto
from numpy import floating
import torch

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


# === experiment constants/settings ===
SEGMENT_LENGTH = 250
POP_SIZE = 50
MAX_GENERATIONS = 5000
TIME_LIMIT = 60 * 60 * 0.5  # max run time in seconds
HIDDEN_SIZE = 8
SIM_STEPS = 7500  # running at 500 steps per second
OUTPUT_DELTA = 0.05 # change in output per step, to smooth out controls
NUM_HIDDEN_LAYERS = 1
FITNESS_MODE = "segment_median"  # Options: "segment_median", "simple"
INTERACTIVE_MODE = False  # If True, show and ask every X generations; if False, run to max
# IMPORTANT NOTE: in interactive mode, it is required to close the viewer window to continue running
RECORD_LAST = True  # If True, record a video of the last individual
BATCH_SIZE = 20  # Number of individuals to evaluate before running interactive prompts
RECORD_BATCH = True  # If True, record a video of the best individual every BATCH_SIZE generations
RECORD_LAST = True  # If True, record a video of the last individual
SEEDING_ATTEMPTS = 2  # Number of attempts to seed the initial population
SEEDING_GENERATIONS = 10  # Number of generations to run for seeding

# # Staged evolution settings
# STAGED_EVOLUTION = False  # If True, use staged evolution
# STAGE1_GENERATIONS = 5  # Number of generations for stage 1
# STAGE2_GENERATIONS = 10  # Number of generations for stage 2
# STAGE3_GENERATIONS = MAX_GENERATIONS - STAGE1_GENERATIONS - STAGE2_GENERATIONS  # Remaining generations for stage 3
# ===========================


# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
PARALLEL_CORES = 1 if DEVICE == "cuda" else multiprocessing.cpu_count() - 1  # leave one core free for the system itself

plt.ioff()  # Turn off interactive mode for plotting to avoid blocking

def numpy_nn_controller_move_with_weights(model, data, to_track, net: torch.nn.Module, input_size, hidden_size, output_size, history) -> None:
    # `weights` is expected to be a flat numpy array of parameters (float)
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    weights: floating[np.ndarray] = net if isinstance(net, np.ndarray) else np.array(net)

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
        x = sigmoid(np.dot(x, ws[i]))
    outputs = sigmoid(np.dot(x, ws[-1])) - 0.5  # Center around 0
    data.ctrl += outputs * OUTPUT_DELTA
    data.ctrl = np.clip(data.ctrl, -np.pi / 2, np.pi / 2)
    history.append(to_track[0].xpos.copy())

def initialize_world_and_robot() -> Any:
    mujoco.set_mjcb_control(None)

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = SimpleFlatWorld()

    # Initialise robot body
    # YOU MUST USE THE GECKO BODY
    gecko_core = gecko()     # DO NOT CHANGE

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    
    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mujoco.MjData(model) # type: ignore

    # Initialise data tracking
    # to_track is automatically updated every time step
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    return model, data, to_track

def run_bot_session(net: torch.nn.Module, method: str, options: dict = None) -> list:
    # Accept either a torch.nn.Module or a tensor-like object containing parameters.
    if isinstance(net, torch.nn.Module):
        try:
            weights_tensor = torch.cat([p.detach().cpu().flatten() for p in net.parameters()])
        except Exception:
            # Fallback: if module has no parameters, try to treat it as a tensor-like object
            weights_tensor = torch.tensor([])
    elif isinstance(net, torch.Tensor):
        weights_tensor = net.detach().cpu()
    elif isinstance(net, np.ndarray):
        weights_tensor = torch.from_numpy(net).float()
    else:
        # Last resort: try to convert to numpy then tensor
        try:
            arr = np.asarray(net)
            weights_tensor = torch.from_numpy(arr).float()
        except Exception:
            raise TypeError("Unsupported network type for run_bot_session")

    weights = weights_tensor.numpy().flatten()
    model, data, to_track = initialize_world_and_robot()

    # Initialise history tracking
    history = []
    
    mujoco.set_mjcb_control(lambda m, d: numpy_nn_controller_move_with_weights(m, d, to_track, weights, model.nq, HIDDEN_SIZE, model.nu, history))

    
    if method == "record":
        video_path = Path(__file__).parent / "output" / "videos"
        video_path.mkdir(exist_ok=True)
        video_file = ""
        if options:
            video_file += f"{options.get('filename','recording')} mode {options.get('mode','unknown')}_fit {options.get('fitness',0.0):.4f}"

        video_recorder = VideoRecorder(output_folder=video_path, file_name=video_file, width=1200, height=960, fps=30)
        video_renderer(
            model,
            data,
            duration=10 + SIM_STEPS / 500,
            video_recorder=video_recorder,
        )
        mujoco.set_mjcb_control(None)
        print(f"Recorded episode saved to {video_path}/{video_file}")
    elif method == "viewer":
        viewer.launch(model, data)
    elif method == "headless":
        mujoco.mj_step(model, data, nstep=SIM_STEPS)

    mujoco.set_mjcb_control(None)

    return history

def calc_origin_distance(history: list) -> float:
    start = np.array(history[0][:2])
    end = np.array(history[-1][:2])
    return np.linalg.norm(end - start)

def calc_median_segment_distance(history: list) -> float:
    segment_fits = []
    for i in range(0, len(history), SEGMENT_LENGTH):
        segment = history[i:i + SEGMENT_LENGTH]
        if len(segment) < SEGMENT_LENGTH:
            continue
        start_pos = segment[0][:2]
        end_pos = segment[-1][:2]
        segment_fit = np.linalg.norm(end_pos - start_pos)
        segment_fits.append(segment_fit)
    median_segment_fit = np.median(segment_fits) if segment_fits else 0.0
    return median_segment_fit

def fitness(history: list) -> float:
    segment_count = SIM_STEPS // SEGMENT_LENGTH

    # calculate distance from origin
    origin_distance = calc_origin_distance(history)
    normalized_origin_distance = origin_distance / segment_count if segment_count > 0 else 0.0

    if FITNESS_MODE == "simple":
        fit = normalized_origin_distance
    elif FITNESS_MODE == "segment_median":
        # calculate median distance per segment to reward steady movement
        median_segment_fit = calc_median_segment_distance(history)

        # combine with origin distance to promote outward movement
        fit = (normalized_origin_distance + median_segment_fit) / 2

    else:
        raise ValueError(f"Unknown FITNESS_MODE: {FITNESS_MODE}")
    return fit

def evaluate_network(net: torch.nn.Module) -> float:
    history = run_bot_session(net, method="headless")
    if len(history) < 2:
        return 0.0
    fit = fitness(history)
    return fit

def show_qpos_history(history: list) -> None:
    fit = fitness(history)
    origin_distance = calc_origin_distance(history)
    median_segment_distance = calc_median_segment_distance(history)

    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')
    
    # Add labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position') 

    plt.title(f'Robot Trajectory - Fitness ({FITNESS_MODE}): {fit:.5f}, Origin Distance: {origin_distance:.2f}, Median Segment Distance: {median_segment_distance:.5f}')
    plt.legend()
    plt.grid(True)

    # Set equal aspect ratio and center at (0,0)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 0.3) * 1.5 
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)

    plt.show(block=False)
    plt.draw()
    plt.pause(0.1)  # Pause to update the plot


def evolve():
    start_time = time.time()

    # input size is the number of position sensors (qpos)
    model, data, to_track = initialize_world_and_robot()
    input_size = model.nq
    output_size = model.nu  # number of actuators
    hidden_size = HIDDEN_SIZE
    num_hidden_layers = NUM_HIDDEN_LAYERS

    # Create a small nn.Module template that contains a single flat parameter vector
    class FlatParamNet(torch.nn.Module):
        def __init__(self, total_params: int):
            super().__init__()
            self.flat = torch.nn.Parameter(torch.randn(total_params))

        def forward(self, *args, **kwargs):
            raise NotImplementedError("FlatParamNet is used only as a parameter container")

    # compute total number of scalar weights used by the controller (all dense layers)
    layer_sizes = [input_size] + [hidden_size] * NUM_HIDDEN_LAYERS + [output_size]
    weight_shapes = [(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
    total_params = sum(a * b for a, b in weight_shapes)

    problem: NEProblem = NEProblem(
        objective_sense="max",
        network=lambda: FlatParamNet(total_params),
        network_eval_func=evaluate_network,
        initial_bounds=(-1.0, 1.0),
        device=DEVICE,
        num_actors=PARALLEL_CORES,
    )


    def get_searcher() -> SearchAlgorithm:
        return SNES(
            problem=problem,
            popsize=POP_SIZE,
            stdev_init=0.1,
            # re_evaluate=False,
            # elitist=ELITIST,
        )
        
    
    # run a few generations to get a good initial population
    best_evolution: tuple[SearchAlgorithm, float] = None
    for e in range(SEEDING_ATTEMPTS):
        print(f"Initial evolution phase {e+1}/{SEEDING_ATTEMPTS}: running {SEEDING_GENERATIONS} generations to seed population")
        searcher = get_searcher()
        searcher.run(SEEDING_GENERATIONS)
        seed_best_fit = searcher.status['best_eval']
        print(f"  Best fitness after seeding: {seed_best_fit:.5f}")
        if best_evolution is None or seed_best_fit > best_evolution[1]:
            best_evolution = (searcher, seed_best_fit)

    searcher, best_fit = best_evolution
    print(f"Seeding complete. Initial best fitness: {best_fit:.5f}. Starting main evolution.")

    # Use a concrete logger implementation to avoid NotImplementedError
    logger = evotorch.logging.StdOutLogger(
        searcher=searcher,
        interval=1,
        after_first_step=True
    )

    generation = 0

    def terminate() -> bool:
        if TIME_LIMIT > 0 and (time.time() - start_time) > TIME_LIMIT:
            print("Time limit reached, terminating evolution.")
            return True
        if MAX_GENERATIONS > 0 and generation >= MAX_GENERATIONS:
            print("Max generations reached, terminating evolution.")
            return True
        return False
    
    interactive_mode = INTERACTIVE_MODE

    # Main evolutionary loop
    while not terminate():
        for _ in range(BATCH_SIZE if (MAX_GENERATIONS <= 0 or generation + BATCH_SIZE <= MAX_GENERATIONS) else (MAX_GENERATIONS - generation)):
            gen_start_time = time.time()
            searcher.step()
            generation += 1
            gen_end_time = time.time()
            print(f"Generation {generation}, Best Fitness: {searcher.status['best_eval']:.5f}, Time: {gen_end_time - gen_start_time:.5f} seconds")
            
        
        best_individual: Solution = searcher.status['best']
        best_net = problem.parameterize_net(best_individual)

        if interactive_mode:
            history = run_bot_session(best_net, method="headless")
            run_bot_session(best_net, method="viewer")
            show_qpos_history(history)
            user_input = input("Continue evolution? (y)es, (n)o, (s)kip interactive: ").strip().lower()
            if user_input == 'n':
                print("Evolution terminated by user.")
                break
            elif user_input == 's':
                interactive_mode = False
                print("Skipping further interactive prompts.")

        else:
            print("Running in non-interactive mode, continuing evolution.")
        if RECORD_BATCH:
            run_bot_session(best_net, method="record", options={"filename": "auto_recording", "mode": FITNESS_MODE, "fitness": searcher.status['best_eval']})
            save_genotype(best_net, searcher.status['best_eval'])


    best_individual: Solution = searcher.status['best']
    best_net = problem.parameterize_net(best_individual)
    best_fit = searcher.status['best_eval']

    if RECORD_LAST:
        print("Recording final best individual...")
        run_bot_session(best_net, method="record", options={"mode": FITNESS_MODE, "fitness": best_fit, "filename": "final_best"})
    
    if INTERACTIVE_MODE:
        print("Showing final best individual in viewer...")
        history = run_bot_session(best_net, method="headless")
        run_bot_session(best_net, method="viewer")
        show_qpos_history(history)
        input("Press Enter to continue...")

    return best_net, best_fit


def save_genotype(net: torch.nn.Module, fitness: float) -> None:
    output_path = Path(__file__).parent / "output" / "genotypes"
    output_path.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = output_path / f"genotype_{FITNESS_MODE}_fit {fitness:.4f}_{timestamp}.pt"
    torch.save(net.state_dict(), filename)
    print(f"Saved best genotype to {filename}")


def main():
    _, best_fit = evolve()
    print(f"Evolution complete. Best fitness: {best_fit:.5f}")

def try_multiple():
    for hidden_layers in [1, 2, 3]:
        global NUM_HIDDEN_LAYERS
        NUM_HIDDEN_LAYERS = hidden_layers
        for hidden_size in [8,16]:
            global HIDDEN_SIZE
            HIDDEN_SIZE = hidden_size
        print(f"=== Running experiments with NUM_HIDDEN_LAYERS = {NUM_HIDDEN_LAYERS} ===")
        for mode in ["segment_median", "simple"]:
            try:
                global FITNESS_MODE
                FITNESS_MODE = mode
                print(f"=== Evolution run with FITNESS_MODE = {FITNESS_MODE} ===")
                best_net, best_fit = evolve()
                save_genotype(best_net, best_fit)
                print(f"Run complete. Best fitness: {best_fit:.5f}")
            except Exception as e:
                print(f"Error during evolution: {e}")

    
    

if __name__ == "__main__":
    # main()
    try_multiple()