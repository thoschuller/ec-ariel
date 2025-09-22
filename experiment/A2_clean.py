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
import torch
from numpy import floating
from torch import nn

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


# === experiment constants/settings ===
SEGMENT_LENGTH = 250
POP_SIZE = 60
MAX_GENERATIONS = 15000
TIME_LIMIT = 60 * 60 * 1  # max run time in seconds
HIDDEN_SIZE = 16
SIM_STEPS = 2500  # running at 500 steps per second
# OUTPUT_DELTA = 0.05 # change in output per step, to smooth out controls
NUM_HIDDEN_LAYERS = 3
FITNESS_MODE = "segment_median"  # Options: "segment_median", "simple"
INTERACTIVE_MODE = True  # If True, show and ask every X generations; if False, run to max
# IMPORTANT NOTE: in interactive mode, it is required to close the viewer window to continue running
RECORD_LAST = True  # If True, record a video of the last individual
BATCH_SIZE = 20  # Number of individuals to evaluate before running interactive prompts
RECORD_BATCH = True  # If True, record a video of the best individual every BATCH_SIZE generations
RECORD_LAST = True  # If True, record a video of the last individual
SEEDING_ATTEMPTS = 3  # Number of attempts to seed the initial population
SEEDING_GENERATIONS = 4  # Number of generations to run for seeding

# # Staged evolution settings
# STAGED_EVOLUTION = False  # If True, use staged evolution
# STAGE1_GENERATIONS = 5  # Number of generations for stage 1
# STAGE2_GENERATIONS = 10  # Number of generations for stage 2
# STAGE3_GENERATIONS = MAX_GENERATIONS - STAGE1_GENERATIONS - STAGE2_GENERATIONS  # Remaining generations for stage 3
# ===========================


# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
PARALLEL_CORES = multiprocessing.cpu_count() - 1  # leave one core free for the system itself

plt.ioff()  # Turn off interactive mode for plotting to avoid blocking

# Per-process cache to avoid recompiling the world/model on every evaluation
_MODEL_CACHE: dict = {}

def neuralnet_move(model: mujoco.MjModel, data: mujoco.MjData, to_track: list, nn_model: torch.nn.Module, history: list) -> None:
    # Get inputs, in this case the positions of the actuator motors (hinges)
    # Ensure inputs are on the same device as the model. If the model has no parameters,
    # fall back to the configured DEVICE.
    # Determine device once and reuse; avoid creating new tensors if possible
    try:
        device = next(nn_model.parameters()).device
    except StopIteration:
        device = torch.device(DEVICE)

    # Use torch.as_tensor to avoid an extra copy when data.qpos is already a numpy array
    inputs = torch.as_tensor(data.qpos, dtype=torch.float32, device=device)

    # Run the inputs through the neural network (no grad for speed)
    with torch.no_grad():
        outputs = nn_model(inputs).cpu().numpy()

    outputs = outputs * (0.5 * np.pi)  # Scale the outputs to actuator range [-0.5π, 0.5π]

    # check for any out-of-range values
    if np.any(outputs < -np.pi / 2) or np.any(outputs > np.pi / 2):
        print("Warning: Neural network output out of range!")

    # Scale the outputs to acceptable range
    data.ctrl = np.clip(outputs, -np.pi / 2, np.pi / 2)

    # Update history
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

def run_bot_session(network: torch.nn.Module, method: str, options: dict = None) -> list:
    model, data, to_track = initialize_world_and_robot()

    # Initialise history tracking
    history = []
    
    mujoco.set_mjcb_control(lambda m,d: neuralnet_move(m, d, to_track, network, history))

    
    if method == "record":
        video_path = Path(__file__).parent / "output" / "videos"
        video_path.mkdir(exist_ok=True)
        video_file = ""
        if options:
            video_file += f"mode {options.get('mode','unknown')}_fit {options.get('fitness',0.0):.4f}"
            if options.get('filename'):
                video_file = f"{options.get('filename')}"
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_file += f"_{timestamp}.mp4"
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

def evaluate_network(nn_model: torch.nn.Module) -> float:
    history = run_bot_session(nn_model, method="headless")
    if len(history) < 2:
        return 0.0
    fit = fitness(history)
    return fit
    
def create_nn_model(input_size: int, hidden_size: int, output_size: int, num_hidden_layers: int) -> torch.nn.Module:
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
            super(NeuralNetwork, self).__init__()
            layers = [nn.Linear(input_size, hidden_size), nn.LeakyReLU(0.01)]
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Linear(hidden_size, output_size))
            layers.append(nn.Tanh())
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    net = NeuralNetwork(input_size, hidden_size, output_size, num_hidden_layers)
    # Move model to configured device to keep tensors consistent with DEVICE
    try:
        net = net.to(torch.device(DEVICE))
    except Exception:
        # Fallback: if DEVICE is invalid or unavailable, leave on default device
        pass
    return net

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

    problem: NEProblem = NEProblem(
        objective_sense="max",
        network=lambda: create_nn_model(input_size, hidden_size, output_size, num_hidden_layers),
        network_eval_func=evaluate_network,
        initial_bounds=(-1.0, 1.0),
        device=DEVICE,
        num_actors=PARALLEL_CORES
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
            
        if interactive_mode:
            best_individual: Solution = searcher.status['best']
            best_net = problem.parameterize_net(best_individual)
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
            run_bot_session(best_net, method="headless", options={"filename": "auto_recording"})


    best_individual: Solution = searcher.status['best']
    best_net = problem.parameterize_net(best_individual)
    best_fit = searcher.status['best_eval']

    if RECORD_LAST:
        print("Recording final best individual...")
        run_bot_session(best_individual, method="record", options={"mode": FITNESS_MODE, "fitness": best_fit})
    
    if INTERACTIVE_MODE:
        print("Showing final best individual in viewer...")
        history = run_bot_session(best_individual, method="headless")
        run_bot_session(best_individual, method="viewer")
        show_qpos_history(history)
        input("Press Enter to continue...")

    return best_net, best_fit





def main():
    _, best_fit = evolve()
    print(f"Evolution complete. Best fitness: {best_fit:.5f}")

def try_multiple():
    for evo in range(3):
        for mode in ["simple", "segment_median"]:
            global FITNESS_MODE
            FITNESS_MODE = mode
            print(f"=== Evolution run {evo+1} with FITNESS_MODE = {FITNESS_MODE} ===")
            main()
    
    

if __name__ == "__main__":
    main()