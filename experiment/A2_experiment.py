# Third-party libraries
from typing import Any

import evotorch.logging
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
from evotorch import Problem
from evotorch.algorithms import SNES
import time
import os

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

# Keep track of data / history
HISTORY = []

# === Experiment Constants ===
SEGMENT_LENGTH = 100
POP_SIZE = 50
MAX_GENERATIONS = 50
HIDDEN_SIZE = 16
SIM_STEPS = 50
OUTPUT_DELTA = 0.05
NUM_HIDDEN_LAYERS = 2
FITNESS_MODE = "simple"  # Options: "segment_median", "simple"
INTERACTIVE_MODE = True  # If True, show and ask every 5 generations; if False, run to max
RECORD_LAST = True
# In interactive mode, you need to close the mujoco viewer window to continue with the next batch
# ===========================

#TODO: proper cuda implementation and speed test

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

def numpy_nn_controller_move(model, data, to_track) -> None:
    """
    example neural network controller, based on the seminar version
    """
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # Simple 3-layer neural network
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu

    # Initialize the networks weights randomly
    w1 = np.random.randn(input_size, hidden_size) * 0.1
    w2 = np.random.randn(hidden_size, hidden_size) * 0.1
    w3 = np.random.randn(hidden_size, output_size) * 0.1

    # Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = data.qpos

    # Run the inputs through the layers of the network.
    layer1 = sigmoid(np.dot(inputs, w1))
    layer2 = sigmoid(np.dot(layer1, w2))
    outputs = sigmoid(np.dot(layer2, w3))

    # Scale the outputs to acceptable range
    data.ctrl = np.clip(outputs, -np.pi / 2, np.pi / 2)

    # save movements in history
    HISTORY.append(to_track[0].xpos.copy())

# convert weights to prevent stubborn errors
def assign_flat_weights_to_model(model, flat_weights):
    """
    Assign a flat list of weights to a PyTorch model's parameters.
    """
    flat_weights = np.asarray(flat_weights, dtype=np.float32)
    state_dict = model.state_dict()
    param_keys = list(state_dict.keys())
    param_shapes = [tuple(state_dict[k].shape) for k in param_keys]
    param_sizes = [np.prod(shape) for shape in param_shapes]

    idx = 0
    for key, shape, size in zip(param_keys, param_shapes, param_sizes):
        chunk = flat_weights[idx:idx + size]
        if chunk.size != size:
            raise ValueError(
                f"Parameter {key}: expected {size} values, got {chunk.size}"
            )
        state_dict[key] = torch.tensor(chunk.reshape(shape), dtype=torch.float32)
        idx += size

    model.load_state_dict(state_dict, strict=True)



def torch_nn_controller(model, data, to_track, weights, input_size, hidden_size, output_size, NUM_HIDDEN_LAYERS) -> None:
    """
Improved neural network controller with variable layers and weights from torch
based on https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
"""

    # torch_device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    torch_device = torch.device(DEVICE)

    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, NUM_HIDDEN_LAYERS):
            super(NeuralNetwork, self).__init__()
            layers = []
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.Sigmoid())
            for _ in range(NUM_HIDDEN_LAYERS - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.Sigmoid())
            layers.append(nn.Linear(hidden_size, output_size))
            layers.append(nn.Sigmoid())
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    model = NeuralNetwork(input_size, hidden_size, output_size, NUM_HIDDEN_LAYERS).to(torch_device)

    assign_flat_weights_to_model(model, weights)


    # get outputs
    inputs = torch.tensor(data.qpos, dtype=torch.float32, device=torch_device)
    # outputs = model(inputs).cpu().detach().numpy() - 0.5  # Center around 0
    outputs = model(inputs).detach().numpy() - 0.5  # Center around 0
    data.ctrl += outputs * OUTPUT_DELTA
    # check and print whether any control signal is larger than pi/2
    if np.any(np.abs(data.ctrl) > (np.pi / 2) + 0.05):
        print("Warning: Control signal out of bounds:", data.ctrl)
    data.ctrl = np.clip(data.ctrl, -np.pi / 2, np.pi / 2) # should be redundant with above scaling
    HISTORY.append(to_track[0].xpos.copy())



# fitness function
def fitness() -> floating[Any] | float | Any:
    """
    Calculates fitness based on history as euclidian distance or median of segment distances + mean euclidian distance per segment
    """

    start = np.array(HISTORY[0][:2])
    end = np.array(HISTORY[-1][:2])
    euclidian_distance = np.linalg.norm(end - start)

    if FITNESS_MODE == "simple":
        return euclidian_distance # Simple: total distance from start to end
    # Default: segment_median
    positions = np.array(HISTORY)
    segment_length = SEGMENT_LENGTH
    num_segments = len(positions) // segment_length
    distances = []
    for i in range(0, num_segments - 1):
        start_pos = positions[i * segment_length][:2]
        end_pos = positions[(i + 1) * segment_length][:2]
        dist = np.linalg.norm(end_pos - start_pos)
        distances.append(dist)
    # combine with euclidean distance to favor straight movement
    return np.median(distances) + euclidian_distance / num_segments

def show_qpos_history(history: list, fitness: float):
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
    total_dist = np.linalg.norm(pos_data[-1, :2] - pos_data[0, :2])
    plt.title(f'Robot Path in XY Plane\nFitness (XY progress): {fitness:.4f}, Total Dist: {total_dist:.4f}')
    plt.legend()
    plt.grid(True)

    # Set equal aspect ratio and center at (0,0)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 0.3)  # At least 1.0 to avoid empty plots
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)

    plt.show()


# Save genotype and fitness
def save_genotype(weights, fitness, filename=None):
    # create folder if not exists
    if not os.path.exists(".\\output"):
        os.makedirs(".\\output")


    if filename is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f".\\output\\best_genotype_{FITNESS_MODE}_{fitness}_{timestamp}.npz"
    np.savez(filename, weights=weights, fitness=fitness)
    print(f"Saved best genotype to {filename}")


# Load and rerun genotype
def load_and_rerun_genotype(filename, input_size, hidden_size, output_size, NUM_HIDDEN_LAYERS):
    data = np.load(filename)
    weights = data['weights']
    fitness_val = data['fitness']
    print(f"Loaded genotype from {filename} with fitness {fitness_val}")
    fit = run_episode(weights, input_size, hidden_size, output_size, NUM_HIDDEN_LAYERS, render=True, show_gui=True)
    print(f"Re-run fitness: {fit}")


# Show simulation in GUI
def show_in_gui(weights, input_size, hidden_size, output_size, NUM_HIDDEN_LAYERS):
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    def controller(m, d):
        torch_nn_controller(m, d, to_track, weights, input_size, hidden_size, output_size, NUM_HIDDEN_LAYERS)

    mujoco.set_mjcb_control(controller)
    mujoco.viewer.launch(model, data)
    mujoco.set_mjcb_control(None)


# Update run_episode to support show_gui
def run_episode(weights, input_size, hidden_size, output_size, NUM_HIDDEN_LAYERS, render=False, show_gui=False):
    global HISTORY
    HISTORY = []
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    def controller(m, d):
        torch_nn_controller(m, d, to_track, weights, input_size, hidden_size, output_size, NUM_HIDDEN_LAYERS)

    mujoco.set_mjcb_control(controller)
    for _ in range(SIM_STEPS):
        mujoco.mj_step(model, data)
    mujoco.set_mjcb_control(None)
    fit = fitness() if len(HISTORY) > 1 else 0.0
    if render:
        show_qpos_history(HISTORY, fit)
    if show_gui:
        mujoco.viewer.launch(model, data)
    return fit

def run_episode_star(args):
    """
    Wrapper to unpack arguments for parallel execution
    """
    return run_episode(*args)

def record_episode(weights, input_size, hidden_size, output_size, NUM_HIDDEN_LAYERS, fitness, filename=None, duration=30):
    if filename is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"recorded_episode_{FITNESS_MODE}_{fitness}_{timestamp}.mp4"
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    def controller(m, d):
        torch_nn_controller(m, d, to_track, weights, input_size, hidden_size, output_size, NUM_HIDDEN_LAYERS)

    mujoco.set_mjcb_control(controller)

    # Non-default VideoRecorder options
    PATH_TO_VIDEO_FOLDER = "./output/videos"
    video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER, file_name=filename)

    # Render with video recorder
    video_renderer(
        model,
        data,
        duration=duration,
        video_recorder=video_recorder,
    )
    mujoco.set_mjcb_control(None)
    print(f"Recorded episode saved to {PATH_TO_VIDEO_FOLDER}/{filename}")


def main():
    # Dynamically determine input/output sizes
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    input_size = len(data.qpos)
    hidden_size = HIDDEN_SIZE
    output_size = model.nu
    # Calculate total weights for configurable layers (weights + biases)
    layer_sizes = [input_size] + [hidden_size] * NUM_HIDDEN_LAYERS + [output_size]
    weight_shapes = [(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
    bias_shapes = [(layer_sizes[i + 1],) for i in range(len(layer_sizes) - 1)]
    total_weights = sum(a * b + b for a, b in weight_shapes)  # weights + biases for each layer

    # based on https://docs.evotorch.ai/v0.4.0/quickstart/#problem-definition

    def evo_fitness(weights):
        # weights kan een (ReadOnly) torch.Tensor op cuda zijn
        if isinstance(weights, torch.Tensor):
            w = weights.numpy()
            # w = weights.detach().cpu().numpy()
        else:
            w = np.asarray(weights, dtype=np.float32)
        return run_episode(w, input_size, hidden_size, output_size, NUM_HIDDEN_LAYERS)

    problem = Problem("max", evo_fitness, solution_length=total_weights, initial_bounds=(-1.0, 1.0), device=DEVICE)

    searcher = SNES(problem, popsize=POP_SIZE, stdev_init=0.025)
    logger = evotorch.logging.StdOutLogger(searcher)


    generations = MAX_GENERATIONS
    gen = 0
    while gen < generations:

        searcher.run(5)
        gen += 5
        best_weights = searcher.status['best']

        best_fit = run_episode(np.array(best_weights), input_size, hidden_size, output_size, NUM_HIDDEN_LAYERS, render=True)
        print("Current best fitness:", best_fit)

        if INTERACTIVE_MODE:
            print("Showing best solution in MuJoCo GUI...")
            show_in_gui(np.array(best_weights), input_size, hidden_size, output_size, NUM_HIDDEN_LAYERS)
            cont = input("Continue for 5 more generations? (y/n): ").strip().lower()
            if cont != 'y':
                print("Stopping evolution.")
                break
    input("Best weights found, run final episode with rendering and GUI... (press Enter to continue)")

    final_fit = run_episode(np.array(best_weights), input_size, hidden_size, output_size, NUM_HIDDEN_LAYERS, render=True)
    save_genotype(best_weights, final_fit)
    show_in_gui(np.array(best_weights), input_size, hidden_size, output_size, NUM_HIDDEN_LAYERS)
    print("Final fitness:", final_fit)
    print("mean fitness of last population:", searcher.status['mean_eval'])
    if RECORD_LAST:
        record_episode(np.array(best_weights), input_size, hidden_size, output_size, NUM_HIDDEN_LAYERS, final_fit)


if __name__ == "__main__":
    main()
