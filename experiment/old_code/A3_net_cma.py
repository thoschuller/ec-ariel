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
# import csv
from typing import Any
from collections.abc import Callable

import numpy as np
import mujoco
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
import sys
import copy

from typing import cast

# CMA-ES library
from cma import CMAEvolutionStrategy # pyright: ignore[reportMissingTypeStubs]
from functools import partial

# Local libraries
from ariel.utils.renderers import tracking_video_renderer, single_frame_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.environments import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.ec.a001 import Individual
# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph 
from ariel.simulation.controllers.controller import Controller
from networkx import DiGraph  # Add this import
import constants as constants









def show_qpos_history(history: list[np.ndarray], save: bool = False) -> None:
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
    world = cast(Callable[[], Any], constants.SIM_WORLD)()
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Create temporary background image
    output_path = Path(__file__).parent / "output" / "plots"
    output_path.mkdir(exist_ok=True)
    background_path = output_path / "temp_background.png"

    single_frame_renderer(
        model,
        data,
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
    if constants.SIM_WORLD == SimpleFlatWorld:
        spawn_pos = [0, 0, 0]
    elif constants.SIM_WORLD == OlympicArena:
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
    bg_title = f'Robot Trajectory - Fitness ({constants.FITNESS_MODE}): {fit:.5f}'
    bg_title += f'\nOrigin Distance: {origin_distance:.3f}, Median Segment Distance: {median_segment_distance:.5f}'
    background_axis.set_title(bg_title, fontsize=12)
    background_axis.legend()

    plt.tight_layout()

    if constants.INTERACTIVE_MODE:
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
    def nn_controller_callback(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
        outputs = numpy_nn_controller_move_with_weights(m, d, weights, len(d.qpos), constants.HIDDEN_SIZE, m.nu)
        return outputs

    ctrl = Controller(
        controller_callback_function=nn_controller_callback,
        alpha=constants.OUTPUT_DELTA,
    )
    return ctrl

def create_individual(weight_shapes: list[tuple[int, int]]) -> Individual:
    """Create a random individual with Glorot/Xavier weight initialization."""
    genotype = sample_glorot_flat(weight_shapes)
    ind = Individual()
    ind.genotype = genotype.tolist()  # Store as list to avoid numpy ambiguity
    ind.requires_eval = True
    return ind


def analyze_trajectory_with_plots(weights: np.ndarray, save_plots: bool = True) -> dict[str, Any]:
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

def main() -> None:
    evolve_using_cma_es()

if __name__ == "__main__":
    main()
