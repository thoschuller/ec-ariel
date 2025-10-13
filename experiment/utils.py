from terminal import console
import numpy as np
import time
from pathlib import Path
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    save_graph_as_json,
)
from typing import Any
import mujoco
import constants
from ariel.utils.renderers import single_frame_renderer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Type Checking
from networkx import DiGraph
from ariel.utils.tracker import Tracker
import evaluator as evaluator


def save_brain_genotype(
    weights: np.ndarray, fitness: float = 0.0, filename: str = None
) -> None:
    """
    Saves a genotype (numpy network weights) to a .npy file.
    """
    output_path = constants.OUTPUT / "genotypes"
    output_path.mkdir(exist_ok=True, parents=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    prefix = filename if filename else "genotype_cma"
    file_path = output_path / f"{prefix}_fit_{fitness:.4f}_{timestamp}.npy"
    np.save(file_path, weights)
    console.log(f"Saved best genotype to {file_path}.npy")


def load_brain_genotype(file_path: str) -> np.ndarray:
    """
    Loads a genotype (numpy network weights) from a .npy file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Genotype file not found: {file_path}")
    weights = np.load(path)
    console.log(f"Loaded genotype from {file_path}")
    return weights


def save_body_to_json(
    gecko_graph: DiGraph, filename: str = None # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
) -> None:  
    """
    Saves the body structure to a JSON file for visualization.
    """
    output_path = constants.OUTPUT / "genotypes"
    output_path.mkdir(exist_ok=True, parents=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    prefix = filename if filename else "body_structure"
    file_path = output_path / f"{prefix}_{timestamp}.json"
    save_graph_as_json(gecko_graph, file_path)
    console.log(f"Saved body structure to {file_path}")


def save_xpos_history(tracker: Tracker, fitness: float = None) -> None:
    history = tracker.history["xpos"][0]

    try:
        # Convert list of [x,y,z] positions to numpy array
        pos_data = np.array(history)

        # Only use valid (finite) positions
        finite_mask = np.all(np.isfinite(pos_data), axis=1)
        if not np.any(finite_mask):
            msg = "[red] [ERROR] No valid positions to plot in xpos history. Skipping plot."
            console.log(msg)
            import sys

            sys.stdout.flush()
            return
        valid_pos_data = pos_data[finite_mask]
        SPAWN_POS = constants.POSITIONS[0][0]
        TARGET_POSITION = constants.POSITIONS[2][1]

        plots_dir = constants.OUTPUT / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)

        # Initialize world to get the background
        mujoco.set_mjcb_control(None)
        world = constants.SIM_WORLD()

        # Add some objects to the world
        start_sphere = r"""
        <mujoco>
            <worldbody>
                <geom name="green_sphere"
                size=".1"
                rgba="0 1 0 1"/>
            </worldbody>
        </mujoco>
        """
        end_sphere = r"""
        <mujoco>
            <worldbody>
                <geom name="red_sphere"
                size=".1"
                rgba="1 0 0 1"/>
            </worldbody>
        </mujoco>
        """
        target_box = r"""
        <mujoco>
            <worldbody>
                <geom name="magenta_box"
                    size=".1 .1 .1"
                    type="box"
                    rgba="1 0 1 0.75"/>
            </worldbody>
        </mujoco>
        """
        spawn_box = r"""
        <mujoco>
            <worldbody>
                <geom name="gray_box"
                size=".1 .1 .1"
                type="box"
                rgba="0.5 0.5 0.5 0.5"/>
            </worldbody>
        </mujoco>
        """

        # Starting point of robot
        adjustment = np.array((0, 0, TARGET_POSITION[2] + 1))
        world.spawn(
            mujoco.MjSpec.from_string(start_sphere),
            position=valid_pos_data[0] + (adjustment * 1.5),
            correct_collision_with_floor=False,
        )

        # End point of robot (last valid position)
        world.spawn(
            mujoco.MjSpec.from_string(end_sphere),
            position=valid_pos_data[-1] + (adjustment * 1.5),
            correct_collision_with_floor=False,
        )

        # Target position
        world.spawn(
            mujoco.MjSpec.from_string(target_box),
            position=TARGET_POSITION + adjustment,
            correct_collision_with_floor=False,
        )

        # Spawn position of robot
        world.spawn(
            mujoco.MjSpec.from_string(spawn_box),
            position=SPAWN_POS,
            correct_collision_with_floor=False,
        )

        # Section border boxes
        border_box = r"""
        <mujoco>
            <worldbody>
                <geom name="border_box"
                    size=".1 .1 .1"
                    type="box"
                    rgba="0 0 1 0.5"/>
            </worldbody>
        </mujoco>
        """
        # Border between section 1-2
        world.spawn(
            mujoco.MjSpec.from_string(border_box),
            position=constants.POSITIONS[1][0],
            correct_collision_with_floor=False,
        )
        # Border between section 2-3
        world.spawn(
            mujoco.MjSpec.from_string(border_box),
            position=constants.POSITIONS[2][0],
            correct_collision_with_floor=False,
        )

        # Draw path box only if there are at least 2 valid positions
        if len(valid_pos_data) > 1:
            last_value = valid_pos_data[0]
            for i in range(1, len(valid_pos_data)):
                position = valid_pos_data[i]
                distance = np.abs(np.array(position - last_value)) / 2
                distance_as_size = (
                    f'"{(distance[0] + 0.01):.2f} 0.05 {(distance[2] + 0.01):.2f}"'
                )
                path_box = rf"""
                <mujoco>
                    <worldbody>
                        <geom name="yellow_sphere"
                            type="box"
                            size={distance_as_size}
                            rgba="1 1 0 0.9"
                        />
                    </worldbody>
                </mujoco>
                """
                world.spawn(
                    mujoco.MjSpec.from_string(path_box),
                    position=position + (adjustment * 1.25),
                    correct_collision_with_floor=False,
                )
                last_value = position

        model = world.spec.compile()
        data = mujoco.MjData(model)
        save_path = str(constants.DATA / "background.png")
        single_frame_renderer(
            model,
            data,
            save_path=save_path,
            save=True,
            width=200,
            height=600,
            cam_fovy=8,
            cam_pos=[2.1, 0, 50],
            cam_quat=[-0.7071, 0, 0, 0.7071],
        )

        # Setup background image
        img = plt.imread(save_path)
        _, ax = plt.subplots()
        ax.imshow(img)

        # Add legend to the plot
        plt.rc("legend", fontsize="small")
        red_patch = mpatches.Patch(color="red", label="End Position")
        gray_patch = mpatches.Patch(color="gray", label="Spawn Position")
        green_patch = mpatches.Patch(color="green", label="Start Position")
        magenta_patch = mpatches.Patch(color="magenta", label="Target Position")
        yellow_patch = mpatches.Patch(color="yellow", label="Robot Path")
        ax.legend(
            handles=[
                green_patch,
                red_patch,
                magenta_patch,
                gray_patch,
                yellow_patch,
            ],
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
        )

        # Add labels and title
        ax.set_xlabel("Y Position")
        ax.set_ylabel("X Position")
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        # Title
        plt.title(
            "Robot Path in XY Plane - Fitness: "
            + (f"{fitness:.4f}" if fitness else "N/A")
        )

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"fit_{fitness:.4f}_xpos_history_{timestamp}.png"

        # Show results
        plt.savefig(plots_dir / filename)

        console.log(f"Saved xpos history plot to {plots_dir / filename}")
    except Exception as e:
        msg = f"[red] [ERROR] Exception in save_xpos_history: {e}"
        console.log(f"[red]Failed to save xpos history plot: {e}[/red]")
        console.log(msg)
        import traceback

        traceback.print_exc()
        import sys

        sys.stdout.flush()


def numpy_tolist(obj: Any) -> list[Any] | tuple[Any, ...] | dict[Any, Any] | Any:
    """Recursively convert numpy arrays in a structure to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [numpy_tolist(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_tolist(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: numpy_tolist(v) for k, v in obj.items()}
    else:
        return obj


def load_json_as_digraph( # pyright: ignore[reportUnknownParameterType]
    file_path: str,  
) -> DiGraph:   # pyright: ignore[reportMissingTypeArgument]
    """
    Load a body structure from a JSON file and convert it to a DiGraph.
    """
    from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
        load_graph_from_json,
    )

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Body structure file not found: {file_path}")
    graph = load_graph_from_json(path)
    console.log(f"Loaded body structure from {file_path}")
    return graph


def plot_and_record_saved_phenotype(
    brain_file: str,
    body_file: str,
    duration: float = constants.STAGE_SETTINGS["FULL"]["DURATION"],
    video_filename: str = None,
) -> None:
    """
    Loads a saved brain and body, runs a single simulation with video recording, and plots the result.
    """
    try:
        # Use default spawn position from constants
        spawn_pos = constants.POSITIONS[0][0]

        # Prepare options for video recording
        options = {}
        if video_filename:
            options["video_filename"] = video_filename

        tracker = run_saved_phenotype(
            brain_file=brain_file,
            body_file=body_file,
            duration=duration,
            method="record",
            options=options,
        )

        # Compute fitness if possible (optional, can be None)
        fitness = evaluator.fitness(
            tracker, spawn=spawn_pos, goal=constants.POSITIONS[2][1], bonus=True
        )

        # Plot the result
        save_xpos_history(tracker, fitness=fitness)
        console.log(f"Plotted and recorded saved phenotype. With fitness: {fitness}")
    except Exception as e:
        msg = f"[red] [ERROR] Exception in plot_and_record_saved_phenotype: {e}"
        console.log(
            f"[red] [ERROR] Failed to plot and record saved phenotype: {e}[/red]"
        )
        console.log(msg)
        import traceback

        traceback.print_exc()
        import sys

        sys.stdout.flush()


def run_saved_phenotype(
    brain_file: str,
    body_file: str,
    duration: float = constants.STAGE_SETTINGS["FULL"]["DURATION"],
    method: str = "headless",
    options: dict[str, Any] = None,
) -> Tracker:
    """
    Loads a saved brain and body, runs a single simulation without plotting.
    """
    tracker = Tracker()  # Default empty tracker in case of failure
    try:
        # Load brain weights and body graph
        weights = load_brain_genotype(brain_file)
        body_graph = load_json_as_digraph(body_file)

        # Import run_bot_session here to avoid circular imports
        from session_runner import run_bot_session

        # Use default spawn position from constants
        spawn_pos = constants.POSITIONS[0][0]

        # Run a single simulation (no extra training)
        tracker = run_bot_session(
            weights=weights,
            method=method,
            gecko_body=body_graph,
            duration=duration,
            spawn_pos=spawn_pos,
            options=options,
        )
        console.log(f"Simulation completed.")
    except Exception as e:
        msg = f"[red] [ERROR] Exception in run_saved_phenotype: {e}"
        console.log(f"[red] [ERROR] Failed to run saved phenotype: {e}[/red]")
        console.log(msg)
        import traceback

        traceback.print_exc()
        import sys

        sys.stdout.flush()

    return tracker


def plot_saved_phenotype(
    brain_file: str,
    body_file: str,
    duration: float = constants.STAGE_SETTINGS["FULL"]["DURATION"],
    method: str = "headless",
) -> None:
    """
    Loads a saved brain and body, runs a single simulation, and plots the result.
    """
    try:

        tracker = run_saved_phenotype(
            brain_file=brain_file, body_file=body_file, duration=duration, method=method
        )

        # Compute fitness if possible (optional, can be None)
        fitness = evaluator.fitness(
            tracker,
            spawn=constants.POSITIONS[0][0],
            goal=constants.POSITIONS[2][1],
            bonus=True,
        )

        # Plot the result
        save_xpos_history(tracker, fitness=fitness)
        console.log(f"Plotted saved phenotype. With fitness: {fitness}")
    except Exception as e:
        msg = f"[red] [ERROR] Exception in plot_saved_phenotype: {e}[/red]"
        console.log(f"[red] [ERROR] Failed to plot saved phenotype: {e}[/red]")
        console.log(msg)
        import traceback

        traceback.print_exc()
        import sys

        sys.stdout.flush()
