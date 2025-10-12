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

TARGET_POSITION = constants.POSITIONS[2][1]
SPAWN_POS = constants.POSITIONS[0][0]

# Type Checking
from networkx import DiGraph
from ariel.utils.tracker import Tracker

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


def save_body_to_json(gecko_graph: DiGraph, filename: str = None) -> None: # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
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
        # Convert list of [x,y,z] positions to numpy array
        pos_data = np.array(history)

        # Starting point of robot
        adjustment = np.array((0, 0, TARGET_POSITION[2] + 1))
        world.spawn(
            mujoco.MjSpec.from_string(start_sphere),
            position=pos_data[0] + (adjustment * 1.5),
            correct_collision_with_floor=False,
        )

        # End point of robot
        world.spawn(
            mujoco.MjSpec.from_string(end_sphere),
            position=pos_data[-1] + (adjustment * 1.5),
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

        last_value = None
        for i in range(len(pos_data)):
            position = pos_data[i]
            if last_value is not None:
                distance = np.abs(np.array(position - last_value)) / 2
            else:
                distance = np.array((0.01, 0.01, 0.01))
            last_value = position

            distance_as_size: str = (
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
            "Robot Path in XY Plane - Fitness: " + (f"{fitness:.4f}" if fitness else "N/A")
        )
        
        
        
        

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        filename =  f"fit_{fitness:.4f}_xpos_history_{timestamp}.png"

        # Show results
        plt.savefig(plots_dir / filename)

        console.log(f"Saved xpos history plot to {plots_dir / filename}")
    except Exception as e:
        console.log(f"[red]Failed to save xpos history plot: {e}[/red]")




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