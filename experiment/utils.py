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
        plots_dir = constants.OUTPUT / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)

        # Initialize world to get the background
        mujoco.set_mjcb_control(None)
        world = constants.SIM_WORLD()
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
        w, h, _ = img.shape

        # Convert list of [x,y,z] positions to numpy array
        pos_data = np.array(history)

        # Calculate initial position
        x0, y0 = int(h * 0.483), int(w * 0.815)
        xc, yc = int(h * 0.483), int(w * 0.9205)
        ym0, ymc = 0, constants.POSITIONS[0][0][0]

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

        # Plot x,y trajectory
        ax.plot(x0, y0, "kx", label="[0, 0, 0]")
        ax.plot(xc, yc, "go", label="Start")
        ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
        ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")

        # Add labels and title
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend()

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