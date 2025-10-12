import numpy as np
import mujoco
from pathlib import Path

def load_brain_genotype(file_path: str) -> np.ndarray:
    """
    Loads a genotype (numpy network weights) from a .npy file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Genotype file not found: {file_path}")
    weights = np.load(path)
    print(f"Loaded genotype from {file_path}")
    return weights


NUM_HIDDEN_LAYERS = 1
WEIGHTS = load_brain_genotype("best_brain.npy")
HIDDEN_SIZE = 16

        # Define controller callback
def controller_callback(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray: # pyright: ignore[reportUnusedFunction]
    outputs = _controller(
        data=d,
        weights=WEIGHTS,
        input_size=len(d.qpos),
        hidden_size=HIDDEN_SIZE,
        output_size=m.nu,
    )
    return outputs

def _controller(
    data: mujoco.MjData,
    weights: np.ndarray,
    input_size: int,
    hidden_size: int,
    output_size: int,
) -> np.ndarray:
    # `weights` is expected to be a flat numpy array of parameters (float)

    # Dynamically unpack weights for multiple hidden layers
    layer_sizes = (
        [input_size] + [hidden_size] * NUM_HIDDEN_LAYERS + [output_size]
    )
    weight_shapes = [
        (layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
    ]
    weight_sizes = [a * b for a, b in weight_shapes]
    indices = np.cumsum([0] + weight_sizes)
    ws = [
        weights[indices[i] : indices[i + 1]].reshape(weight_shapes[i])
        for i in range(len(weight_shapes))
    ]

    # Forward pass
    inputs = data.qpos
    x = inputs
    for i in range(NUM_HIDDEN_LAYERS):
        x = np.tanh(np.dot(x, ws[i]))
    outputs = np.tanh(np.dot(x, ws[-1]))
    outputs = outputs * np.pi

    return outputs
