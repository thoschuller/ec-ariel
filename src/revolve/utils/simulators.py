"""TODO(jmdm): description of script.

Author:     jmdm
Date:       2025-05-02

Py Ver:     3.12

OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro

Status:     In progress ⚙️

This code is provided "As Is"

Sources:
    1.

Notes:
    *

Todo:
    [ ]

"""

# Third-party libraries
import mujoco
from PIL import Image
from rich.console import Console

# Global functions
console = Console()


def render_single_frame(xml: str) -> None:
    """Render a single frame of a MuJoCo simulation."""
    # MuJoCo basics
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Enable joint visualization option:
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # Call rendering engine
    with mujoco.Renderer(model) as renderer:
        # Move simulation forward one iteration/step
        mujoco.mj_step(model, data)

        # Update rendering engine
        renderer.update_scene(data, scene_option=scene_option)

        # Generate frame using rendering engine
        frame = renderer.render()

        # Convert frame into an image which can be shown
        img = Image.fromarray(frame)
        img.show()
