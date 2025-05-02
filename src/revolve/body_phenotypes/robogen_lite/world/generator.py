"""
TODO(jmdm): description of script.

Author:     jmdm
Date:       YYYY-MM-DD

Py Ver:     3.12

OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro

Status:     Complete ✅
Status:     To Improve ⬆️
Status:     In progress ⚙️
Status:     Broken ⚠️

This code is provided "As Is"

Sources:
    1.

Notes:
    *

Todo:
    [ ]

"""

# Standard library
from pathlib import Path

# Third-party libraries
import mujoco
import numpy as np
from rich.console import Console

# Local libraries

BRICK_WEIGHT = 0.05901
USE_DEGREES = False

CWD = Path.cwd()
SCRIPT_NAME = __file__.split("/")[-1][:-3]
DATA = f"{CWD}/__data__"
SEED = 42

# Global functions
console = Console()
console_err = Console(stderr=True, style="bold red")
RNG = np.random.default_rng(seed=SEED)

# Third-party libraries
from rich.traceback import install

# Global functions
install(show_locals=True)


class World:
    """XML class for all things that are not bodies or actuators."""

    def __init__(
        self,
        floor_size: tuple[float, float, float] = (1, 1, 1),  # meters
    ) -> None:
        """
        Create a basic specification for MuJoCo.

        :param tuple[float, float, float] floor_size: the size of the floor geom
        """
        # Fixed parameters
        grid_name = "grid"

        # Passed parameters
        self.floor_size = floor_size

        # --- Root ---
        spec = mujoco.MjSpec()
        spec.option.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        spec.compiler.degree = USE_DEGREES
        spec.compiler.balanceinertia = True
        spec.compiler.discardvisual = False

        # --- Assets ---
        spec.add_texture(
            name=grid_name,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.1, 0.2, 0.3],
            rgb2=[0.2, 0.3, 0.4],
            width=300,  # pixels
            height=300,  # pixels
        )
        spec.add_material(
            name=grid_name,
            textures=["", f"{grid_name}"],
            texrepeat=[3, 3],
            texuniform=True,
            reflectance=0.2,
        )

        # --- Worldbody ---
        spec.worldbody.add_light(
            name="light",
            pos=[0, 0, 1],
        )
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[0, 0, 0],
            size=self.floor_size,
            material=grid_name,
        )

        # Save specification
        self.spec: mujoco.MjSpec = spec


def main() -> None:
    """Entry point."""
    return


if __name__ == "__main__":
    main()
