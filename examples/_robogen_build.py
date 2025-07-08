"""Robogen robot using MjSpec.

Author:     jmdm
Date:       2025-07-06
Py Ver:     3.12
OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro
Status:     Completed ✅

References
----------
    [1] https://github.com/google-deepmind/dm_control/blob/main/dm_control/mjcf/README.md
    [2] https://mujoco.readthedocs.io/en/stable/python.html#attachment
    [3] https://quaternion.readthedocs.io/en/latest/quaternion/

"""

# Standard library
from pathlib import Path

# Third-party libraries
import mujoco
from mujoco import viewer
from rich.console import Console
from rich.traceback import install

from revolve.body_phenotypes.robogen_lite.config import (
    ModuleFaces,
    ModuleRotationsTheta,
)
from revolve.body_phenotypes.robogen_lite.modules.brick import BrickModule
from revolve.body_phenotypes.robogen_lite.modules.core import CoreModule
from revolve.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from revolve.environments.simple_world import SimpleWorld

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)

# Global functions
install(show_locals=True)
console = Console(width=180)


class DummyRobotTestAttach:
    """Robot model using MuJoCo MjSpec with attachment points."""

    def __init__(self) -> None:
        """Initialize the robot model."""
        core = CoreModule()

        for side_i in ModuleFaces:
            if side_i in core.sites:
                module_1 = HingeModule()
                core.sites[side_i].attach_body(
                    body=module_1.body,
                    prefix=f"{module_1.index}-{side_i.value}-",
                )

                module_2 = BrickModule()
                module_1.sites[ModuleFaces.FRONT].attach_body(
                    body=module_2.body,
                    prefix=f"{module_2.index}-{side_i.value}-",
                )
                for side_j in ModuleFaces:
                    if side_j in module_2.sites:
                        module_3 = HingeModule()
                        module_2.sites[side_j].attach_body(
                            body=module_3.body,
                            prefix=f"{module_3.index}-{side_j.value}-",
                        )

        # Save model specification
        self.spec: mujoco.MjSpec = core.spec


class DummyRobotTestRotate:
    """Robot model using MuJoCo MjSpec with rotation."""

    def __init__(self) -> None:
        """Initialize the robot model."""
        core = CoreModule()

        core.sites[ModuleFaces.BOTTOM].attach_body(
            body=BrickModule().body,
            prefix="brick-bottom-",
        )

        for idx, rot_i in enumerate(ModuleRotationsTheta):
            module_1 = HingeModule()
            module_1.rotate(angle=rot_i.value)
            console.log(rot_i)
            list(core.sites.values())[idx].attach_body(
                body=module_1.body,
                prefix=f"{rot_i.name}={module_1.index}-{ModuleFaces.FRONT.name}-",
            )

            module_2 = BrickModule()
            module_2.rotate(angle=90)
            module_1.sites[ModuleFaces.FRONT].attach_body(
                body=module_2.body,
                prefix=f"{module_2.index}-{ModuleFaces.FRONT.name}-",
            )

            module_3 = HingeModule()
            module_2.sites[ModuleFaces.RIGHT].attach_body(
                body=module_3.body,
                prefix=f"{module_3.index}-{ModuleFaces.FRONT.name}-",
            )

        # Save model specification
        self.spec: mujoco.MjSpec = core.spec


class DummyRobotTestCtrl:
    """Robot model using MuJoCo MjSpec with rotation."""

    def __init__(self) -> None:
        """Initialize the robot model."""
        core = CoreModule()

        core.sites[ModuleFaces.BOTTOM].attach_body(
            body=BrickModule().body,
            prefix="brick-bottom-",
        )

        for side_i in (ModuleFaces.FRONT, ModuleFaces.BACK):
            if side_i in core.sites:
                module_1 = HingeModule()
                core.sites[side_i].attach_body(
                    body=module_1.body,
                    prefix=f"{module_1.index}-{side_i.value}-",
                )

                module_2 = BrickModule()
                module_1.sites[ModuleFaces.FRONT].attach_body(
                    body=module_2.body,
                    prefix=f"{module_2.index}-{side_i.value}-",
                )
                for side_j in ModuleFaces:
                    if side_j in module_2.sites:
                        module_3 = HingeModule()
                        module_2.sites[side_j].attach_body(
                            body=module_3.body,
                            prefix=f"{module_3.index}-{side_j.value}-",
                        )
        # Save model specification
        self.spec: mujoco.MjSpec = core.spec


def main() -> None:
    """Entry point."""
    # MuJoCo configuration
    viz_options = mujoco.MjvOption()  # visualization of various elements

    # Visualization of the corresponding model or decoration element
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = True

    # MuJoCo basics
    world = SimpleWorld()
    # robot = DummyRobotTestCtrl()
    # robot = DummyRobotTestRotate()
    robot = DummyRobotTestAttach()

    # Set random colors for geoms
    for i in range(len(robot.spec.geoms)):
        robot.spec.geoms[i].rgba[-1] = 0.5

    # Spawn the robot at the world
    world.spawn(robot.spec, spawn_position=(0, 0, 0.2))

    # Compile the model
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Save the model to XML
    xml = world.spec.to_xml()
    with (DATA / f"{SCRIPT_NAME}.xml").open("w", encoding="utf-8") as f:
        f.write(xml)

    # Number of actuators and DoFs
    console.log(f"{model.nq=}, {model.nu=}")

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # Launch viewer
    viewer.launch(model=model, data=data)


if __name__ == "__main__":
    main()
