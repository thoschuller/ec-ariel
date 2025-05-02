"""
Generator for the Robogen Lite 'brick' module.

Author:     jmdm
Date:       2025-05-02

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
    * See: src/revolve/body_phenotypes/base_configuration.py

Todo:
    [ ]

"""

# Standard library
from pathlib import Path

# Third-party libraries
import mujoco
from rich.console import Console
from rich.traceback import install

# Local libraries
from revolve.body_phenotypes.base_configuration import (
    BaseConfiguration,
    Dimensions,
    Units,
)

# Third-party libraries
from revolve.body_phenotypes.robogen_lite.brick import (
    BRICK_CONFIG_PATH,
    BRICK_XML_PATH,
)
from revolve.body_phenotypes.robogen_lite.module import (
    AttachmentDirections,
    Module,
)

# Global functions
install(show_locals=True)
console = Console()


class BrickConfiguration(BaseConfiguration):
    """Defines a valid 'brick' configuration."""


class Brick(Module[BrickConfiguration]):
    """
    XML class for the Robogen Lite brick.

    User isn't expected to change the default values defined here.
    """

    # Configuration
    config_path_obj: Path = BRICK_CONFIG_PATH
    xml_path_obj: Path = BRICK_XML_PATH
    base_model = BrickConfiguration

    # Names
    _brick_name: str = "brick"

    # Geom types
    _brick_type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_BOX

    def set_class_attributes(self) -> None:
        """
        Set the parameters of the brick.

        :param _type_ config: The configuration object containing the parameters.
        """
        # Dimensions
        self.width = self.config.dimensions.width
        self.height = self.config.dimensions.depth
        self.depth = self.config.dimensions.height
        self.dimensions = [self.width, self.height, self.depth]

        # Mass
        self.mass = self.config.mass

        # Color
        self.color = self.config.color

    def config_generate(self) -> None:
        """
        Generate a default brick configuration.

        :return BrickConfiguration: A default brick configuration.

        Notes:
            * Values estimated using OrcaSlicer.

        """
        # Define units
        units = Units(length="mm", mass="g")

        # Define dimensions
        dimensions = Dimensions(width=75, depth=75, height=75)

        # Final configuration
        self.config = BrickConfiguration(
            mass=58.99,
            color=(0.5, 0.1, 0.13, 1),
            dimensions=dimensions,
            units=units,
        )

    def _create_mjspec(self) -> None:
        """Create the MuJoCo spec object."""
        # Root
        spec = mujoco.MjSpec()

        # Body
        brick = spec.worldbody.add_body(
            name=self._brick_name,
        )
        brick.add_geom(
            name=self._brick_name,
            type=self._brick_type,
            rgba=self.color,
            size=[self.width, self.height, self.depth],
            mass=self.mass,
            pos=[0, 0, 0],
        )

        # Sites
        brick.add_site(
            name=str(AttachmentDirections.FRONT.value),
            pos=[self.width, 0, 0],
        )
        brick.add_site(
            name=str(AttachmentDirections.LEFT.value),
            pos=[0, -self.height, 0],
        )
        brick.add_site(
            name=str(AttachmentDirections.RIGHT.value),
            pos=[0, self.height, 0],
        )
        brick.add_site(
            name=str(AttachmentDirections.UP.value),
            pos=[0, 0, self.depth],
        )
        brick.add_site(
            name=str(AttachmentDirections.DOWN.value),
            pos=[0, 0, -self.depth],
        )

        # Save specification
        self.spec: mujoco.MjSpec = spec


def compile_brick() -> None:
    """Compile the brick module."""
    console.rule("START: Compile Brick Module\n")
    brick = Brick()
    brick.config_generate()
    brick.config_dump_as_json()
    brick.create_mjspec()
    brick.dump_mjspec_as_xml()
    console.rule("END: Compile Brick Module\n")


def main() -> None:
    """Entry point."""
    compile_brick()


if __name__ == "__main__":
    main()
