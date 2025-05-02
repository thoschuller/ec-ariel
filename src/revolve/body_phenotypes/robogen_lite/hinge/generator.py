"""
Generator for the Robogen Lite 'Hinge' module.

Author:     jmdm
Date:       2025-05-02

Py Ver:     3.12

OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro

Status:     Complete ✅

This code is provided "As Is"

Sources:
    1.

Notes:
    *  See: src/revolve/body_phenotypes/base_configuration.py

Todo:
    [ ]

"""

# Standard library
from pathlib import Path

# Third-party libraries
import mujoco
import numpy as np
from pydantic import BaseModel
from rich.console import Console
from rich.traceback import install

# Local libraries
from revolve.body_phenotypes import BaseConfiguration
from revolve.body_phenotypes.base_configuration import (
    Dimensions,
    Units,
)
from revolve.body_phenotypes.robogen_lite.hinge import (
    HINGE_CONFIG_PATH,
    HINGE_XML_PATH,
)
from revolve.body_phenotypes.robogen_lite.module import (
    AttachmentDirections,
    Module,
)

# Global functions
console = Console()

# Global functions
install(show_locals=True)


class HingeConfiguration(BaseModel):
    """Defines a valid 'hinge' configuration."""

    kp: float = 1
    kv: float = 1

    stator: BaseConfiguration
    rotor: BaseConfiguration


class Hinge(Module[HingeConfiguration]):
    """
    XML class for the Robogen Lite hinge.

    User isn't expected to change the default values defined here.
    """

    # Configuration
    config_path_obj: Path = HINGE_CONFIG_PATH
    xml_path_obj: Path = HINGE_XML_PATH
    base_model = HingeConfiguration

    # Names
    _hinge_name: str = "hinge"
    _stator_name: str = "stator"
    _rotor_name: str = "rotor"
    _joint_name: str = "servo"

    # Geom types
    _stator_type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_BOX
    _rotor_type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_BOX

    # Joint
    _joint_axis: tuple[int, int, int] = (0, 0, 1)
    _joint_type: mujoco.mjtJoint = mujoco.mjtJoint.mjJNT_HINGE

    # Shrink factor to prevent z-fighting (fraction of rotor.width)
    _shrink: float = 0.9999

    # Actuator attributes
    _dyntype = mujoco.mjtDyn.mjDYN_NONE
    _gaintype = mujoco.mjtGain.mjGAIN_FIXED
    _biastype = mujoco.mjtBias.mjBIAS_AFFINE
    _trntype = mujoco.mjtTrn.mjTRN_JOINT

    _dynprm = np.zeros(10)
    _gainprm = np.zeros(10)
    _biasprm = np.zeros(10)

    def set_class_attributes(self) -> None:
        """Process and save object parameters."""
        # Check if config is set
        if self.config is None:
            msg = "Hinge configuration not set!\n"
            msg += "Please set the configuration using 'default_config()' or load a config file.\n"
            raise ValueError(msg)

        # Stator
        self.stator_width = self.config.stator.dimensions.width
        self.stator_depth = self.config.stator.dimensions.depth
        self.stator_height = self.config.stator.dimensions.height

        # Rotors
        self.rotor_width = self.config.rotor.dimensions.width * self._shrink
        self.rotor_depth = self.config.rotor.dimensions.depth * self._shrink
        self.rotor_height = self.config.rotor.dimensions.height * self._shrink

        # Masses
        self.stator_mass = self.config.stator.mass
        self.rotor_mass = self.config.rotor.mass

        # Colors
        self.stator_color = self.config.stator.color
        self.rotor_color = self.config.rotor.color

        # Derived parameters
        self.width = self.stator_width + self.rotor_width
        self.height = self.stator_height + self.rotor_height
        self.depth = self.stator_depth + self.rotor_depth
        self.dimensions = [self.width, self.height, self.depth]

        # Actuator parameters
        kp = self.config.kp
        kv = self.config.kv
        self._gainprm[0] = kp
        self._biasprm[:3] = [0, -kp, -kv]

    def config_generate(self) -> None:
        """
        Generate the default configuration.

        Notes:
            * I used Orca Slicer to get estimated weights and dimensions for the stator and rotor.
            * The servo motor is assumed to be 55g (from the datasheet).
            * I assume the weight of the servo motor is evenly distributed between the stator and rotor.

        """
        # Define units
        units = Units(length="mm", mass="g")

        # Hardware weight
        servo_motor = 55

        # Stator
        s_dims = Dimensions(width=52.75, depth=54, height=52)
        stator = BaseConfiguration(
            mass=14.08 + (servo_motor / 2),
            color=(0, 0.55, 0.88, 1),
            dimensions=s_dims,
            units=units,
        )

        # Rotor
        r_dims = Dimensions(width=44.74, depth=52, height=52)
        rotor = BaseConfiguration(
            mass=11.32 + (servo_motor / 2),
            color=(0, 0.25, 1, 1),
            dimensions=r_dims,
            units=units,
        )

        # Final configuration
        self.config = HingeConfiguration(stator=stator, rotor=rotor)

    def _create_mjspec(self) -> None:
        """Create the MuJoCo spec object."""
        # Root
        spec = mujoco.MjSpec()

        # Body
        hinge = spec.worldbody.add_body(
            name=self._hinge_name,
        )

        # Stator (fixed part)
        stator = hinge.add_body(
            name=self._stator_name,
            pos=[-self.stator_width, 0, 0],
        )
        stator.add_geom(
            name=self._stator_name,
            type=self._stator_type,
            rgba=self.stator_color,
            size=[
                self.stator_width,
                self.stator_depth,
                self.stator_height,
            ],
            mass=self.stator_mass,
        )

        # Rotor (moving part)
        rotor = hinge.add_body(
            name=self._rotor_name,
        )
        rotor.add_joint(
            name=self._joint_name,
            type=self._joint_type,
            axis=self._joint_axis,
        )
        rotor.add_geom(
            name=self._rotor_name,
            type=self._rotor_type,
            rgba=self.rotor_color,
            size=[
                self.rotor_width,
                self.rotor_depth,
                self.rotor_height,
            ],
            mass=self.rotor_mass,
            pos=[self.rotor_width, 0, 0],
        )
        rotor.add_site(
            name=str(AttachmentDirections.FRONT.value),
            pos=[self.rotor_width, 0, 0],
        )

        # Contact exclusion
        spec.add_exclude(
            bodyname1=self._stator_name,
            bodyname2=self._rotor_name,
        )

        # --- Actuator(s) ---
        spec.add_actuator(
            name=self._joint_name,
            dyntype=self._dyntype,
            gaintype=self._gaintype,
            biastype=self._biastype,
            dynprm=self._dynprm,
            gainprm=self._gainprm,
            biasprm=self._biasprm,
            trntype=self._trntype,
            target=self._joint_name,
        )

        # Save specification
        self.spec: mujoco.MjSpec = spec


def compile_hinge() -> None:
    """Compile the hinge module."""
    console.rule("START: Compile Hinge Module\n")
    hinge = Hinge()
    hinge.config_generate()
    hinge.config_dump_as_json()
    hinge.create_mjspec()
    hinge.dump_mjspec_as_xml()
    console.rule("END: Compile Hinge Module\n")


def main() -> None:
    """TODO."""
    compile_hinge()


if __name__ == "__main__":
    main()
