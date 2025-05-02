"""
Hinge generator for the Robogen Lite module.

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
from revolve.body_phenotypes.base_configuration import Dimensions, Units
from revolve.body_phenotypes.robogen_lite.hinge import (
    HINGE_CONFIG_PATH,
    HINGE_XML_PATH,
)

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

# Global functions
install(show_locals=True)


class HingeConfiguration(BaseModel):
    """Defines a valid 'hinge' configuration."""

    kp: float = 1
    kv: float = 1

    stator: BaseConfiguration
    rotor: BaseConfiguration


class Hinge:
    """
    XML class for the Robogen Lite hinge.

    User isn't expected to change the default values defined here.
    """

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

    def __init__(self, hinge_config: HingeConfiguration) -> None:
        """
        Create a hinge module.

        :param HingeConfiguration hinge_config: configuration object passed during initialisation.
        """
        # Parameters
        self.set_parameters(hinge_config)

        # Create MjSpec
        self.define_specification()

    def set_parameters(self, hinge_config: HingeConfiguration) -> None:
        """
        Process and save object parameters.

        :param HingeConfiguration hinge_config: configuration object passed during initialisation.
        """
        # Stator
        self.stator_width = hinge_config.stator.dimensions.width
        self.stator_depth = hinge_config.stator.dimensions.depth
        self.stator_height = hinge_config.stator.dimensions.height

        # Rotors
        self.rotor_width = hinge_config.rotor.dimensions.width * self._shrink
        self.rotor_depth = hinge_config.rotor.dimensions.depth * self._shrink
        self.rotor_height = hinge_config.rotor.dimensions.height * self._shrink

        # Masses
        self.stator_mass = hinge_config.stator.mass
        self.rotor_mass = hinge_config.rotor.mass

        # Colors
        self.stator_color = hinge_config.stator.color
        self.rotor_color = hinge_config.rotor.color

        # Derived parameters
        self.width = self.stator_width + self.rotor_width
        self.height = self.stator_height + self.rotor_height
        self.depth = self.stator_depth + self.rotor_depth
        self.dimensions = [self.width, self.height, self.depth]

        # Actuator parameters
        kp = hinge_config.kp
        kv = hinge_config.kv
        self._gainprm[0] = kp
        self._biasprm[:3] = [0, -kp, -kv]

    def define_specification(self) -> None:
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


def default_hinge_config() -> HingeConfiguration:
    """
    Generate the default configuration for the hinge module.

    :return HingeConfiguration: default hinge configuration

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

    # Validate configuration
    return HingeConfiguration(stator=stator, rotor=rotor)


def default_hinge_config_dump_as_json() -> None:
    """Save locally (to physical_parameters.jsonc) the default configuration for the hinge module."""
    hinge_config = default_hinge_config()
    config_fileobj = Path(HINGE_CONFIG_PATH)
    config_fileobj.write_text(
        data=hinge_config.model_dump_json(indent=2),
        encoding="utf-8",
    )
    console.log(hinge_config)
    console.log(
        f"[bold green] --> Saved hinge config to '{config_fileobj.name}' :white_check_mark:",
    )


def default_hinge_config_load_from_json() -> HingeConfiguration:
    """Load locally defined (in physical_parameters.jsonc) default configuration for the hinge module."""
    config_fileobj = Path(HINGE_CONFIG_PATH)
    json_data = config_fileobj.read_text(encoding="utf-8")
    hinge_config = HingeConfiguration.model_validate_json(json_data=json_data)
    console.log(hinge_config)
    return hinge_config


def default_hinge() -> Hinge:
    """Generate the default hinge module."""
    hinge_config = default_hinge_config()
    return Hinge(hinge_config)


def default_hinge_dump_as_xml() -> None:
    """Save locally (to hinge.xml) the default XML for the hinge module."""
    hinge = default_hinge()
    xml_str = hinge.spec.to_xml()
    xml_fileobj = Path(HINGE_XML_PATH)
    xml_fileobj.write_text(
        data=xml_str,
        encoding="utf-8",
    )
    console.log(xml_str)
    console.log(
        f"[bold green] --> Saved hinge XML to '{xml_fileobj.name}' :white_check_mark:",
    )


def default_hinge_load_as_xml() -> str:
    """Load locally defined (in hinge.xml) default XML for the hinge module."""
    xml_fileobj = Path(HINGE_XML_PATH)
    xml_str = xml_fileobj.read_text(encoding="utf-8")
    console.log(xml_str)
    return xml_str


def compile_hinge() -> None:
    """Compile the hinge module."""
    console.rule("START: Compile Hinge Module\n")
    default_hinge_config_dump_as_json()
    default_hinge_dump_as_xml()
    console.rule("END: Compile Hinge Module\n")


def main() -> None:
    """TODO."""
    compile_hinge()


if __name__ == "__main__":
    main()
