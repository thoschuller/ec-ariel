"""High-level configuration for Robogen Lite body phenotypes.

Date:       2025-06-25
Status:     To Improve ⬆️

Notes
-----
    * Using Enums ensures that the same naming is used across the codebase.

Todo
----
    [ ] Extend ALLOWED_FACES to include TOP and BOTTOM faces.

"""

# Standard library
from enum import Enum

# Third-party libraries
from pydantic import BaseModel


class ModuleType(Enum):
    """Enum for module types."""

    CORE = 0
    BRICK = 1
    HINGE = 2
    DEAD = 3


class ModuleFaces(Enum):
    """Enum for module attachment points."""

    FRONT = 0
    BACK = 1
    RIGHT = 2
    LEFT = 3
    TOP = 4
    BOTTOM = 5


class ModuleRotationsIdx(Enum):
    """Enum for module rotations as indices."""

    DEG_0 = 0
    DEG_90 = 1
    DEG_180 = 2
    DEG_270 = 3


class ModuleRotationsTheta(Enum):
    """Enum for module rotations in degrees."""

    DEG_0 = 0
    DEG_90 = 90
    DEG_180 = 180
    DEG_270 = 270


class ModuleInstance(BaseModel):
    """
    ModuleInstance represents a single module in the system.

    Parameters
    ----------
    BaseModel : pydantic.BaseModel
        The base model class from Pydantic.
    """

    type: ModuleType
    rotation: ModuleRotationsIdx
    links: dict[ModuleFaces, int]


# Define allowed faces for each module type
ALLOWED_FACES: dict[ModuleType, list[ModuleFaces]] = {
    ModuleType.CORE: [
        ModuleFaces.FRONT,
        ModuleFaces.BACK,
        ModuleFaces.RIGHT,
        ModuleFaces.LEFT,
    ],
    ModuleType.BRICK: [ModuleFaces.FRONT, ModuleFaces.RIGHT, ModuleFaces.LEFT],
    ModuleType.HINGE: [ModuleFaces.FRONT],
    ModuleType.DEAD: [],
}

# Define allowed rotations for each module type
ALLOWED_ROTATIONS: dict[ModuleType, list[ModuleRotationsIdx]] = {
    ModuleType.CORE: [ModuleRotationsIdx.DEG_0],
    ModuleType.BRICK: [
        ModuleRotationsIdx.DEG_0,
        ModuleRotationsIdx.DEG_90,
        ModuleRotationsIdx.DEG_180,
        ModuleRotationsIdx.DEG_270,
    ],
    ModuleType.HINGE: [
        ModuleRotationsIdx.DEG_0,
        ModuleRotationsIdx.DEG_90,
        ModuleRotationsIdx.DEG_180,
        ModuleRotationsIdx.DEG_270,
    ],
    ModuleType.DEAD: [ModuleRotationsIdx.DEG_0],
}

# Global constants
IDX_OF_CORE = 0

# Derived system parameters
NUM_OF_TYPES_OF_MODULES = len(ModuleType)
NUM_OF_FACES = len(ModuleFaces)
NUM_OF_ROTATIONS = len(ModuleRotationsIdx)
