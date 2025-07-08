"""Robogen Lite Modules."""

# Standard library
from enum import Enum

# Local libraries
from .brick import BrickModule
from .core import CoreModule
from .hinge import HingeModule


class ModuleTypeInstances(Enum):
    """Enum for module types."""

    CORE = CoreModule
    BRICK = BrickModule
    HINGE = HingeModule
    DEAD = None


__all__ = [
    "BrickModule",
    "CoreModule",
    "HingeModule",
    "ModuleTypeInstances",
]
