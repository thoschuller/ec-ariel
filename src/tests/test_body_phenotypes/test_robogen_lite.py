"""
Test: src/revolve/body_phenotypes/robogen_lite.

Author:     jmdm
Date:       2025-04-30

Py Ver:     3.12

OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro

Notes:
    *

Todo:
    [ ]

"""

# Local libraries
from revolve.body_phenotypes.base_configuration import (
    BaseConfiguration,
    Dimensions,
    Units,
)


def test_create_dimensions() -> None:
    """Test the creation of dimensions."""
    Dimensions(width=5, depth=5, height=5)


def test_create_units() -> None:
    """Test the creation of units."""
    Units(length="cm", mass="g")


def test_create_base_configuration() -> None:
    """Test the creation of a base configuration."""
    units = Units(length="m", mass="kg")
    dims = Dimensions(width=5, depth=5, height=5)
    BaseConfiguration(
        mass=1,
        color=(0, 0.55, 0.88, 1),
        dimensions=dims,
        units=units,
    )


def test_create_base_configuration_with_unit_conversion() -> None:
    """Test the creation of a base configuration with unit conversion."""
    units = Units(length="cm", mass="g")
    dims = Dimensions(width=5, depth=5, height=5)
    BaseConfiguration(
        mass=1,
        color=(0, 0.55, 0.88, 1),
        dimensions=dims,
        units=units,
    )
