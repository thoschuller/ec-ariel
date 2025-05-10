"""Test the base configuration of a body phenotype.

Author:     jmdm
Date:       2025-04-30
Py Ver:     3.12

Sources:
    1. https://docs.astral.sh/ruff/rules/pytest-raises-too-broad/
"""

# Third-party libraries
import pytest

# Local libraries
from revolve.body_phenotypes import DEFAULT_UNIT_OF_LENGTH, DEFAULT_UNIT_OF_MASS
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
    Units(length="m", mass="g")


def test_create_invalid_units() -> None:
    """Test the creation of invalid units."""
    with pytest.raises(ValueError, match="isn't a valid unit of"):
        Units(length="xxx", mass="xxx")


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
    # Define the units
    units_args = {
        "length": "cm",
        "mass": "g",
    }
    units = Units(**units_args)

    # Define the dimensions
    dims_args = {
        "width": 5,
        "depth": 5,
        "height": 5,
    }
    dims = Dimensions(**dims_args)

    # Create the base configuration
    BaseConfiguration(
        mass=1,
        color=(0, 0.55, 0.88, 1),
        dimensions=dims,
        units=units,
    )

    # Ensure that the units are converted correctly
    assert units.length == DEFAULT_UNIT_OF_LENGTH
    assert units.mass == DEFAULT_UNIT_OF_MASS

    # Ensure that the original objects are not modified
    assert units.length == units_args["length"]
    assert units.mass == units_args["mass"]
    assert dims.width == dims_args["width"]
    assert dims.depth == dims_args["depth"]
    assert dims.height == dims_args["height"]
