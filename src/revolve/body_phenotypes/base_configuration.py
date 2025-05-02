"""
TODO.

Author:     jmdm
Date:       2025-04-29

Py Ver:     3.12

OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro

Status:     Complete ✅

This code is provided "As Is"

Sources:
    1. https://docs.pydantic.dev/latest/concepts/models/
    2. https://docs.pydantic.dev/latest/concepts/json/
    3. https://docs.pydantic.dev/latest/examples/files/

Notes:
    *  Okay, I get it, no abstractions we said, but listen for a second, when I wrote this code I had the following idea in mind: at the simplest level, we will be running the simulations with pre-generated XML files, that is much faster than generating them on the fly.
        Now, lets assume we do run the 'generate' functions, there is only a small set of reasons why you may want to do that (that I can think of):
        1. You changed the default parameters and you need a new XML.
        2. You are generating parametric modules(eg. evolving bricks with different sizes).
        With the former, the only thing you need to do is simply change the default values.
        With the latter, it's good practice to use a standard API such that you can easily apply the evolve parameters.
        If you need something REALLY different, you probably shouldn't be using the Robogen Lite folder. You should either create a new set of modules, or work directly with MujoCo MjSpec.
        Also, in Revolve 2 the units where going absolutely everywhere, this is the easiest and most elegant solution I could think of.

Todo:
    [ ] move defaults into __init__?

"""

# Standard library
from copy import deepcopy
from pathlib import Path
from typing import Self

# Third-party libraries
from astropy import units as u
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    field_validator,
    model_validator,
)
from rich.traceback import install

# Global constants
MIN_WEIGHT = 1e-6
MAX_WEIGHT = 1e6
MIN_LENGTH = 1e-6
MAX_LENGTH = 1e6

_MASS_DEF_UNIT = u.si.kg
MASS_DEF_UNIT = _MASS_DEF_UNIT.name
_LENGTH_DEF_UNIT = u.si.m
LENGTH_DEF_UNIT = _LENGTH_DEF_UNIT.name

UNITS_OF_LENGTH = _LENGTH_DEF_UNIT.find_equivalent_units(
    include_prefix_units=True,
)
UNITS_OF_MASS = _MASS_DEF_UNIT.find_equivalent_units(
    include_prefix_units=True,
)

# Global functions
install(show_locals=True)


class Dimensions(BaseModel):
    """Dimensions of bounding box (lengths) of a model."""

    # PyDantic configuration
    model_config = ConfigDict(extra="forbid")

    # Size of (bounding) box
    width: NonNegativeFloat
    depth: NonNegativeFloat
    height: NonNegativeFloat


class Units(BaseModel):
    """Define units of model."""

    # PyDantic configuration
    model_config = ConfigDict(extra="forbid")

    # Allowed units
    length: str = LENGTH_DEF_UNIT
    mass: str = MASS_DEF_UNIT

    @model_validator(mode="after")
    def compatible_units(self) -> Self:
        """
        Ensure passed units are valid.

        :raises ValueError: if the passed units for 'length' are not defined in astropy.
        :raises ValueError: if the passed units for 'mass' are not defined in astropy.
        :return Self: return self if there are no errors (standard pydantic syntax).
        """
        # Check length units
        if self.length not in UNITS_OF_LENGTH:
            msg = f"{self.length=} isn't a valid unit of length!"
            raise ValueError(msg)

        # Check mass units
        if self.mass not in UNITS_OF_MASS:
            msg = f"{self.mass=} isn't a valid unit of mass!"
            raise ValueError(msg)
        return self


class BaseConfiguration(BaseModel):
    """A PyDantic model that defines the basic properties of model."""

    # PyDantic configuration
    model_config = ConfigDict(extra="forbid")

    # User defined
    color: tuple[  # RGBA
        NonNegativeFloat,
        NonNegativeFloat,
        NonNegativeFloat,
        NonNegativeFloat,
    ]
    units: Units
    stl_filepath: str = ""

    # Pre-conversion
    mass: NonNegativeFloat
    dimensions: Dimensions

    @field_validator("units", mode="before")
    @classmethod
    def make_copy_of_units(cls, units: Units) -> Units:
        """
        Make a copy of the passed units (useful if the user wants to reuse arguments).

        :param Units units: user defined units.
        :return Units: a deep copy of the user defined 'Units' class.
        """
        return deepcopy(units)

    @field_validator("units", mode="before")
    @classmethod
    def make_copy_of_dimensions(cls, dimensions: Dimensions) -> Dimensions:
        """
        Make a copy of the passed dimensions (useful if the user wants to reuse arguments).

        :param Dimensions dimensions: user defined dimensions.
        :return Dimensions: a deep copy of the user defined 'Dimensions' class.
        """
        return deepcopy(dimensions)

    @field_validator("stl_filepath", mode="after")
    @classmethod
    def file_exists(cls, file_path: str) -> str:
        """
        Check if the STL file exists.

        :param str file_path: path to the STL file.
        :raises FileNotFoundError: if the STL file does not exist.
        :return str: the STL file path.
        """
        if file_path and not Path(file_path).exists():
            msg = f"Did not find the STL file at {file_path=}!\n"
            raise FileNotFoundError(msg)
        return file_path

    @model_validator(mode="after")
    def mass_to_si(self) -> Self:
        """
        Convert mass to SI units.

        :return Self: return self if there are no errors (standard pydantic syntax).
        """
        mass_with_units = self.mass * getattr(u, self.units.mass)
        self.units.mass = MASS_DEF_UNIT
        self.mass = float(mass_with_units.to(MASS_DEF_UNIT).value)
        return self

    @model_validator(mode="after")
    def dimensions_to_si(self) -> Self:
        """
        Convert dimensions to SI units.

        :return Self: return self if there are no errors (standard pydantic syntax).
        """
        for dimension in self.dimensions.model_fields_set:
            dim = getattr(self.dimensions, dimension)
            dim_with_units = dim * getattr(u, self.units.length)
            setattr(
                self.dimensions,
                dimension,
                float(dim_with_units.to(LENGTH_DEF_UNIT).value),
            )
        self.units.length = LENGTH_DEF_UNIT
        return self

    @model_validator(mode="after")
    def check_mass(self) -> Self:
        """
        Check if mass is within limits.

        :raises ValueError: if the mass is not within limits.
        :return Self: return self if there are no errors (standard pydantic syntax).
        """
        if not (MIN_WEIGHT < self.mass < MAX_WEIGHT):
            msg = f"Mass {self.mass=} is not within limits!\n"
            msg += f"Mass should be between {MIN_WEIGHT} and {MAX_WEIGHT} kg!"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def check_dimensions(self) -> Self:
        """
        Check if dimensions are within limits.

        :raises ValueError: if the dimensions are not within limits.
        :return Self: return self if there are no errors (standard pydantic syntax).
        """
        for dimension in self.dimensions.model_fields_set:
            dim = getattr(self.dimensions, dimension)
            if not (MIN_LENGTH < dim < MAX_LENGTH):
                msg = f"Dimension {dimension} {dim=} is not within limits!\n"
                msg += f"Dimension should be between {MIN_LENGTH} and {MAX_LENGTH} m!"
                raise ValueError(msg)
        return self
