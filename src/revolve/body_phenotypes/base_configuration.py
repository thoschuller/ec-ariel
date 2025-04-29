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
    [ ]

"""

# Standard library
from pathlib import Path
from typing import Literal, Self

# Third-party libraries
from pydantic import BaseModel, model_validator


class Dimensions(BaseModel):
    """Dimensions of bounding box."""

    width: float
    height: float
    depth: float


class Shell(BaseModel):
    """Description of 3D printed shell (for approximate weight calculation)."""

    material: Literal["PLA", "Unknown"] = "PLA"
    density: float = 1.25
    weight: float
    stl_filepath: str = ""

    @model_validator(mode="after")
    def file_exists(self) -> Self:
        """Check that the STL file exists."""
        # Check if file exists
        if self.stl_filepath and not Path(self.stl_filepath).exists():
            msg = f"Did not find the STL file at {self.stl_filepath=}!\n"
            raise FileNotFoundError(msg)
        return self


class Units(BaseModel):
    """Defines the allowed units (extending this requires an update in 'unit_conversion')."""

    dimensions: Literal["meters", "centimeters", "millimeters"] = "centimeters"
    weight: Literal["kilograms", "grams"] = "grams"
    density: Literal["kg/m3", "g/cm3"] = "g/cm3"


class BaseConfiguration(BaseModel):
    """General required information."""

    weight: float
    dimensions: Dimensions
    shell: Shell

    units: Units

    @model_validator(mode="after")
    def unit_conversion(self) -> Self:
        """Convert values to standard units."""
        # Weight should be in kilograms.
        match self.units.weight:
            case "grams":
                self.weight /= 1000
                self.shell.weight /= 1000
        self.units.weight = "kilograms"

        # Distances should be in centimeters.
        match self.units.dimensions:
            case "meters":
                self.dimensions.width *= 100
                self.dimensions.height *= 100
                self.dimensions.depth *= 100
            case "millimeters":
                self.dimensions.width /= 10
                self.dimensions.height /= 10
                self.dimensions.depth /= 10
        self.units.dimensions = "centimeters"

        # Density should be in kilograms per meter cube.
        match self.units.density:
            case "g/cm3":
                self.shell.density *= 1000
        self.units.density = "kg/m3"

        return self

    @model_validator(mode="after")
    def sensible_values(self) -> Self:
        """Check that all the values are within 'reasonable' ranges (based on my opinion lol)."""
        # Check weight
        min_weight = 1e-5
        max_weight = 1e5
        if not min_weight < self.weight < max_weight:
            msg = f"\tExpected weight to be between {min_weight=} and {max_weight=}!\n"
            msg += f"\tInstead got: {self.weight=}\n"
            msg += "\tAre you sure you're using the correct units?\n"
            raise ValueError(msg)

        # Check dimensions
        min_dim = 1e-5
        max_dim = 1e5
        for dimension in self.dimensions.model_fields_set:
            if not min_dim < getattr(self.dimensions, dimension) < max_dim:
                msg = f"\tExpected weight to be between {min_dim=} and {max_dim=}!\n"
                msg += f"\tInstead got: dimension={getattr(self.dimensions, dimension)}\n"
                msg += "\tAre you sure you're using the correct units?\n"
                raise ValueError(msg)
        return self
