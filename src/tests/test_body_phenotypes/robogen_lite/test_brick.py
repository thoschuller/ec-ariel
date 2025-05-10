"""Test the Brick class of the Robogen Lite body phenotypes.

Author:     jmdm
Date:       2025-04-30
Py Ver:     3.12
"""

# Local libraries
from revolve.body_phenotypes.base_configuration import Dimensions, Units
from revolve.body_phenotypes.robogen_lite.brick.generator import (
    Brick,
    BrickConfiguration,
)
from revolve.body_phenotypes.robogen_lite.world.generator import World


def test_create_brick_config() -> None:
    """Test the creation of a brick configuration."""
    # Define units
    units = Units(length="mm", mass="g")

    # Define dimensions
    dimensions = Dimensions(width=75, depth=75, height=75)

    # Brick configuration
    BrickConfiguration(
        mass=58.99,
        color=(0.5, 0.1, 0.13, 1),
        dimensions=dimensions,
        units=units,
    )


def test_default_brick_config_dumping_and_loading() -> None:
    """Test the default brick configuration dumping and loading."""
    body = Brick()
    body.config_generate()
    body.config_dump_as_json()
    body.config_load_from_json()


def test_brick_model_xml_dumping_and_loading() -> None:
    """Test the XML dumping and loading of brick model."""
    body = Brick()
    body.config_generate()
    body.create_mjspec()
    body.dump_mjspec_as_xml()
    body.load_mjspec_from_xml()


def test_custom_brick_config_dumping_and_loading() -> None:
    """Test the custom brick configuration dumping and loading."""
    # Define units
    units = Units(length="mm", mass="g")

    # Define dimensions
    dimensions = Dimensions(width=75, depth=75, height=75)

    # Brick configuration
    config = BrickConfiguration(
        mass=58.99,
        color=(0.5, 0.1, 0.13, 1),
        dimensions=dimensions,
        units=units,
    )

    # Brick
    body = Brick()
    body.config = config
    body.create_mjspec()
    body.spec.compile()


def test_generate_brick_model_compile() -> None:
    """Test the compilation of brick model."""
    # Brick
    body = Brick()
    body.config_generate()
    body.create_mjspec()

    # World
    world = World()
    spawn_site = world.spec.worldbody.add_site()
    spawn_site.attach_body(
        body=body.spec.worldbody,
        prefix="child-",
        suffix="",
    )
    world.spec.compile()
