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
from revolve.body_phenotypes.robogen_lite.hinge.generator import (
    Hinge,
    HingeConfiguration,
    default_hinge_config_dump_as_json,
    default_hinge_config_load_from_json,
)
from revolve.body_phenotypes.robogen_lite.world.generator import World


def test_create_hinge_config() -> None:
    """Test the creation of a hinge configuration."""
    units = Units(length="cm", mass="g")
    s_dims = Dimensions(width=5, depth=5, height=5)
    stator = BaseConfiguration(
        mass=1,
        color=(0, 0.55, 0.88, 1),
        dimensions=s_dims,
        units=units,
    )
    r_dims = Dimensions(width=5, depth=5, height=5)
    rotor = BaseConfiguration(
        mass=0.5,
        color=(0, 0.25, 1, 1),
        dimensions=r_dims,
        units=units,
    )
    HingeConfiguration(stator=stator, rotor=rotor)


def test_default_hinge_config_dump_as_json() -> None:
    """Test the default hinge configuration dump as JSON."""
    default_hinge_config_dump_as_json()


def test_default_hinge_config_load_from_json() -> None:
    """Test the default hinge configuration load from JSON."""
    default_hinge_config_dump_as_json()
    default_hinge_config_load_from_json()


def test_generate_hinge_model_xml() -> None:
    """Test the generation of hinge model XML."""
    hinge_config = default_hinge_config_load_from_json()
    body = Hinge(hinge_config)
    world = World()
    spawn_site = world.spec.worldbody.add_site()
    spawn_site.attach_body(body=body.spec.worldbody, prefix="child-", suffix="")
    world.spec.to_xml()


def test_generate_hinge_model_compile() -> None:
    """Test the compilation of hinge model."""
    hinge_config = default_hinge_config_load_from_json()
    body = Hinge(hinge_config)
    world = World()
    spawn_site = world.spec.worldbody.add_site()
    spawn_site.attach_body(body=body.spec.worldbody, prefix="child-", suffix="")
    world.spec.compile()
