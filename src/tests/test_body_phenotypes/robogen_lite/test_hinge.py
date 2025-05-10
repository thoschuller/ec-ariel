"""Test: src/revolve/body_phenotypes/robogen_lite (hinge).

Author:     jmdm
Date:       2025-04-30

Py Ver:     3.12

OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro

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


def test_default_hinge_config_dumping_and_loading() -> None:
    """Test the default hinge configuration dumping and loading."""
    body = Hinge()
    body.config_generate()
    body.config_dump_as_json()
    body.config_load_from_json()


def test_hinge_model_xml_dumping_and_loading() -> None:
    """Test the XML dumping and loading of hinge model."""
    body = Hinge()
    body.config_generate()
    body.create_mjspec()
    body.dump_mjspec_as_xml()
    body.load_mjspec_from_xml()


def test_custom_hinge_config_dumping_and_loading() -> None:
    """Test the custom hinge configuration dumping and loading."""
    # Hinge configuration
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
    config = HingeConfiguration(stator=stator, rotor=rotor)

    # Hinge
    body = Hinge()
    body.config = config
    body.create_mjspec()
    body.spec.compile()


def test_generate_hinge_model_compile() -> None:
    """Test the compilation of hinge model."""
    # Hinge
    body = Hinge()
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
