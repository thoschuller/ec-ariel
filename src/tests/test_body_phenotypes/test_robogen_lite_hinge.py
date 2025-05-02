"""
Test: src/revolve/body_phenotypes/robogen_lite.

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
    hinge = Hinge()
    hinge.config_generate()
    hinge.config_dump_as_json()
    hinge.config_load_from_json()


def test_hinge_model_xml_dumping_and_loading() -> None:
    """Test the XML dumping and loading of hinge model."""
    hinge = Hinge()
    hinge.config_generate()
    hinge.create_mjspec()
    hinge.dump_mjspec_as_xml()
    hinge.load_mjspec_from_xml()


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
    hinge_config = HingeConfiguration(stator=stator, rotor=rotor)

    # Hinge
    hinge = Hinge()
    hinge.config = hinge_config
    hinge.create_mjspec()
    hinge.spec.compile()


def test_generate_hinge_model_compile() -> None:
    """Test the compilation of hinge model."""
    # Hinge
    hinge = Hinge()
    hinge.config_generate()
    hinge.create_mjspec()

    # World
    world = World()
    spawn_site = world.spec.worldbody.add_site()
    spawn_site.attach_body(
        body=hinge.spec.worldbody,
        prefix="child-",
        suffix="",
    )
    world.spec.compile()
