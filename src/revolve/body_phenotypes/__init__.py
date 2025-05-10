"""Body Genotypes."""

# Third-party libraries
from astropy import units as u

# ================================================
#                FILE INFO
# ================================================
# Name of the configuration file
DEFAULT_CONFIG_NAME = "physical_parameters.jsonc"

# ================================================
#                UNITS
# ================================================
# Use degrees instead of radians
USE_DEGREES = False

# Global constants
MIN_WEIGHT = 1e-4  # 0.000,01 kg
MAX_WEIGHT = 1e4  # 10,000 kg
MIN_LENGTH = 1e-4  # 0.000,01 m
MAX_LENGTH = 1e4  # 10,000 m

# Default units (astropy)
_DEFAULT_UNIT_OF_MASS_OBJ = u.si.kg
_DEFAULT_UNIT_OF_LENGTH_OBJ = u.si.m

# Default units (string)
DEFAULT_UNIT_OF_MASS = _DEFAULT_UNIT_OF_MASS_OBJ.name
DEFAULT_UNIT_OF_LENGTH = _DEFAULT_UNIT_OF_LENGTH_OBJ.name

# Alternative names for units (astropy)
UNITS_OF_LENGTH = _DEFAULT_UNIT_OF_LENGTH_OBJ.find_equivalent_units(
    include_prefix_units=True,
)
UNITS_OF_MASS = _DEFAULT_UNIT_OF_MASS_OBJ.find_equivalent_units(
    include_prefix_units=True,
)

# These objects are not needed outside this module
del _DEFAULT_UNIT_OF_MASS_OBJ
del _DEFAULT_UNIT_OF_LENGTH_OBJ
