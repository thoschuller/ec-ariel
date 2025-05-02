"""Robogen-Lite hinge."""

# Standard library
from pathlib import Path

# Global constants
DEFAULT_CONFIG_NAME = "physical_parameters.jsonc"
DEFAULT_XML_NAME = "hinge.xml"
HINGE_CONFIG_PATH = Path(__file__).parent / DEFAULT_CONFIG_NAME
HINGE_XML_PATH = Path(__file__).parent / DEFAULT_XML_NAME

# Local libraries
__all__ = ["HINGE_CONFIG_PATH"]
