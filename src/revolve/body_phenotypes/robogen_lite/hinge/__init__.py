"""Robogen-Lite hinge."""

# Standard library
from pathlib import Path

# Local libraries
from revolve.body_phenotypes import DEFAULT_CONFIG_NAME

# Global constants
HINGE_XML_NAME = "hinge.xml"
HINGE_CONFIG_PATH = Path(__file__).parent / DEFAULT_CONFIG_NAME
HINGE_XML_PATH = Path(__file__).parent / HINGE_XML_NAME
