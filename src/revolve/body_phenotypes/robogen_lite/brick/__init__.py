"""Robogen-Lite brick."""

# Standard library
from pathlib import Path

# Local libraries
from revolve.body_phenotypes import DEFAULT_CONFIG_NAME

# Global constants
BRICK_XML_NAME = "brick.xml"
BRICK_CONFIG_PATH = Path(__file__).parent / DEFAULT_CONFIG_NAME
BRICK_XML_PATH = Path(__file__).parent / BRICK_XML_NAME
