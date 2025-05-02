"""
TODO(jmdm): description of script.

Author:     jmdm
Date:       2025-05-02

Py Ver:     3.12

OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro

Status:     Complete ✅
Status:     To Improve ⬆️
Status:     In progress ⚙️
Status:     Broken ⚠️

This code is provided "As Is"

Sources:
    1.

Notes:
    *

Todo:
    [ ]

"""

# Standard library
from abc import abstractmethod
from enum import Enum
from pathlib import Path

# Third-party libraries
import mujoco
from pydantic import BaseModel
from rich.console import Console
from rich.traceback import install

# Global functions
console = Console()

# Global functions
install(show_locals=True)


class AttachmentDirections(Enum):
    """Enum for attachment types."""

    UP = "up"
    DOWN = "down"

    LEFT = "left"
    RIGHT = "right"

    FRONT = "front"
    BACK = "back"


class Module[T: BaseModel]:
    """Base class for all modules."""

    config_path_obj: Path
    xml_path_obj: Path

    base_model: type[T]
    config: T
    spec: mujoco.MjSpec

    @abstractmethod
    def set_class_attributes(self) -> None:
        """TODO."""

    @abstractmethod
    def _create_mjspec(self) -> None:
        """TODO."""

    @abstractmethod
    def config_generate(self) -> None:
        """TODO."""

    def config_load_from_json(self) -> None:
        """TODO."""
        # Load config
        json_data = self.config_path_obj.read_text(encoding="utf-8")
        self.config = self.base_model.model_validate_json(json_data=json_data)

    def config_dump_as_json(self) -> None:
        """TODO."""
        # Save config
        self.config_path_obj.write_text(
            data=self.config.model_dump_json(indent=4),
            encoding="utf-8",
        )

        # Print config
        console.log(self.config)
        console.log(
            f"[bold green] --> Saved config to '{self.config_path_obj.name}' :white_check_mark:\n",
        )

    def create_mjspec(self) -> None:
        """TODO."""
        # Parameters
        self.set_class_attributes()

        # Create MjSpec
        self._create_mjspec()

    def load_mjspec_from_xml(self) -> None:
        """TODO."""
        xml_str = self.xml_path_obj.read_text(encoding="utf-8")
        self.spec = mujoco.MjSpec.from_string(xml_str)

    def dump_mjspec_as_xml(self) -> None:
        """TODO."""
        # Save XML
        xml_str = self.spec.to_xml()
        self.xml_path_obj.write_text(
            data=xml_str,
            encoding="utf-8",
        )

        # Print XML
        console.log(xml_str)
        console.log(
            f"[bold green] --> Saved brick config to '{self.xml_path_obj.name}' :white_check_mark:\n",
        )
