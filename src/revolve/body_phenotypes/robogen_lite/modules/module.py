"""TODO(jmdm): description of script.

Date:       2025-07-08
Status:     Completed ✅
"""


class Module:
    """Base class for all modules."""

    # Count instances
    index = 0

    def __init__(self) -> None:
        """Initialize the module."""
        Module.index += 1
