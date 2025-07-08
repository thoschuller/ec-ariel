"""TODO(jmdm): description of script.

Date:       2025-07-08
Status:     Completed ✅
"""

# Standard library
from abc import ABC, abstractmethod


class Module(ABC):
    """Base class for all modules."""

    def __init__(self, index: int) -> None:
        """
        Initialize the module with an index.

        Parameters
        ----------
        index : int
            The index of the module.
        """
        self.index = index

    @abstractmethod
    def rotate(self, angle: float) -> None:
        """
        Rotate the module by a certain angle.

        Parameters
        ----------
        angle : float
            The angle to rotate the module by, in degrees.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        msg = f"{self.__class__.__name__} does not implement 'rotate' method."
        raise NotImplementedError(msg)
