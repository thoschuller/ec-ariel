from _typeshed import Incomplete
from typing import Callable, Generic, TypeVar

R = TypeVar('R')

class Thunk(Generic[R]):
    """
    A simple lazy evaluation implementation that lets you delay
    execution of a function.  It properly handles releasing the
    function once it is forced.
    """
    f: Callable[[], R] | None
    r: R | None
    __slots__: Incomplete
    def __init__(self, f: Callable[[], R]) -> None: ...
    def force(self) -> R: ...
