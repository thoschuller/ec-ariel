import sys
import time
from rich.console import Console
from rich.progress import Progress
from typing import Any
import constants

LOG_PATH = constants.OUTPUT / "logs" / f"evolver-{time.strftime('%Y%m%d-%H%M%S')}.txt"

class DualWriter:
    def __init__(self, *files: Any) -> None:
        self.files: tuple[Any, ...] = files
    def write(self, data: str) -> None:
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self) -> None:
        for f in self.files:
            f.flush()
            
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
log_file = open(LOG_PATH, "w", encoding="utf-8", buffering=1)

dual_writer = DualWriter(sys.stdout, log_file)

console = Console(file=dual_writer)
progress = Progress(console=console)

