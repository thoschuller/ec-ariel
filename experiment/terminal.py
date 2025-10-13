import sys
import time
from rich.console import Console
from rich.progress import Progress
from typing import Any
import constants
from rich import pretty


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

_console = Console(file=log_file)
progress = Progress()

class CustomConsole:
    def __getattr__(self, name: str) -> Any:
        return getattr(_console, name)
    def log(self, message: str) -> None:
        # If progress is started, use its console to log without interrupting the bar
        if progress.live and progress.live.is_started:
            progress.console.log(message, _stack_offset=2)
        _console.log(message, _stack_offset=2)
    def rule(self, *args: Any, **kwargs: Any) -> None:
        if progress.live and progress.live.is_started:
            progress.console.rule(*args, **kwargs)
        _console.rule(*args, **kwargs)
    
console = CustomConsole()

pretty.install(console=console)
