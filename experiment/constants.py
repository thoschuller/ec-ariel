from pathlib import Path
from ariel.simulation.environments import OlympicArena
import multiprocessing

# --- DATA SETUP --- #
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)
OUTPUT = CWD / "output"
OUTPUT.mkdir(exist_ok=True)

SIM_WORLD = OlympicArena
SEED = 42
SEGMENT_LENGTH = 250

BRAIN_POP_SIZE = 20
BRAIN_MAX_GENERATIONS = 125
BRAIN_BATCH_SIZE = 25
BRAIN_TIME_LIMIT = 60 * 60

HIDDEN_SIZE = 8
OUTPUT_DELTA = 0.05
LATERAL_PENALTY_FACTOR = 0.1
NUM_HIDDEN_LAYERS = 1

BODY_POP_SIZE = 8
BODY_MAX_GENERATIONS = 50
BODY_TIME_LIMIT = 60 * 60 * 7  # in seconds
UNIFORM_CROSSOVER = True

RECORD_EVERY_INDIVIDUAL = False
# RECORD_LAST = False
# RECORD_BATCH = False
# INTERACTIVE_MODE = False

DETAILED_LOGGING = False

PARALLEL = True
DEVICE = "cpu"
PARALLEL_CORES = (
    multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1
)

NUM_OF_MODULES = 30

STAGE_SETTINGS = {
    1: {
        "SECTIONED_MODE": True,
        "DURATION": 10,
    },
    2: {
        "SECTIONED_MODE": True,
        "DURATION": 30,
    },
    3: {
        "SECTIONED_MODE": False,
        "DURATION": 90,
    },
    "FULL": {
        "SECTIONED_MODE": False,
        "DURATION": 120,
    },
}

# TODO: check
POSITIONS = [
    ([-0.8, 0, 0.1], [1.5, 0, 0.1]),  # Section 1: Flat
    ([1.5, 0, 0.1], [2.5, 0, 0.1]),  # Section 2: Rugged
    ([2.75, 0, 0.1], [5.0, 0, 0.5]),  # Section 3: Inclined
]
