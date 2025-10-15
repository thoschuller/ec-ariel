from pathlib import Path
import multiprocessing
from ariel.simulation.environments import OlympicArena

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

BRAIN_POP_SIZE = 38
BRAIN_MAX_GENERATIONS = None # set to None to train until time limit, but was set to 1200 for matching random baseline iterations
BRAIN_BATCH_SIZE = 50
BRAIN_TIME_LIMIT = 60 * 15

HIDDEN_SIZE = 16
OUTPUT_DELTA = 0.05
LATERAL_PENALTY_FACTOR = 0.25
NUM_HIDDEN_LAYERS = 1

BODY_POP_SIZE = 10
BODY_MAX_GENERATIONS = None # set to None to train until time limit, but was set to 97 for matching random baseline iterations
BODY_TIME_LIMIT = 60 * 60 * 25.5  # in seconds
UNIFORM_CROSSOVER = True
BODY_BATCH_SIZE = 1

EXTRA_TRAINING_TIME = 60 * 60 * 4

DETAILED_LOGGING = False
BODY_EVO_RECORD_BATCH = True
BODY_EVO_RECORD_LAST = True
BRAIN_EVO_RECORD_BATCH = True
BRAIN_EVO_RECORD_LAST = True

RANDOM_BASELINE = False

PARALLEL = True
DEVICE = "cpu"
PARALLEL_CORES = (
    multiprocessing.cpu_count() if multiprocessing.cpu_count() > 1 else 1
)

NUM_OF_MODULES = 30

STAGE_SETTINGS = {
    1: {
        "SECTIONED_MODE": True,
        "DURATION": 15,
        "MAX_STAGNATION": 15,
        "MAX_STAGNATION_DELTA": 0.0001,
    },
    2: {
        "SECTIONED_MODE": True,
        "DURATION": 30,
        "MAX_STAGNATION": 10,
        "MAX_STAGNATION_DELTA": 0.001,
    },
    3: {
        "SECTIONED_MODE": False,
        "DURATION": 90,
        "MAX_STAGNATION": 10,
        "MAX_STAGNATION_DELTA": 0.0001,
    },
    "FULL": {
        "SECTIONED_MODE": False,
        "DURATION": 120,
        "MAX_STAGNATION": 6,
        "MAX_STAGNATION_DELTA": 0.0001,
    },
}

# TODO: check
POSITIONS = [
    ([-0.8, 0, 0.1], [0.5, 0, 0.1]),  # Section 1: Flat
    ([0.5, 0, 0.1], [2.6, 0, 0.1]),  # Section 2: Rugged
    ([2.5, 0, 0.1], [5.0, 0, 0.5]),  # Section 3: Inclined
]
