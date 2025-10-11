"""Assignment 3 template code."""

# Standard library
from pathlib import Path
import time
from typing import Any, Literal, cast
import copy

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
# from mujoco import viewer

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.a001 import Individual
from ariel.ec.a003 import Population
from ariel.ec.a004 import EAStep, EA
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer
# from ariel.utils.tracker import Tracker
# from ariel.simulation.controllers.controller import Controller
import A3_net_cma as a3cma

from rich.console import Console
from rich.traceback import install
from rich.progress import progress
from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- RANDOM GENERATOR SETUP --- #
SEED = 42


# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
# TARGET_POSITION = [5, 0, 0.5]
NDE = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
HPD = HighProbabilityDecoder(NUM_OF_MODULES)
POP_SIZE = 8
TIME_LIMIT = 60*60*5 # in seconds
MAX_GENERATIONS = None







config_overrides = {
    "MAX_GENERATIONS": 75,
    "MULTI_EVAL_RUNS": 1,
    # "CONSOLE": console,
    # "progress": progress,
    "SECTIONED_MODE": True,  # Start with sectioned training
    "DURATION": 20,
}










def show_best_individual(individual: Individual) -> None:
    """Show the best individual in the viewer"""
    console.rule("Showing best individual")
    p_matrices = NDE.forward(np.array(individual.genotype[0]))
    hpd = HPD
    a3cma.run_weights_only(method="viewer", weights=np.array(individual.genotype[1]), gecko_body=hpd.probability_matrices_to_graph(p_matrices[0], p_matrices[1], p_matrices[2]))

def show_best_of_population(population: Population) -> Population:
    """Show the best individual of a population"""
    best_individual = max(population, key=lambda ind: ind.fitness)
    show_best_individual(best_individual)
    return population




if __name__ == "__main__":
    body_evolution()
