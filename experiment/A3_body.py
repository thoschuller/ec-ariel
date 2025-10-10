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
from rich.progress import Progress
from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

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


# Fancy console messages and progress bars
install()
(CWD / "output" / "logs").mkdir(parents=True, exist_ok=True)
# console = Console(file=dual_writer, emoji=False, markup=False)
console = Console(file = open(CWD / "output" / "logs" / (time.strftime("%Y%m%d-%H%M%S") + "-evolution.txt"), "a"), emoji=False, markup=False)
# console = Console()
console.rule(f"Body evolution started.")
PROGRESS = Progress(console=console)


# def fitness_function(history: list[list[float]]) -> float:
#     xt, yt, zt = TARGET_POSITION
#     xc, yc, zc = history[-1]

#     # Minimize the distance --> maximize the negative distance
#     cartesian_distance = np.sqrt(
#         (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
#     )
#     return -cartesian_distance


def show_xpos_history(history: list[list[float]]) -> None:
    # Create a tracking camera
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(
        model,
        data,
        camera=camera,
        save_path=save_path,
        save=True,
    )

    # Setup background image
    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Calculate initial position
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]

    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    # Plot x,y trajectory
    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")

    # Add labels and title
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()

    # Title
    plt.title("Robot Path in XY Plane")

    # Show results
    plt.show()


# def nn_controller(
#     model: mj.MjModel,
#     data: mj.MjData,
# ) -> npt.NDArray[np.float64]:
#     # Simple 3-layer neural network
#     input_size = len(data.qpos)
#     hidden_size = 8
#     output_size = model.nu

#     # Initialize the networks weights randomly
#     # Normally, you would use the genes of an individual as the weights,
#     # Here we set them randomly for simplicity.
#     w1 = RNG.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
#     w2 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
#     w3 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))

#     # Get inputs, in this case the positions of the actuator motors (hinges)
#     inputs = data.qpos

#     # Run the inputs through the lays of the network.
#     layer1 = np.tanh(np.dot(inputs, w1))
#     layer2 = np.tanh(np.dot(layer1, w2))
#     outputs = np.tanh(np.dot(layer2, w3))

#     # Scale the outputs
#     return outputs * np.pi


# def experiment(
#     robot: Any,
#     controller: Controller,
#     duration: int = 15,
#     mode: ViewerTypes = "viewer",
# ) -> None:
#     """Run the simulation with random movements."""
#     # ==================================================================== #
#     # Initialise controller to controller to None, always in the beginning.
#     mj.set_mjcb_control(None)  # DO NOT REMOVE

#     # Initialise world
#     # Import environments from ariel.simulation.environments
#     world = OlympicArena()

#     # Spawn robot in the world
#     # Check docstring for spawn conditions
#     world.spawn(robot.spec, spawn_position=SPAWN_POS)

#     # Generate the model and data
#     # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
#     model = world.spec.compile()
#     data = mj.MjData(model)

#     # Reset state and time of simulation
#     mj.mj_resetData(model, data)

#     # Pass the model and data to the tracker
#     if controller.tracker is not None:
#         controller.tracker.setup(world.spec, data)

#     # Set the control callback function
#     # This is called every time step to get the next action.
#     args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
#     kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

#     mj.set_mjcb_control(
#         lambda m, d: controller.set_control(m, d, *args, **kwargs), # type: ignore
#     )

#     # ------------------------------------------------------------------ #
#     match mode:
#         case "simple":
#             # This disables visualisation (fastest option)
#             simple_runner(
#                 model,
#                 data,
#                 duration=duration,
#             )
#         case "frame":
#             # Render a single frame (for debugging)
#             save_path = str(DATA / "robot.png")
#             single_frame_renderer(model, data, save=True, save_path=save_path)
#         case "video":
#             # This records a video of the simulation
#             path_to_video_folder = str(DATA / "videos")
#             video_recorder = VideoRecorder(output_folder=path_to_video_folder)

#             # Render with video recorder
#             video_renderer(
#                 model,
#                 data,
#                 duration=duration,
#                 video_recorder=video_recorder,
#             )
#         case "launcher":
#             # This opens a liver viewer of the simulation
#             viewer.launch(
#                 model=model,
#                 data=data,
#             )
#         case "no_control":
#             # If mj.set_mjcb_control(None), you can control the limbs manually.
#             mj.set_mjcb_control(None)
#             viewer.launch(
#                 model=model,
#                 data=data,
#             )
#     # ==================================================================== #

# class robot to hold its genotype and phenotype
# class RobotGenotype(): 
#     def __init__(self, body_genotype: list[list[float]] = [], brain_genotype: np.ndarray = None) -> None:
#         self.body_genotype = body_genotype
#         self.brain_genotype = brain_genotype

# class RobotPhenotype:
#     def __init__(self, body_phenotype: DiGraph[Any] = DiGraph(), brain_phenotype: Any = None) -> None:
#         self.body_phenotype = body_phenotype
#         self.brain_phenotype = brain_phenotype

# class Robot:
#     def __init__(self, genotype: RobotGenotype = RobotGenotype(), phenotype: RobotPhenotype = RobotPhenotype()) -> None:
#         self.genotype = genotype
#         self.phenotype = phenotype

#         if self.phenotype.body_phenotype == DiGraph() and len(self.genotype["body_genotype"]) > 0:
#             p_matrices = NDE.forward(self.genotype["body_genotype"])

#             # Decode the high-probability graph
#             hpd = HighProbabilityDecoder(NUM_OF_MODULES)
#             self.phenotype.body_phenotype = hpd.probability_matrices_to_graph(
#                 p_matrices[0],
#                 p_matrices[1],
#                 p_matrices[2],
#             )

#         if self.phenotype.brain_phenotype is None and self.genotype["brain_genotype"] != np.ndarray(None):
#             self.phenotype.brain_phenotype = a3cma.get_controller_from_weights(self.genotype["brain_genotype"])

def tolist_recursive(obj: Any) -> list[Any] | tuple[Any, ...] | dict[Any, Any] | Any:
    """Recursively convert numpy arrays in a structure to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [tolist_recursive(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(tolist_recursive(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: tolist_recursive(v) for k, v in obj.items()}
    else:
        return obj

config_overrides = {
    "MAX_GENERATIONS": 75,
    "MULTI_EVAL_RUNS": 1,
    "CONSOLE": console,
    "PROGRESS": PROGRESS,
    "SECTIONED_MODE": True,  # Start with sectioned training
    "DURATION": 20,
}


def create_individual() -> Individual:
    """Create a new individual with Glorot initialization."""
    individual = Individual()
    individual.requires_init = True
    individual.requires_eval = True
    return individual

def create_population(size: int) -> Population:
    """Create a population of individuals."""
    initialization_task = PROGRESS.add_task("[green]Creating individuals...", total=size)
    population = []
    for _ in range(size):
        population.append(create_individual())
        PROGRESS.update(initialization_task, advance=1)
    PROGRESS.remove_task(initialization_task)
    return population

def initialize_individual(individual: Individual, config_overrides: dict[str, Any] = config_overrides) -> Individual:
    """train and evaluate a single individual, set its fitness attribute"""
    individual.genotype = ( tolist_recursive([
        RNG.random(64).astype(np.float32),
        RNG.random(64).astype(np.float32),
        RNG.random(64).astype(np.float32),
    ]),
    [] # brain genotype will be set after training
    )
    individual = train_individual_brain(individual, config_overrides)
    individual.requires_init = False
    individual.requires_eval = False
    return individual

def initialize_population(population: Population, config_overrides: dict[str, Any] = config_overrides) -> Population:
    """initialize a population of individuals"""
    new_population = []
    for idx, individual in enumerate(population):
        console.log(f"Initializing individual number {idx+1}/{len(population)}")
        if individual.requires_init:
            new_population.append(initialize_individual(individual, config_overrides))
        else:
            new_population.append(individual)
        console.log(f"individual initialized with fitness {individual.fitness:.4f}")
    return new_population

def train_individual_brain(individual: Individual, config_overrides: dict[str, Any] = config_overrides) -> Individual:
    """train the brain of a single individual, keep its body unchanged"""
    console.log("Starting brain training...")
    if individual.requires_init:
        raise ValueError("Individual must be initialized before training its brain.")
    else:
        p_matrices = NDE.forward(np.array(individual.genotype[0]))
        training_result = a3cma.evolve_using_cma_es(
            gecko_body=copy.deepcopy(HPD.probability_matrices_to_graph(
                p_matrices[0],
                p_matrices[1],
                p_matrices[2],
            )),
            config_overrides=config_overrides
        )
        individual.genotype = (individual.genotype[0], training_result["genotype"])
        individual.fitness = training_result["fitness"]
        individual.requires_eval = False
    console.log("Brain training completed.")
    return individual

def train_and_evaluate_individual(individual: Individual, config_overrides: dict[str, Any] = config_overrides) -> Individual:
    """train and evaluate a single individual, set its fitness attribute"""
    if individual.requires_init:
        individual = initialize_individual(individual)
    else:
        individual = train_individual_brain(individual, config_overrides)
        individual.requires_eval = False
    return individual

def evaluate_population(population: Population) -> Population:
    """evaluate a population of individuals"""
    console.log("Starting population evaluation...")
    new_population = []
    re_evaluated = 0
    eval_inds = [ind for ind in population if ind.requires_eval]
    evaluation_task = PROGRESS.add_task("[green]Evaluating individuals...", total=len(eval_inds))
    for individual in eval_inds:
        new_population.append(train_and_evaluate_individual(individual))
        re_evaluated += 1
        PROGRESS.update(evaluation_task, advance=1)
    for individual in [ind for ind in population if not ind.requires_eval]:
        new_population.append(individual)
    PROGRESS.stop_task(evaluation_task)
    evaluation_time = PROGRESS.get_timer(evaluation_task)
    PROGRESS.remove_task(evaluation_task)
    console.log(f"Re-evaluated {re_evaluated}/{len(population)} individuals in {evaluation_time:.2f} seconds.")
    return new_population

def parent_selection(population: Population) -> Population:
    #TODO: implement a better selection mechanism
    """Tournament selection"""
    console.log("Starting parent selection...")
    task = PROGRESS.add_task("[green]Selecting parents...", total=len(population)//2)
    PROGRESS.start_task(task)

    # Shuffle population to avoid bias
    np.random.shuffle(population)

    # Tournament selection
    for idx in range(0, len(population) - 1, 2):
        ind_i = population[idx]
        ind_j = population[idx + 1]

        # Compare fitness values and update tags
        if ind_i.fitness > ind_j.fitness:
            ind_i.tags['ps'] = True
            ind_j.tags['ps'] = False
        else:
            ind_i.tags['ps'] = False
            ind_j.tags['ps'] = True
        PROGRESS.update(task, advance=1)
    PROGRESS.stop_task(task)
    task_time = PROGRESS.get_timer(task)
    PROGRESS.remove_task(task)
    console.log(f"Parent selection completed in {task_time:.2f} seconds.")
    return population

def survivor_selection(population: Population) -> Population:
    console.log("Starting survivor selection...")
    task = PROGRESS.add_task("[green]Selecting survivors...")

    # Shuffle population to avoid bias
    np.random.shuffle(population)
    current_pop_size = len(population)

    # Iterate in pairs, never go out of bounds
    for idx in range(0, len(population) - 1, 2):
        ind_i = population[idx]
        ind_j = population[idx + 1]

        # Kill worse individual
        if ind_i.fitness > ind_j.fitness:
            ind_j.alive = False
        else:
            ind_i.alive = False

        # Termination condition
        current_pop_size -= 1
        if current_pop_size <= POP_SIZE:
            break

    # Remove dead individuals to maintain population size
    survivors = [ind for ind in population if getattr(ind, 'alive', True)]
    # If too many, trim to POP_SIZE
    if len(survivors) > POP_SIZE:
        survivors = survivors[:POP_SIZE]

    PROGRESS.stop_task(task)
    task_time = PROGRESS.get_timer(task)
    PROGRESS.remove_task(task)
    console.log(f"Survivor selection completed in {task_time:.2f} seconds.")

    return survivors

class Crossover:    
    @staticmethod
    def uniform(
        parent_i: list[list[float]],
        parent_j: list[list[float]],
    ) -> tuple[list[list[float]], list[list[float]]]:
        child1, child2 = [], []
        for i in range(len(parent_i)):
            mask = np.random.randint(0, 2, size=len(parent_i[i])).astype(bool)
            c1 = np.array(parent_i[i], dtype=np.float32)
            c2 = np.array(parent_j[i], dtype=np.float32)
            c1[mask] = np.array(parent_j[i], dtype=np.float32)[mask]
            c2[mask] = np.array(parent_i[i], dtype=np.float32)[mask]
            child1.append(c1.tolist())
            child2.append(c2.tolist())
        return child1, child2

def crossover_individuals(ind1 : Individual, ind2: Individual) -> tuple[Individual, Individual]:
    parent_i = ind1.model_copy(deep=True)
    parent_j = ind2.model_copy(deep=True)

    # Decide which to crossover and which to clone directly

    if np.random.random() < 0.25:
        child_i = parent_i
        child_j = parent_j
    
    else:
        child_i = Individual()
        child_j = Individual()
        body_genotype_i, body_genotype_j = Crossover.uniform(
            cast("list[list[float]]", parent_i.genotype[0]),
            cast("list[list[float]]", parent_j.genotype[0]),
        )
        child_i.genotype = (body_genotype_i, None)
        child_i.requires_eval = True
        child_j.genotype = (body_genotype_j, None)
        child_j.requires_eval = True


    child_i.tags['mut'] = True
    child_j.tags['mut'] = True

    ind1.tags['ps'] = False
    ind2.tags['ps'] = False

    return child_i, child_j

def crossover(population: Population) -> Population:
    """Crossover individuals tagged for parent selection"""
    # Shuffle population to avoid bias

    console.log("Starting crossover...")
    parents = [ind for ind in population if ind.tags.get('ps', False)]

    task = PROGRESS.add_task("[green]Crossover...", total=len(parents)//2)
    PROGRESS.start_task(task)

    np.random.shuffle(parents)
    for idx in range(0, len(parents) - 1, 2):
            parent_i = parents[idx]
            parent_j = parents[idx+1]
            child_i, child_j = crossover_individuals(parent_i, parent_j)
            population.extend([child_i, child_j])
            PROGRESS.update(task, advance=1)
    PROGRESS.stop_task(task)
    task_time = PROGRESS.get_timer(task)
    PROGRESS.remove_task(task)
    console.log(f"Crossover completed in {task_time:.2f} seconds.")

    return population

def mutate_individual(individual: Individual, mutation_probability: float = 0.5, mutation_stddev: float = 0.1) -> Individual:
    """Mutate an individual's body genotype with given probability and stddev"""
    body_genotype = cast("list[list[float]]", individual.genotype[0])
    mutated_body_genotype = []
    for gene_array in body_genotype:
        gene_array_np = np.array(gene_array, dtype=np.float32)  # Ensure numpy array
        mutation_mask = RNG.random(gene_array_np.shape) > mutation_probability
        mutations = RNG.normal(0, mutation_stddev, gene_array_np.shape)
        new_gene_array = gene_array_np + mutation_mask * mutations
        mutated_body_genotype.append(new_gene_array.astype(np.float32).tolist())  # Convert back to list
    mutated_individual = Individual()
    mutated_individual.genotype = (mutated_body_genotype, None)
    mutated_individual.requires_eval = True
    mutated_individual.tags = individual.tags.copy()
    mutated_individual.tags['mut'] = False
    return mutated_individual

def mutate(population: Population, mutation_probability: float = 0.5, mutation_stddev: float = 0.1) -> Population:
    """Mutate individuals tagged for mutation"""
    console.log("Starting mutation...")
    new_population = []
    mutable_inds = [ind for ind in population if ind.tags.get('mut', True)]
    task = PROGRESS.add_task("[green]Mutating individuals...", total=len(mutable_inds))
    PROGRESS.start_task(task)
    for individual in mutable_inds:
        mutated = mutate_individual(individual, mutation_probability, mutation_stddev)
        new_population.append(mutated)
        PROGRESS.update(task, advance=1)
    for individual in [ind for ind in population if not ind.tags.get('mut', True)]:
        new_population.append(individual)

    
    PROGRESS.stop_task(task)
    task_time = PROGRESS.get_timer(task)
    PROGRESS.remove_task(task)
    console.log(f"Mutation completed in {task_time:.2f} seconds.")

    return new_population

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

def body_evolution() -> tuple[float, list[list[float]], np.ndarray, DiGraph]: #type: ignore
    """full evolution of body and brain genotypes
    returns fitness, body_genotype, brain_genotype, body_phenotype"""

    start_time = time.time()

    try:

        # get_pool()

        # Create initial population
        console.rule("Creating initial population")
        population = evaluate_population(initialize_population(create_population(POP_SIZE)))
        console.log(f"Initial population created with {len(population)} individuals.")
        ops = [
            EAStep("evaluation", evaluate_population),
            # EAStep("show_best", show_best_of_population),
            EAStep("parent_selection", parent_selection),
            EAStep("crossover", crossover),
            EAStep("mutation", mutate),
            EAStep("evalutation", evaluate_population),
            EAStep("survivor_selection", survivor_selection),
        ]
        ea = EA(
            population=population,
            operations=ops,
            quiet=False,
        )

        def terminate() -> bool:
            if MAX_GENERATIONS and ea.current_generation >= MAX_GENERATIONS:
                console.log("Reached maximum generations.")
                return True
            if time.time() - start_time >= TIME_LIMIT:
                console.log("Reached time limit.")
                return True
            return False
        
        evolution_task = PROGRESS.add_task("[green]Evolving bodies...", total=MAX_GENERATIONS if MAX_GENERATIONS else TIME_LIMIT if TIME_LIMIT else None)
        PROGRESS.start()

        # Ensure output/genotypes and output/weights directories exist
        (CWD / "output" / "genotypes").mkdir(parents=True, exist_ok=True)
        (CWD / "output" / "weights").mkdir(parents=True, exist_ok=True)
        (CWD / "output" / "plots").mkdir(parents=True, exist_ok=True)

        # Prepare CSV for logging fitness
        import csv
        fitness_log_path = CWD / "output" / "logs" / f"fitness_log-{time.strftime('%Y%m%d-%H%M%S')}.csv"
        # Write header if file does not exist
        if not fitness_log_path.exists():
            with open(fitness_log_path, mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["generation", "fitness"])

        ea.fetch_population()
        for ind in ea.population:
                with open(fitness_log_path, mode="a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([ea.current_generation, ind.fitness])

        while not terminate():
            console.log(f"Running evolution step for generation {ea.current_generation}...")
            ea.step()
            best_ind = ea.get_solution('best', only_alive=False)
            best_fitness = best_ind.fitness
            # Compute average fitness (only for alive individuals)
            ea.fetch_population()
            # Log to CSV
            for ind in ea.population:
                with open(fitness_log_path, mode="a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([ea.current_generation, ind.fitness])

            console.log(f"Generation {ea.current_generation}: Best Fitness = {best_fitness:.4f}, Population Size = {ea.population_size}")
            runtime = time.time() - start_time
            PROGRESS.update(evolution_task, completed=ea.current_generation if MAX_GENERATIONS else runtime, description=f"[green]Evolving bodies... Generation {ea.current_generation}, Best Fitness: {best_fitness:.4f}, runtime: {runtime // 3600}h {(runtime % 3600) // 60}m {(runtime % 60):.0f}s")
            console.log(f"Running best individual of generation {ea.current_generation} with fitness {best_fitness:.4f} for recording...")
            p_matrices = NDE.forward(np.array(best_ind.genotype[0]))
            hpd = HPD
            gecko_body = hpd.probability_matrices_to_graph(p_matrices[0], p_matrices[1], p_matrices[2])
            # Save JSON of best body
            save_graph_as_json(
                gecko_body,
                CWD / "output" / "genotypes" / f"best_body_genotype_gen{ea.current_generation}_fit{best_fitness:.4f}.json",
            )
            # Save weights of best brain
            weights_path = CWD / "output" / "weights" / f"best_brain_weights_gen{ea.current_generation}_fit{best_fitness:.4f}.npy"
            np.save(weights_path, np.array(best_ind.genotype[1]))
            # Save plot of best individual's trajectory if available
            try:
                # Try to get tracker from a3cma if available
                if hasattr(best_ind, 'tracker') and getattr(best_ind, 'tracker', None) is not None:
                    tracker = best_ind.tracker
                elif hasattr(best_ind, 'history') and getattr(best_ind, 'history', None) is not None:
                    tracker = best_ind
                else:
                    tracker = None
                # If tracker has history, plot it
                if tracker is not None and hasattr(tracker, 'history') and 'xpos' in tracker.history:
                    xpos_history = tracker.history['xpos'][0]
                    plt.figure()
                    show_xpos_history(xpos_history)
                    plt.savefig(CWD / "output" / "plots" / f"best_path_gen{ea.current_generation}_fit{best_fitness:.4f}.png")
                    plt.close()
                else:
                    no_plot_reason = "no tracker" if tracker is None else "no history" if not hasattr(tracker, 'history') else "no xpos in history"
                    console.log(f"[yellow]Warning: No tracker history available for generation {ea.current_generation}, {no_plot_reason}, skipping path plot.")
            except Exception as e:
                console.log(f"[yellow]Warning: Could not save plot for generation {ea.current_generation}: {e}")

            # Record video as before
            a3cma.run_weights_only(method="record", weights=np.array(best_ind.genotype[1]), gecko_body=gecko_body, options={"filename": f"best_body_individual_gen{ea.current_generation}", "fitness": best_fitness}, duration=120)

            # Stage transitions based on fitness thresholds
            if config_overrides.get("SECTIONED_MODE", False):
                # Stage 1: Sectioned training (fitness in [-1, 0])
                # -1 = no progress, 0 = all sections complete
                # When sectioned fitness >= -0.2, sections are performing very well
                if best_fitness >= -0.2:
                    console.rule(f"Reached sectioned fitness threshold with fitness {best_fitness:.4f}. Switching to full arena mode.")
                    config_overrides["SECTIONED_MODE"] = False
                    config_overrides["MULTI_EVAL_RUNS"] = 1
                    config_overrides["DURATION"] = 50
                    a3cma.set_config(config_overrides)
            else:
                # Stage 2: Full arena training (fitness in [0, 2+])
                # 0 = no progress, 1 = reached goal, 2 = reached goal quickly
                current_duration = config_overrides.get("DURATION", 0)
                if best_fitness >= 0.6 and isinstance(current_duration, int) and current_duration < 60:
                    console.rule(f"Reached full arena fitness threshold with fitness {best_fitness:.4f}. Increasing duration.")
                    config_overrides["MULTI_EVAL_RUNS"] = 1
                    config_overrides["DURATION"] = 100
                    a3cma.set_config(config_overrides)

        PROGRESS.remove_task(evolution_task)
        PROGRESS.stop()

        console.rule("Evolution process finished.")
        console.log("Best Fitness:", ea.get_solution('best', only_alive=False).fitness)
        console.log("Saving best individual...")
        p_matrices = NDE.forward(np.array(ea.get_solution('best', only_alive=False).genotype[0]))
        hpd = HPD
        save_graph_as_json(
            hpd.probability_matrices_to_graph(
                p_matrices[0],
                p_matrices[1],
                p_matrices[2],
            ),
            CWD / "output" / "genotypes" / f"best_body_genotype_final_fit{ea.get_solution('best', only_alive=False).fitness:.4f}.json",
        )



    finally:
        if PROGRESS:
            PROGRESS.stop()
        # close_pool()
        console.log("Evolution process completed.")


    final_ind = ea.get_solution('best', only_alive=False)
    final_fit: float = final_ind.fitness
    final_body_genotype: list[list[float]] = cast("list[list[float]]", final_ind.genotype[0])
    final_brain_genotype: np.ndarray = np.array(final_ind.genotype[1])
    p_matrices = NDE.forward(np.array(final_body_genotype))
    hpd = HPD
    final_graph = hpd.probability_matrices_to_graph(p_matrices[0], p_matrices[1], p_matrices[2])
    return final_fit, final_body_genotype, final_brain_genotype, final_graph


# def main() -> None:
#     """Entry point."""
#     # ? ------------------------------------------------------------------ #
#     genotype_size = 64
#     type_p_genes = RNG.random(genotype_size).astype(np.float32)
#     conn_p_genes = RNG.random(genotype_size).astype(np.float32)
#     rot_p_genes = RNG.random(genotype_size).astype(np.float32)

#     genotype = [
#         type_p_genes,
#         conn_p_genes,
#         rot_p_genes,
#     ]

#     p_matrices = NDE.forward(genotype)

#     # Decode the high-probability graph
#     robot_graph: DiGraph[Any] = HPD.probability_matrices_to_graph(
#         p_matrices[0],
#         p_matrices[1],
#         p_matrices[2],
#     )

#     # ? ------------------------------------------------------------------ #
#     # Save the graph to a file
#     save_graph_as_json(
#         robot_graph,
#         DATA / "robot_graph.json",
#     )

#     # ? ------------------------------------------------------------------ #
#     # Default CONFIGURATION
#     # "SIM_WORLD": OlympicArena,
#     # "SEED": 42,
#     # "SEGMENT_LENGTH": 250,
#     # "POP_SIZE": 30,
#     # "MAX_GENERATIONS": 125,
#     # "TIME_LIMIT": 60*15, 
#     # "HIDDEN_SIZE": 8,
#     # "DURATION": 15,
#     # "OUTPUT_DELTA": 0.05,
#     # "NUM_HIDDEN_LAYERS": 1,
#     # "FITNESS_MODE": "lateral_adjusted",
#     # "UNIFORM_CROSSOVER": True,
#     # "LATERAL_PENALTY_FACTOR": 0.1,
#     # "MULTI_EVAL": True,
#     # "INTERACTIVE_MODE": False,
#     # "PARALLEL": True,
#     # "RECORD_LAST": True,
#     # "BATCH_SIZE": 25,
#     # "RECORD_BATCH": True,
#     # "DETAILED_LOGGING": True,
#     # "DEVICE": "cpu",
#     # "PARALLEL_CORES": multiprocessing.cpu_count()-1 if multiprocessing.cpu_count() > 1 else 1,
#     # "MULTI_RUN_OPTIONS": {},
#     # "MULTI_EVAL_RUNS": 1,
#     # "RNG": np.random.default_rng(42),
#     # "FITNESS_FUNCTION": None,  # Allow override
#     # "GECKO_BODY": None,        # Allow override
#     # "CONSOLE": None,
#     # "MUTATION_PROBABILITY": 0.5,
#     # "MUTATION_STDDEV": 0.1,
#     # "SAVE_PLOTS": True
#     # ? ------------------------------------------------------------------ #

#     result = a3cma.evolve_using_cma_es(
#         gecko_body=robot_graph,
#         config_overrides={"MAX_GENERATIONS": 60},
#     )

#     fitness = result["fitness"]
#     tracker = result["tracker"]

#     show_xpos_history(tracker.history["xpos"][0])

#     # fitness = fitness_function(tracker.history["xpos"][0])
#     msg = f"Fitness of generated robot: {fitness}"
#     console.log(msg)

if __name__ == "__main__":
    body_evolution()
