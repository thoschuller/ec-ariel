# Standard library
import time
from typing import cast
import copy

import numpy as np

# from mujoco import viewer

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.ec.a001 import Individual
from ariel.ec.a003 import Population
from ariel.ec.a004 import EAStep, EA
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding

# from ariel.utils.tracker import Tracker
# from ariel.simulation.controllers.controller import Controller
import constants as constants
from terminal import console, progress
from networkx import DiGraph, is_isomorphic
from utils import numpy_tolist
import brain_train as braintrain
import utils as utils
from rich import pretty
import evaluator as evaluator

RNG = np.random.default_rng(constants.SEED)
HPD = HighProbabilityDecoder(constants.NUM_OF_MODULES)
NDE = NeuralDevelopmentalEncoding(number_of_modules=constants.NUM_OF_MODULES)
GENOTYPE_SIZE = 64
import session_runner as runner

current_stage: int | str = 1
def _create_individual() -> Individual:
    """glorot initialization of body genotype, brain genotype is not initialized yet"""
    individual = Individual()
    
    limit = np.sqrt(6 / (64 + 64))
    type_p_genes = RNG.uniform(-limit, limit, GENOTYPE_SIZE)
    conn_p_genes = RNG.uniform(-limit, limit, GENOTYPE_SIZE)
    rot_p_genes = RNG.uniform(-limit, limit, GENOTYPE_SIZE)
    
    individual.genotype = (
        numpy_tolist(
            [
                type_p_genes,
                conn_p_genes,
                rot_p_genes
            ]
        ),
        [],  # brain genotype will be set after training
    )
    
    individual.requires_init = True
    individual.requires_eval = True
    return individual




def _create_population(size: int) -> Population:
    """Create a population of individuals."""
    creation_task = progress.add_task(
        "[green]Creating individuals...", total=size
    )
    population = []        
    while len(population) < size:
        new_ind = _create_individual()
        unique = True
        gecko_body = HPD.probability_matrices_to_graph(
                NDE.forward(np.array(new_ind.genotype[0]))[0],
                NDE.forward(np.array(new_ind.genotype[0]))[1],
                NDE.forward(np.array(new_ind.genotype[0]))[2],
            )
        for k in range(len(population)):  
            if population[k].genotype[0] == new_ind.genotype[0]:
                console.log(f"Duplicate body genotype found with an existing individual. No sense in keeping both. Discarding individual and creating new.")
                unique = False
                break          
            p_matrices_k = NDE.forward(np.array(population[k].genotype[0]))
            gecko_body_k = HPD.probability_matrices_to_graph(
                p_matrices_k[0], p_matrices_k[1], p_matrices_k[2]
            )
            if is_isomorphic(gecko_body, gecko_body_k):
                console.log(f"Duplicate body phenotype found with an existing individual. No sense in keeping both. Discarding individual and creating new.")
                unique = False
                break
        if unique:
            population.append(train_and_evaluate_individual(new_ind))
            progress.update(creation_task, advance=1)
        
    progress.remove_task(creation_task)
    return population

def train_and_evaluate_individual_brain(individual: Individual) -> Individual:
    """train the brain of a single individual, keep its body unchanged"""
    console.log("Starting brain training...")
    
    p_matrices = NDE.forward(np.array(individual.genotype[0]))
    training_result = braintrain.evolve_using_cma_es(
        gecko_body=
            HPD.probability_matrices_to_graph(
                p_matrices[0],
                p_matrices[1],
                p_matrices[2],
            
        ),
        duration=constants.STAGE_SETTINGS[current_stage]["DURATION"],
        sectioned=constants.STAGE_SETTINGS[current_stage]["SECTIONED_MODE"],
        stagnation_threshold=constants.STAGE_SETTINGS[current_stage]["MAX_STAGNATION_DELTA"],
        max_stagnation=constants.STAGE_SETTINGS[current_stage]["MAX_STAGNATION"]
    )
    individual.genotype = (individual.genotype[0], training_result[0])
    individual.fitness = training_result[1]
    individual.requires_eval = False
    individual.requires_init = False
    
    console.log("Brain training completed.")
    return individual

def train_and_evaluate_individual(individual: Individual) -> Individual:
    """train and evaluate a single individual, set its fitness attribute"""
    individual = train_and_evaluate_individual_brain(individual)
    individual.requires_init = False
    individual.requires_eval = False
    return individual

def evaluate_individual(individual: Individual) -> Individual:
    console.log("Evaluating individual...")
    
    if individual.requires_init:
        individual = train_and_evaluate_individual_brain(individual)
        individual.requires_init = False
        individual.requires_eval = False
        return individual
    
    p_matrices = NDE.forward(np.array(individual.genotype[0]))
    result = evaluator.evaluate_individual(    
        genotype_list=individual.genotype[1],
        gecko_body=
            HPD.probability_matrices_to_graph(
                p_matrices[0],
                p_matrices[1],
                p_matrices[2],
            
        ),
        duration=constants.STAGE_SETTINGS[current_stage]["DURATION"],
        sectioned=constants.STAGE_SETTINGS[current_stage]["SECTIONED_MODE"]
    )
    individual.fitness = result
    individual.requires_eval = False
    
    console.log(f"Individual evaluated with fitness: {individual.fitness}")
    return individual

def reset_tags(population: Population) -> Population:
    for ind in population:
        ind.tags["mut"] = False
        ind.tags["ps"] = False
    return population

def reset_fitness(population: Population) -> Population:
    console.log("Re-evaluating ALL fitnesses")
    eval_task = progress.add_task("Evaluating full population", total=len(population))
    for ind in population:
        retrain_individual(ind)
        ind.requires_eval = False
        progress.update(eval_task, advance=1)
    console.log("Finished evaluating full population")
    progress.remove_task(eval_task)
    return population
    


def evaluate_population(population: Population) -> Population:
    """evaluate a population of individuals"""
    start_time = time.time()
    new_population = []
    re_evaluated = 0
    alive_pop = [ind for ind in population if getattr(ind, "alive", True)]
    eval_inds = [ind for ind in alive_pop if ind.requires_eval == True]
    console.log(f"Starting population evaluation for {len(eval_inds)} individuals...")
    evaluation_task = progress.add_task(
        "[green]Evaluating individuals...", total=len(eval_inds)
    )
    for individual in eval_inds:
        new_population.append(evaluate_individual(individual))
        re_evaluated += 1
        progress.update(evaluation_task, advance=1)
    for individual in [ind for ind in population if ind.requires_eval == False]:
        new_population.append(individual)
    evaluation_time = time.time() - start_time
    console.log(
        f"Re-evaluated {re_evaluated}/{len(population)} individuals in {evaluation_time:.2f} seconds."
    )
    progress.remove_task(evaluation_task)
    return new_population


def parent_selection(population: Population) -> Population:
    # TODO: implement a better selection mechanism
    """Tournament selection"""
    console.log("Starting parent selection...")
    task = progress.add_task("[green]Selecting parents...", total=len(population) // 2)
    start_time = time.time()
    progress.start_task(task)

    # Shuffle population to avoid bias
    np.random.shuffle(population)

    # Tournament selection
    for idx in range(0, len(population) - 1, 2):
        ind_i = population[idx]
        ind_j = population[idx + 1]

        # Compare fitness values and update tags
        if ind_i.fitness > ind_j.fitness:
            ind_i.tags["ps"] = True
            ind_j.tags["ps"] = False
        else:
            ind_i.tags["ps"] = False
            ind_j.tags["ps"] = True
        progress.update(task, advance=1)
    progress.stop_task(task)
    task_time = time.time() - start_time
    progress.remove_task(task)
    console.log(f"Parent selection completed in {task_time:.2f} seconds.")
    return population


def survivor_selection(population: Population) -> Population:
    console.log("Starting survivor selection...")
    task = progress.add_task("[green]Selecting survivors...")
    start_time = time.time()
    
    _population = [ind for ind in population if getattr(ind, "alive", True)]

    # Shuffle population to avoid bias
    np.random.shuffle(_population)
    current_pop_size = len(_population)

    # Iterate in pairs, never go out of bounds
    for idx in range(0, len(_population) - 1, 2):
        ind_i = _population[idx]
        ind_j = _population[idx + 1]

        # Kill worse individual
        if ind_i.fitness > ind_j.fitness:
            ind_j.alive = False
        else:
            ind_i.alive = False

        # Termination condition
        current_pop_size -= 1
        if current_pop_size <= constants.BODY_POP_SIZE:
            break

    # Remove dead individuals to maintain population size
    survivors = [ind for ind in _population if getattr(ind, "alive", True)]
    # If too many, trim to POP_SIZE
    if len(survivors) > constants.BODY_POP_SIZE:
        survivors.sort(key=lambda ind: ind.fitness, reverse=True)
        survivors = survivors[: constants.BODY_POP_SIZE]

    for ind in _population:
        if ind not in survivors:
            ind.alive = False

    task_time = time.time() - start_time
    progress.remove_task(task)
    console.log(f"Population size was {len(_population)}, is now {len(survivors)} out of {constants.BODY_POP_SIZE}")
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


def crossover_individuals(
    ind1: Individual, ind2: Individual
) -> tuple[Individual, Individual]:
    parent_i = ind1.model_copy(deep=True)
    parent_j = ind2.model_copy(deep=True)

    # Decide which to crossover and which to clone directly
    
    child_i = Individual()
    child_j = Individual()

    if np.random.random() < 0.25:
        for child in [child_i, child_j]:
            child.requires_init = False
            child.requires_eval = False
        child_j.genotype = parent_j.genotype
        child_j.fitness = parent_j.fitness
        
        
        child_i.genotype = parent_i.genotype
        child_i.fitness = parent_i.fitness

    else:
        body_genotype_i, body_genotype_j = Crossover.uniform(
            cast("list[list[float]]", parent_i.genotype[0]),
            cast("list[list[float]]", parent_j.genotype[0]),
        )
        child_i.genotype = (body_genotype_i, None)
        child_j.genotype = (body_genotype_j, None)
        
        for child in [child_i, child_j]:
            child.requires_init = True
            child.requires_eval = True

    for child in [child_i, child_j]:
        if(np.random.random() < 0.5):
            child.tags["mut"] = True

    ind1.tags["ps"] = False
    ind2.tags["ps"] = False

    return child_i, child_j


def crossover(population: Population) -> Population:
    """Crossover individuals tagged for parent selection"""
    # Shuffle population to avoid bias

    console.log("Starting crossover...")
    start_time = time.time()
    parents = [ind for ind in population if ind.tags.get("ps", False) == True]

    task = progress.add_task("[green]Crossover...", total=len(parents) // 2)
    progress.start_task(task)

    np.random.shuffle(parents)
    for idx in range(0, len(parents) - 1, 2):
        parent_i = parents[idx]
        parent_j = parents[idx + 1]
        child_i, child_j = crossover_individuals(parent_i, parent_j)
        population.extend([child_i, child_j])
        progress.update(task, advance=1)
        parent_i.tags["ps"] = False
        parent_j.tags["ps"] = False
    task_time = time.time() - start_time
    progress.remove_task(task)
    console.log(f"Crossover completed in {task_time:.2f} seconds.")

    return population


def mutate_individual(
    individual: Individual,
    mutation_probability: float = 0.5,
    mutation_stddev: float = 0.1,
) -> Individual:
    """Mutate an individual's body genotype with given probability and stddev"""
    body_genotype = cast("list[list[float]]", individual.genotype[0])
    mutated_body_genotype = []
    for gene_array in body_genotype:
        gene_array_np = np.array(gene_array, dtype=np.float32)  # Ensure numpy array
        mutation_mask = RNG.random(gene_array_np.shape) > mutation_probability
        mutations = RNG.normal(0, mutation_stddev, gene_array_np.shape)
        new_gene_array = gene_array_np + mutation_mask * mutations
        mutated_body_genotype.append(
            new_gene_array.astype(np.float32).tolist()
        )  # Convert back to list
    individual.genotype = (mutated_body_genotype, individual.genotype[1])
    retrain_individual(individual)
    individual.tags["mut"] = False
    return individual


def mutate(
    population: Population,
    mutation_probability: float = 0.5,
    mutation_stddev: float = 0.1,
) -> Population:
    """Mutate individuals tagged for mutation"""
    console.log("Starting mutation...")
    start_time = time.time()
    mutable_inds = [ind for ind in population if ind.tags.get("mut", False) == True]
    task = progress.add_task("[green]Mutating individuals...", total=len(mutable_inds))
    progress.start_task(task)
    for individual in mutable_inds:
        mutate_individual(individual, mutation_probability, mutation_stddev)
        progress.update(task, advance=1)

    progress.stop_task(task)
    task_time = time.time() - start_time
    progress.remove_task(task)
    console.log(f"Mutation completed in {task_time:.2f} seconds.")

    return population

def retrain_individual(individual: Individual) -> Individual:
    """retrain the brain of a single individual with current weights as initial weights"""
    console.log("Starting brain retraining...")
    
    if individual.requires_eval:
        individual = train_and_evaluate_individual_brain(individual)
        individual.requires_eval = False
        individual.requires_init = False
        return individual
    
    p_matrices = NDE.forward(np.array(individual.genotype[0]))
    training_result = braintrain.evolve_using_cma_es(
        gecko_body=
            HPD.probability_matrices_to_graph(
                p_matrices[0],
                p_matrices[1],
                p_matrices[2],
            
        ),
        duration=constants.STAGE_SETTINGS[current_stage]["DURATION"],
        sectioned=constants.STAGE_SETTINGS[current_stage]["SECTIONED_MODE"],
        stagnation_threshold=constants.STAGE_SETTINGS[current_stage]["MAX_STAGNATION_DELTA"],
        max_stagnation=constants.STAGE_SETTINGS[current_stage]["MAX_STAGNATION"],
        initial_weights=np.array(individual.genotype[1], dtype=np.float32)
    )
    individual.genotype = (individual.genotype[0], training_result[0])
    individual.fitness = training_result[1]
    individual.requires_eval = False
    individual.requires_init = False

    console.log("Brain retraining completed.")
    return individual


def body_evolution() -> tuple[float, list[list[float]], np.ndarray, DiGraph]:  # type: ignore
    """full evolution of body and brain genotypes
    returns fitness, body_genotype, brain_genotype, body_phenotype"""
    console.rule(f"Body evolution started.")
    evolution_task = progress.add_task("[green]Evolving bodies...", total=(
        constants.BODY_MAX_GENERATIONS
        if constants.BODY_MAX_GENERATIONS
        else constants.BODY_TIME_LIMIT if constants.BODY_TIME_LIMIT else None
    ), start=False)
    
    global current_stage
    
    start_time = time.time()


    # Create initial population
    console.rule("Creating initial population")
    population = _create_population(constants.BODY_POP_SIZE)

    #DEBUG: Record full run and plot
    random_ind_int = RNG.integers(0, len(population))
    random_ind = population[int(random_ind_int)]
    debug_p_matrices = NDE.forward(np.array(random_ind.genotype[0]))
    debug_gecko_body = HPD.probability_matrices_to_graph(*debug_p_matrices)
    
    console.log(f"Recording a random individual from initial population for debugging purposes...")
    console.log(f"Parameters given: duration={constants.STAGE_SETTINGS['FULL']['DURATION']}, spawn_pos={constants.POSITIONS[0][0]}")
    console.log(debug_gecko_body)
    
    console.log(f"Initial population created with {len(population)} individuals.")
    ops = [
        EAStep("reset tags", reset_tags),
        EAStep("evaluation", evaluate_population),
        # EAStep("show_best", show_best_of_population),
        EAStep("parent_selection", parent_selection),
        EAStep("crossover", crossover),
        EAStep("mutation", mutate),
        EAStep("evaluation", evaluate_population),
        EAStep("survivor_selection", survivor_selection),
    ]
    ea = EA(
        population=population,
        operations=ops,
        quiet=False,
    )

    def terminate() -> bool:
        if (
            constants.BODY_MAX_GENERATIONS
            and ea.current_generation >= constants.BODY_MAX_GENERATIONS
        ):
            console.log("Reached maximum generations.")
            return True
        if time.time() - start_time >= constants.BODY_TIME_LIMIT:
            console.log("Reached time limit.")
            return True
        return False

    # Prepare CSV for logging fitness
    import csv

    fitness_log_path = (
        constants.OUTPUT
        / "logs"
        / f"fitness_log-{time.strftime('%Y%m%d-%H%M%S')}.csv"
    )
    # Write header if file does not exist
    if not fitness_log_path.exists():
        with open(fitness_log_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["generation", "fitness"])

    
    ea.fetch_population()
    for ind in [ind for ind in ea.population if getattr(ind, "alive", True)]:
        with open(fitness_log_path, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([ea.current_generation, ind.fitness])
    ea.commit_population()

    progress.start_task(evolution_task)

    while not terminate():
        console.log(
            f"Running evolution step for generation {ea.current_generation}..."
        )
        ea.step()
        best_ind = ea.get_solution("best", only_alive=False)
        best_fitness = best_ind.fitness
        # Compute average fitness (only for alive individuals)
        ea.fetch_population()
        # Log to CSV
        for ind in [ind for ind in ea.population if getattr(ind, "alive", True)]:
            with open(fitness_log_path, mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([ea.current_generation, ind.fitness])
        ea.commit_population()

        console.log(
            f"Generation {ea.current_generation}: Best Fitness = {best_fitness:.4f}, Population Size = {ea.population_size}"
        )
        runtime = time.time() - start_time
        progress.update(
            evolution_task,
            completed=(
                ea.current_generation if constants.BODY_MAX_GENERATIONS else runtime
            ),
            description=f"[green]Evolving bodies... Generation {ea.current_generation}, Best Fitness: {best_fitness:.4f}, runtime: {runtime // 3600}h {(runtime % 3600) // 60}m {(runtime % 60):.0f}s",
        )
        
        console.log(f"Saving best individual of generation {ea.current_generation} with fitness {best_fitness:.4f}...")
        p_matrices = NDE.forward(np.array(best_ind.genotype[0]))
        gecko_body = HPD.probability_matrices_to_graph(
            p_matrices[0], p_matrices[1], p_matrices[2]
        )
        # Save JSON of best body
        utils.save_body_to_json(
            gecko_body,
            filename=f"best_body_gen{ea.current_generation}_fit{best_fitness:.4f}",
        )
        # Save weights of best brain
        utils.save_brain_genotype(
            np.array(best_ind.genotype[1]),
            filename=f"best_brain_weights_gen{ea.current_generation}_fit{best_fitness:.4f}.npy",
        )
        tracker = runner.run_bot_session(
            method="headless",
            weights=np.array(best_ind.genotype[1]),
            gecko_body=copy.deepcopy(gecko_body),
            duration=constants.STAGE_SETTINGS["FULL"]["DURATION"],
            spawn_pos=constants.POSITIONS[0][0],
        )
        utils.save_xpos_history(tracker, fitness=best_fitness)
            
            
        
        if ea.current_generation % constants.BODY_BATCH_SIZE == 0 or ea.current_generation == 1:
            console.log(
                f"Running best individual of generation {ea.current_generation} with fitness {best_fitness:.4f} for recording..."
            )
            # Record full run and plot
            tracker = runner.run_bot_session(
                method="record",
                weights=np.array(best_ind.genotype[1]),
                gecko_body=copy.deepcopy(gecko_body),
                options={
                    "filename": f"best_body_individual_gen{ea.current_generation}",
                    "fitness": best_fitness,
                },
                duration=constants.STAGE_SETTINGS["FULL"]["DURATION"],
                spawn_pos=constants.POSITIONS[0][0],
            )

        # Stage transitions based on fitness thresholds
        if current_stage == 1:
            # Stage 1: Sectioned training (fitness in [0, 1])
            # 0 = no progress, 1 = all sections complete
            # When sectioned fitness >= 0.35, bots have proven basic locomotion
            if (
                best_fitness >= 0.35
                or (time.time() - start_time) > 0.2 * constants.BODY_TIME_LIMIT
                or (constants.BODY_MAX_GENERATIONS is not None and ea.current_generation >= 0.2 * constants.BODY_MAX_GENERATIONS) # pyright: ignore[reportUnnecessaryComparison]
            ):
                console.rule(
                    f"Reached sectioned fitness threshold 1 with fitness {best_fitness:.4f}. Switching to stage 2."
                )
                current_stage = 2
                ea.fetch_population()
                ea.population = reset_fitness([ind for ind in ea.population if getattr(ind, "alive", True)])
                ea.commit_population()


        elif current_stage == 2:
            # Stage 2: Sectioned training (fitness >= 0.25). Higher duration
            # 0 = no progress, 1 = all sections complete
            # When sectioned fitness >= 0.75, sections are performing very well
            if (
                best_fitness >= 0.75
                or (time.time() - start_time) > 0.4 * constants.BODY_TIME_LIMIT
                or (constants.BODY_MAX_GENERATIONS is not None and ea.current_generation >= 0.6 * constants.BODY_MAX_GENERATIONS) # pyright: ignore[reportUnnecessaryComparison]
            ):
                console.rule(
                    f"Reached sectioned fitness threshold 2 with fitness {best_fitness:.4f}. Switching to stage FULL - LENGTH."
                )
                current_stage = "FULL"
                ea.fetch_population()
                ea.population = reset_fitness([ind for ind in ea.population if getattr(ind, "alive", True)])
                ea.commit_population()


        elif current_stage == "FULL":
            # Stage Full training (fitness >= 1). High duration
            # 2 = reached goal, up to 3 for time bonus
            if (
                best_fitness >= 2
                or (time.time() - start_time) > 0.8 * constants.BODY_TIME_LIMIT
                or (constants.BODY_MAX_GENERATIONS is not None and ea.current_generation >= 0.8 * constants.BODY_MAX_GENERATIONS) # pyright: ignore[reportUnnecessaryComparison]
            ):
                console.rule(
                    f"Reached full fitness threshold 1 with fitness {best_fitness:.4f}. Switching to stage 3."
                )
                current_stage = 3
                ea.fetch_population()
                ea.population = reset_fitness([ind for ind in ea.population if getattr(ind, "alive", True)])
                ea.commit_population()

        elif current_stage == 3:
            # Stage 3: Full training (fitness >= 2). Lowered duration for faster iterations
            # 2 = reached goal, up to 3 for time bonus
            # at 2.5 fitness, the bots reach the end in 27 seconds
            if (
                best_fitness >= 2.9
            ):
                console.rule(
                    f"Reached full fitness threshold 2 with fitness {best_fitness:.4f}. Ending evolution."
                )
                break
        progress.update(evolution_task, completed=ea.current_generation if constants.BODY_MAX_GENERATIONS else (time.time() - start_time) if constants.BODY_TIME_LIMIT is not None else None) # pyright: ignore[reportUnnecessaryComparison]

    progress.remove_task(evolution_task)

    console.rule("Evolution process finished.")
    console.log(f"Best Fitness: {ea.get_solution("best", only_alive=False).fitness}")
    console.log("Saving best individual...")
    p_matrices = NDE.forward(
        np.array(ea.get_solution("best", only_alive=False).genotype[0])
    )
    hpd = HPD
    
    # record and save best individual
    console.log("Recording best individual of the entire evolution...")
    utils.save_body_to_json(
        hpd.probability_matrices_to_graph(
            p_matrices[0], p_matrices[1], p_matrices[2]
        ),
        filename=f"best_body_final_fit{ea.get_solution('best', only_alive=False).fitness:.4f}",
    )
    utils.save_brain_genotype(
        np.array(ea.get_solution("best", only_alive=False).genotype[1]),
        filename=f"best_brain_weights_final_fit{ea.get_solution('best', only_alive=False).fitness:.4f}.npy",
    )
    tracker = runner.run_bot_session(
        method="record",
        spawn_pos=constants.POSITIONS[0][0],
        weights=np.array(ea.get_solution("best", only_alive=False).genotype[1]),
        gecko_body=
            hpd.probability_matrices_to_graph(
                p_matrices[0], p_matrices[1], p_matrices[2]
            
        ),
        options={
            "filename": f"best_body_individual_final",
            "fitness": ea.get_solution("best", only_alive=False).fitness,
        },
        duration=constants.STAGE_SETTINGS["FULL"]["DURATION"],
    )
    utils.save_xpos_history(
        tracker,
        fitness=ea.get_solution("best", only_alive=False).fitness,
    ) 

    console.log("Evolution process completed.")

    final_ind = ea.get_solution("best", only_alive=False)
    final_fit: float = final_ind.fitness
    final_body_genotype: list[list[float]] = cast(
        "list[list[float]]", final_ind.genotype[0]
    )
    final_brain_genotype: np.ndarray = np.array(final_ind.genotype[1])
    p_matrices = NDE.forward(np.array(final_body_genotype))
    hpd = HPD
    final_graph = hpd.probability_matrices_to_graph(
        p_matrices[0], p_matrices[1], p_matrices[2]
    )
    return final_fit, final_body_genotype, final_brain_genotype, final_graph

if __name__ == "__main__":
    progress.start()
    try:
        pretty.install()
        body_evolution()
    finally:
        progress.stop()
