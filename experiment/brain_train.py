import numpy as np
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
import constants as constants
from terminal import console, progress
import session_runner as runner
from evaluator import minimized_fitness_evaluation
from ariel.utils.tracker import Tracker
import multiprocessing
from functools import partial
from cma import CMAEvolutionStrategy  # type: ignore[reportMissingTypeStubs]
import time
from networkx import DiGraph
from utils import (
    save_brain_genotype,
    save_xpos_history,
    load_brain_genotype,
    load_json_as_digraph,
)


def train_individual_from_files(  # pyright: ignore[reportUnknownParameterType]
    body_file: str | None = None, body_graph: DiGraph | None = None, weights_file: str | None = None, np_weights: np.ndarray | None = None, time_limit: float = 60 * 60 * 1, record_batch: bool = constants.BRAIN_EVO_RECORD_BATCH, record_last: bool = constants.BRAIN_EVO_RECORD_LAST  # type: ignore
) -> tuple[list[float], float, Tracker | None]:
    """
    Load a genotype and body structure from files for training or evaluation.
    """
    weights = None
    if np_weights is not None:
        weights = np_weights
    elif weights_file is not None:
        weights = load_brain_genotype(weights_file)
    if body_graph is None and body_file is not None:
        body_graph = load_json_as_digraph(body_file)
    elif body_graph is None:
        raise ValueError("Either body_graph or body_file must be provided.")
    return evolve_using_cma_es(
        gecko_body=body_graph,
        duration=constants.STAGE_SETTINGS["FULL"]["DURATION"],
        sectioned=False,
        stagnation_threshold=0.0,
        max_stagnation=np.inf,
        record_batch=100 if record_batch else None,
        record_last=True,
        save_all=False,
        initial_weights=weights,
        time_limit=time_limit,
        plot_all_batches=True,
    )


def sample_glorot_flat(weight_shapes: list[tuple[int, int]]) -> np.ndarray:
    """
    Sample weights using Glorot/Xavier initialization for tanh networks.
    For each weight matrix with shape (fan_in, fan_out):
    range = ±sqrt(6 / (fan_in + fan_out))
    """
    parts = []
    for fan_in, fan_out in weight_shapes:
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        W = np.random.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)
        parts.append(W.reshape(-1))
    return np.concatenate(parts, dtype=np.float32)


def evolve_using_cma_es(
    gecko_body: DiGraph,  # type: ignore
    duration: float,
    sectioned: bool,
    stagnation_threshold: float,
    max_stagnation: int,
    record_batch: int = None,
    record_last: bool = False,
    initial_weights: np.ndarray | None = None,
    save_all: bool = False,
    time_limit: float = constants.BRAIN_TIME_LIMIT,  # type:ignore
    plot_all_batches: bool = False,
) -> tuple[list[float], float, Tracker | None]:
    """
    Main evolutionary loop using CMA-ES. Returns (best_individual.genotype, best_fitness, best_tracker).
    """

    eval_func = partial(
        minimized_fitness_evaluation,
        gecko_body=gecko_body,
        duration=duration,
        sectioned=sectioned,
    )
    pool = None
    if constants.PARALLEL and constants.PARALLEL_CORES > 1:
        pool = multiprocessing.Pool(constants.PARALLEL_CORES)
        console.log(f"Using multiprocessing pool with {constants.PARALLEL_CORES} cores")
        eval_func = partial(
            minimized_fitness_evaluation,
            gecko_body=gecko_body,
            duration=duration,
            sectioned=sectioned,
        )
    else:
        console.log("Running in single-threaded mode")

    console.rule("[green]Starting CMA-ES Run")

    if initial_weights is None:
        model, _, _ = runner.quick_spawn(
            gecko_body=construct_mjspec_from_graph(gecko_body),
            spawn_pos=constants.POSITIONS[0][0],
        )

        input_size = model.nq
        output_size = model.nu
        hidden_size = constants.HIDDEN_SIZE
        num_hidden_layers = constants.NUM_HIDDEN_LAYERS
        layer_sizes = [input_size] + [hidden_size] * num_hidden_layers + [output_size]
        weight_shapes = [
            (layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
        ]
        total_params = sum(a * b for a, b in weight_shapes)

        console.log(
            f"Population Size: {constants.BRAIN_POP_SIZE}, Total Params: {total_params}"
        )

        # Initialize CMA-ES
        initial_solution = sample_glorot_flat(weight_shapes)
    else:
        initial_solution = initial_weights
        total_params = len(initial_solution)
        console.log(
            f"Population Size: {constants.BRAIN_POP_SIZE}, Total Params from file: {total_params}"
        )

    sigma = 0.2  # Initial step size
    options = {
        "popsize": constants.BRAIN_POP_SIZE,
        "maxiter": constants.BRAIN_MAX_GENERATIONS,
        "verb_log": 0,  # Reduce logging
        "verb_disp": 1 if constants.DETAILED_LOGGING else 0,
    }
    es = CMAEvolutionStrategy(initial_solution, sigma, options)

    evolution_start_time = time.time()

    time_spent_evaluating = 0.0

    brain_evo_task = progress.add_task(
        f"CMA-ES Evolution Progress. Current best: ...",
        total=(
            constants.BRAIN_MAX_GENERATIONS
            if constants.BRAIN_MAX_GENERATIONS
            else time_limit
        ),
    )

    try:
        # Stagnation detection variables
        stagnation_generations = 0
        best_fitness_so_far = None

        while not es.stop():
            if time_limit > 0 and (time.time() - evolution_start_time) > time_limit:
                console.log("Time limit reached, terminating CMA-ES.")
                break
            
            if (constants.CWD / "STOP_BODY").is_file():
                console.log("STOP_BRAIN file detected. Terminating evolution.")
                break

            solutions = es.ask()

            start_eval = time.time()
            if constants.PARALLEL and pool:
                fitnesses = pool.map(eval_func, solutions)
            else:
                fitnesses = [eval_func(x) for x in solutions]
            time_spent_evaluating += time.time() - start_eval

            es.tell(solutions, fitnesses)

            es.disp(constants.BRAIN_BATCH_SIZE)

            current_best_fitness = -es.result.fbest
            if (
                best_fitness_so_far is None
                or (current_best_fitness - best_fitness_so_far) > stagnation_threshold
            ):
                best_fitness_so_far = current_best_fitness
                stagnation_generations = 0
            else:
                stagnation_generations += 1
                if stagnation_generations >= max_stagnation:
                    console.log(
                        f"Stagnation detected: no improvement in {max_stagnation} generations. Stopping early."
                    )
                    break

            if record_batch and es.countiter % record_batch == 0:
                console.log(f"Recording batch at iteration {es.countiter}")
                tracker = runner.run_bot_session(
                    np.array(es.result.xbest, dtype=np.float32),
                    method="record",
                    gecko_body=gecko_body,
                    duration=duration,
                    spawn_pos=constants.POSITIONS[0][0],
                    options={
                        "filename": f"brain_evo_gen{es.countiter}",
                        "fitness": current_best_fitness,
                    },
                )

            if es.countiter % constants.BRAIN_BATCH_SIZE == 0 and plot_all_batches:
                save_brain_genotype(
                    np.array(es.result.xbest, dtype=np.float32),
                    fitness=current_best_fitness,
                    filename=f"brain_evo_gen{es.countiter}_best_brain",
                )
                tracker = runner.run_bot_session(
                    np.array(es.result.xbest, dtype=np.float32),
                    method="headless",
                    gecko_body=gecko_body,
                    duration=duration,
                    spawn_pos=constants.POSITIONS[0][0],
                    options={
                        "filename": f"brain_evo_gen{es.countiter}",
                        "fitness": current_best_fitness,
                    },
                )
                save_xpos_history(tracker, fitness=current_best_fitness)

            elif save_all:
                save_brain_genotype(
                    np.array(es.result.xbest, dtype=np.float32),
                    fitness=current_best_fitness,
                    filename=f"brain_evo_gen{es.countiter}_best_brain",
                )

            progress.update(
                brain_evo_task,
                completed=(
                    es.countiter
                    if constants.BRAIN_MAX_GENERATIONS
                    else (time.time() - evolution_start_time)
                ),
                description=f"CMA-ES Evolution Progress. Current best: {current_best_fitness}",
            )

    finally:
        if pool:
            pool.close()
            pool.join()

    result = es.result
    best_weights_list = result.xbest
    best_weights = np.array(best_weights_list, dtype=np.float32)
    best_fitness = -result.fbest

    if record_last:
        console.log(f"Recording final run at iteration {es.countiter}")
        tracker = runner.run_bot_session(
            best_weights,
            method="record",
            gecko_body=gecko_body,
            duration=duration,
            spawn_pos=constants.POSITIONS[0][0],
            options={
                "filename": f"brain_evo_final_gen{es.countiter}",
                "fitness": best_fitness,
            },
        )

        save_brain_genotype(
            best_weights_list, fitness=best_fitness, filename="brain_evo_best_brain"
        )
        save_xpos_history(tracker, fitness=best_fitness)

    tracker = None

    if sectioned == False:
        tracker = runner.run_bot_session(
            best_weights,
            method="headless",
            gecko_body=gecko_body,
            duration=constants.STAGE_SETTINGS["FULL"]["DURATION"],
            spawn_pos=constants.POSITIONS[0][0],
        )

    console.rule(
        f"CMA-ES complete in {(time.time() - evolution_start_time)/60:.2f} minutes. Best fitness: {best_fitness:.5f}"
    )
    console.log(
        f"Total time spent evaluating: {time_spent_evaluating:.2f} seconds, {time_spent_evaluating/(time.time() - evolution_start_time)*100:.2f}% of total time."
    )
    console.log(f"Best fitness: {best_fitness:.5f}")

    progress.remove_task(brain_evo_task)

    # Return as Individual for compatibility
    return best_weights.tolist(), best_fitness, tracker
