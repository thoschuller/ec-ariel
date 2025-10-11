import numpy as np
import experiment.constants as constants
from experiment.terminal import console, progress
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
import session_runner as runner
from evaluator import minimized_fitness_evaluation
from ariel.utils.tracker import Tracker
import multiprocessing
from functools import partial
from cma import CMAEvolutionStrategy  # type: ignore[reportMissingTypeStubs]
import time


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
    gecko_body: CoreModule, duration: float, sectioned: bool
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

    model, _, _, _ = runner.initialize_world_and_robot(
        gecko_body=gecko_body, spawn_pos=constants.POSITIONS[0][0]
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
    sigma = 0.1  # Initial step size
    options = {
        "popsize": constants.BRAIN_POP_SIZE,
        "maxiter": constants.BRAIN_MAX_GENERATIONS,
        "verb_log": 0,  # Reduce logging
        "verb_disp": 1 if constants.DETAILED_LOGGING else 0,
    }
    es = CMAEvolutionStrategy(initial_solution, sigma, options)

    evolution_start_time = time.time()

    time_spent_evaluating = 0.0
    try:

        brain_evo_task = progress.add_task(
            f"CMA-ES Evolution Progress. Current best: ...",
            total=constants.BRAIN_MAX_GENERATIONS,
        )

        while not es.stop():
            if (
                constants.BRAIN_TIME_LIMIT > 0
                and (time.time() - evolution_start_time) > constants.BRAIN_TIME_LIMIT
            ):
                console.log("Time limit reached, terminating CMA-ES.")
                break

            solutions = es.ask()
            if constants.PARALLEL and pool:
                fitnesses = pool.map(eval_func, solutions)
            else:
                fitnesses = [eval_func(x) for x in solutions]

            es.tell(solutions, fitnesses)

            es.disp(constants.BRAIN_BATCH_SIZE)

            progress.update(
                brain_evo_task,
                advance=1,
                description=f"CMA-ES Evolution Progress. Current best: {-es.result.fbest}",
            )

    finally:
        if pool:
            pool.close()
            pool.join()

    result = es.result
    best_weights_list = result.xbest
    best_weights = np.array(best_weights_list, dtype=np.float32)
    best_fitness = -result.fbest

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

    # Return as Individual for compatibility
    return best_weights.tolist(), best_fitness, tracker
