import numpy as np
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
import constants as constants
from terminal import console, progress
import session_runner as runner
from evaluator import minimized_fitness_evaluation
from ariel.utils.tracker import Tracker
from functools import partial
import time
from networkx import DiGraph
from utils import (
    save_brain_genotype,
    save_xpos_history,
    load_json_as_digraph,
)

def random_eval(args):
    weight_shapes, body_graph, duration, sectioned = args
    random_weights = sample_glorot_flat(weight_shapes)
    eval_func = partial(
        minimized_fitness_evaluation,
        gecko_body=body_graph,
        duration=duration,
        sectioned=sectioned,
    )
    fitness = -eval_func(random_weights)
    return random_weights, fitness

def train_individual_from_files(  # pyright: ignore[reportUnknownParameterType]
    body_file: str | None = None, body_graph: DiGraph | None = None, weights_file: str | None = None, np_weights: np.ndarray | None = None, time_limit: float = 60 * 60 * 1, record_batch: bool = constants.BRAIN_EVO_RECORD_BATCH, record_last: bool = constants.BRAIN_EVO_RECORD_LAST  # type: ignore
) -> tuple[list[float], float, Tracker | None]:
    """
    Load a genotype and body structure from files for training or evaluation.
    """
    
        # Prepare CSV log file
    import csv
    log_dir = constants.OUTPUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"random_brain_baseline_log-{time.strftime('%Y%m%d-%H%M%S')}.csv"
    with open(log_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["iteration", "fitness"])
    
    if body_graph is None and body_file is not None:
        body_graph = load_json_as_digraph(body_file)
    elif body_graph is None:
        raise ValueError("Either body_graph or body_file must be provided.")

    # Determine number of random samples (iterations) to match real training
    num_iterations = constants.BRAIN_MAX_GENERATIONS if hasattr(constants, 'BRAIN_MAX_GENERATIONS') else 100

    # Prepare model and weight shapes
    model, _, _ = runner.quick_spawn(
        gecko_body=construct_mjspec_from_graph(body_graph),
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
    
    
    progress_task = progress.add_task("Random baseline iterations", total=num_iterations)


    duration = constants.STAGE_SETTINGS["FULL"]["DURATION"]
    sectioned = False
    args_list = [(weight_shapes, body_graph, duration, sectioned) for _ in range(num_iterations)]
    results = []
    if constants.PARALLEL is True and constants.PARALLEL_CORES > 1:
        import multiprocessing
        with multiprocessing.Pool(constants.PARALLEL_CORES if hasattr(constants, 'PARALLEL_CORES') else None) as pool:
            for res in pool.imap_unordered(random_eval, args_list):
                results.append(res)
                progress.update(progress_task, advance=1)
    else:
        for args in args_list:
            results.append(random_eval(args))
            progress.update(progress_task, advance=1)
    progress.remove_task(progress_task)

    # Log all iterations to CSV
    with open(log_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for i, (_, fitness) in enumerate(results):
            writer.writerow([i, fitness])

    # Find the best result
    best_weights, best_fitness = max(results, key=lambda x: x[1])
    console.log(f"Random baseline: best fitness out of {num_iterations} samples: {best_fitness:.5f}")

    tracker = None
    # Optionally record the best run
    if record_last:
        tracker = runner.run_bot_session(
            best_weights,
            method="record",
            gecko_body=body_graph,
            duration=constants.STAGE_SETTINGS["FULL"]["DURATION"],
            spawn_pos=constants.POSITIONS[0][0],
            options={
                "filename": "random_baseline_final",
                "fitness": best_fitness,
            },
        )
        save_brain_genotype(
            best_weights.tolist(), fitness=best_fitness, filename="random_baseline_brain"
        )
        save_xpos_history(tracker, fitness=best_fitness)

    # Optionally run headless for tracker
    if tracker is None:
        tracker = runner.run_bot_session(
            best_weights,
            method="headless",
            gecko_body=body_graph,
            duration=constants.STAGE_SETTINGS["FULL"]["DURATION"],
            spawn_pos=constants.POSITIONS[0][0],
        )

    return best_weights.tolist(), best_fitness, tracker


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
    Fake brain training that generates random baseline weights without evolution.
    Returns (random_weights, baseline_fitness, tracker).
    """

    console.rule("[green]Starting Random Baseline Generation (No Training)")

    start_time = time.time()
    



    # Generate random weights using the same architecture as the real training
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

        console.log(f"Generating random weights with {total_params} parameters")

        # Generate random weights using Glorot initialization
        random_weights = sample_glorot_flat(weight_shapes)
    else:
        random_weights = initial_weights
        total_params = len(initial_weights)
        console.log(f"Using provided weights with {total_params} parameters")

    # Evaluate the random weights once to get baseline fitness
    console.log("Evaluating random baseline weights...")
    
    eval_func = partial(
        minimized_fitness_evaluation,
        gecko_body=gecko_body,
        duration=duration,
        sectioned=sectioned,
    )
    
    baseline_fitness = -eval_func(random_weights)  # Convert back from minimized fitness
    
    console.log(f"Baseline fitness with random weights: {baseline_fitness:.5f}")

    # Optional recording and saving
    if record_last:
        console.log("Recording baseline run")
        tracker = runner.run_bot_session(
            random_weights,
            method="record",
            gecko_body=gecko_body,
            duration=duration,
            spawn_pos=constants.POSITIONS[0][0],
            options={
                "filename": "random_baseline_final",
                "fitness": baseline_fitness,
            },
        )

        save_brain_genotype(
            random_weights.tolist(), fitness=baseline_fitness, filename="random_baseline_brain"
        )
        save_xpos_history(tracker, fitness=baseline_fitness)

    tracker = None

    if sectioned == False:
        tracker = runner.run_bot_session(
            random_weights,
            method="headless",
            gecko_body=gecko_body,
            duration=constants.STAGE_SETTINGS["FULL"]["DURATION"],
            spawn_pos=constants.POSITIONS[0][0],
        )

    elapsed_time = time.time() - start_time
    console.rule(
        f"Random baseline generation complete in {elapsed_time:.2f} seconds. Baseline fitness: {baseline_fitness:.5f}"
    )
    console.log(f"Baseline fitness: {baseline_fitness:.5f}")

    # Return in the same format as the original function
    return random_weights.tolist(), baseline_fitness, tracker
