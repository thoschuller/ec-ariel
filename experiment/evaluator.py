# from typing import Any, cast
import constants as constants

# Environment/world class used in some fitness modes
# from ariel.simulation.environments import OlympicArena
from ariel.utils.tracker import Tracker
import numpy as np
import mujoco
from terminal import console
import session_runner as runner
from networkx import DiGraph

def fitness(
    tracker: Tracker,
    spawn: list[float],
    goal: list[float],
    bonus: bool = False,
) -> float:
    """Evaluate fitness base on history, spawn and goal positions. Penalizes lateral deviation."""

    history = tracker.history
    if not history:
        raise ValueError("No history data available from tracker.")
    
    # Check if required key exists
    if "xpos" not in history:
        raise ValueError(f"Missing 'xpos' key in history. Available keys: {list(history.keys())}")
    
    # Get the position data for the first tracked object (index 0)
    if 0 not in history["xpos"]:
        raise ValueError(f"No tracked object at index 0. Available indices: {list(history['xpos'].keys())}")
    
    xpos_data = history["xpos"][0]
    if len(xpos_data) < 2:
        raise ValueError(f"Insufficient position history data: only {len(xpos_data)} entries.")

    if spawn[1] != goal[1]:
        raise NotImplementedError("Goals with lateral displacement not supported yet.")

    # change start to a few seconds later to account for weird spawns
    start = xpos_data[3]

    # Calculate progress toward goal
    goal_distance = goal[0] - start[0]
    straight_distance = xpos_data[-1][0] - start[0]
    correct_direction: bool = goal_distance * straight_distance > 0
    if not correct_direction:
        return 0.0

    lateral_deviation = abs(xpos_data[-1][1] - start[1])
    countable_distance = (
        abs(straight_distance) - lateral_deviation * constants.LATERAL_PENALTY_FACTOR
    )

    basic_fitness = max(0.0, countable_distance / abs(goal_distance))

    if basic_fitness < 1 or not bonus:
        return basic_fitness

    # time can be determined by the position of the first entry in history beyond goal
    finish_index = int(np.argmax(abs(np.array(xpos_data)[:, 0]) >= abs(goal[0])))

    finish_point = finish_index / len(xpos_data)

    return min(1.0, basic_fitness + (1 - finish_point))


def evaluate_individual(
    genotype_list: list[float],
    gecko_body: DiGraph, # pyright: ignore
    duration: float,
    sectioned: bool = False,
) -> float:
    mujoco.set_mjcb_control(None)
    weights = np.array(genotype_list, dtype=np.float32)
    try:
        # Use sectioned fitness if enabled
        if sectioned:
            fitnesses = []

            for spawn, goal in constants.POSITIONS:
                tracker = runner.run_bot_session(
                    weights,
                    method="headless",
                    gecko_body=gecko_body,
                    spawn_pos=spawn,
                    duration=duration,
                )
                try:
                    fit = fitness(
                        tracker,
                        spawn=spawn,
                        goal=goal,
                    )
                    fitnesses.append(fit)
                except Exception as e:
                    console.log(f"Fitness calculation failed for section {spawn} to {goal}: {e}")
                    fitnesses.append(-10000.0)
                    break
            return np.mean(fitnesses) if fitnesses else -1000.0 #type: ignore

        tracker = runner.run_bot_session(
            weights,
            method="headless",
            gecko_body=gecko_body,
            duration=duration,
            spawn_pos=constants.POSITIONS[0][0],
        )
        fit = fitness(
            tracker=tracker,
            spawn=constants.POSITIONS[0][0],
            goal=constants.POSITIONS[2][1],
            bonus=True,
        )

        return fit+1
    except Exception as e:
        console.log(f"Evaluation failed for individual: {type(e).__name__}: {str(e)}")
        import traceback
        console.log(f"Traceback: {traceback.format_exc()}")
        return -1000.0
    finally:
        mujoco.set_mjcb_control(None)


def minimized_fitness_evaluation(
    genotype_list: list[float],
    gecko_body: DiGraph, # pyright: ignore
    duration: float,
    sectioned: bool = False,
) -> float:
    """Convert fitness to a minimization objective."""
    return -evaluate_individual(genotype_list, gecko_body, duration, sectioned)
