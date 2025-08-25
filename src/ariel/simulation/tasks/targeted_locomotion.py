def distance_to_target_ff(initial_position, target_position) -> float:
    """
    Calculate the Euclidean distance between the current position and the target position.

    Args:
        initial_position (tuple): The current position as (x, y).
        target_position (tuple): The target position as (x, y).

    Returns:
        float: The distance between the two positions.
    """
    return ((initial_position[0] - target_position[0]) ** 2 + 
            (initial_position[1] - target_position[1]) ** 2) ** 0.5