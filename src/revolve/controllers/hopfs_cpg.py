"""Hopf-based Central Pattern Generator (CPG).

Date:       2025-05-03
Status:     In progress ⚙️

Sources
-------
    [1] https://www.sciencedirect.com/science/article/pii/S2667379722000353#sec4

Notes
-----
    *

Todo
----
    [ ]

"""

# Stand
from typing import Any

import numpy as np
from rich.progress import track

SEED = 42
RNG = np.random.default_rng(seed=SEED)


class HopfCPG:
    def __init__(self, num_neurons: int, dt: float = 0.02) -> None:
        self.num_neurons = num_neurons
        self.dt = dt

        # Initialize state variables
        self.x = np.zeros(num_neurons)
        self.y = np.zeros(num_neurons)
        self.x_dot = np.zeros(num_neurons)
        self.y_dot = np.zeros(num_neurons)

        # Initialize parameters with default values
        # Learning rate
        self.alpha = np.ones(num_neurons) * 1.0

        # Angular frequency (1Hz default)
        self.omega = np.ones(num_neurons) * 2 * np.pi * 0.001

        # Amplitude
        self.A = np.ones(num_neurons) * 1.0

        # Coupling coefficient
        self.h = 0.1

        # Phase differences
        self.phase_diff = np.zeros((
            num_neurons,
            num_neurons,
        ))

        # Set up ring topology phase differences (neighbors are π/2 apart)
        for i in range(num_neurons):
            for j in [i - 1, i + 1]:
                if 0 <= j < num_neurons:
                    self.phase_diff[i, j] = RNG.uniform(-np.pi / 2, np.pi / 2)

    def step(
        self,
    ) -> tuple[
        np.ndarray[tuple[int, ...], np.dtype[Any]],
        np.ndarray[tuple[int, ...], np.dtype[Any]],
    ]:
        new_x = np.zeros_like(self.x)
        new_y = np.zeros_like(self.y)

        for i in range(self.num_neurons):
            r_squared = self.x[i] ** 2 + self.y[i] ** 2

            # Hopf oscillator dynamics
            x_dot = (
                self.alpha[i] * (self.A[i] ** 2 - r_squared) * self.x[i]
                - self.omega[i] * self.y[i]
            )
            y_dot = (
                self.omega[i] * self.x[i]
                + self.alpha[i] * (self.A[i] ** 2 - r_squared) * self.y[i]
            )

            # Coupling terms from neighboring oscillators
            for j in [i - 1, i + 1]:
                if 0 <= j < self.num_neurons:
                    # Rotation matrix for phase difference
                    rot_matrix = np.array([
                        [
                            np.cos(self.phase_diff[i, j]),
                            -np.sin(self.phase_diff[i, j]),
                        ],
                        [
                            np.sin(self.phase_diff[i, j]),
                            np.cos(self.phase_diff[i, j]),
                        ],
                    ])

                    # Apply coupling
                    coupled = (
                        self.h * rot_matrix @ np.array([self.x[j], self.y[j]])
                    )
                    x_dot += coupled[0]
                    y_dot += coupled[1]

            # Update states using Euler integration
            new_x[i] = self.x[i] + x_dot * self.dt
            new_y[i] = self.y[i] + y_dot * self.dt

        self.x, self.y = new_x, new_y
        return self.x.copy(), self.y.copy()

    def reset(self) -> None:
        """Reset the oscillator states to small random values."""
        self.x = RNG.uniform(-0.1, 0.1, self.num_neurons)
        self.y = RNG.uniform(-0.1, 0.1, self.num_neurons)

    def simulate(
        self,
        steps: int,
    ) -> tuple[
        np.ndarray[tuple[int, ...], np.dtype[Any]],
        np.ndarray[tuple[int, ...], np.dtype[Any]],
    ]:
        # Main simulation
        x_history = np.zeros((steps, self.num_neurons))
        y_history = np.zeros((steps, self.num_neurons))

        for t in track(range(steps)):
            x, y = self.step()
            x_history[t] = x
            y_history[t] = y

        return x_history, y_history
