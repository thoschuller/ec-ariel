"""TODO(jmdm): description of script.

Date:       2025-07-08
Status:     Completed ✅
"""

# Standard library
from typing import TYPE_CHECKING

# Third-party libraries
from networkx import Graph

# Local libraries
from revolve.body_phenotypes.robogen_lite.config import (
    IDX_OF_CORE,
    ModuleFaces,
    ModuleRotationsTheta,
)
from revolve.body_phenotypes.robogen_lite.modules import ModuleTypeInstances
from revolve.body_phenotypes.robogen_lite.modules.core import CoreModule

if TYPE_CHECKING:
    from revolve.body_phenotypes.robogen_lite.modules.module import Module


def construct_mjspec_from_graph(graph: Graph) -> CoreModule:
    """
    Construct a MuJoCo specification from a graph representation.

    Parameters
    ----------
    graph : Graph
        A graph representation of the robot's structure.

    Returns
    -------
    CoreModule
        The core module of the robot, which contains all other modules.
    """
    modules: dict[int, Module] = {}
    for node in graph.nodes:
        maybe_module = graph.nodes[node]["type"]
        module_rotation = graph.nodes[node]["rotation"]

        maybe_module = ModuleTypeInstances[maybe_module].value

        # Check that the module is not None (aka 'DEAD')
        if maybe_module:
            module = maybe_module(index=node)
            module.rotate(
                ModuleRotationsTheta[module_rotation].value,
            )
            modules[node] = module
        else:
            modules[node] = None

    for edge in graph.edges:
        from_module = edge[0]
        to_module = edge[1]
        face = graph.edges[edge]["face"]

        # Check if both modules exist (not 'DEAD')
        if modules[to_module]:
            modules[from_module].sites[ModuleFaces[face]].attach_body(
                body=modules[to_module].body,
                prefix=f"{modules[from_module].index}-{modules[to_module].index}-{ModuleFaces[face].value}-",
            )

    return modules[IDX_OF_CORE]
