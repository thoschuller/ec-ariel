"""Pre-built MuJoCo environments."""

from .amphitheatre_heightmap import AmphitheatreTerrainWorld
from .boxy_heightmap import BoxyRugged
from .crater_heightmap import CraterTerrainWorld
from .djoser_pyramid import PyramidWorld
from .olympic_arena import OlympicArena
from .rugged_heightmap import RuggedTerrainWorld
from .simple_flat_world import SimpleFlatWorld
from .simple_tilted_world import TiltedFlatWorld

__all__ = [
    "AmphitheatreTerrainWorld",
    "BoxyRugged",
    "CraterTerrainWorld",
    "OlympicArena",
    "PyramidWorld",
    "RuggedTerrainWorld",
    "SimpleFlatWorld",
    "TiltedFlatWorld",
]
