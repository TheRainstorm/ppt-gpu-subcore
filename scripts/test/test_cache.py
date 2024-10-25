

from sdcm import *
from cache_simulator import *

l1_cache_parameter = {
    "cache_line_size": 32,
    "capacity": 32 * 4,
    "associativity": 1,
}

cache_line_access = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 2],
    [0, 0, 0, 3],
    [0, 0, 0, 4],
    [0, 0, 0, 5],
    [0, 0, 0, 6],
    [0, 0, 0, 7],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 2],
    [0, 0, 0, 3],
    [0, 0, 0, 4],
    [0, 0, 0, 5],
    [0, 0, 0, 6],
    [0, 0, 0, 7],
]

cache_simulate(cache_line_access, l1_cache_parameter)
model(cache_line_access, l1_cache_parameter)