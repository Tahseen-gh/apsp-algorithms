"""All-pairs shortest path algorithms.

Five approaches to the same problem, four exact and one approximate:

===========================  ==================================  ==============
Algorithm                    Entry point                         Complexity
===========================  ==================================  ==============
Floyd-Warshall               :func:`floyd_warshall`              O(n^3)
Johnson's                    :func:`johnson`                     O(n^2 log n + n*m)
Repeated Dijkstra            :func:`all_pairs_dijkstra`          O(n * (m + n log n))
Repeated Bellman-Ford        :func:`all_pairs_bellman_ford`      O(n^2 * m)
Sketch approximation         :func:`sketch_approx_apsp`          O(k * (m + n log n))
===========================  ==================================  ==============

The single-source routines :func:`dijkstra` and :func:`bellman_ford` are also
exported, since Johnson's algorithm and the sketch oracle are built on them.
"""

from algorithms.bellman_ford import all_pairs_bellman_ford, bellman_ford
from algorithms.dijkstra import all_pairs_dijkstra, dijkstra
from algorithms.floyd_warshall import floyd_warshall
from algorithms.johnson import johnson
from algorithms.sketch import sketch_approx_apsp

__all__ = [
    "all_pairs_bellman_ford",
    "all_pairs_dijkstra",
    "bellman_ford",
    "dijkstra",
    "floyd_warshall",
    "johnson",
    "sketch_approx_apsp",
]
