"""Sketch-based approximate all-pairs shortest paths using random pivots.

The only approximation algorithm in this repo, and the only one that does not
materialise the full n x n distance matrix. Instead of computing every pair, it
samples ``k`` random pivot vertices, records each vertex's distance to them --
that list is the vertex's "sketch" -- and estimates ``d(u, v)`` as the cheapest
detour through a pivot both endpoints share.

This is the idea behind distance oracles in the Thorup-Zwick line of work: pay
a small, bounded amount of preprocessing, then answer queries in time that
depends on the sketch size rather than the graph size.

Accuracy improves as ``k`` grows, trading approximation quality for
preprocessing time, and the estimate is ``INF`` whenever two vertices share no
pivot that reaches them both.

Accuracy caveat: the sketch stores only *from-pivot* distances, so an estimate
adds ``d(p -> u)`` to ``d(p -> v)``. That is a genuine ``u -> v`` walk only if
the first leg can be traversed backwards, which holds on undirected graphs but
not on the directed graphs the generators here produce. On directed input the
estimate is therefore not a guaranteed upper bound and can come in *below* the
true distance -- including finite estimates for pairs with no path at all. An
oracle keyed on both ``d(u -> p)`` and ``d(p -> v)`` would be needed to restore
the one-sided error guarantee.

Time complexity: O(k * (m + n log n)) preprocessing for k pivots, plus O(k^2)
per distance query.

Best for: large graphs where an exact APSP matrix is too expensive to compute
or store.
"""

import random

from algorithms.dijkstra import dijkstra
from graph import INF


def sketch_approx_apsp(graph, k=5):
    """Build a pivot-based distance oracle for ``graph``.

    Samples ``k`` pivots with replacement, runs Dijkstra from each, and stores
    ``(pivot, distance)`` pairs per vertex. See the module docstring for why
    the resulting estimates are not a one-sided approximation on directed input.

    Unlike the other four algorithms this returns a *query function*, not a
    distance matrix. Timing a call to ``sketch_approx_apsp`` therefore measures
    preprocessing only -- the k Dijkstra runs -- which is what makes it the
    fastest entry in the benchmark.

    Args:
        graph: The :class:`~graph.Graph` to preprocess.
        k: Number of random pivots. Larger k means better estimates and slower
            preprocessing. Pivots are chosen with replacement, so duplicates
            are possible.

    Returns:
        A function ``approx_dist(u, v)`` returning the estimated distance
        between two vertices, or ``INF`` if they share no reachable pivot.
    """
    sketch = {v: [] for v in graph.get_vertices()}

    for _ in range(k):
        pivot = random.choice(graph.get_vertices())
        dists = dijkstra(graph, pivot)
        for v in dists:
            sketch[v].append((pivot, dists[v]))

    def approx_dist(u, v):
        """Estimate the distance between ``u`` and ``v`` via a shared pivot."""
        min_dist = INF
        for (p1, d1) in sketch[u]:
            for (p2, d2) in sketch[v]:
                if p1 == p2:
                    min_dist = min(min_dist, d1 + d2)
        return min_dist

    return approx_dist
