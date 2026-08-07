"""Sketch-based approximate all-pairs shortest paths using random pivots.

The only approximation algorithm in this repo, and the only one that does not
materialise the full n x n distance matrix. Instead of computing every pair, it
samples ``k`` random pivot vertices, records each vertex's distances to and from
them -- that pair of lists is the vertex's "sketch" -- and estimates ``d(u, v)``
as the cheapest detour through a pivot both endpoints share.

This is the idea behind distance oracles in the Thorup-Zwick line of work: pay
a small, bounded amount of preprocessing, then answer queries in time that
depends on the sketch size rather than the graph size.

Each vertex stores two numbers per pivot: ``d(v -> p)``, found by searching the
reverse graph from ``p``, and ``d(p -> v)``, found by searching forwards. An
estimate then reads ``d(u -> p) + d(p -> v)``, which traverses each leg in the
direction it actually points, so it always corresponds to a real ``u -> v``
walk. The estimate is therefore a true upper bound: never below the shortest
distance, possibly above it, and ``INF`` exactly when no sampled pivot lies on
any path between the two vertices.

Accuracy improves as ``k`` grows, trading approximation quality for
preprocessing time.

Time complexity: O(k * (m + n log n)) preprocessing for k pivots -- two Dijkstra
runs per pivot rather than one, which is a constant factor, not a change in
order -- plus O(k^2) per distance query.

Best for: large graphs where an exact APSP matrix is too expensive to compute
or store.
"""

import random

from algorithms.dijkstra import dijkstra
from graph import INF


def sketch_approx_apsp(graph, k=5):
    """Build a pivot-based distance oracle for ``graph``.

    Samples ``k`` pivots with replacement and, for each, runs Dijkstra twice:
    once on the graph to get ``d(pivot -> v)`` and once on its reverse to get
    ``d(v -> pivot)``. Every vertex therefore stores a
    ``(pivot, dist_to_pivot, dist_from_pivot)`` triple per sample, which is what
    lets a query compose the two legs in the correct direction.

    Unlike the other four algorithms this returns a *query function*, not a
    distance matrix. Timing a call to ``sketch_approx_apsp`` therefore measures
    preprocessing only -- the Dijkstra runs and the reverse-graph build -- which
    is what makes it the fastest entry in the benchmark.

    Requires non-negative edge weights, since the preprocessing is Dijkstra.

    Args:
        graph: The :class:`~graph.Graph` to preprocess.
        k: Number of random pivots. Larger k means better estimates and slower
            preprocessing. Pivots are chosen with replacement, so duplicates
            are possible.

    Returns:
        A function ``approx_dist(u, v)`` returning an upper bound on the
        distance between two vertices, or ``INF`` if no sampled pivot lies on a
        path between them.
    """
    sketch = {v: [] for v in graph.get_vertices()}
    reverse_graph = graph.reverse()

    for _ in range(k):
        pivot = random.choice(graph.get_vertices())
        dists_from_pivot = dijkstra(graph, pivot)
        dists_to_pivot = dijkstra(reverse_graph, pivot)
        for v in graph.get_vertices():
            sketch[v].append((pivot, dists_to_pivot[v], dists_from_pivot[v]))

    def approx_dist(u, v):
        """Estimate the distance between ``u`` and ``v`` via a shared pivot."""
        min_dist = INF
        for (p1, d_u_to_p, _) in sketch[u]:
            for (p2, _, d_p_to_v) in sketch[v]:
                if p1 == p2:
                    min_dist = min(min_dist, d_u_to_p + d_p_to_v)
        return min_dist

    return approx_dist
