"""Johnson's all-pairs shortest path algorithm.

Johnson's algorithm gets the best of both worlds on sparse graphs with negative
edges: it removes the negative weights with one Bellman-Ford pass, then runs the
much faster Dijkstra from every vertex.

The trick is a potential function ``h``. Adding a virtual vertex ``q`` with a
zero-weight edge to every vertex and running Bellman-Ford from it yields
``h[v]``, the cheapest way to reach ``v``. Reweighting each edge as
``w'(u, v) = w(u, v) + h[u] - h[v]`` makes every weight non-negative (by the
triangle inequality ``h[v] <= h[u] + w``) while preserving which path is
shortest -- the ``h`` terms telescope along a path, so every ``u -> v`` path
shifts by the same constant ``h[u] - h[v]``. Dijkstra can then run safely, and
subtracting the shift restores the true distances.

Time complexity: O(n^2 log n + n*m) with a binary heap -- one O(n*m)
Bellman-Ford pass, then n Dijkstra runs.

Best for: sparse graphs with negative edge weights but no negative cycles.
"""

from algorithms.bellman_ford import bellman_ford
from algorithms.dijkstra import dijkstra


def johnson(graph):
    """Compute all-pairs shortest distances via potential reweighting.

    Runs in four stages: build an extended graph with a virtual source, derive
    the potentials ``h`` from it with Bellman-Ford, reweight every edge to be
    non-negative, then run Dijkstra from each vertex and undo the shift.

    Note: this reweights ``graph`` in place through
    :meth:`~graph.Graph.update_edge_weight` -- the edge weights are left in
    their reweighted form when the function returns, and the caller's graph is
    modified. Only the copy used for the Bellman-Ford stage is isolated. Pass
    ``graph.copy()`` if the original weights are needed afterwards.

    Args:
        graph: The :class:`~graph.Graph` to search. Modified in place.

    Returns:
        A nested dict where ``dist[u][v]`` is the shortest distance from ``u``
        to ``v``, or the string ``"Negative cycle detected"`` if the graph
        contains a negative-weight cycle.
    """
    g_ext = graph.copy()
    q = g_ext.num_vertices
    g_ext.num_vertices += 1
    for v in range(q):
        g_ext.add_edge(q, v, 0)

    h = bellman_ford(g_ext, q)
    if h is None:
        return "Negative cycle detected"

    for u, v, w in graph.edges:
        graph.update_edge_weight(u, v, w + h[u] - h[v])

    dist = {}
    for u in range(graph.num_vertices):
        dist[u] = dijkstra(graph, u)

    for u in dist:
        for v in dist[u]:
            dist[u][v] += h[v] - h[u]

    return dist
