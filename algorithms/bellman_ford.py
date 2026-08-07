"""Bellman-Ford, and repeated Bellman-Ford for all-pairs shortest paths.

Bellman-Ford solves single-source shortest paths and, unlike Dijkstra, tolerates
negative edge weights. It also detects negative-weight cycles, which is what
makes it usable as the preprocessing step in Johnson's algorithm.

Time complexity:
    * Single source: O(n * m).
    * All pairs: O(n^2 * m).

Best for: graphs with negative edge weights (and, in this repo, as the
reweighting step inside Johnson's algorithm).
"""

from graph import INF


def bellman_ford(graph, source):
    """Compute shortest distances from ``source``, tolerating negative weights.

    Relaxes every edge ``n - 1`` times, which is the most rounds any shortest
    path can need since a simple path visits at most ``n`` vertices. One extra
    sweep then acts as the negative-cycle test: if any edge still improves after
    ``n - 1`` rounds, some cycle has negative total weight and no finite
    shortest path exists.

    Unreachable vertices stay at ``INF``; because ``INF + w == INF``, they are
    never mistaken for an improvement.

    Args:
        graph: The :class:`~graph.Graph` to search.
        source: Vertex ID to start from.

    Returns:
        A dict mapping every vertex to its distance from ``source``, or ``None``
        if the graph contains a negative-weight cycle.
    """
    dist = {v: INF for v in graph.get_vertices()}
    dist[source] = 0

    for _ in range(graph.num_vertices - 1):
        for u, v, w in graph.edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    for u, v, w in graph.edges:
        if dist[u] + w < dist[v]:
            return None  # Negative-weight cycle

    return dist


def all_pairs_bellman_ford(graph):
    """Solve APSP by running :func:`bellman_ford` from every vertex.

    Correct in the presence of negative edges, but the slowest option here by a
    wide margin: O(n^2 * m) means the work grows with both the vertex count
    squared and the edge count, so it degrades fastest as graphs get denser.
    It is included as the baseline that Johnson's algorithm is designed to beat
    on exactly the same class of inputs.

    Args:
        graph: The :class:`~graph.Graph` to search.

    Returns:
        A nested dict where ``dist[u][v]`` is the shortest distance from ``u``
        to ``v``, or the string ``"Negative cycle detected"`` if any source
        reveals a negative-weight cycle.
    """
    dist = {}
    for u in range(graph.num_vertices):
        res = bellman_ford(graph, u)
        if res is None:
            return "Negative cycle detected"
        dist[u] = res
    return dist
