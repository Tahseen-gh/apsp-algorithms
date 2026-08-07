"""Dijkstra's algorithm, and repeated Dijkstra for all-pairs shortest paths.

Dijkstra solves the single-source shortest path (SSSP) problem on graphs with
non-negative edge weights. Running it once from every vertex solves APSP.

Time complexity:
    * Single source: O(m log n) with a binary heap.
    * All pairs: O(n * (m + n log n)).

Best for: sparse graphs with non-negative edge weights.
"""

import heapq

from graph import INF


def dijkstra(graph, source):
    """Compute shortest distances from ``source`` to every vertex.

    Uses a binary heap as the priority queue. Because ``heapq`` offers no
    decrease-key, an improved vertex is pushed again rather than updated in
    place; the stale copies are discarded by the ``d > dist[u]`` check when
    they surface. This leaves at most one heap entry per relaxation, so the
    heap holds O(m) entries rather than O(n).

    Correctness requires non-negative weights -- with a negative edge, a vertex
    can be finalised before its true shortest path is found.

    Args:
        graph: The :class:`~graph.Graph` to search.
        source: Vertex ID to start from.

    Returns:
        A dict mapping every vertex to its distance from ``source``, with
        ``INF`` for unreachable vertices.
    """
    dist = {v: INF for v in graph.get_vertices()}
    dist[source] = 0
    heap = [(0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, weight in graph.neighbors(u):
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(heap, (dist[v], v))
    return dist


def all_pairs_dijkstra(graph):
    """Solve APSP by running :func:`dijkstra` from every vertex.

    The simplest correct APSP strategy for non-negative weights, and on sparse
    graphs the fastest of the exact algorithms here: n heap-based searches over
    m edges beat Floyd-Warshall's dense O(n^3) triple loop whenever m is well
    below n^2.

    Args:
        graph: The :class:`~graph.Graph` to search. Must have no negative
            edge weights.

    Returns:
        A nested dict where ``dist[u][v]`` is the shortest distance from ``u``
        to ``v``, or ``INF`` if ``v`` is unreachable from ``u``.
    """
    dist = {}
    for u in range(graph.num_vertices):
        dist[u] = dijkstra(graph, u)
    return dist
