"""Dijkstra's algorithm, and repeated Dijkstra for all-pairs shortest paths.

Dijkstra solves the single-source shortest path (SSSP) problem on graphs with
non-negative edge weights. Running it once from every vertex solves APSP.

This particular implementation keeps no settled set, so it is label-correcting
and tolerates negative edges in practice -- see :func:`dijkstra` for what that
buys and what it costs.

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

    Note there is no settled/visited set: the ``d > dist[u]`` test rejects only
    *stale* heap entries, not vertices that have already been expanded. A vertex
    whose distance later improves is therefore pushed and expanded again, which
    makes this label-correcting rather than textbook Dijkstra. The practical
    consequence is that it stays correct on graphs with negative edges (as long
    as there is no negative cycle), where a settled-set implementation would
    finalise a vertex too early and return a wrong answer.

    That robustness is not free: the O(m log n) bound assumes each vertex is
    expanded once, which negative edges can break. Re-expansions are cheap on
    the benchmark's acyclic negative graphs but are exponential in the worst
    case.

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
