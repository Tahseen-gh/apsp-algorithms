"""The Floyd-Warshall all-pairs shortest path algorithm.

Floyd-Warshall is the direct, classic APSP algorithm: rather than solving n
single-source problems, it fills in one distance matrix by dynamic programming.

Time complexity: O(n^3), independent of the edge count.
Space complexity: O(n^2).

Best for: dense graphs, and small-to-medium graphs generally.
"""

from graph import INF


def floyd_warshall(graph):
    """Compute all-pairs shortest distances by dynamic programming.

    The invariant on the outer loop is that after iteration ``k``, ``dist[i][j]``
    holds the shortest path from ``i`` to ``j`` whose intermediate vertices all
    come from ``{0, ..., k}``. Each new ``k`` asks a single question for every
    pair: is routing through ``k`` cheaper than what we already have? After the
    final ``k`` every vertex is an admissible waypoint, so the matrix holds true
    shortest paths.

    Note the loop order -- ``k`` must be outermost. Making it the inner loop
    would consult vertex sets that have not been computed yet and silently
    produce wrong answers.

    The run time does not depend on how many edges exist, which is why this wins
    on dense graphs and loses badly on sparse ones: the triple loop costs n^3
    whether the graph has n^2 edges or none.

    Args:
        graph: The :class:`~graph.Graph` to search.

    Returns:
        An ``n x n`` list of lists where ``dist[i][j]`` is the shortest distance
        from ``i`` to ``j``, ``0`` on the diagonal, and ``INF`` when no path
        exists.
    """
    n = graph.num_vertices
    dist = [[INF] * n for _ in range(n)]
    for u in range(n):
        dist[u][u] = 0
    for u, v, w in graph.edges:
        dist[u][v] = w
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist
