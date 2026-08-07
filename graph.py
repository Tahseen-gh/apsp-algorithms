"""Graph representation and random graph generators.

This module holds the shared data structure used by every algorithm in
``algorithms/``, plus the two generators used to build benchmark inputs.

The graph is stored twice, in two complementary forms:

* ``edges`` -- a flat ``(u, v, w)`` edge list, which the relaxation-based
  algorithms (Bellman-Ford, Floyd-Warshall) sweep over directly.
* ``adj`` -- an adjacency map ``u -> [(v, w), ...]``, which the frontier-based
  algorithms (Dijkstra) use to expand a vertex's neighbours in O(deg(u)).

Keeping both means no algorithm pays a conversion cost at run time, at the
price of storing each edge twice.
"""

import random
from collections import defaultdict

#: Sentinel for "no path known yet". ``float('inf')`` is used rather than a
#: large integer so that ``INF + w == INF`` and every comparison against a real
#: path length behaves correctly without special-casing.
INF = float('inf')


class Graph:
    """A weighted directed graph over vertices ``0 .. num_vertices - 1``.

    Edges are directed: ``add_edge(u, v, w)`` records ``u -> v`` only. The
    generators below exploit this by emitting a single edge per vertex pair.

    Attributes:
        num_vertices: Number of vertices; vertex IDs are ``range(num_vertices)``.
        edges: Flat list of ``(u, v, w)`` tuples in insertion order.
        adj: ``defaultdict`` mapping ``u -> [(v, w), ...]`` for outgoing edges.
    """

    def __init__(self, num_vertices):
        """Create an edgeless graph with ``num_vertices`` vertices."""
        self.num_vertices = num_vertices
        self.edges = []
        self.adj = defaultdict(list)

    def add_edge(self, u, v, w):
        """Add a directed edge ``u -> v`` of weight ``w`` to both views."""
        self.edges.append((u, v, w))
        self.adj[u].append((v, w))

    def neighbors(self, u):
        """Return the outgoing ``(vertex, weight)`` pairs for ``u``.

        Returns an empty list for a vertex with no outgoing edges (``adj`` is a
        ``defaultdict``, so this never raises).
        """
        return self.adj[u]

    def get_vertices(self):
        """Return all vertex IDs as a list."""
        return list(range(self.num_vertices))

    def copy(self):
        """Return an independent copy carrying the same vertices and edges.

        Used by Johnson's algorithm, which needs to bolt an extra source vertex
        onto the graph without disturbing the caller's copy.
        """
        new_g = Graph(self.num_vertices)
        for u, v, w in self.edges:
            new_g.add_edge(u, v, w)
        return new_g

    def reverse(self):
        """Return a new graph with every edge direction flipped.

        Searching the reverse graph from a vertex ``p`` yields the distances
        *into* ``p`` -- ``d(u -> p)`` for every ``u`` -- which a forward search
        cannot produce on a directed graph. The sketch oracle needs exactly
        that to compose a real path through a pivot.

        On an undirected graph the reverse is isomorphic to the original, so
        this is only load-bearing for directed input.
        """
        new_g = Graph(self.num_vertices)
        for u, v, w in self.edges:
            new_g.add_edge(v, u, w)
        return new_g

    def update_edge_weight(self, u, v, new_w):
        """Set the weight of every ``u -> v`` edge to ``new_w``, in place.

        Updates both the edge list and the adjacency map so the two views stay
        consistent. This is the reweighting hook Johnson's algorithm uses.

        Note: this scans the whole edge list on each call, so it costs O(m) per
        edge updated -- see the complexity notes in the README for what that
        means for Johnson's measured run time on dense graphs.
        """
        for i in range(len(self.edges)):
            if self.edges[i][0] == u and self.edges[i][1] == v:
                self.edges[i] = (u, v, new_w)
        self.adj[u] = [(x, new_w if x == v else w) for x, w in self.adj[u]]


def generate_sparse_graph(num_vertices, edge_probability=0.05):
    """Generate a sparse random graph in the Erdos-Renyi G(n, p) style.

    Each ordered pair ``u < v`` gets an edge with probability
    ``edge_probability``, weighted uniformly at random from 1 to 10. The
    expected edge count is ``p * n * (n - 1) / 2``, so at the default p = 0.05
    a 100-vertex graph carries roughly 250 edges.

    Because edges are only emitted for ``u < v``, the result is a directed
    acyclic graph -- every edge points from a lower to a higher vertex ID.
    Many vertex pairs are therefore unreachable and keep a distance of ``INF``.

    Args:
        num_vertices: Number of vertices in the generated graph.
        edge_probability: Independent probability of each candidate edge.

    Returns:
        A newly built :class:`Graph`.
    """
    graph = Graph(num_vertices)
    for u in range(num_vertices):
        for v in range(u+1, num_vertices):  # Avoid duplicates (since undirected graph)
            if random.random() < edge_probability:  # 5% chance to add an edge
                weight = random.randint(1, 10)  # Random weight between 1 and 10
                graph.add_edge(u, v, weight)
    return graph


def generate_dense_graph(num_vertices):
    """Generate a dense random graph containing every ``u < v`` edge.

    This is :func:`generate_sparse_graph` with the coin flip removed, giving
    exactly ``n * (n - 1) / 2`` edges -- 4,950 at n = 100 and 124,750 at
    n = 500. Weights are again uniform from 1 to 10, and the same ``u < v``
    orientation makes the result acyclic.

    Args:
        num_vertices: Number of vertices in the generated graph.

    Returns:
        A newly built :class:`Graph`.
    """
    graph = Graph(num_vertices)
    for u in range(num_vertices):
        for v in range(u+1, num_vertices):  # Avoid duplicates (since undirected graph)
            weight = random.randint(1, 10)  # Random weight between 1 and 10
            graph.add_edge(u, v, weight)
    return graph
