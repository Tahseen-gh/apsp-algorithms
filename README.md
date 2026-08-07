# APSP Algorithms

Implementations and a benchmark harness for five approaches to the **all-pairs
shortest path** (APSP) problem: given a weighted graph, find the shortest
distance between *every* pair of vertices.

Four of the five are exact; the fifth trades accuracy for speed. They span a
wide range of design choices — dynamic programming over a distance matrix,
repeated single-source searches, potential reweighting, and random-pivot
sketching — which is what makes comparing them interesting: each one wins in a
different corner of the input space.

Graduate coursework, written in pure Python with no dependencies beyond
`matplotlib` for the benchmark plot.

## Layout

```
graph.py              Graph class (edge list + adjacency map) and the random graph generators
algorithms/
  floyd_warshall.py   Floyd-Warshall
  johnson.py          Johnson's algorithm
  dijkstra.py         Dijkstra, and repeated Dijkstra for APSP
  bellman_ford.py     Bellman-Ford, and repeated Bellman-Ford for APSP
  sketch.py           Sketch-based approximation using random pivots
benchmark.py          Timing harness and plot generation
```

`Graph` stores each edge twice — once in a flat `(u, v, w)` list and once in an
adjacency map — so the relaxation-based algorithms can sweep all edges and the
frontier-based ones can expand a single vertex, neither paying a conversion
cost.

## The algorithms

Throughout, **n** is the vertex count and **m** the edge count.

### Floyd-Warshall — `floyd_warshall(graph)`

Dynamic programming over the distance matrix. After iteration `k`, `dist[i][j]`
holds the shortest `i → j` path whose intermediate vertices all come from
`{0..k}`; each new `k` asks one question per pair — is routing through `k`
cheaper? — so after the last `k` the matrix is exact.

- **Time:** O(n³), independent of m. **Space:** O(n²).
- **Wins on:** dense graphs and small-to-medium n. Its cost is fixed by n
  alone, so density is free — but that same property makes it wasteful on
  sparse graphs, where it pays for n² pairs that mostly have no edge.
- Handles negative edges; does not detect negative cycles.

### Johnson's algorithm — `johnson(graph)`

Designed to beat repeated Bellman-Ford on sparse graphs with negative weights.
It adds a virtual vertex connected to everything at zero cost, runs one
Bellman-Ford pass from it to get a potential `h`, then reweights every edge as
`w'(u,v) = w(u,v) + h[u] - h[v]`. The triangle inequality makes every `w'`
non-negative, and because the `h` terms telescope along a path, every `u → v`
path shifts by the same constant — so the *shortest* path is unchanged and
Dijkstra can safely run from each vertex. Subtracting the shift restores true
distances.

- **Time:** O(n² log n + n·m) with a binary heap.
- **Wins on:** sparse graphs with negative edges but no negative cycles — the
  one regime where Dijkstra alone is invalid and repeated Bellman-Ford is
  ruinously slow.
- Returns `"Negative cycle detected"` if the graph has a negative cycle.

### Repeated Dijkstra — `all_pairs_dijkstra(graph)`

Run Dijkstra from every vertex. A binary heap serves as the priority queue;
since `heapq` has no decrease-key, improved vertices are pushed again and stale
entries are skipped on pop.

- **Time:** O(n · (m + n log n)).
- **Wins on:** sparse graphs with non-negative weights. When m ≪ n², n heap
  searches beat Floyd-Warshall's fixed n³ comfortably. In practice (see below)
  it is the fastest exact algorithm here in *every* configuration tested.
- Requires non-negative weights — a negative edge can finalise a vertex before
  its true shortest path is found.

### Repeated Bellman-Ford — `all_pairs_bellman_ford(graph)`

Run Bellman-Ford from every vertex. Each run relaxes all m edges n−1 times —
the most rounds any simple path can need — then does one extra sweep as a
negative-cycle test.

- **Time:** O(n² · m).
- **Wins on:** nothing, on speed. It is the baseline Johnson's algorithm is
  built to replace, and it is included to show the size of that gap. Its merit
  is simplicity and negative-cycle detection.
- Handles negative edges; returns `"Negative cycle detected"` when one exists.

### Sketch-based approximation — `sketch_approx_apsp(graph, k=5)`

The only approximation, and the only one that never builds an n × n matrix. It
samples `k` random pivots, runs Dijkstra from each, and stores every vertex's
distances to those pivots — that list is the vertex's *sketch*. A query
`d(u,v)` is then estimated as the cheapest detour through a pivot the two share.
This is the idea behind distance oracles in the Thorup–Zwick line of work.

- **Time:** O(k · (m + n log n)) preprocessing, O(k²) per query.
- **Wins on:** large graphs where an exact matrix is too expensive to compute or
  even store. Accuracy rises with `k`.
- Returns a **query function**, not a matrix. See
  [Notes and caveats](#notes-and-caveats) for what that means for the timings,
  and for the accuracy limits on directed input.

### Summary

| Algorithm | Time complexity | Negative weights | Best regime |
|---|---|---|---|
| Floyd-Warshall | O(n³) | Yes (no cycle detection) | Dense, small-to-medium n |
| Johnson's | O(n² log n + n·m) | Yes, with detection | Sparse **with** negative weights |
| Repeated Dijkstra | O(n · (m + n log n)) | No | Sparse, non-negative weights |
| Repeated Bellman-Ford | O(n² · m) | Yes, with detection | Simplicity / detection only |
| Sketch approximation | O(k · (m + n log n)) | No | Very large graphs, approximate answers |

## Benchmark

### Setup

Four configurations — two sizes crossed with two density regimes:

| Configuration | Vertices | Edges | Generator |
|---|---|---|---|
| 100 nodes (Sparse) | 100 | ~250 | `generate_sparse_graph(100)`, p = 0.05 |
| 100 nodes (Dense) | 100 | 4,950 | `generate_dense_graph(100)` |
| 500 nodes (Sparse) | 500 | ~6,200 | `generate_sparse_graph(500)`, p = 0.05 |
| 500 nodes (Dense) | 500 | 124,750 | `generate_dense_graph(500)` |

Sparse graphs include each candidate edge with probability 0.05; dense graphs
include all of them. Weights are uniform integers from 1 to 10 in both cases.
Every algorithm is timed once per configuration with `time.time()` around a
single run.

### Running it

```bash
pip install -r requirements.txt
python benchmark.py
```

The plot is written to `results/apsp_benchmark.png` — nothing opens a window, so
this works headless.

```
python benchmark.py --sizes 100              # only the 100-vertex configs
python benchmark.py -a dijkstra -a sketch    # only some algorithms (repeatable)
python benchmark.py --seed 42                # reproducible graphs
python benchmark.py -o results/run.png       # choose the output path
python benchmark.py -p 0.10                  # denser "sparse" graphs
python benchmark.py --help                   # all options
```

**The full default sweep takes roughly an hour**, almost all of it in the two
slowest cells: repeated Bellman-Ford and Johnson's on the 500-node dense graph.
Use `--sizes 100` or `-a` to iterate quickly.

### Results

Measured with `--seed 1` on Python 3.11, in seconds:

| Algorithm | 100 sparse | 100 dense | 500 sparse | 500 dense |
|---|---|---|---|---|
| Floyd-Warshall | 0.062 | 0.077 | 9.427 | 10.058 |
| Johnson's | 0.008 | 1.049 | 2.051 | ~670 † |
| Repeated Dijkstra | 0.001 | 0.017 | 0.196 | 2.198 |
| Repeated Bellman-Ford | 0.219 | 4.209 | 162.634 | ~3270 † |
| Sketch approximation | 0.000 | 0.001 | 0.002 | 0.022 |

† Extrapolated, not measured — these two cells take roughly 11 and 55 minutes
respectively. Both scale on a countable quantity, so the projection is
straightforward: repeated Bellman-Ford performs `n(n-1)m` edge relaxations, and
the sparse 500-node cell fixes the rate at ~9.5M/s, giving 3.11e10 / 9.5e6 for
the dense cell. Johnson's is dominated by its O(m²) reweighting (see point 4
below), and the dense 100-node cell fixes that rate at ~23M/s.

Four things stand out:

1. **Floyd-Warshall barely notices density.** 9.43 s sparse versus 10.06 s dense
   at n = 500, for a 20× difference in edge count — a clean confirmation that
   O(n³) is independent of m. It is the only algorithm here with that property.

2. **Repeated Dijkstra wins everywhere**, including the dense cases where
   Floyd-Warshall is theoretically favoured. Both are O(n³) on a dense graph,
   and the constant factors decide it: Dijkstra touches only real edges and
   skips unreachable work, while Floyd-Warshall's triple loop runs all n³
   iterations as interpreted Python regardless. On a compiled implementation
   with a contiguous matrix, the dense column would likely flip.

3. **Bellman-Ford degrades fastest with density**, as O(n²·m) predicts: 19× the
   edges at n = 100 costs it 19× the time, while Floyd-Warshall is flat.

4. **Johnson's is slower than the plain repeated Dijkstra it wraps.** That is
   the reweighting, not the algorithm: `Graph.update_edge_weight` scans the
   whole edge list on each call, so reweighting all m edges costs O(m²) and
   dominates everything else on dense graphs. Indexing the edge list by
   `(u, v)` would remove it.

The headline caveat: **every generated graph has non-negative weights**, so the
one thing Johnson's and Bellman-Ford exist for is never exercised. On these
inputs Dijkstra is always legal, and an algorithm that pays extra to tolerate
negative edges can only lose. Reading this as "Johnson's is bad" would invert
what the benchmark actually shows — it measures overhead in a regime where the
feature being paid for is not needed.

## Notes and caveats

Things worth knowing before drawing conclusions from these numbers. All of them
describe existing behaviour — the algorithm implementations are unchanged.

- **The generated graphs are directed acyclic graphs.** `add_edge(u, v, w)`
  records `u → v` only, and both generators emit edges only for `u < v`, so
  every edge points from a lower to a higher vertex ID. Roughly half of all
  vertex pairs are therefore unreachable and keep a distance of `INF`. The
  inline comments describing the graphs as undirected reflect the intent, not
  the resulting structure.

- **`johnson()` reweights its input in place.** It returns with the graph's edge
  weights left in reweighted form. The benchmark shares one graph object across
  all five algorithms per configuration, as the original notebook did, so the
  three algorithms that run after Johnson's see the reweighted graph. Shortest
  path *structure* is preserved by reweighting, so the timings remain valid, but
  pass `graph.copy()` if you need the original weights back.

- **The sketch timing measures preprocessing only.** `sketch_approx_apsp`
  returns a query function, so timing the call captures the `k` Dijkstra runs
  and none of the O(k²)-per-pair query cost. Its numbers are not comparable
  like-for-like with the four algorithms that materialise a full matrix.

- **The sketch can underestimate on directed graphs.** It stores only
  *from-pivot* distances, so an estimate adds `d(p → u)` to `d(p → v)`. That is
  a real `u → v` walk only if the first leg can be traversed backwards — true on
  undirected graphs, false here. On the test graphs it returns finite estimates
  for some pairs that have no path at all. An oracle keyed on both `d(u → p)`
  and `d(p → v)` would restore the usual one-sided error guarantee.

- **Single-run timings.** Each cell is one measurement, unrepeated, so the
  sub-millisecond entries are close to timer noise. The order-of-magnitude gaps
  are solid; small differences are not.

## Requirements

Python 3.7+ and `matplotlib` (for the benchmark plot only — `graph.py` and
everything in `algorithms/` are pure standard library).

```bash
pip install -r requirements.txt
```
