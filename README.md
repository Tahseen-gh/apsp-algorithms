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
  searches beat Floyd-Warshall's fixed n³ comfortably — and in practice it is
  the fastest exact algorithm here in *every* configuration tested, dense
  included. See [finding 1](#1-repeated-dijkstra-beats-floyd-warshall-even-on-dense-graphs).
- Textbook Dijkstra requires non-negative weights, but **this implementation
  keeps no settled set**, so it re-expands improved vertices and stays correct
  with negative edges absent a negative cycle. See
  [finding 3](#3-this-dijkstra-is-not-textbook-dijkstra) — it is a real
  robustness gain that costs the O(m log n) guarantee.

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
samples `k` random pivots and, for each, runs Dijkstra twice — once forwards for
`d(p → v)` and once on the reversed graph for `d(v → p)`. Every vertex keeps
both numbers per pivot; that is the vertex's *sketch*. A query `d(u,v)` is
estimated as the cheapest detour through a pivot the two share,
`min over p of d(u → p) + d(p → v)`. This is the idea behind distance oracles in
the Thorup–Zwick line of work.

Because each leg is traversed in the direction it actually points, the estimate
always corresponds to a real walk, making it a true **upper bound** on the
distance: never too small, sometimes too large, and `INF` exactly when no
sampled pivot lies on a path between the two vertices.

- **Time:** O(k · (m + n log n)) preprocessing, O(k²) per query. Two Dijkstra
  runs per pivot is a constant factor, not a change in order.
- **Wins on:** large graphs where an exact matrix is too expensive to compute or
  even store. Accuracy rises with `k` — on a 40-vertex directed test graph it
  returns the exact distance for 90% of pairs at k = 5 and 98% at k = 20.
- Requires non-negative weights, since the preprocessing is Dijkstra.
- Returns a **query function**, not a matrix. See
  [Notes and caveats](#notes-and-caveats) for what that means for the timings.

### Summary

| Algorithm | Time complexity | Negative weights | Best regime |
|---|---|---|---|
| Floyd-Warshall | O(n³) | Yes (no cycle detection) | Dense, small-to-medium n |
| Johnson's | O(n² log n + n·m) | Yes, with detection | Sparse **with** negative weights |
| Repeated Dijkstra | O(n · (m + n log n)) | Yes here † | Sparse, non-negative weights |
| Repeated Bellman-Ford | O(n² · m) | Yes, with detection | Simplicity / detection only |
| Sketch approximation | O(k · (m + n log n)) | Yes here † | Very large graphs, approximate answers |

† Not a property of Dijkstra's algorithm — a property of *this* implementation,
which omits the settled set and is therefore label-correcting. Correct without
negative cycles, but without the O(m log n) bound. See
[finding 3](#3-this-dijkstra-is-not-textbook-dijkstra).

## Benchmark

### Setup

Every size is run in three regimes, so a sweep of *k* sizes is *3k*
configurations:

| Regime | Directed? | Weights | Edge entries at n | Generator |
|---|---|---|---|---|
| Sparse | undirected | 1…10 | ≈ p·n·(n−1) | `generate_sparse_graph(n)`, p = 0.05 |
| Dense | undirected | 1…10 | n·(n−1) | `generate_dense_graph(n)` |
| Negative | directed | −5…10 | ≈ p·n·(n−1)/2 | `generate_negative_weight_graph(n)`, p = 0.05 |

The default sweep uses **n = 50, 100, 150** (nine configurations, about a
minute). `--full` uses **n = 100, 500** — the original coursework sizes, and the
source of the numbers in [Results](#results):

| Configuration | Vertices | Edge entries |
|---|---|---|
| 100 nodes (Sparse) | 100 | ~500 |
| 100 nodes (Dense) | 100 | 9,900 |
| 100 nodes (Negative) | 100 | ~250 |
| 500 nodes (Sparse) | 500 | ~12,400 |
| 500 nodes (Dense) | 500 | 249,500 |
| 500 nodes (Negative) | 500 | ~6,200 |

Sparse graphs include each candidate edge with probability 0.05; dense graphs
include all of them. Both are undirected, stored as two directed entries per
connection — so "edge entries" is twice the number of distinct connections, and
the dense 500-node graph is 20× the sparse one, exactly 1/p.

The negative configurations exist because the first two never justify why
Johnson's and Bellman-Ford are in the repo at all: with non-negative weights a
plain Dijkstra is always available and always faster. **These graphs are
directed on purpose.** An undirected edge of weight −3 is traversable in both
directions and is therefore already a cycle of weight −6, so *every* undirected
graph carrying a negative weight has a negative cycle, and the two algorithms
being tested would do nothing but detect it and return. Emitting edges only from
lower to higher vertex IDs makes the graph acyclic by construction, so shortest
paths stay well defined at any weight range.

Every algorithm is timed once per configuration with `time.time()` around a
single run.

### Running it

```bash
pip install -r requirements.txt
python benchmark.py
```

That is the **quick sweep** and takes **a little over a minute**: sizes 50, 100
and 150 across all three regimes, nine configurations in total. Three points per
regime is enough to see each algorithm's growth curve rather than isolated
measurements. The plot is written to `results/apsp_benchmark.png` — nothing
opens a window, so this works headless.

#### The full sweep

```bash
python benchmark.py --full
```

This runs the original coursework sizes, 100 and 500. **It takes hours, not
minutes** — the numbers in [Results](#results) come from it. Almost all of the
time goes to a single cell, the 500-node dense graph, where Bellman-Ford's
O(n²·m) and Johnson's O(m²) reweighting both meet 249,500 edge entries:
Bellman-Ford alone runs for over an hour there.

Cost grows steeply with n in the dense regime, because the dense generator makes
m itself quadratic. Between the quick sweep's largest cell and the full sweep's,
n grows 3.3× while Bellman-Ford's time grows roughly 200×.

#### Other options

```
python benchmark.py --sizes 100 200 300      # any sizes you like
python benchmark.py -a dijkstra -a sketch    # only some algorithms (repeatable)
python benchmark.py --no-negative            # skip the negative-weight configs
python benchmark.py --seed 42                # reproducible graphs
python benchmark.py -o results/run.png       # choose the output path
python benchmark.py -p 0.10                  # denser "sparse" graphs
python benchmark.py --help                   # all options
```

`--sizes` and `--full` are mutually exclusive; `--sizes` accepts any vertex
counts, so it is the general escape hatch in both directions.

### Results

The benchmark is currently being re-measured with `--full`: making the sparse
and dense generators undirected doubled their edge counts and added the
negative-weight configurations, so every previously published number is stale.
Run `python benchmark.py` for the quick sweep in the meantime.

## Notes and caveats

Things worth knowing before drawing conclusions from these numbers.

- **The negative-weight graphs are directed, and so have unreachable pairs.**
  `generate_negative_weight_graph` emits edges only for `u < v` to stay acyclic
  (see [Setup](#setup)), so roughly half of all vertex pairs have no path and
  keep a distance of `INF`. The sparse and dense generators are undirected and
  do not have this property. This means the negative column is not directly
  comparable to the other two: it is a different graph shape, not just a
  different weight range.

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

- **The sketch estimate is an upper bound, never an underestimate.** Earlier it
  stored only *from-pivot* distances and estimated `d(p → u) + d(p → v)`, which
  is a real `u → v` walk only on undirected graphs; on directed input it
  returned finite distances for pairs with no path at all. It now stores
  `d(u → p)` alongside `d(p → v)` so both legs run in the direction they point.

- **Single-run timings.** Each cell is one measurement, unrepeated, so the
  sub-millisecond entries are close to timer noise. The order-of-magnitude gaps
  are solid; small differences are not.

## Requirements

Python 3.7+ and `matplotlib` (for the benchmark plot only — `graph.py` and
everything in `algorithms/` are pure standard library).

```bash
pip install -r requirements.txt
```
