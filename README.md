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
- **Wins on:** dense graphs and small-to-medium n — measured 3.4× faster than
  repeated Dijkstra at n = 500 dense. Its cost is fixed by n alone, so density
  is free; that same property makes it wasteful on sparse graphs, where it pays
  for n² pairs that mostly have no edge.
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
- **Wins on:** sparse graphs with non-negative weights, by 5.7–6.9× over
  Floyd-Warshall in the measured range. When m ≪ n², n heap searches beat
  Floyd-Warshall's fixed n³ comfortably — but on dense input both are O(n³) and
  Floyd-Warshall's smaller constant wins. See
  [finding 1](#1-the-classic-regime-split-holds-cleanly--floyd-warshall-for-dense-dijkstra-for-sparse).
- Textbook Dijkstra requires non-negative weights, but **this implementation
  keeps no settled set**, so it re-expands improved vertices and stays correct
  with negative edges absent a negative cycle. See
  [finding 5](#5-this-dijkstra-is-not-textbook-dijkstra) — it is a real
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
- Inherits the Dijkstra implementation's tolerance for negative edges on graphs
  without negative cycles (see
  [finding 5](#5-this-dijkstra-is-not-textbook-dijkstra)).
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
[finding 5](#5-this-dijkstra-is-not-textbook-dijkstra).

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

Measured with `python benchmark.py --full --seed 1` on Python 3.11, in seconds.
The full sweep took 1 h 53 m; every cell below is measured, none extrapolated.

| Algorithm | 100 sparse | 100 dense | 100 negative | 500 sparse | 500 dense | 500 negative |
|---|---|---|---|---|---|---|
| Floyd-Warshall | 0.068 | **0.056** | 0.067 | 7.922 | **7.664** | 10.401 |
| Johnson's | 0.029 | 4.354 | **0.008** | 8.078 | 3318.876 | 2.467 |
| Repeated Dijkstra | **0.012** | 0.090 | **0.001** | **1.147** | 25.724 | **0.290** |
| Repeated Bellman-Ford | 0.319 | 6.738 | 0.247 | 252.290 | 5553.765 | 185.007 |
| Sketch approximation | 0.002 | 0.014 | 0.001 | 0.031 | 0.729 | 0.070 |

Bold marks the fastest *exact* algorithm per column; the sketch is excluded from
that comparison because it only measures preprocessing (see
[Notes and caveats](#notes-and-caveats)).

![Execution time of APSP algorithms across the six full-sweep configurations](docs/benchmark-full-sweep.png)

Log scale — the algorithms span five orders of magnitude. Floyd-Warshall is
the flat line; every other series spikes on the dense configurations.

## Findings

### 1. The classic regime split holds cleanly — Floyd-Warshall for dense, Dijkstra for sparse

Each algorithm wins exactly where the textbook says it should, and the margins
are wide enough to be unambiguous:

| n = 500 | Floyd-Warshall | Repeated Dijkstra | Winner |
|---|---|---|---|
| sparse (12,358 entries) | 7.922 | 1.147 | Dijkstra, 6.9× |
| dense (249,500 entries) | 7.664 | 25.724 | **Floyd-Warshall, 3.4×** |

The mechanism is visible in the asymptotics. Repeated Dijkstra is
O(n·(m + n log n)), so its cost rides on m; Floyd-Warshall is O(n³) and ignores
m entirely. On the sparse graph m ≈ 25n and Dijkstra wins comfortably. On the
dense graph m ≈ n², both become O(n³), and Floyd-Warshall's constant factor is
smaller — a flat triple loop over a preallocated matrix beats n heap-driven
searches with their pushes, pops and staleness checks.

The crossover is not at 500 either: the quick sweep shows Floyd-Warshall ahead
in the dense regime at n = 50, 100 and 150 as well, by a consistent ~1.6×. There
is no size in the tested range where repeated Dijkstra wins on dense input.

**This corrects an earlier result from this repo.** Before the generator fix,
these same measurements showed the opposite — repeated Dijkstra beating
Floyd-Warshall 4.6× on the 500-node dense graph, which looked like a genuine
inversion of the textbook advice. It was an artifact. The old generators emitted
edges only for `u < v`, so every graph was a DAG in which a search from vertex
`u` could only reach vertices numbered above it. Each Dijkstra run explored half
the graph on average, and runs from high-numbered vertices were nearly free,
while Floyd-Warshall's n³ loop paid full price regardless. Making the graphs
genuinely undirected removed the discount, and Dijkstra's 500-node dense time
went from 2.198 s to 25.724 s — an 11.7× jump on a graph with only twice the
edges.

The methodological lesson is the useful part: a bug in input generation produced
a plausible, quotable, and completely wrong headline result. It survived because
it was *interesting* — an inverted textbook claim invites explanation rather than
suspicion, and the explanation drafted for it (interpreted-Python constant
factors) was perfectly reasonable. Nothing about the number itself looked wrong.

### 2. Floyd-Warshall's runtime is independent of the edge count

The cleanest empirical result in the sweep, and the one that survived the
generator fix unchanged. Between 500 sparse and 500 dense the edge count rises
**20.2×** (12,358 → 249,500 entries), and Floyd-Warshall goes from 7.922 s to
7.664 s — it gets 3% *faster*, which is noise around a flat line.

That is O(n³) with no m term, visible directly. The contrast with Bellman-Ford
across the identical pair of graphs makes the point sharper:

| 500 sparse → 500 dense | Time | Growth |
|---|---|---|
| Edge entries | 12,358 → 249,500 | 20.2× |
| Floyd-Warshall | 7.922 → 7.664 | **0.97×** |
| Repeated Bellman-Ford | 252.290 → 5553.765 | **22.0×** |

Bellman-Ford's 22.0× against an edge growth of 20.2× tracks its O(n²·m) bound
almost exactly. Floyd-Warshall is the only algorithm here whose cost is
predictable from the vertex count alone — which is precisely what makes it
reliable on dense input and wasteful on sparse input.

### 3. Johnson's bottleneck is the graph representation, not the algorithm

Johnson's takes **3318.876 s** on the 500-node dense graph — 55 minutes, and
**129× slower** than the repeated Dijkstra it is built on top of. Since Johnson's
*is* one Bellman-Ford pass plus n Dijkstra runs, and those n runs cost 25.724 s
on their own, essentially the entire runtime is overhead.

The overhead is `Graph.update_edge_weight`, which scans the whole edge list on
every call. Reweighting all m edges therefore costs **O(m²)**, or 6.2 × 10¹⁰
operations at 249,500 entries. The negative-weight configuration confirms the
diagnosis: it has only 6,216 entries, so m² predicts a 1,611× reduction, and the
measured drop is 1,345× (3318.876 s → 2.467 s). That is the right order for a
prediction spanning three decades.

Indexing the edge list by `(u, v)` would drop this to O(m) and bring Johnson's
back in line with its O(n² log n + n·m) bound. Left as-is deliberately — the
brief was to preserve the original implementations.

### 4. The negative-weight regime is the only place Johnson's design pays off

On the negative configurations Johnson's beats repeated Bellman-Ford by **75×**
at n = 500 (2.467 s vs 185.007 s) and 31× at n = 100. That is exactly the
comparison the algorithm was designed to win, and the original non-negative-only
benchmark could never show it.

The reason is structural: Johnson's does *one* O(n·m) Bellman-Ford pass to
compute potentials and then n cheap Dijkstra runs, where repeated Bellman-Ford
does n expensive Bellman-Ford runs. The saving grows with n, which is why the
margin more than doubles from n = 100 to n = 500.

This is also why the dense column should not be read as "Johnson's is slow."
There it pays for a capability the input does not require: with non-negative
weights Dijkstra alone is always legal, so reweighting is pure loss. Measure an
algorithm outside the regime it was designed for and it will lose to something
simpler — a statement about the benchmark, not the algorithm.

### 5. This "Dijkstra" is not textbook Dijkstra

Discovered while adding the negative-weight configurations. The implementation
keeps **no settled/visited set** — the `d > dist[u]` test rejects only *stale*
heap entries, not vertices that have already been expanded. A vertex whose
distance later improves is pushed and expanded again, making this
label-correcting rather than true Dijkstra.

The consequence is that it returns **correct** distances on graphs with negative
edges, provided there is no negative cycle. Verified against Floyd-Warshall
across 25 random negative-weight graphs (zero disagreements) and on the standard
counterexample where textbook Dijkstra fails:

```
0 →1 (w=2), 0 →2 (w=5), 1 →2 (w=-4)     true d(0,2) = -2
settled-set Dijkstra: 5 (wrong — finalises vertex 2 before the shortcut)
this implementation: -2 (correct — re-expands vertex 2)
```

So the benchmark runs all five algorithms on the negative configurations rather
than skipping the two built on Dijkstra. The robustness is not free: the
O(m log n) bound assumes one expansion per vertex, and negative edges break that
assumption — re-expansions are cheap on these acyclic graphs but exponential in
the worst case.


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
