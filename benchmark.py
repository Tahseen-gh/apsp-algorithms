"""Time the five APSP algorithms across sparse and dense graphs, then plot.

Run from the command line:

    python benchmark.py                        # quick default sweep, ~2 minutes
    python benchmark.py --full                 # the 100/500 sweep, hours
    python benchmark.py --sizes 100 200 300    # any sizes you like
    python benchmark.py -a dijkstra -a johnson # only some algorithms
    python benchmark.py --no-negative          # skip the negative configs
    python benchmark.py -o results/run.png     # choose the output path

Every sweep covers three regimes per size: sparse (undirected, p = 0.05), dense
(undirected, complete), and negative (directed, sparse, weights from -5 to 10).
The negative configurations exist so that Johnson's algorithm and Bellman-Ford
are exercised on the input class they were designed for, rather than only on
inputs where a plain Dijkstra would do.

The default sizes are deliberately small so the whole thing finishes in a couple
of minutes. Cost grows steeply with n -- Bellman-Ford is O(n^2 * m) and the
dense generator makes m itself quadratic, so the 500-node dense cell alone runs
for over an hour. Use ``--full`` when you want that, and expect to wait.

The plot is always written to a file; nothing requires a display, so this runs
fine over SSH or in CI.
"""

import argparse
import os
import random
import sys
import time

import matplotlib

# Select a non-interactive backend before pyplot is imported, so that the
# script runs on headless machines.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

from algorithms import (  # noqa: E402  (kept below the backend selection)
    all_pairs_bellman_ford,
    all_pairs_dijkstra,
    floyd_warshall,
    johnson,
    sketch_approx_apsp,
)
from graph import (  # noqa: E402
    generate_dense_graph,
    generate_negative_weight_graph,
    generate_sparse_graph,
)

#: Maps the CLI slug for each algorithm to its display name and callable. The
#: order here is the order runs happen and the order series appear in the plot.
#:
#: All five run on every configuration, including the negative-weight ones.
#: That is safe only because of an implementation detail: this Dijkstra keeps no
#: settled set, so it re-expands improved vertices and stays correct without
#: negative cycles. A textbook Dijkstra would be invalid there -- see the
#: README's findings section.
ALGORITHMS = [
    ("floyd-warshall", "Floyd-Warshall", floyd_warshall),
    ("johnson", "Johnson's Algorithm", johnson),
    ("dijkstra", "Repeated Dijkstra", all_pairs_dijkstra),
    ("bellman-ford", "Bellman-Ford", all_pairs_bellman_ford),
    ("sketch", "Sketch Approximate", sketch_approx_apsp),
]

#: Sizes for the default run. Chosen so a bare ``python benchmark.py`` finishes
#: in a couple of minutes while still giving three points per regime -- enough
#: to see the growth curves rather than isolated measurements.
DEFAULT_SIZES = [50, 100, 150]

#: Sizes for ``--full``: the original coursework sweep. Takes hours, almost all
#: of it in the 500-node dense cell, where Bellman-Ford's O(n^2 * m) and
#: Johnson's O(m^2) reweighting both hit 249,500 edge entries.
FULL_SIZES = [100, 500]

DEFAULT_OUTPUT = os.path.join("results", "apsp_benchmark.png")


def measure_execution_time(algorithm, graph):
    """Return the wall-clock seconds taken to run ``algorithm`` on ``graph``.

    Uses a single untimed-warmup-free run, matching the original coursework
    measurement. The result is discarded -- only the elapsed time matters.

    Note that :func:`~algorithms.sketch.sketch_approx_apsp` returns a query
    function rather than a distance matrix, so its measurement covers pivot
    preprocessing only, not the cost of answering queries.
    """
    start_time = time.time()
    algorithm(graph)
    return time.time() - start_time


def build_configurations(sizes, edge_probability, include_negative=True):
    """Build the configurations to benchmark: sparse, dense, negative per size.

    The sparse and dense graphs are undirected with weights from 1 to 10. The
    negative configuration is sparse and directed, which is what keeps it free
    of negative cycles -- see
    :func:`~graph.generate_negative_weight_graph` for why that matters.

    Args:
        sizes: Vertex counts to test, in order.
        edge_probability: Edge probability for the sparse graphs.
        include_negative: Whether to add the negative-weight configuration.

    Returns:
        A list of ``(label, graph, has_negative_weights)`` tuples.
    """
    configurations = []
    for size in sizes:
        configurations.append(
            ("%d nodes (Sparse)" % size,
             generate_sparse_graph(size, edge_probability), False)
        )
        configurations.append(
            ("%d nodes (Dense)" % size, generate_dense_graph(size), False)
        )
        if include_negative:
            configurations.append(
                ("%d nodes (Negative)" % size,
                 generate_negative_weight_graph(size, edge_probability), True)
            )
    return configurations


def run_benchmark(configurations, selected):
    """Time every selected algorithm against every configuration.

    Each configuration's graph object is shared by all algorithms in the sweep,
    as in the original notebook. That matters because
    :func:`~algorithms.johnson.johnson` reweights its input in place, so the
    algorithms that run after it see the reweighted graph. Shortest-path
    *structure* is preserved by the reweighting, so this does not invalidate the
    timings, but it is worth knowing when interpreting them.

    Args:
        configurations: ``(label, graph, has_negative)`` triples from
            :func:`build_configurations`.
        selected: ``(slug, display_name, callable)`` entries to time.

    Returns:
        A dict mapping display name to a list of durations, one per
        configuration and in configuration order.
    """
    execution_times = {name: [] for _, name, _ in selected}

    for label, graph, _ in configurations:
        print("\n%s -- %d edges" % (label, len(graph.edges)))
        for _, name, algorithm in selected:
            sys.stdout.write("  %-22s " % name)
            sys.stdout.flush()
            elapsed = measure_execution_time(algorithm, graph)
            execution_times[name].append(elapsed)
            print("%8.3f s" % elapsed)

    return execution_times


def plot_execution_times(execution_times, labels, output_path):
    """Plot each algorithm's timings across configurations and save to disk.

    Args:
        execution_times: Mapping of algorithm name to per-configuration seconds.
        labels: Configuration labels for the x axis, in the same order.
        output_path: Where to write the figure. Parent directories are created
            as needed; the extension picks the format (.png, .pdf, .svg, ...).

    Returns:
        The path written to.
    """
    algorithms = list(execution_times.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    for algorithm in algorithms:
        ax.plot(labels, execution_times[algorithm], marker="o", label=algorithm)

    ax.set_xlabel("Graph Configuration")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Execution Time of APSP Algorithms")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def print_summary(execution_times, labels):
    """Print the collected timings as a plain-text table."""
    name_width = max(len(name) for name in execution_times)
    column_width = max([len(label) for label in labels] + [10])

    print("\nResults (seconds)")
    header = "%-*s" % (name_width, "Algorithm")
    for label in labels:
        header += "  %*s" % (column_width, label)
    print(header)
    print("-" * len(header))

    for name, timings in execution_times.items():
        row = "%-*s" % (name_width, name)
        for value in timings:
            row += "  %*.3f" % (column_width, value)
        print(row)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark five all-pairs shortest path algorithms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help="path to write the plot to",
    )
    sizes_group = parser.add_mutually_exclusive_group()
    sizes_group.add_argument(
        "-s",
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        metavar="N",
        help="vertex counts to benchmark",
    )
    sizes_group.add_argument(
        "--full",
        action="store_true",
        help="run the full %s sweep instead; takes hours, not minutes"
        % " and ".join(str(s) for s in FULL_SIZES),
    )
    parser.add_argument(
        "-p",
        "--edge-probability",
        type=float,
        default=0.05,
        help="edge probability for the sparse graphs",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        dest="algorithms",
        action="append",
        choices=[slug for slug, _, _ in ALGORITHMS],
        metavar="NAME",
        help="run only this algorithm (repeatable); choices: "
        + ", ".join(slug for slug, _, _ in ALGORITHMS),
    )
    parser.add_argument(
        "--no-negative",
        dest="include_negative",
        action="store_false",
        help="skip the negative-weight configurations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed the RNG for reproducible graphs (default: unseeded)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the benchmark sweep and write the plot. Returns a process exit code."""
    args = parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

    selected = ALGORITHMS
    if args.algorithms:
        chosen = set(args.algorithms)
        selected = [entry for entry in ALGORITHMS if entry[0] in chosen]

    sizes = FULL_SIZES if args.full else args.sizes
    if args.full:
        print("Running the full %s sweep -- this takes hours, not minutes."
              % " and ".join(str(s) for s in FULL_SIZES))

    configurations = build_configurations(
        sizes, args.edge_probability, args.include_negative
    )
    labels = [label for label, _, _ in configurations]

    execution_times = run_benchmark(configurations, selected)

    print_summary(execution_times, labels)

    output_path = plot_execution_times(execution_times, labels, args.output)
    print("\nPlot written to %s" % output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
