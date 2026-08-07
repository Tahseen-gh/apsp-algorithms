"""Time the five APSP algorithms across sparse and dense graphs, then plot.

Run from the command line:

    python benchmark.py                        # full default sweep
    python benchmark.py --sizes 100            # just the 100-vertex configs
    python benchmark.py -a dijkstra -a johnson # only some algorithms
    python benchmark.py -o results/run.png     # choose the output path

The default sweep reproduces the coursework setup: 100 and 500 vertices, each
in a sparse (p = 0.05) and a dense (complete) regime, giving four
configurations per algorithm. See the README for the run times to expect --
the 500-vertex dense configuration is very slow for the O(n^2 * m) algorithms.

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
from graph import generate_dense_graph, generate_sparse_graph  # noqa: E402

#: Maps the CLI slug for each algorithm to its display name and callable. The
#: order here is the order runs happen and the order series appear in the plot.
ALGORITHMS = [
    ("floyd-warshall", "Floyd-Warshall", floyd_warshall),
    ("johnson", "Johnson's Algorithm", johnson),
    ("dijkstra", "Repeated Dijkstra", all_pairs_dijkstra),
    ("bellman-ford", "Bellman-Ford", all_pairs_bellman_ford),
    ("sketch", "Sketch Approximate", sketch_approx_apsp),
]

DEFAULT_SIZES = [100, 500]
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


def build_configurations(sizes, edge_probability):
    """Build the (label, graph) pairs to benchmark, sparse then dense per size.

    Args:
        sizes: Vertex counts to test, in order.
        edge_probability: Edge probability for the sparse graphs.

    Returns:
        A list of ``(label, graph)`` tuples ordered sparse-then-dense within
        each size, matching the original notebook's sweep.
    """
    configurations = []
    for size in sizes:
        configurations.append(
            ("%d nodes (Sparse)" % size, generate_sparse_graph(size, edge_probability))
        )
        configurations.append(
            ("%d nodes (Dense)" % size, generate_dense_graph(size))
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
        configurations: ``(label, graph)`` pairs from :func:`build_configurations`.
        selected: ``(slug, display_name, callable)`` triples to time.

    Returns:
        A dict mapping display name to a list of durations, one per
        configuration and in configuration order.
    """
    execution_times = {name: [] for _, name, _ in selected}

    for label, graph in configurations:
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
    parser.add_argument(
        "-s",
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        metavar="N",
        help="vertex counts to benchmark",
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

    configurations = build_configurations(args.sizes, args.edge_probability)
    labels = [label for label, _ in configurations]

    execution_times = run_benchmark(configurations, selected)

    print_summary(execution_times, labels)

    output_path = plot_execution_times(execution_times, labels, args.output)
    print("\nPlot written to %s" % output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
