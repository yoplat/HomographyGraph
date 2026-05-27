"""
benchmark_noise_topology.py — Joint noise × topology benchmark.

Sweeps noise levels (σ) across multiple graph topologies and records the
median angular error of every synchronization method.  The result is a 2-D
grid of line plots — one subplot per topology, all methods on each — plus a
summary heatmap that shows which method wins at each (topology, σ) cell.

Usage
-----
    python benchmark_noise_topology.py               # run all topologies
    python benchmark_noise_topology.py --trials 10   # faster smoke test
    python benchmark_noise_topology.py --no-heatmap  # skip the summary plot

Output files
------------
    exp_noise_topology_grid.png    — grid of error-vs-noise line plots
    exp_noise_topology_heatmap.png — winner heatmap across the 2-D space
"""

import argparse
import sys
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Path fix so this file can be run directly from any working directory.
# ---------------------------------------------------------------------------
import os

sys.path.insert(0, os.path.dirname(__file__))

from graph import (
    Graph,
    build_synthetic_graph,
    build_linear_graph,
    build_circular_graph,
    build_grid_graph,
    build_multilane_graph,
    calculate_angular_error,
)

# ---------------------------------------------------------------------------
# Synchronization methods — plain only (outlier_density=0 throughout, so IRLS
# is not needed and would be wasted computation).
# ---------------------------------------------------------------------------

METHODS: Dict[str, callable] = {
    "Tree": lambda g: g.synchronize_tree(),
    "Sphere": lambda g: g.synchronize_iterative(avg_method="sphere", max_iters=100),
    "Euclidean": lambda g: g.synchronize_iterative(avg_method="euclidean", max_iters=100),
    "Direction": lambda g: g.synchronize_iterative(avg_method="direction", max_iters=100),
    "LSH": lambda g: g.synchronize_spectral(method="lsh"),
    "GSH": lambda g: g.synchronize_spectral(method="gsh"),
}

STYLE: Dict[str, dict] = {
    "Tree":      {"color": "black",  "marker": "s", "linestyle": "--"},
    "Sphere":    {"color": "red",    "marker": "o", "linestyle": "-"},
    "Euclidean": {"color": "blue",   "marker": "^", "linestyle": "-"},
    "Direction": {"color": "green",  "marker": "v", "linestyle": "-"},
    "LSH":       {"color": "purple", "marker": "D", "linestyle": "-."},
    "GSH":       {"color": "orange", "marker": "x", "linestyle": "-."},
}

# ---------------------------------------------------------------------------
# Topology configurations: representative subset for the 2-D sweep
# ---------------------------------------------------------------------------

TOPOLOGY_CONFIGS: List[Dict] = [
    {
        "label": "Linear (bw=1)",
        "topology": "linear",
        "extra": {"bandwidth": 1},
    },
    {
        "label": "Linear (bw=3)",
        "topology": "linear",
        "extra": {"bandwidth": 3},
    },
    {
        "label": "Circular (chord=3)",
        "topology": "circular",
        "extra": {"chord_step": 3},
    },
    {
        "label": "Grid (8-conn)",
        "topology": "grid",
        "extra": {"diagonal_edges": True},
    },
    {
        "label": "Multilane (3 lanes)",
        "topology": "multilane",
        "extra": {"num_lanes": 3, "bandwidth": 3},
    },
    {
        "label": "Random (E-R)",
        "topology": "random",
        "extra": {},
    },
]


# ---------------------------------------------------------------------------
# Graph builder dispatcher (mirrors _build_for_topology_config in benchmark.py)
# ---------------------------------------------------------------------------


def _build_graph(
    cfg: Dict,
    n: int,
    sigma: float,
    hole_density: float,
) -> Tuple[Graph, Dict[int, np.ndarray]]:
    topo = cfg["topology"]
    extra = cfg.get("extra", {})

    if topo == "random":
        return build_synthetic_graph(
            n=n, sigma=sigma, hole_density=hole_density
        )

    elif topo == "linear":
        return build_linear_graph(
            n=n,
            sigma=sigma,
            hole_density=hole_density,
            bandwidth=extra.get("bandwidth", 1),
        )

    elif topo == "circular":
        return build_circular_graph(
            n=n,
            sigma=sigma,
            hole_density=hole_density,
            chord_step=extra.get("chord_step", 1),
        )

    elif topo == "grid":
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        return build_grid_graph(
            rows=rows,
            cols=cols,
            sigma=sigma,
            hole_density=hole_density,
            diagonal_edges=extra.get("diagonal_edges", False),
        )

    elif topo == "multilane":
        graph, gt, _ = build_multilane_graph(
            n=n,
            sigma=sigma,
            hole_density=hole_density,
            num_lanes=extra.get("num_lanes", 3),
            bandwidth=extra.get("bandwidth", 1),
            cross_connect=True,
            diagonal_cross_density=0.15,
        )
        return graph, gt

    else:
        raise ValueError(f"Unknown topology '{topo}'.")


# ---------------------------------------------------------------------------
# Core 2-D sweep
# ---------------------------------------------------------------------------


def experiment_noise_vs_topology(
    noise_levels: List[float],
    n: int = 25,
    hole_density: float = 0.3,
    n_trials: int = 20,
    configs: List[Dict] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Sweep noise levels across all topology configurations.

    For every (topology, σ) pair run ``n_trials`` independent trials and
    record the median angular error of each method.

    Parameters
    ----------
    noise_levels : list of float
        Noise standard deviations to evaluate.
    n : int
        Number of nodes per graph.
    hole_density : float
        Fraction of edges to drop (same across all topologies).
    n_trials : int
        Independent trials per (topology, σ) cell.
    configs : list of dict, optional
        Topology configurations; defaults to TOPOLOGY_CONFIGS.

    Returns
    -------
    results : dict
        Nested as  results[topology_label][method_label] = [median_error, ...]
        where the list has one entry per noise level.
    """
    if configs is None:
        configs = TOPOLOGY_CONFIGS

    print("\n" + "=" * 68)
    print("BENCHMARK: Noise level vs. Graph topology")
    print(
        f"  n={n}, ρ={hole_density}, σ∈{noise_levels}, "
        f"trials={n_trials}, topologies={len(configs)}"
    )
    print("=" * 68)

    # results[topo_label][method_label] → list of median errors (one per σ)
    results: Dict[str, Dict[str, List[float]]] = {
        cfg["label"]: {m: [] for m in METHODS} for cfg in configs
    }

    total_cells = len(configs) * len(noise_levels)
    cell = 0

    for cfg in configs:
        topo_label = cfg["label"]
        print(f"\n  Topology: {topo_label}")

        for sigma in noise_levels:
            cell += 1
            trial_errors: Dict[str, List[float]] = {m: [] for m in METHODS}

            for _ in range(n_trials):
                base_graph, ground_truth = _build_graph(
                    cfg, n=n, sigma=sigma, hole_density=hole_density
                )

                for method_label, sync_fn in METHODS.items():
                    g = base_graph.copy()
                    g.normalize()
                    sync_fn(g)
                    err = calculate_angular_error(g, ground_truth)
                    trial_errors[method_label].append(err)

            for m in METHODS:
                med = float(np.median(trial_errors[m]))
                results[topo_label][m].append(med)

            best_method = min(METHODS, key=lambda m: results[topo_label][m][-1])
            best_err = results[topo_label][best_method][-1]
            print(
                f"    σ={sigma:.3f} [{cell:3d}/{total_cells}]  "
                + "  ".join(
                    f"{m}: {results[topo_label][m][-1]:6.2f}°" for m in METHODS
                )
                + f"   ← best: {best_method}"
            )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_grid(
    results: Dict[str, Dict[str, List[float]]],
    noise_levels: List[float],
    save_path: str = None,
) -> None:
    """
    Grid of line plots: one subplot per topology, error vs noise for all methods.

    Layout adapts to the number of topologies (2 columns, ⌈n/2⌉ rows).
    """
    topo_labels = list(results.keys())
    n_topos = len(topo_labels)
    ncols = 2
    nrows = (n_topos + 1) // 2

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 5 * nrows), squeeze=False
    )

    for idx, topo_label in enumerate(topo_labels):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        for method in METHODS:
            s = STYLE.get(method, {})
            ax.plot(
                noise_levels, results[topo_label][method], label=method, **s
            )

        ax.set_title(topo_label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Noise σ")
        ax.set_ylabel("Error (degrees)")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Hide any unused axes in the last row.
    for idx in range(n_topos, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.suptitle(
        "Error vs Noise — across graph topologies\n"
        "(median angular error, degrees; lower is better)",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Grid plot saved to {save_path}")
    plt.show()


def plot_heatmap(
    results: Dict[str, Dict[str, List[float]]],
    noise_levels: List[float],
    save_path: str = None,
) -> None:
    """
    Summary heatmap: topology (rows) × noise level (columns).

    Each cell shows the best median error achieved by any method, and its
    background colour encodes the error magnitude.  The winning method name
    is annotated in each cell.
    """
    topo_labels = list(results.keys())
    method_labels = list(METHODS.keys())
    n_topos = len(topo_labels)
    n_sigma = len(noise_levels)

    # Best error and winning method for each (topo, σ) cell.
    best_errors = np.zeros((n_topos, n_sigma))
    best_methods = [[""] * n_sigma for _ in range(n_topos)]

    for i, topo in enumerate(topo_labels):
        for j in range(n_sigma):
            errs = {m: results[topo][m][j] for m in method_labels}
            winner = min(errs, key=errs.get)
            best_errors[i, j] = errs[winner]
            best_methods[i][j] = winner

    fig, ax = plt.subplots(
        figsize=(max(10, n_sigma * 1.4), max(4, n_topos * 0.9))
    )

    im = ax.imshow(
        best_errors, aspect="auto", cmap="YlOrRd", interpolation="nearest"
    )
    plt.colorbar(im, ax=ax, label="Best error (degrees)")

    ax.set_xticks(range(n_sigma))
    ax.set_xticklabels(
        [f"{s:.3f}" for s in noise_levels], rotation=45, ha="right"
    )
    ax.set_yticks(range(n_topos))
    ax.set_yticklabels(topo_labels)
    ax.set_xlabel("Noise σ")
    ax.set_title(
        "Best achievable error per (topology, noise) cell\n"
        "— cell text = winning method —",
        fontsize=11,
        fontweight="bold",
    )

    # Annotate each cell with the winning method (short name).
    for i in range(n_topos):
        for j in range(n_sigma):
            method_name = best_methods[i][j]
            # Shorten for readability inside tight cells.
            short = method_name[:3]
            brightness = best_errors[i, j] / (best_errors.max() + 1e-9)
            text_color = "white" if brightness > 0.6 else "black"
            ax.text(
                j,
                i,
                f"{short}\n{best_errors[i, j]:.1f}°",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
                fontweight="bold",
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Heatmap saved to {save_path}")
    plt.show()


def plot_method_comparison(
    results: Dict[str, Dict[str, List[float]]],
    noise_levels: List[float],
    save_path: str = None,
) -> None:
    """
    One subplot per method: error vs noise, one line per topology.

    Complementary to plot_grid — useful for comparing how a single method
    behaves across different topologies.
    """
    method_labels = list(METHODS.keys())
    topo_labels = list(results.keys())
    n_methods = len(method_labels)

    # Auto-pick a colour palette for topologies.
    cmap = plt.get_cmap("tab10")
    topo_colors = {
        t: cmap(k / max(len(topo_labels) - 1, 1))
        for k, t in enumerate(topo_labels)
    }
    topo_markers = ["o", "s", "^", "v", "D", "x", "P", "*"]

    ncols = 3
    nrows = (n_methods + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(15, 5 * nrows), squeeze=False
    )

    for idx, method in enumerate(method_labels):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        for k, topo in enumerate(topo_labels):
            ax.plot(
                noise_levels,
                results[topo][method],
                label=topo,
                color=topo_colors[topo],
                marker=topo_markers[k % len(topo_markers)],
                linestyle="-",
                linewidth=1.5,
                markersize=5,
            )

        ax.set_title(method, fontsize=11, fontweight="bold")
        ax.set_xlabel("Noise σ")
        ax.set_ylabel("Error (degrees)")
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)

    for idx in range(n_methods, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.suptitle(
        "Per-method error vs noise across topologies\n"
        "(each line = one topology; shows topology sensitivity per method)",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Method-comparison plot saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark noise sensitivity across graph topologies."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        metavar="N",
        help="Independent trials per (topology, σ) cell (default: 20).",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=25,
        metavar="N",
        help="Number of graph nodes (default: 25).",
    )
    parser.add_argument(
        "--hole-density",
        type=float,
        default=0.3,
        metavar="RHO",
        help="Fraction of edges to drop (default: 0.3).",
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Skip the winner-heatmap figure.",
    )
    parser.add_argument(
        "--no-method-plot",
        action="store_true",
        help="Skip the per-method topology-comparison figure.",
    )
    args = parser.parse_args()

    np.random.seed(42)

    noise_levels = [0.02, 0.04, 0.10, 0.15, 0.20, 0.25]

    results = experiment_noise_vs_topology(
        noise_levels=noise_levels,
        n=args.nodes,
        hole_density=args.hole_density,
        n_trials=args.trials,
        configs=TOPOLOGY_CONFIGS,
    )

    plot_grid(
        results,
        noise_levels,
        save_path="exp_noise_topology_grid.png",
    )

    if not args.no_method_plot:
        plot_method_comparison(
            results,
            noise_levels,
            save_path="exp_noise_topology_methods.png",
        )

    if not args.no_heatmap:
        plot_heatmap(
            results,
            noise_levels,
            save_path="exp_noise_topology_heatmap.png",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
