"""
benchmark.py — Synthetic comparison of synchronization methods for 3×3 homographies.

This script runs multiple synchronization algorithms on synthetic graphs
and compares their accuracy and execution time.

The comparison covers two families of methods:

  • **Iterative** [1]: Madhavan, Fusiello & Arrigoni (sphere / euclidean / direction)
  • **Spectral** [2]: Schroeder, Bartoli, Georgel & Navab (LSH / GSH)
  • **Spanning tree** baseline

Four experimental axes are swept:
  1. Number of nodes (graph size)           — ``experiment_vary_nodes``
  2. Noise level (measurement perturbation) — ``experiment_vary_noise``
  3. Hole density (fraction of missing edges) — ``experiment_vary_holes``
  4. Graph topology (linear / circular / grid / multilane / random)
     — ``experiment_vary_topology``

Error metric
------------
The synchronization solution is defined up to a global transformation C.
We align the estimate to the ground truth by solving for C using the
direction-based average, then measure the angular distance (in degrees)
between each estimated and true vertex (Eq. 12 of [1]).

References
----------
[1] R. Madhavan, A. Fusiello, F. Arrigoni, "Synchronization of Projective
    Transformations", 2024.
[2] P. Schroeder, A. Bartoli, P. Georgel, N. Navab, "Closed-Form Solutions
    to Multiple-View Homography Estimation", IEEE WACV 2011.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple

from graph import (
    Graph,
    generate_random_sl3,
    add_noise,
    add_outlier_edge,
    calculate_angular_error,
    build_synthetic_graph,
    build_linear_graph,
    build_circular_graph,
    build_grid_graph,
    build_multilane_graph,
)

# Reproducibility.
SEED = 42
np.random.seed(SEED)


# ===================================================================
# Registry of synchronization methods to benchmark
# ===================================================================

# Each entry is (label → callable).  The callable takes a Graph, runs
# synchronization in-place, and returns nothing.
METHODS = {
    "Tree": lambda g: g.synchronize_tree(),
    "Sphere": lambda g: g.synchronize_iterative(
        avg_method="sphere", max_iters=100
    ),
    "Euclidean": lambda g: g.synchronize_iterative(
        avg_method="euclidean", max_iters=100
    ),
    "Direction": lambda g: g.synchronize_iterative(
        avg_method="direction", max_iters=100
    ),
    "LSH": lambda g: g.synchronize_spectral(method="lsh"),
    "GSH": lambda g: g.synchronize_spectral(method="gsh"),
}


# ===================================================================
# Single-trial runner
# ===================================================================


def run_single_trial(
    n: int,
    sigma: float,
    hole_density: float,
    outlier_density: float,
    methods: Dict[str, callable],
    topology: str = "multilane",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Run one trial: build a synthetic graph, run all methods, return errors
    and execution times.

    Parameters
    ----------
    n : int
        Number of nodes.
    sigma : float
        Noise standard deviation.
    hole_density : float
        Fraction of removable edges to drop.
    outlier_density : float
        Fraction of remaining edges replaced by outliers.
    methods : dict
        {label → sync callable}.
    topology : str
        Which graph builder to use.  One of ``"random"``, ``"linear"``,
        ``"circular"``, ``"grid"``, ``"multilane"`` (default).

    Returns
    -------
    errors : dict[str, float]
        Angular error (degrees) for each method label.
    times : dict[str, float]
        Execution time (seconds) for each method label.
    """
    # ── Build graph according to requested topology ──────────────────────
    if topology == "random":
        base_graph, ground_truth = build_synthetic_graph(
            n=n, sigma=sigma, hole_density=hole_density,
            outlier_density=outlier_density,
        )
    elif topology == "linear":
        base_graph, ground_truth = build_linear_graph(
            n=n, sigma=sigma, hole_density=hole_density,
            outlier_density=outlier_density, bandwidth=3,
        )
    elif topology == "circular":
        base_graph, ground_truth = build_circular_graph(
            n=n, sigma=sigma, hole_density=hole_density,
            outlier_density=outlier_density, chord_step=3,
        )
    elif topology == "grid":
        # Choose the most square-like layout for n nodes.
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        base_graph, ground_truth = build_grid_graph(
            rows=rows, cols=cols, sigma=sigma,
            hole_density=hole_density, outlier_density=outlier_density,
            diagonal_edges=True,
        )
    elif topology == "multilane":
        base_graph, ground_truth, _ = build_multilane_graph(
            n=n, num_lanes=3, sigma=sigma,
            hole_density=hole_density, outlier_density=outlier_density,
            bandwidth=3, cross_connect=True, diagonal_cross_density=0.15,
        )
    else:
        raise ValueError(f"Unknown topology '{topology}'.")

    # ── Run every method on an independent copy ──────────────────────────
    errors = {}
    times = {}

    for label, sync_fn in methods.items():
        # Work on a deep copy so each method starts from the same state.
        g = base_graph.copy()
        g.normalize()

        t0 = time.perf_counter()
        sync_fn(g)
        elapsed = time.perf_counter() - t0

        err = calculate_angular_error(g, ground_truth)
        errors[label] = err
        times[label] = elapsed

    return errors, times


# ===================================================================
# Experiment sweeps (original three axes)
# ===================================================================


def experiment_vary_nodes(
    node_counts: List[int],
    sigma: float = 0.05,
    hole_density: float = 0.5,
    outlier_density: float = 0.0,
    n_trials: int = 20,
    topology: str = "multilane",
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Experiment 1: vary the number of nodes, keeping noise and hole density fixed.

    Mirrors Fig. 2 of [1].  As the graph grows, more redundant measurements
    become available, and iterative / spectral methods should improve — unlike
    the spanning tree whose error accumulates with depth.

    Returns error_results and time_results, each mapping method labels to
    lists of median values (one per node count).
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Varying number of nodes")
    print(
        f"  topology={topology}, σ={sigma}, ρ={hole_density}, "
        f"γ={outlier_density}, trials={n_trials}"
    )
    print("=" * 60)

    error_results = {label: [] for label in METHODS}
    time_results = {label: [] for label in METHODS}

    for n in node_counts:
        trial_errors = {label: [] for label in METHODS}
        trial_times = {label: [] for label in METHODS}

        for t in range(n_trials):
            errs, tms = run_single_trial(
                n=n, sigma=sigma, hole_density=hole_density,
                outlier_density=outlier_density, methods=METHODS,
                topology=topology,
            )
            for label in METHODS:
                trial_errors[label].append(errs[label])
                trial_times[label].append(tms[label])

        for label in METHODS:
            med_err = np.median(trial_errors[label])
            med_time = np.median(trial_times[label])
            error_results[label].append(med_err)
            time_results[label].append(med_time)

        print(
            f"  n={n:3d} | "
            + " | ".join(
                f"{label}: {error_results[label][-1]:.2f}°"
                for label in METHODS
            )
        )

    return error_results, time_results


def experiment_vary_noise(
    noise_levels: List[float],
    n: int = 25,
    hole_density: float = 0.5,
    outlier_density: float = 0.0,
    n_trials: int = 20,
    topology: str = "multilane",
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Experiment 2: vary the noise level, keeping graph size and connectivity fixed.

    Mirrors Fig. 4 (left) of [1].  All methods degrade with increasing noise,
    but the iterative and spectral approaches should degrade more gracefully
    than the spanning tree.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Varying noise level")
    print(
        f"  topology={topology}, n={n}, ρ={hole_density}, "
        f"γ={outlier_density}, trials={n_trials}"
    )
    print("=" * 60)

    error_results = {label: [] for label in METHODS}
    time_results = {label: [] for label in METHODS}

    for sigma in noise_levels:
        trial_errors = {label: [] for label in METHODS}
        trial_times = {label: [] for label in METHODS}

        for t in range(n_trials):
            errs, tms = run_single_trial(
                n=n, sigma=sigma, hole_density=hole_density,
                outlier_density=outlier_density, methods=METHODS,
                topology=topology,
            )
            for label in METHODS:
                trial_errors[label].append(errs[label])
                trial_times[label].append(tms[label])

        for label in METHODS:
            med_err = np.median(trial_errors[label])
            med_time = np.median(trial_times[label])
            error_results[label].append(med_err)
            time_results[label].append(med_time)

        print(
            f"  σ={sigma:.3f} | "
            + " | ".join(
                f"{label}: {error_results[label][-1]:.2f}°"
                for label in METHODS
            )
        )

    return error_results, time_results


def experiment_vary_holes(
    hole_densities: List[float],
    n: int = 25,
    sigma: float = 0.05,
    outlier_density: float = 0.0,
    n_trials: int = 20,
    topology: str = "multilane",
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Experiment 3: vary the fraction of missing edges (hole density).

    Mirrors Fig. 3 of [1].  With more missing edges, less redundancy is
    available for error compensation, and all methods converge towards
    spanning-tree performance.  The spectral methods (LSH/GSH) handle
    missing data natively, which is one of their key advantages.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Varying hole density")
    print(
        f"  topology={topology}, n={n}, σ={sigma}, "
        f"γ={outlier_density}, trials={n_trials}"
    )
    print("=" * 60)

    error_results = {label: [] for label in METHODS}
    time_results = {label: [] for label in METHODS}

    for rho in hole_densities:
        trial_errors = {label: [] for label in METHODS}
        trial_times = {label: [] for label in METHODS}

        for t in range(n_trials):
            errs, tms = run_single_trial(
                n=n, sigma=sigma, hole_density=rho,
                outlier_density=outlier_density, methods=METHODS,
                topology=topology,
            )
            for label in METHODS:
                trial_errors[label].append(errs[label])
                trial_times[label].append(tms[label])

        for label in METHODS:
            med_err = np.median(trial_errors[label])
            med_time = np.median(trial_times[label])
            error_results[label].append(med_err)
            time_results[label].append(med_time)

        print(
            f"  ρ={rho:.2f} | "
            + " | ".join(
                f"{label}: {error_results[label][-1]:.2f}°"
                for label in METHODS
            )
        )

    return error_results, time_results


# ===================================================================
# Experiment 4: vary graph topology
# ===================================================================


# Each topology configuration specifies how to build the graph and
# a human-readable label for display.
TOPOLOGY_CONFIGS = [
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
        "label": "Circular (chord=1)",
        "topology": "circular",
        "extra": {"chord_step": 1},
    },
    {
        "label": "Circular (chord=3)",
        "topology": "circular",
        "extra": {"chord_step": 3},
    },
    {
        "label": "Grid (4-conn)",
        "topology": "grid",
        "extra": {"diagonal_edges": False},
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
        "label": "Random (Erdős–Rényi)",
        "topology": "random",
        "extra": {},
    },
]


def _build_for_topology_config(
    cfg: dict,
    n: int,
    sigma: float,
    hole_density: float,
    outlier_density: float,
) -> Tuple[Graph, Dict[int, np.ndarray]]:
    """
    Internal helper: build a graph from a topology configuration dict.

    Dispatches to the appropriate ``build_*`` function based on the
    ``"topology"`` key, forwarding any extra keyword arguments from
    ``cfg["extra"]``.
    """
    topo = cfg["topology"]
    extra = cfg.get("extra", {})

    if topo == "random":
        return build_synthetic_graph(
            n=n, sigma=sigma, hole_density=hole_density,
            outlier_density=outlier_density, **extra,
        )
    elif topo == "linear":
        return build_linear_graph(
            n=n, sigma=sigma, hole_density=hole_density,
            outlier_density=outlier_density,
            bandwidth=extra.get("bandwidth", 1),
        )
    elif topo == "circular":
        return build_circular_graph(
            n=n, sigma=sigma, hole_density=hole_density,
            outlier_density=outlier_density,
            chord_step=extra.get("chord_step", 1),
        )
    elif topo == "grid":
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        return build_grid_graph(
            rows=rows, cols=cols, sigma=sigma,
            hole_density=hole_density, outlier_density=outlier_density,
            diagonal_edges=extra.get("diagonal_edges", False),
        )
    elif topo == "multilane":
        graph, gt, _ = build_multilane_graph(
            n=n, sigma=sigma, hole_density=hole_density,
            outlier_density=outlier_density,
            num_lanes=extra.get("num_lanes", 3),
            bandwidth=extra.get("bandwidth", 1),
            cross_connect=True, diagonal_cross_density=0.15,
        )
        return graph, gt
    else:
        raise ValueError(f"Unknown topology '{topo}'.")


def experiment_vary_topology(
    n: int = 36,
    sigma: float = 0.05,
    hole_density: float = 0.2,
    outlier_density: float = 0.0,
    n_trials: int = 20,
    configs: List[dict] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Experiment 4: compare all methods across different graph topologies.

    For each topology configuration, runs ``n_trials`` independent trials
    and reports the median angular error and execution time per method.

    Parameters
    ----------
    n : int
        Number of nodes (36 gives a nice 6×6 grid and clean lane splits).
    sigma : float
        Noise standard deviation.
    hole_density : float
        Fraction of removable edges to drop.
    outlier_density : float
        Fraction of remaining edges replaced by outliers.
    n_trials : int
        Number of independent trials per configuration.
    configs : list of dict, optional
        Custom topology configurations.  Defaults to ``TOPOLOGY_CONFIGS``.

    Returns
    -------
    error_results : dict[topology_label → dict[method_label → median_error]]
    time_results  : dict[topology_label → dict[method_label → median_time]]
    """
    if configs is None:
        configs = TOPOLOGY_CONFIGS

    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Varying graph topology")
    print(
        f"  n={n}, σ={sigma}, ρ={hole_density}, "
        f"γ={outlier_density}, trials={n_trials}"
    )
    print("=" * 60)

    error_results: Dict[str, Dict[str, float]] = {}
    time_results: Dict[str, Dict[str, float]] = {}

    for cfg in configs:
        topo_label = cfg["label"]
        trial_errors = {label: [] for label in METHODS}
        trial_times = {label: [] for label in METHODS}

        for _ in range(n_trials):
            base_graph, ground_truth = _build_for_topology_config(
                cfg, n=n, sigma=sigma,
                hole_density=hole_density, outlier_density=outlier_density,
            )

            for method_label, sync_fn in METHODS.items():
                g = base_graph.copy()
                g.normalize()

                t0 = time.perf_counter()
                sync_fn(g)
                elapsed = time.perf_counter() - t0

                err = calculate_angular_error(g, ground_truth)
                trial_errors[method_label].append(err)
                trial_times[method_label].append(elapsed)

        # Collect medians.
        error_results[topo_label] = {
            m: np.median(trial_errors[m]) for m in METHODS
        }
        time_results[topo_label] = {
            m: np.median(trial_times[m]) for m in METHODS
        }

        row = " | ".join(
            f"{m}: {error_results[topo_label][m]:.2f}°" for m in METHODS
        )
        print(f"  {topo_label:25s} | {row}")

    return error_results, time_results


# ===================================================================
# Plotting utilities
# ===================================================================

# Visual style for each method family.
STYLE = {
    "Tree": {"color": "black", "marker": "s", "linestyle": "--"},
    "Sphere": {"color": "red", "marker": "o", "linestyle": "-"},
    "Euclidean": {"color": "blue", "marker": "^", "linestyle": "-"},
    "Direction": {"color": "green", "marker": "v", "linestyle": "-"},
    "LSH": {"color": "purple", "marker": "D", "linestyle": "-."},
    "GSH": {"color": "orange", "marker": "x", "linestyle": "-."},
}


def plot_results(
    x_values: List[float],
    error_data: Dict[str, List[float]],
    time_data: Dict[str, List[float]],
    xlabel: str,
    title: str,
    save_path: str = None,
) -> None:
    """
    Plot error (left) and execution time (right) side by side,
    using consistent colours and markers for each method.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Error plot ---
    for label, values in error_data.items():
        s = STYLE.get(label, {})
        ax1.plot(x_values, values, label=label, **s)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Error (degrees)")
    ax1.set_title(f"{title} — Error")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- Time plot ---
    for label, values in time_data.items():
        s = STYLE.get(label, {})
        ax2.plot(x_values, values, label=label, **s)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title(f"{title} — Execution time")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")
    plt.show()


def plot_topology_results(
    error_data: Dict[str, Dict[str, float]],
    time_data: Dict[str, Dict[str, float]],
    title: str = "Topology comparison",
    save_path: str = None,
) -> None:
    """
    Plot experiment 4 results as grouped bar charts.

    One bar group per topology, one bar per method.
    Left subplot: error.  Right subplot: execution time.
    """
    topo_labels = list(error_data.keys())
    method_labels = list(METHODS.keys())
    n_topos = len(topo_labels)
    n_methods = len(method_labels)

    x = np.arange(n_topos)
    bar_width = 0.8 / n_methods

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for i, method in enumerate(method_labels):
        errors = [error_data[t][method] for t in topo_labels]
        times = [time_data[t][method] for t in topo_labels]
        color = STYLE.get(method, {}).get("color", "gray")

        ax1.bar(x + i * bar_width, errors, bar_width,
                label=method, color=color, alpha=0.85)
        ax2.bar(x + i * bar_width, times, bar_width,
                label=method, color=color, alpha=0.85)

    for ax, ylabel, subtitle in [
        (ax1, "Error (degrees)", "Error"),
        (ax2, "Time (seconds)", "Execution time"),
    ]:
        ax.set_xlabel("Topology")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — {subtitle}")
        ax.set_xticks(x + bar_width * (n_methods - 1) / 2)
        ax.set_xticklabels(topo_labels, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")
    plt.show()


# ===================================================================
# Main
# ===================================================================


def main():
    """
    Run all four experiments and produce comparison plots.

    The experimental setup follows the structure in Sec. 5.2 of [1]:
      • Experiment 1: n ∈ {10, 20, 30, 50, 75, 100}, σ=0.05, ρ=0.5
      • Experiment 2: σ ∈ [0.01, 0.15], n=25, ρ=0.5
      • Experiment 3: ρ ∈ [0.0, 0.9], n=25, σ=0.05
      • Experiment 4: topology ∈ {linear, circular, grid, multilane, random}
    """
    # ---- Experiment 1: varying nodes ----
    node_counts = [10, 20, 30, 50, 75, 100]
    err1, time1 = experiment_vary_nodes(
        node_counts, sigma=0.05, hole_density=0.5, n_trials=20
    )
    plot_results(
        node_counts, err1, time1,
        xlabel="Number of Nodes",
        title="Varying graph size",
        save_path="exp1_vary_nodes.png",
    )

    # ---- Experiment 2: varying noise ----
    noise_levels = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    err2, time2 = experiment_vary_noise(
        noise_levels, n=25, hole_density=0.5, n_trials=20
    )
    plot_results(
        noise_levels, err2, time2,
        xlabel="Noise σ",
        title="Varying noise level",
        save_path="exp2_vary_noise.png",
    )

    # ---- Experiment 3: varying hole density ----
    hole_densities = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
    err3, time3 = experiment_vary_holes(
        hole_densities, n=25, sigma=0.05, n_trials=20
    )
    plot_results(
        hole_densities, err3, time3,
        xlabel="Hole Density ρ",
        title="Varying hole density",
        save_path="exp3_vary_holes.png",
    )

    # ---- Experiment 4: varying topology ----
    err4, time4 = experiment_vary_topology(
        n=36, sigma=0.05, hole_density=0.2, n_trials=20
    )
    plot_topology_results(
        err4, time4,
        title="Topology comparison",
        save_path="exp4_vary_topology.png",
    )

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)
    print("""
    Method comparison — Iterative [1] vs. Spectral [2]:

    Iterative (Sphere/Euclidean):
      + Handles large graphs efficiently (per-node updates, O(n·d) per iter)
      + Sphere and Euclidean use L1-based averaging → inherently robust to outliers
      + Sphere is the best-performing variant overall
      - Requires multiple iterations to converge
      - Direction averaging uses L2 norm → less robust

    Spectral (LSH/GSH):
      + Single SVD solve — no iterative convergence needed
      + Handles missing data natively (no edge hallucination required)
      + GSH corrects the weighting bias present in LSH
      - SVD cost is O((3n)^3) — less scalable for very large graphs
      - Not inherently robust to outliers (would need IRLS wrapper)

    Spanning Tree (baseline):
      + Fastest (single BFS traversal)
      - Accumulates errors from root to leaves
      - Does not exploit redundant measurements
      - Performance degrades as graph size increases

    Topology observations (Experiment 4):
      • Linear graphs (low bandwidth) stress error accumulation the most,
        widening the gap between tree and iterative/spectral methods.
      • Circular graphs provide loop closure, which helps all methods.
      • Grid graphs with 8-connectivity offer dense local redundancy,
        favouring spectral approaches.
      • Multilane graphs are a realistic proxy for multi-strip mosaicking.
    """)


if __name__ == "__main__":
    main()
