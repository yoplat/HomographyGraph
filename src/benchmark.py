"""
benchmark.py — Synthetic comparison of synchronization methods for 3×3 homographies.

This script generates random ground-truth projective transformations, creates
noisy relative measurements on a graph with controlled connectivity, runs
multiple synchronization algorithms, and compares their accuracy.

The comparison covers two families of methods:

  • **Iterative** [1]: Madhavan, Fusiello & Arrigoni (sphere / euclidean / direction)
  • **Spectral** [2]: Schroeder, Bartoli, Georgel & Navab (LSH / GSH)
  • **Spanning tree** baseline

Three experimental axes are swept (mirroring the evaluation in [1]):
  1. Number of nodes (graph size)          — Section "Experiment 1"
  2. Noise level (measurement perturbation) — Section "Experiment 2"
  3. Hole density (fraction of missing edges) — Section "Experiment 3"

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

from operator import gt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import time
from typing import Dict, List, Tuple, Literal, Optional
from graph import Graph

# Reproducibility.
SEED = 42
np.random.seed(SEED)


# ===================================================================
# Ground-truth generation utilities
# ===================================================================


def generate_random_sl3() -> np.ndarray:
    """
    Generate a random 3×3 matrix in SL(3) (unit determinant).

    We draw entries from a standard normal distribution, then normalise
    by the cube root of the determinant.  This produces a realistic
    spread of projective transformations for testing.
    """
    A = np.random.randn(3, 3)
    det = np.linalg.det(A)
    if det == 0:
        return generate_random_sl3()  # retry on degenerate draw
    return A / np.cbrt(det)


def add_noise(matrix: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """
    Perturb a 3×3 matrix with additive Gaussian noise.

    The noise is applied in the tangent space of the matrix manifold,
    following the approach in Sec. 5.2 of [1]: flatten the matrix to
    a 9-vector, add a noise vector of controlled angular magnitude,
    then reshape.  For small sigma this is equivalent to a perturbation
    on the sphere in R^9.

    Parameters
    ----------
    matrix : np.ndarray
        The noise-free 3×3 matrix.
    sigma : float
        Standard deviation of the additive Gaussian noise per entry.

    Returns
    -------
    np.ndarray
        The perturbed 3×3 matrix.
    """
    noise = np.random.normal(0, sigma, (3, 3))
    return matrix + noise


def add_outlier_edge(graph: Graph, i: int, j: int) -> None:
    """
    Replace the edge (i, j) with a completely random relative homography,
    simulating a gross outlier (wrong match).
    """
    random_H = generate_random_sl3()
    graph.edges[(i, j)] = random_H
    graph.edges[(j, i)] = np.linalg.inv(random_H)


# ===================================================================
# Error computation
# ===================================================================


def calculate_angular_error(
    graph: Graph, ground_truth: Dict[int, np.ndarray]
) -> float:
    """
    Compute the mean angular error between estimated and true homographies.

    Since the synchronization solution is defined up to a global
    transformation C, we first align by solving:
        C = X̂_0^{-1} · X_0
    (using the first vertex as anchor), then measure:
        e_i = arccos( |vec(X̂_i C) · vec(X_i)| )
    for each vertex (Eq. 12 of [1]).

    Parameters
    ----------
    graph : Graph
        Synchronised graph.
    ground_truth : dict
        Mapping from vertex id to ground-truth 3×3 matrix.

    Returns
    -------
    float
        Mean angular error in degrees.
    """
    # Pick the first vertex as the alignment anchor.
    anchor = min(ground_truth.keys())
    est_X0 = graph.get_vertex_proj(anchor)
    true_X0 = ground_truth[anchor]

    # Alignment matrix: Ĉ such that X̂_i · C ≈ X_i.
    C = np.linalg.inv(est_X0) @ true_X0

    errors = []
    for i in graph.vertices:
        est_Xi_aligned = graph.get_vertex_proj(i) @ C
        true_Xi = ground_truth[i]

        # Flatten and normalise to unit vectors for angular comparison.
        v1 = est_Xi_aligned.flatten()
        v1 /= np.linalg.norm(v1)
        v2 = true_Xi.flatten()
        v2 /= np.linalg.norm(v2)

        # Angular distance: arccos(|v1 · v2|).
        # We take |·| because projective transformations are defined
        # up to sign (antipodal equivalence on the sphere).
        cos_theta = np.clip(np.abs(np.dot(v1, v2)), 0, 1)
        errors.append(np.arccos(cos_theta))

    return np.degrees(np.mean(errors))


# ===================================================================
# Synthetic graph construction
# ===================================================================


def build_synthetic_graph(
    n: int,
    sigma: float = 0.05,
    hole_density: float = 0.3,
    outlier_density: float = 0.0,
) -> Tuple[Graph, Dict[int, np.ndarray]]:
    """
    Build a synthetic synchronization graph with controlled noise and
    connectivity.

    Parameters
    ----------
    n : int
        Number of vertices (images).
    sigma : float
        Noise standard deviation for the relative measurements.
    hole_density : float
        Fraction of edges to remove (0 = complete graph, 0.95 = very sparse).
        Connectivity is not guaranteed — the caller should check.
    outlier_density : float
        Fraction of remaining edges that are replaced by random outliers.

    Returns
    -------
    graph : Graph
        The constructed noisy graph (vertices initialised to I).
    ground_truth : dict
        The true absolute homographies.
    """
    graph = Graph()
    ground_truth = {}

    # 1. Generate random ground-truth homographies in SL(3).
    for i in range(n):
        gt_mat = generate_random_sl3()
        ground_truth[i] = gt_mat
        graph.add_vertex(i, np.eye(3))

    # 2. Generate noisy relative measurements.
    #    Z_ij = X_i · X_j^{-1} + noise
    for i in range(n):
        for j in range(i + 1, n):
            # Skip edge with probability = hole_density.
            if np.random.rand() < hole_density:
                continue

            rel_ij = ground_truth[i] @ np.linalg.inv(ground_truth[j])
            noisy_rel = add_noise(rel_ij, sigma=sigma)
            graph.add_edge(i, j, noisy_rel)

    # 3. Inject outlier edges.
    if outlier_density > 0:
        edge_keys = [(i, j) for (i, j) in graph.edges.keys() if i < j]
        n_outliers = int(len(edge_keys) * outlier_density)
        outlier_indices = np.random.choice(
            len(edge_keys), size=n_outliers, replace=False
        )
        for idx in outlier_indices:
            i, j = edge_keys[idx]
            add_outlier_edge(graph, i, j)

    return graph, ground_truth

import numpy as np
from typing import Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Linear (chain) graph: 0 ─ 1 ─ 2 ─ ... ─ (n-1)
# ─────────────────────────────────────────────────────────────────────────────

def build_linear_graph(
    n: int,
    sigma: float = 0.05,
    hole_density: float = 0.0,
    outlier_density: float = 0.0,
    bandwidth: int = 1,
) -> Tuple[Graph, Dict[int, np.ndarray]]:
    """
    Build a chain-structured synchronization graph.

    The backbone  0 ─ 1 ─ 2 ─ … ─ (n-1)  is always fully present,
    guaranteeing every node has at least one neighbour and the graph is
    connected.  ``hole_density`` is applied **only** to shortcut edges
    (k ≥ 2), so it controls redundancy without risking isolation.

    Parameters
    ----------
    n : int
        Number of vertices.
    sigma : float
        Noise standard deviation for relative measurements.
    hole_density : float
        Fraction of *shortcut* edges (k ≥ 2) to drop randomly.
        Backbone edges (k = 1) are never dropped.
    outlier_density : float
        Fraction of remaining edges replaced by random outliers.
    bandwidth : int
        Maximum hop distance for which an edge is considered.
        bandwidth=1  → pure chain (backbone only)
        bandwidth=k  → backbone + shortcuts up to hop distance k

    Returns
    -------
    graph        : Graph
    ground_truth : dict  {node_id → absolute homography}
    """
    graph = Graph()
    ground_truth: Dict[int, np.ndarray] = {}

    # 1. Ground-truth homographies.
    for i in range(n):
        gt_mat = generate_random_sl3()
        ground_truth[i] = gt_mat
        graph.add_vertex(i, np.eye(3))

    # 2. Band-diagonal edges.
    #    k=1 → backbone: always included (no dropout).
    #    k≥2 → shortcuts: subject to hole_density.
    for k in range(1, bandwidth + 1):
        is_backbone = (k == 1)
        for i in range(n - k):
            j = i + k
            if not is_backbone and np.random.rand() < hole_density:
                continue
            rel_ij = ground_truth[i] @ np.linalg.inv(ground_truth[j])
            graph.add_edge(i, j, add_noise(rel_ij, sigma=sigma))

    # 3. Outlier injection.
    if outlier_density > 0:
        edge_keys = [(i, j) for (i, j) in graph.edges.keys() if i < j]
        n_outliers = int(len(edge_keys) * outlier_density)
        for idx in np.random.choice(len(edge_keys), size=n_outliers, replace=False):
            i, j = edge_keys[idx]
            add_outlier_edge(graph, i, j)

    return graph, ground_truth



# ─────────────────────────────────────────────────────────────────────────────
# Circular (ring / cycle) graph: 0 ─ 1 ─ … ─ (n-1) ─ 0
# ─────────────────────────────────────────────────────────────────────────────

def build_circular_graph(
    n: int,
    sigma: float = 0.05,
    hole_density: float = 0.0,
    outlier_density: float = 0.0,
    chord_step: int = 1,
) -> Tuple[Graph, Dict[int, np.ndarray]]:
    if n < 3:
        raise ValueError("Circular graph requires n ≥ 3.")

    graph = Graph()
    ground_truth: Dict[int, np.ndarray] = {}

    for i in range(n):
        gt_mat = generate_random_sl3()
        ground_truth[i] = gt_mat
        graph.add_vertex(i, np.eye(3))

    seen: set = set()
    for k in range(1, chord_step + 1):
        is_ring = (k == 1)          # ring edges are protected
        for i in range(n):
            j = (i + k) % n
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            if not is_ring and np.random.rand() < hole_density:
                continue
            a, b = key
            rel_ab = ground_truth[a] @ np.linalg.inv(ground_truth[b])
            graph.add_edge(a, b, add_noise(rel_ab, sigma=sigma))

    if outlier_density > 0:
        edge_keys = [(i, j) for (i, j) in graph.edges.keys() if i < j]
        n_outliers = int(len(edge_keys) * outlier_density)
        for idx in np.random.choice(len(edge_keys), size=n_outliers, replace=False):
            i, j = edge_keys[idx]
            add_outlier_edge(graph, i, j)

    return graph, ground_truth



# ─────────────────────────────────────────────────────────────────────────────
# Grid (lattice) graph: nodes on an r × c grid, edges to 4-neighbours
# ─────────────────────────────────────────────────────────────────────────────

def build_grid_graph(
    rows: int,
    cols: int,
    sigma: float = 0.05,
    hole_density: float = 0.0,
    outlier_density: float = 0.0,
    diagonal_edges: bool = False,
) -> Tuple[Graph, Dict[int, np.ndarray]]:
    import networkx as nx

    n = rows * cols
    graph = Graph()
    ground_truth: Dict[int, np.ndarray] = {}
    node_id = lambda r, c: r * cols + c

    for idx in range(n):
        gt_mat = generate_random_sl3()
        ground_truth[idx] = gt_mat
        graph.add_vertex(idx, np.eye(3))

    offsets = [(0, 1), (1, 0)]
    if diagonal_edges:
        offsets += [(1, 1), (1, -1)]

    for r in range(rows):
        for c in range(cols):
            i = node_id(r, c)
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                j = node_id(nr, nc)
                if np.random.rand() < hole_density:
                    continue
                rel_ij = ground_truth[i] @ np.linalg.inv(ground_truth[j])
                graph.add_edge(i, j, add_noise(rel_ij, sigma=sigma))

    # ── Spanning-tree repair ─────────────────────────────────────────────────
    # Build a NetworkX graph from current edges and check connectivity.
    # For every missing tree edge, force-insert it so no node is isolated.
    G_check = nx.Graph()
    G_check.add_nodes_from(range(n))
    G_check.add_edges_from(
        (i, j) for (i, j) in graph.edges.keys() if i < j
    )
    if not nx.is_connected(G_check):
        for i, j in nx.minimum_spanning_edges(
            nx.complement(G_check), data=False   # edges NOT yet in graph
        ):
            # Only add enough edges to connect the components.
            if not nx.is_connected(G_check):
                rel_ij = ground_truth[i] @ np.linalg.inv(ground_truth[j])
                graph.add_edge(i, j, add_noise(rel_ij, sigma=sigma))
                G_check.add_edge(i, j)

    if outlier_density > 0:
        edge_keys = [(i, j) for (i, j) in graph.edges.keys() if i < j]
        n_outliers = int(len(edge_keys) * outlier_density)
        for idx in np.random.choice(len(edge_keys), size=n_outliers, replace=False):
            i, j = edge_keys[idx]
            add_outlier_edge(graph, i, j)

    return graph, ground_truth

def build_multilane_graph(
    n: int,
    num_lanes: int = 3,
    sigma: float = 0.05,
    hole_density: float = 0.0,
    outlier_density: float = 0.0,
    bandwidth: int = 1,
    cross_connect: bool = True,
    diagonal_cross_density: float = 0.0,
) -> Tuple[Graph, Dict[int, np.ndarray]]:
    """
    Build a multi-lane linear synchronization graph.

    Parameters
    ----------
    n : int
        Total number of nodes.  The nodes are distributed across lanes as
        evenly as possible:
            n_per_lane = ceil(n / num_lanes)
        If n is not divisible by num_lanes, the last lane gets fewer nodes
        (n - (num_lanes-1) * n_per_lane).
    num_lanes : int
        Number of parallel lanes.  Defaults to 3.
        Must satisfy num_lanes ≤ n.
    ...  (all other parameters unchanged)
    """
    if num_lanes > n:
        raise ValueError(f"num_lanes ({num_lanes}) cannot exceed n ({n}).")

    n_per_lane = int(np.ceil(n / num_lanes))

    # Actual lane sizes (last lane may be shorter).
    lane_sizes = [n_per_lane] * (num_lanes - 1)
    lane_sizes.append(n - n_per_lane * (num_lanes - 1))   # remainder

    graph = Graph()
    ground_truth: Dict[int, np.ndarray] = {}

    # Node id: running counter across lanes.
    # lane_offsets[lane] = index of first node in that lane.
    lane_offsets = [sum(lane_sizes[:l]) for l in range(num_lanes)]
    node_id = lambda lane, col: lane_offsets[lane] + col

    # 1. Ground-truth homographies.
    for idx in range(n):
        gt_mat = generate_random_sl3()
        ground_truth[idx] = gt_mat
        graph.add_vertex(idx, np.eye(3))

    existing = set()

    def _try_add(i: int, j: int) -> None:
        key = (min(i, j), max(i, j))
        if key in existing:
            return
        existing.add(key)
        a, b = key
        rel = ground_truth[a] @ np.linalg.inv(ground_truth[b])
        graph.add_edge(a, b, add_noise(rel, sigma=sigma))

    # 2. Within-lane edges (backbone protected).
    for lane in range(num_lanes):
        size = lane_sizes[lane]
        for k in range(1, bandwidth + 1):
            is_backbone = (k == 1)
            for col in range(size - k):
                if not is_backbone and np.random.rand() < hole_density:
                    continue
                _try_add(node_id(lane, col), node_id(lane, col + k))

    # 3. Vertical cross edges (never dropped).
    if cross_connect:
        for lane in range(num_lanes - 1):
            shared_cols = min(lane_sizes[lane], lane_sizes[lane + 1])
            for col in range(shared_cols):
                _try_add(node_id(lane, col), node_id(lane + 1, col))

    # 4. Diagonal cross edges.
    if diagonal_cross_density > 0.0:
        for lane in range(num_lanes - 1):
            for col in range(lane_sizes[lane]):
                for d in range(1, bandwidth + 1):
                    for target_col in [col - d, col + d]:
                        if not (0 <= target_col < lane_sizes[lane + 1]):
                            continue
                        if np.random.rand() < diagonal_cross_density:
                            _try_add(
                                node_id(lane,     col),
                                node_id(lane + 1, target_col),
                            )

    # 5. Outlier injection.
    if outlier_density > 0:
        edge_keys = [(i, j) for (i, j) in graph.edges.keys() if i < j]
        n_outliers = int(len(edge_keys) * outlier_density)
        for idx in np.random.choice(len(edge_keys), size=n_outliers, replace=False):
            i, j = edge_keys[idx]
            add_outlier_edge(graph, i, j)

    return graph, ground_truth, (num_lanes, n_per_lane)






LayoutHint = Literal["auto", "spring", "linear", "circular", "grid"]

def visualize_graph(
    graph: Graph,
    ground_truth: Optional[Dict[int, np.ndarray]] = None,
    layout: LayoutHint = "auto",
    grid_shape: Optional[tuple[int, int]] = None,
    outlier_edges: Optional[set[tuple[int, int]]] = None,
    title: str = "Synchronization Graph",
    figsize: tuple[int, int] = (9, 7),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Visualise a synchronisation Graph.

    Draws nodes and edges with layout heuristics that match the underlying
    graph topology.  Optionally highlights outlier edges and annotates
    nodes with their ground-truth index.

    Parameters
    ----------
    graph : Graph
        The graph to draw.
    ground_truth : dict, optional
        {node_id → absolute homography}.  When supplied the node colour
        encodes whether the vertex has been initialised (white) or not.
        Currently used only for the legend; extend to show residuals if needed.
    layout : {"auto", "spring", "linear", "circular", "grid"}
        Spatial layout algorithm.

        * ``"auto"``     – heuristically chosen from the graph's edge
          structure: ring-like → circular, path-like → linear,
          grid-like → grid, otherwise → spring.
        * ``"spring"``   – Fruchterman–Reingold force-directed layout
          (best for random graphs).
        * ``"linear"``   – nodes arranged on a horizontal line in index
          order (best for chain / band-diagonal graphs).
        * ``"circular"`` – nodes equally spaced on a circle (best for
          ring graphs).
        * ``"grid"``     – nodes placed on a 2-D grid.  ``grid_shape``
          must be provided or is inferred as (√n × √n).

    grid_shape : (rows, cols), optional
        Required when ``layout="grid"``.  Ignored for other layouts.
    outlier_edges : set of (i, j) pairs, optional
        Edges to highlight in red.  If not provided the function draws
        all edges in a single neutral colour.
    title : str
        Figure / axes title.
    figsize : (width, height)
        Passed to ``plt.subplots`` when no ``ax`` is supplied.
    ax : matplotlib.axes.Axes, optional
        Draw onto an existing axes instead of creating a new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the drawing.
    """
    # ── 1. Build a NetworkX graph from the custom Graph object ──────────────
    G = nx.Graph()
    G.add_nodes_from(graph.vertices.keys())
    for (i, j) in graph.edges.keys():
        if i < j:
            G.add_edge(i, j)

    n = G.number_of_nodes()
    nodes = sorted(G.nodes())

    # ── 2. Layout auto-detection ─────────────────────────────────────────────
    def _is_path_like(G: nx.Graph) -> bool:
        """True when the graph looks like a (possibly sparse) chain."""
        degrees = [d for _, d in G.degree()]
        return max(degrees) <= 3 and nx.is_connected(G) and not nx.is_biconnected(G)

    def _is_ring_like(G: nx.Graph) -> bool:
        """True when every node has degree 2 or when the graph is a single cycle."""
        degrees = [d for _, d in G.degree()]
        avg_deg = np.mean(degrees)
        return 1.8 <= avg_deg <= 2.4

    def _is_grid_like(G: nx.Graph, n: int) -> bool:
        """True when n is a perfect square and avg degree ≈ 3-4."""
        sq = int(np.round(np.sqrt(n)))
        if sq * sq != n:
            return False
        degrees = [d for _, d in G.degree()]
        return 2.5 <= np.mean(degrees) <= 4.5

    if layout == "auto":
        if _is_ring_like(G):
            layout = "circular"
        elif _is_path_like(G):
            layout = "linear"
        elif _is_grid_like(G, n):
            layout = "grid"
        else:
            layout = "spring"

    # ── 3. Compute positions ─────────────────────────────────────────────────
    pos: Dict[int, np.ndarray] = {}

    if layout == "spring":
        pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(n))

    elif layout == "linear":
        for rank, node in enumerate(nodes):
            pos[node] = np.array([rank / max(n - 1, 1), 0.0])

    elif layout == "circular":
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for node, angle in zip(nodes, angles):
            pos[node] = np.array([np.cos(angle), np.sin(angle)])

    elif layout == "grid":
        if grid_shape is None:
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
        else:
            rows, cols = grid_shape
        for node in nodes:
            r, c = divmod(node, cols)
            pos[node] = np.array([c / max(cols - 1, 1), -r / max(rows - 1, 1)])

    # ── 4. Partition edges ───────────────────────────────────────────────────
    outlier_set: set[tuple[int, int]] = set()
    if outlier_edges is not None:
        outlier_set = {(min(i, j), max(i, j)) for i, j in outlier_edges}

    inlier_edgelist  = []
    outlier_edgelist = []
    for (i, j) in G.edges():
        key = (min(i, j), max(i, j))
        if key in outlier_set:
            outlier_edgelist.append((i, j))
        else:
            inlier_edgelist.append((i, j))

    # ── 5. Draw ──────────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#f7f6f2")

    ax.set_facecolor("#f7f6f2")

    # Inlier edges
    nx.draw_networkx_edges(
        G, pos, edgelist=inlier_edgelist, ax=ax,
        edge_color="#01696f", alpha=0.65, width=1.6,
    )
    # Outlier edges
    if outlier_edgelist:
        nx.draw_networkx_edges(
            G, pos, edgelist=outlier_edgelist, ax=ax,
            edge_color="#a12c7b", alpha=0.85, width=2.2,
            style="dashed",
        )

    # Node fill: white if initialised (all vertices start at I), teal if solved
    node_colors = ["#cedcd8" if np.allclose(graph.vertices[v], np.eye(3)) else "#01696f"
                   for v in nodes]
    nx.draw_networkx_nodes(
        G, pos, nodelist=nodes, ax=ax,
        node_color=node_colors, edgecolors="#01696f",
        node_size=max(80, min(500, 4000 // n)), linewidths=1.8,
    )

    # Labels — hide when the graph is large
    if n <= 40:
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=max(6, min(10, 120 // n)),
            font_color="#28251d", font_weight="bold",
        )

    # ── 6. Stats annotation ──────────────────────────────────────────────────
    n_edges   = G.number_of_edges()
    n_out     = len(outlier_edgelist)
    connected = nx.is_connected(G) if n_edges > 0 else False
    density   = nx.density(G)

    stats = (
        f"n={n}  |  edges={n_edges}  |  density={density:.2f}\n"
        f"connected={'yes' if connected else 'NO ⚠'}  |  "
        f"outliers={n_out} ({100*n_out/max(n_edges,1):.0f}%)"
    )
    ax.text(
        0.02, 0.02, stats,
        transform=ax.transAxes,
        fontsize=8, color="#7a7974",
        verticalalignment="bottom",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9f8f5", edgecolor="#dcd9d5"),
    )

    # ── 7. Legend ────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor="#cedcd8", edgecolor="#01696f", label="vertex  (I init)"),
        mpatches.Patch(facecolor="#01696f", edgecolor="#01696f", label="vertex  (solved)"),
        mpatches.Patch(facecolor="#01696f", alpha=0.65,          label="inlier edge"),
    ]
    if outlier_edgelist:
        legend_items.append(
            mpatches.Patch(facecolor="#a12c7b", alpha=0.85, label="outlier edge")
        )
    ax.legend(
        handles=legend_items, loc="upper right",
        fontsize=8, framealpha=0.9,
        facecolor="#f9f8f5", edgecolor="#dcd9d5",
    )

    ax.set_title(title, fontsize=12, fontweight="bold",
                 color="#28251d", pad=12)
    ax.axis("off")
    plt.tight_layout()
    return ax

# ===================================================================
# Single experiment runner
# ===================================================================

# Registry of all methods to benchmark.  Each entry is:
#   (label, setup_function)
# where setup_function takes a Graph, runs synchronization, and returns nothing.

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


def run_single_trial(
    n: int,
    sigma: float,
    hole_density: float,
    outlier_density: float,
    methods: Dict[str, callable],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Run one trial: build a synthetic graph, run all methods, return errors
    and execution times.

    Returns
    -------
    errors : dict[str, float]
        Angular error (degrees) for each method label.
    times : dict[str, float]
        Execution time (seconds) for each method label.
    """

    num_lanes  = 3
    base_graph, ground_truth, grid_shape = build_multilane_graph(
        n=100,
        num_lanes=num_lanes,
        sigma=sigma,
        hole_density=hole_density,   
        outlier_density=0.05,
        bandwidth=3,
        cross_connect=True,
        diagonal_cross_density=0.15
    )
    # visualize_graph(
    #     base_graph, ground_truth,
    #     layout="grid",
    #     grid_shape=grid_shape,   # rows=lanes, cols=nodes per lane
    #     title=f"Multi-lane graph  (n={n}, {grid_shape[0]} lanes × {grid_shape[1]} cols)",    )    
    # plt.show()

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
# Experiment sweeps
# ===================================================================


def experiment_vary_nodes(
    node_counts: List[int],
    sigma: float = 0.05,
    hole_density: float = 0.5,
    outlier_density: float = 0.0,
    n_trials: int = 20,
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
        f"  σ={sigma}, ρ={hole_density}, γ={outlier_density}, "
        f"trials={n_trials}"
    )
    print("=" * 60)

    error_results = {label: [] for label in METHODS}
    time_results = {label: [] for label in METHODS}

    for n in node_counts:
        trial_errors = {label: [] for label in METHODS}
        trial_times = {label: [] for label in METHODS}

        for t in range(n_trials):
            start_time = time.time_ns()
            errs, tms = run_single_trial(
                n=n,
                sigma=sigma,
                hole_density=hole_density,
                outlier_density=outlier_density,
                methods=METHODS,
            )
            print(
                f"Single Trial time take: {(time.time_ns() - start_time) / 1_000_000_000}s"
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
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Experiment 2: vary the noise level, keeping graph size and connectivity fixed.

    Mirrors Fig. 4 (left) of [1].  All methods degrade with increasing noise,
    but the iterative and spectral approaches should degrade more gracefully
    than the spanning tree.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Varying noise level")
    print(f"  n={n}, ρ={hole_density}, γ={outlier_density}, trials={n_trials}")
    print("=" * 60)

    error_results = {label: [] for label in METHODS}
    time_results = {label: [] for label in METHODS}

    for sigma in noise_levels:
        trial_errors = {label: [] for label in METHODS}
        trial_times = {label: [] for label in METHODS}

        for t in range(n_trials):
            errs, tms = run_single_trial(
                n=n,
                sigma=sigma,
                hole_density=hole_density,
                outlier_density=outlier_density,
                methods=METHODS,
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
    print(f"  n={n}, σ={sigma}, γ={outlier_density}, trials={n_trials}")
    print("=" * 60)

    error_results = {label: [] for label in METHODS}
    time_results = {label: [] for label in METHODS}

    for rho in hole_densities:
        trial_errors = {label: [] for label in METHODS}
        trial_times = {label: [] for label in METHODS}

        for t in range(n_trials):
            errs, tms = run_single_trial(
                n=n,
                sigma=sigma,
                hole_density=rho,
                outlier_density=outlier_density,
                methods=METHODS,
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


# ===================================================================
# Main
# ===================================================================


def main():
    """
    Run all three experiments and produce comparison plots.

    The experimental setup follows the structure in Sec. 5.2 of [1]:
      • Experiment 1: n ∈ {10, 20, 30, 50, 75, 100}, σ=0.05, ρ=0.5
      • Experiment 2: σ ∈ [0.01, 0.15], n=25, ρ=0.5
      • Experiment 3: ρ ∈ [0.0, 0.9], n=25, σ=0.05
    """
    # ---- Experiment 1: varying nodes ----
    node_counts = [10, 20, 30, 50, 75, 100]
    err1, time1 = experiment_vary_nodes(
        node_counts, sigma=0.05, hole_density=0.5, n_trials=20
    )
    plot_results(
        node_counts,
        err1,
        time1,
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
        noise_levels,
        err2,
        time2,
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
        hole_densities,
        err3,
        time3,
        xlabel="Hole Density ρ",
        title="Varying hole density",
        save_path="exp3_vary_holes.png",
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
    """)


if __name__ == "__main__":
    main()
