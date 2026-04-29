"""
graph.py — Synchronization graph for projective transformations (3×3 homographies).

This module implements a graph whose vertices carry unknown absolute homographies
(X_i ∈ SL(3)) and whose edges carry measured *relative* homographies
(Z_ij ≈ X_i · X_j^{-1}, up to scale).

Two families of synchronization algorithms are provided:

  1. **Iterative** (Madhavan, Fusiello, Arrigoni — ECCV-like approach [1]):
     Each node is updated in turn as the "average" of the estimates provided
     by its neighbours.  Three notions of average are implemented:
       • Euclidean  — simple componentwise mean of unit-norm representatives
       • Direction  — eigenvector of the sum of rank-1 projectors (L2, scale-invariant)
       • Sphere     — L1 geodesic mean on the unit sphere via fixed-point iteration

  2. **Spectral / closed-form** (Schroeder, Bartoli, Georgel, Navab [2]):
     Builds a block matrix from all known inter-frame homographies and
     extracts the absolute homographies from its null space via SVD.
     Two variants handle missing data:
       • LSH — Locally Scaled Homographies
       • GSH — Globally Scaled Homographies

Both families operate in SL(3): every homography is normalised so that
det(H) = 1, which removes the scale ambiguity inherent to projective
transformations and turns the consistency relation Z_ij ≃ X_i X_j^{-1}
into the strict equality Z_ij = X_i X_j^{-1}.

A **spanning-tree** baseline is also included for reference: it chains
relative homographies along a BFS tree, accumulating errors from root
to leaves.

Convention
----------
``edges[(i, j)]`` stores Z_ij = X_i · X_j^{-1}, i.e. the transformation
that maps coordinates *from frame j into frame i*.

References
----------
[1] R. Madhavan, A. Fusiello, F. Arrigoni, "Synchronization of Projective
    Transformations", 2024.
[2] P. Schroeder, A. Bartoli, P. Georgel, N. Navab, "Closed-Form Solutions
    to Multiple-View Homography Estimation", IEEE WACV 2011.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Callable, Literal
from collections import deque

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 120
np.random.seed(SEED)


class Graph:
    """
    A synchronization graph for 3×3 projective transformations.

    Attributes
    ----------
    vertices : dict[int, np.ndarray]
        Maps each vertex uid to its current 3×3 absolute homography estimate.
    edges : dict[tuple[int,int], np.ndarray]
        Maps each directed pair (i, j) to the 3×3 relative homography Z_ij.
        Both directions are stored: Z_ji = Z_ij^{-1}.
    adj : dict[int, set[int]]
        Adjacency list — undirected neighbourhood for each vertex.
    """

    def __init__(self):
        self.vertices: Dict[int, np.ndarray] = {}
        self.edges: Dict[Tuple[int, int], np.ndarray] = {}
        self.adj: Dict[int, set] = {}

        # Registry of averaging functions used by the iterative synchroniser.
        self._averaging_map: Dict[
            str, Callable[[List[np.ndarray]], np.ndarray]
        ] = {
            "euclidean": self._averaging_euclidean,
            "direction": self._averaging_direction,
            "sphere": self._averaging_sphere,
        }

    # ======================================================================
    # Graph construction
    # ======================================================================

    def add_vertex(
        self, uid: int, initial_proj: Optional[np.ndarray] = None
    ) -> None:
        """Add a vertex with an optional initial absolute homography (default: I)."""
        self.vertices[uid] = (
            initial_proj if initial_proj is not None else np.eye(3)
        )
        if uid not in self.adj:
            self.adj[uid] = set()

    def add_edge(self, v1_id: int, v2_id: int, rel_proj: np.ndarray) -> None:
        """
        Add an undirected edge between v1 and v2.

        Parameters
        ----------
        v1_id, v2_id : int
            The two endpoint vertex ids.
        rel_proj : np.ndarray
            The measured relative homography Z_{v1, v2} = X_{v1} · X_{v2}^{-1}.
            The inverse direction is stored automatically.
        """
        # Ensure both vertices exist.
        if v1_id not in self.vertices:
            self.add_vertex(v1_id)
        if v2_id not in self.vertices:
            self.add_vertex(v2_id)

        self.edges[(v1_id, v2_id)] = rel_proj
        self.edges[(v2_id, v1_id)] = np.linalg.inv(rel_proj)

        self.adj[v1_id].add(v2_id)
        self.adj[v2_id].add(v1_id)

    def get_vertex_proj(self, uid: int) -> Optional[np.ndarray]:
        """Return the current absolute homography for vertex *uid*, or None."""
        return self.vertices.get(uid)

    def get_edge_proj(self, u: int, v: int) -> Optional[np.ndarray]:
        """Return the relative homography Z_{u,v}, or None if the edge is absent."""
        return self.edges.get((u, v))

    # ======================================================================
    # SL(3) normalisation
    # ======================================================================

    @staticmethod
    def _norm_matrix(matrix: np.ndarray) -> np.ndarray:
        """
        Project a 3×3 matrix into SL(3) by dividing by the cube root of its
        determinant, so that det(result) = 1.

        For 3×3 real matrices the cube root is always real, which is what
        makes SL(3) normalisation straightforward (unlike the 4×4 case where
        the 4th root can be complex — see Sec. 3 of [1]).
        """
        det = np.linalg.det(matrix)
        if det != 0:
            return matrix / np.cbrt(det)
        return matrix

    def normalize(self) -> None:
        """Normalise every vertex and edge matrix into SL(3)."""
        for uid in self.vertices:
            self.vertices[uid] = self._norm_matrix(self.vertices[uid])
        for pair in self.edges:
            self.edges[pair] = self._norm_matrix(self.edges[pair])

    # ======================================================================
    # Averaging helpers (used by the iterative synchroniser)
    # ======================================================================

    @staticmethod
    def _averaging_euclidean(estimates: List[np.ndarray]) -> np.ndarray:
        """
        Euclidean average (Sec. 4.3 of [1]).

        Normalise each estimate to unit Frobenius norm, compute the
        componentwise mean, then re-normalise.  This ignores the curved
        geometry of the sphere but is fast and works surprisingly well.
        """
        # Map each 3×3 matrix to a unit-norm vector in R^9.
        h_vecs = [h.flatten() / np.linalg.norm(h.flatten()) for h in estimates]
        c = np.mean(h_vecs, axis=0)
        c /= np.linalg.norm(c)
        return c.reshape((3, 3))

    @staticmethod
    def _averaging_direction(estimates: List[np.ndarray]) -> np.ndarray:
        """
        Direction-based average (Sec. 4.1 of [1]).

        Two vectors have the same direction iff the projection of one onto
        the orthogonal complement of the other is zero.  Stacking these
        constraints for all estimates and solving in the least-squares sense
        reduces to finding the eigenvector of
            M = Σ_k  (h_k h_k^T) / (h_k^T h_k)
        corresponding to the *largest* eigenvalue.
        """
        h_vecs = [h.flatten() for h in estimates]
        # Build the 9×9 matrix whose maximum eigenvector is the centroid.
        M = sum(np.outer(h, h) / np.dot(h, h) for h in h_vecs)
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        # eigh returns eigenvalues in ascending order — take the last column.
        c = eigenvectors[:, -1]
        return c.reshape((3, 3))

    @staticmethod
    def _averaging_sphere(estimates: List[np.ndarray]) -> np.ndarray:
        """
        Spherical average — L1 geodesic mean on the unit sphere (Sec. 4.2 of [1]).

        The centroid c minimises Σ_k arccos(c^T h̄_k) subject to ||c|| = 1.
        Setting the gradient to zero gives a fixed-point equation (Eq. 10 in [1]):
            c = α · Σ_k  h̄_k / sqrt(1 − (c^T h̄_k)²)
        which we iterate until convergence.
        """
        # (K, 9) matrix of unit-norm flattened homographies
        H = np.stack([h.flatten() for h in estimates])
        H /= np.linalg.norm(H, axis=1, keepdims=True)

        # Initialise with Euclidean mean
        c = H.mean(axis=0)
        c /= np.linalg.norm(c)

        for _ in range(10):
            dots = np.clip(H @ c, -1.0, 1.0)  # (K,)
            weights = 1.0 / np.maximum(np.sqrt(1 - dots**2), 1e-6)  # (K,)
            c = weights @ H  # (9,)
            c /= np.linalg.norm(c)

        return c.reshape((3, 3))

    # ======================================================================
    # Utility: vertex ordering
    # ======================================================================

    def _sorted_vertices_by_degree(self) -> List[int]:
        """
        Return vertex ids sorted by descending degree.

        Starting the iterative update from high-degree nodes improves
        convergence because they receive more constraints and are therefore
        more stable (see Sec. 5.1 of [1]).
        """
        return sorted(
            self.vertices.keys(),
            key=lambda uid: len(self.adj.get(uid, set())),
            reverse=True,
        )

    # ======================================================================
    # METHOD 1: Iterative synchronization  (Madhavan et al. [1])
    # ======================================================================

    def synchronize_iterative(
        self,
        avg_method: str = "sphere",
        max_iters: int = 100,
    ) -> None:
        """
        Iterative synchronization following Madhavan, Fusiello & Arrigoni [1].

        For each node i (processed in descending degree order):
          1. Compute neighbour estimates: X_{i|j} = Z_{ij} · X_j  for every
             neighbour j of i.
          2. Average these estimates using the chosen method to get a new X_i.
          3. Re-normalise X_i into SL(3).

        Parameters
        ----------
        avg_method : str
            One of ``"euclidean"``, ``"direction"``, ``"sphere"``.
        max_iters : int
            Maximum number of full sweeps over all vertices.
        """
        sorted_verts = self._sorted_vertices_by_degree()
        avg_func = self._averaging_map.get(
            avg_method.lower(), self._averaging_euclidean
        )

        for _ in range(max_iters):
            new_vertices = {}
            for i in sorted_verts:
                neighbours = self.adj.get(i, set())
                if not neighbours:
                    continue

                # Eq. (4) in [1]: X_{i|j} = Z_{ij} · X_j
                estimates = [
                    self.edges[(i, j)] @ self.vertices[j] for j in neighbours
                ]

                # Compute the scale-aware average.
                avg_xi = avg_func(estimates)
                new_vertices[i] = self._norm_matrix(avg_xi)

            # Batch update after a full sweep.
            self.vertices.update(new_vertices)

    # ======================================================================
    # METHOD 2: Spectral synchronization  (Schroeder et al. [2])
    # ======================================================================

    def synchronize_spectral(self, method: str = "lsh") -> None:
        """
        Closed-form spectral synchronization following Schroeder et al. [2].

        Builds a 3n × 3n block matrix from all known inter-frame homographies
        and extracts the absolute homographies from its null space via SVD.

        Parameters
        ----------
        method : str
            ``"lsh"`` — Locally Scaled Homographies  (Sec. 4.3.1 of [2])
            ``"gsh"`` — Globally Scaled Homographies  (Sec. 4.3.2 of [2])

        Notes
        -----
        Both LSH and GSH handle missing data naturally (not all pairs need
        to be connected).  The block matrix is real and 3n × 3n where n is
        the number of vertices.  The SVD cost is O(n³) in the matrix dimension,
        which makes this approach fast for moderate graph sizes but less
        scalable than the iterative method for very large graphs.
        """
        if method.lower() == "lsh":
            self._synchronize_lsh()
        elif method.lower() == "gsh":
            self._synchronize_gsh()
        else:
            raise ValueError(
                f"Unknown spectral method '{method}'. Use 'lsh' or 'gsh'."
            )

    def _synchronize_lsh(self) -> None:
        """
        Locally Scaled Homographies (LSH) — Sec. 4.3.1 of [2].

        Builds the block matrix S (Eq. 21):
            S[k, i] = (γ_{i,k} / ζ_k) · H_{i,k}
        where γ_{i,k} = 1 if the homography from i to k is known, 0 otherwise,
        and ζ_k = Σ_i γ_{i,k} counts known connections to node k (including self).

        In SL(3), the noise-free relation S · U = U holds, so U lies in the
        null space of (S − I).  The 3 right singular vectors of (S − I) with
        the smallest singular values approximate U.
        """
        uids = sorted(self.vertices.keys())
        n = len(uids)
        uid_to_idx = {uid: idx for idx, uid in enumerate(uids)}

        S = np.zeros((3 * n, 3 * n))

        for k_idx, k in enumerate(uids):
            # ζ_k: number of known homographies TO node k (including self H_{k,k}=I).
            zeta_k = 1 + len(self.adj.get(k, set()))

            # Diagonal block: self-homography H_{k,k} = I, scaled by 1/ζ_k.
            r = k_idx * 3
            S[r : r + 3, r : r + 3] = np.eye(3) / zeta_k

            # Off-diagonal blocks: for each neighbour i of k.
            for i in self.adj.get(k, set()):
                i_idx = uid_to_idx[i]
                c = i_idx * 3
                # H_{i,k} in Schroeder's notation = edges[(k, i)] in our convention.
                # (edges[(k,i)] = Z_{ki} = X_k · X_i^{-1}, which maps i→k.)
                H_ik = self.edges[(k, i)]
                S[r : r + 3, c : c + 3] = H_ik / zeta_k

        # Solve: minimise ||(S − I) · Û||  ⟹  Û = 3 right singular vectors
        # of (S − I) with smallest singular values.
        A = S - np.eye(3 * n)
        _, sigma, Vh = np.linalg.svd(A, full_matrices=True)

        # numpy returns singular values in descending order; pick last 3 rows of Vh.
        U_hat = Vh[-3:, :].T  # shape (3n, 3)

        # Assign the 3×3 block for each vertex.
        for idx, uid in enumerate(uids):
            raw = U_hat[idx * 3 : (idx + 1) * 3, :]
            self.vertices[uid] = self._norm_matrix(raw)

    def _synchronize_gsh(self) -> None:
        """
        Globally Scaled Homographies (GSH) — Sec. 4.3.2 of [2].

        Corrects the weighting bias of LSH so that every known inter-frame
        homography influences the cost uniformly.

        Builds the block matrix G (Eq. 25):
            G[k, k] = (1 − ζ_k) · I        (diagonal)
            G[k, i] = γ_{i,k} · H_{i,k}    (off-diagonal, i ≠ k)

        The absolute homographies lie in the null space of G; the 3 right
        singular vectors with smallest singular values provide the solution.
        """
        uids = sorted(self.vertices.keys())
        n = len(uids)
        uid_to_idx = {uid: idx for idx, uid in enumerate(uids)}

        G = np.zeros((3 * n, 3 * n))

        for k_idx, k in enumerate(uids):
            # ζ_k includes self: 1 + degree(k).
            zeta_k = 1 + len(self.adj.get(k, set()))
            r = k_idx * 3

            # Diagonal: (1 − ζ_k) · I
            G[r : r + 3, r : r + 3] = (1 - zeta_k) * np.eye(3)

            # Off-diagonal: H_{i,k} for each known neighbour i ≠ k.
            for i in self.adj.get(k, set()):
                i_idx = uid_to_idx[i]
                c = i_idx * 3
                H_ik = self.edges[(k, i)]
                G[r : r + 3, c : c + 3] = H_ik

        # Null space via SVD.
        _, sigma, Vh = np.linalg.svd(G, full_matrices=True)
        U_hat = Vh[-3:, :].T

        for idx, uid in enumerate(uids):
            raw = U_hat[idx * 3 : (idx + 1) * 3, :]
            self.vertices[uid] = self._norm_matrix(raw)

    # ======================================================================
    # BASELINE: Spanning-tree synchronization
    # ======================================================================

    def synchronize_tree(self, root: Optional[int] = None) -> None:
        """
        Spanning-tree baseline: chain relative homographies along a BFS tree.

        This is the simplest approach — it finds a single path from a root to
        every other node and concatenates transformations along that path.
        It is fast (O(n)) but accumulates errors from root to leaves, so it
        degrades for large or deep graphs.

        Parameters
        ----------
        root : int or None
            The root vertex id.  If None, the vertex with the highest degree
            is chosen (same heuristic as the iterative method).
        """
        if root is None:
            # Pick the vertex with the most connections.
            root = max(
                self.vertices.keys(), key=lambda u: len(self.adj.get(u, set()))
            )

        # X_root = I (the root defines the reference frame).
        self.vertices[root] = np.eye(3)
        visited = {root}
        queue = deque([root])

        while queue:
            parent = queue.popleft()
            for child in self.adj.get(parent, set()):
                if child in visited:
                    continue
                visited.add(child)
                queue.append(child)

                # X_child = Z_{child, parent} · X_parent
                Z_cp = self.edges[(child, parent)]
                self.vertices[child] = self._norm_matrix(
                    Z_cp @ self.vertices[parent]
                )

    # ======================================================================
    # Deep copy (for running multiple methods on the same graph)
    # ======================================================================

    def copy(self) -> "Graph":
        """Return a deep copy of this graph (independent vertex values)."""
        g = Graph()
        g.vertices = {uid: mat.copy() for uid, mat in self.vertices.items()}
        g.edges = {pair: mat.copy() for pair, mat in self.edges.items()}
        g.adj = {uid: set(neighbours) for uid, neighbours in self.adj.items()}
        return g


# =========================================================================
# Ground-truth generation utilities
# =========================================================================


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


# =========================================================================
# Error computation
# =========================================================================


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


# =========================================================================
# Synthetic graph builders
# =========================================================================
#
# Each builder generates ground-truth absolute homographies in SL(3),
# constructs noisy relative measurements on a specific topology, and
# optionally injects outlier edges.  All builders return the same
# (Graph, ground_truth) pair (or a triple when extra layout metadata
# is needed).


def build_synthetic_graph(
    n: int,
    sigma: float = 0.05,
    hole_density: float = 0.3,
    outlier_density: float = 0.0,
) -> Tuple[Graph, Dict[int, np.ndarray]]:
    """
    Build a synthetic synchronization graph with controlled noise and
    connectivity.

    The underlying topology is a complete graph with edges randomly
    removed according to ``hole_density``.  A spanning-tree repair step
    guarantees the result is always connected.

    Parameters
    ----------
    n : int
        Number of vertices (images).
    sigma : float
        Noise standard deviation for the relative measurements.
    hole_density : float
        Fraction of edges to remove (0 = complete graph, 0.95 = very sparse).
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

    # 2b. Spanning-tree repair: if hole removal disconnected the graph,
    #     add the minimum number of missing edges to restore connectivity.
    G_check = nx.Graph()
    G_check.add_nodes_from(range(n))
    G_check.add_edges_from((i, j) for (i, j) in graph.edges.keys() if i < j)
    if not nx.is_connected(G_check):
        for i, j in nx.minimum_spanning_edges(nx.complement(G_check), data=False):
            if not nx.is_connected(G_check):
                rel_ij = ground_truth[i] @ np.linalg.inv(ground_truth[j])
                graph.add_edge(i, j, add_noise(rel_ij, sigma=sigma))
                G_check.add_edge(i, j)

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
        is_backbone = k == 1
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
        for idx in np.random.choice(
            len(edge_keys), size=n_outliers, replace=False
        ):
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
    """
    Build a ring-structured synchronization graph.

    The ring backbone (chord_step=1) is always fully present.
    Additional chord edges (chord_step ≥ 2) are subject to
    ``hole_density``.

    Parameters
    ----------
    n : int
        Number of vertices (must be ≥ 3).
    sigma : float
        Noise standard deviation for relative measurements.
    hole_density : float
        Fraction of *chord* edges (k ≥ 2) to drop randomly.
    outlier_density : float
        Fraction of remaining edges replaced by random outliers.
    chord_step : int
        Maximum chord distance.  chord_step=1 produces a pure ring.

    Returns
    -------
    graph        : Graph
    ground_truth : dict  {node_id → absolute homography}
    """
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
        is_ring = k == 1  # ring edges are protected
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
        for idx in np.random.choice(
            len(edge_keys), size=n_outliers, replace=False
        ):
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
    """
    Build a 2-D grid synchronization graph.

    Nodes are placed on an (rows × cols) lattice with edges to their
    4-neighbours (or 8-neighbours if ``diagonal_edges=True``).  When
    ``hole_density`` disconnects the graph, a spanning-tree repair step
    inserts the minimum number of edges to restore connectivity.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions.
    sigma : float
        Noise standard deviation for relative measurements.
    hole_density : float
        Fraction of edges to drop randomly.
    outlier_density : float
        Fraction of remaining edges replaced by random outliers.
    diagonal_edges : bool
        If True, include diagonal neighbours (8-connectivity).

    Returns
    -------
    graph        : Graph
    ground_truth : dict  {node_id → absolute homography}
    """
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
                if np.random.rand() < hole_density:
                    continue
                j = node_id(nr, nc)
                rel_ij = ground_truth[i] @ np.linalg.inv(ground_truth[j])
                graph.add_edge(i, j, add_noise(rel_ij, sigma=sigma))

    # ── Spanning-tree repair ─────────────────────────────────────────────
    # Build a NetworkX graph from current edges and check connectivity.
    # For every missing tree edge, force-insert it so no node is isolated.
    G_check = nx.Graph()
    G_check.add_nodes_from(range(n))
    G_check.add_edges_from((i, j) for (i, j) in graph.edges.keys() if i < j)
    if not nx.is_connected(G_check):
        for i, j in nx.minimum_spanning_edges(
            nx.complement(G_check),
            data=False,  # edges NOT yet in graph
        ):
            # Only add enough edges to connect the components.
            if not nx.is_connected(G_check):
                rel_ij = ground_truth[i] @ np.linalg.inv(ground_truth[j])
                graph.add_edge(i, j, add_noise(rel_ij, sigma=sigma))
                G_check.add_edge(i, j)

    if outlier_density > 0:
        edge_keys = [(i, j) for (i, j) in graph.edges.keys() if i < j]
        n_outliers = int(len(edge_keys) * outlier_density)
        for idx in np.random.choice(
            len(edge_keys), size=n_outliers, replace=False
        ):
            i, j = edge_keys[idx]
            add_outlier_edge(graph, i, j)

    return graph, ground_truth


# ─────────────────────────────────────────────────────────────────────────────
# Multi-lane linear graph: parallel chains with lateral connections
# ─────────────────────────────────────────────────────────────────────────────


def build_multilane_graph(
    n: int,
    num_lanes: int = 3,
    sigma: float = 0.05,
    hole_density: float = 0.0,
    outlier_density: float = 0.0,
    bandwidth: int = 1,
    cross_connect: bool = True,
    diagonal_cross_density: float = 0.0,
) -> Tuple[Graph, Dict[int, np.ndarray], Tuple[int, int]]:
    """
    Build a multi-lane linear synchronization graph.

    This topology models multi-row camera arrays or multi-strip aerial
    surveys: several parallel chains with optional vertical and diagonal
    cross-connections between adjacent lanes.

    Parameters
    ----------
    n : int
        Total number of nodes.  Distributed across lanes as evenly as
        possible: n_per_lane = ceil(n / num_lanes).  The last lane gets
        the remainder.
    num_lanes : int
        Number of parallel lanes (must satisfy num_lanes ≤ n).
    sigma : float
        Noise standard deviation for relative measurements.
    hole_density : float
        Fraction of *shortcut* edges (k ≥ 2) to drop within each lane.
        Backbone edges (k = 1) are never dropped.
    outlier_density : float
        Fraction of remaining edges replaced by random outliers.
    bandwidth : int
        Maximum intra-lane hop distance for edges.
    cross_connect : bool
        If True, add vertical edges between aligned nodes in adjacent lanes
        (never dropped).
    diagonal_cross_density : float
        Probability of adding diagonal cross-edges between adjacent lanes.

    Returns
    -------
    graph        : Graph
    ground_truth : dict  {node_id → absolute homography}
    layout_shape : (num_lanes, n_per_lane)
        Layout metadata for visualisation.
    """
    if num_lanes > n:
        raise ValueError(f"num_lanes ({num_lanes}) cannot exceed n ({n}).")

    n_per_lane = int(np.ceil(n / num_lanes))

    # Actual lane sizes (last lane may be shorter).
    lane_sizes = [n_per_lane] * (num_lanes - 1)
    lane_sizes.append(n - n_per_lane * (num_lanes - 1))  # remainder

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
        """Add an edge if it does not already exist (deduplicated)."""
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
            is_backbone = k == 1
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
                                node_id(lane, col),
                                node_id(lane + 1, target_col),
                            )

    # 5. Outlier injection.
    if outlier_density > 0:
        edge_keys = [(i, j) for (i, j) in graph.edges.keys() if i < j]
        n_outliers = int(len(edge_keys) * outlier_density)
        for idx in np.random.choice(
            len(edge_keys), size=n_outliers, replace=False
        ):
            i, j = edge_keys[idx]
            add_outlier_edge(graph, i, j)

    return graph, ground_truth, (num_lanes, n_per_lane)


# =========================================================================
# Graph visualisation
# =========================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    for i, j in graph.edges.keys():
        if i < j:
            G.add_edge(i, j)

    n = G.number_of_nodes()
    nodes = sorted(G.nodes())

    # ── 2. Layout auto-detection ─────────────────────────────────────────────
    def _is_path_like(G: nx.Graph) -> bool:
        """True when the graph looks like a (possibly sparse) chain."""
        degrees = [d for _, d in G.degree()]
        return (
            max(degrees) <= 3
            and nx.is_connected(G)
            and not nx.is_biconnected(G)
        )

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

    inlier_edgelist = []
    outlier_edgelist = []
    for i, j in G.edges():
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
        G,
        pos,
        edgelist=inlier_edgelist,
        ax=ax,
        edge_color="#01696f",
        alpha=0.65,
        width=1.6,
    )
    # Outlier edges
    if outlier_edgelist:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=outlier_edgelist,
            ax=ax,
            edge_color="#a12c7b",
            alpha=0.85,
            width=2.2,
            style="dashed",
        )

    # Node fill: white if initialised (all vertices start at I), teal if solved
    node_colors = [
        "#cedcd8" if np.allclose(graph.vertices[v], np.eye(3)) else "#01696f"
        for v in nodes
    ]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes,
        ax=ax,
        node_color=node_colors,
        edgecolors="#01696f",
        node_size=max(80, min(500, 4000 // n)),
        linewidths=1.8,
    )

    # Labels — hide when the graph is large
    if n <= 40:
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_size=max(6, min(10, 120 // n)),
            font_color="#28251d",
            font_weight="bold",
        )

    # ── 6. Stats annotation ──────────────────────────────────────────────────
    n_edges = G.number_of_edges()
    n_out = len(outlier_edgelist)
    connected = nx.is_connected(G) if n_edges > 0 else False
    density = nx.density(G)

    stats = (
        f"n={n}  |  edges={n_edges}  |  density={density:.2f}\n"
        f"connected={'yes' if connected else 'NO ⚠'}  |  "
        f"outliers={n_out} ({100 * n_out / max(n_edges, 1):.0f}%)"
    )
    ax.text(
        0.02,
        0.02,
        stats,
        transform=ax.transAxes,
        fontsize=8,
        color="#7a7974",
        verticalalignment="bottom",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="#f9f8f5", edgecolor="#dcd9d5"
        ),
    )

    # ── 7. Legend ────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(
            facecolor="#cedcd8", edgecolor="#01696f", label="vertex  (I init)"
        ),
        mpatches.Patch(
            facecolor="#01696f", edgecolor="#01696f", label="vertex  (solved)"
        ),
        mpatches.Patch(facecolor="#01696f", alpha=0.65, label="inlier edge"),
    ]
    if outlier_edgelist:
        legend_items.append(
            mpatches.Patch(
                facecolor="#a12c7b", alpha=0.85, label="outlier edge"
            )
        )
    ax.legend(
        handles=legend_items,
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
        facecolor="#f9f8f5",
        edgecolor="#dcd9d5",
    )

    ax.set_title(title, fontsize=12, fontweight="bold", color="#28251d", pad=12)
    ax.axis("off")
    plt.tight_layout()
    return ax
