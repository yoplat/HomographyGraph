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
from typing import Dict, List, Tuple, Optional, Callable, Any
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
        self._averaging_map: Dict[str, Callable[[List[np.ndarray]], np.ndarray]] = {
            "euclidean": self._averaging_euclidean,
            "direction": self._averaging_direction,
            "sphere":    self._averaging_sphere,
        }

    # ======================================================================
    # Graph construction
    # ======================================================================

    def add_vertex(self, uid: int, initial_proj: Optional[np.ndarray] = None) -> None:
        """Add a vertex with an optional initial absolute homography (default: I)."""
        self.vertices[uid] = initial_proj if initial_proj is not None else np.eye(3)
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
        # Unit-norm representatives in R^9.
        h_vecs = [h.flatten() / np.linalg.norm(h.flatten()) for h in estimates]

        # Initialise with the Euclidean mean.
        c = np.mean(h_vecs, axis=0)
        c /= np.linalg.norm(c)

        for _ in range(10):  # Fixed-point iterations (converges fast).
            weights = []
            for h in h_vecs:
                dot = np.clip(np.dot(c, h), -1.0, 1.0)
                denom = np.sqrt(1 - dot ** 2)
                weights.append(1.0 / max(denom, 1e-6))
            c = np.average(h_vecs, axis=0, weights=weights)
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
                estimates = [self.edges[(i, j)] @ self.vertices[j] for j in neighbours]

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
            raise ValueError(f"Unknown spectral method '{method}'. Use 'lsh' or 'gsh'.")

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
            S[r:r+3, r:r+3] = np.eye(3) / zeta_k

            # Off-diagonal blocks: for each neighbour i of k.
            for i in self.adj.get(k, set()):
                i_idx = uid_to_idx[i]
                c = i_idx * 3
                # H_{i,k} in Schroeder's notation = edges[(k, i)] in our convention.
                # (edges[(k,i)] = Z_{ki} = X_k · X_i^{-1}, which maps i→k.)
                H_ik = self.edges[(k, i)]
                S[r:r+3, c:c+3] = H_ik / zeta_k

        # Solve: minimise ||(S − I) · Û||  ⟹  Û = 3 right singular vectors
        # of (S − I) with smallest singular values.
        A = S - np.eye(3 * n)
        _, sigma, Vh = np.linalg.svd(A, full_matrices=True)

        # numpy returns singular values in descending order; pick last 3 rows of Vh.
        U_hat = Vh[-3:, :].T  # shape (3n, 3)

        # Assign the 3×3 block for each vertex.
        for idx, uid in enumerate(uids):
            raw = U_hat[idx*3:(idx+1)*3, :]
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
            G[r:r+3, r:r+3] = (1 - zeta_k) * np.eye(3)

            # Off-diagonal: H_{i,k} for each known neighbour i ≠ k.
            for i in self.adj.get(k, set()):
                i_idx = uid_to_idx[i]
                c = i_idx * 3
                H_ik = self.edges[(k, i)]
                G[r:r+3, c:c+3] = H_ik

        # Null space via SVD.
        _, sigma, Vh = np.linalg.svd(G, full_matrices=True)
        U_hat = Vh[-3:, :].T

        for idx, uid in enumerate(uids):
            raw = U_hat[idx*3:(idx+1)*3, :]
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
            root = max(self.vertices.keys(), key=lambda u: len(self.adj.get(u, set())))

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
                self.vertices[child] = self._norm_matrix(Z_cp @ self.vertices[parent])

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
