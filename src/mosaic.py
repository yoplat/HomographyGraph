"""
mosaic.py — Image mosaicking via homography synchronization.

This module provides the full pipeline for stitching a set of images into a
single mosaic using synchronized homographies:

  1. **Feature extraction** — SIFT keypoints + descriptors (CPU).
  2. **Descriptor matching** — GPU BFMatcher when CUDA is available, CPU fallback.
  3. **Pairwise homography estimation** — Lowe's ratio test + RANSAC.
  4. **Graph construction** — vertices = images, edges = relative homographies.
  5. **Synchronization** — either iterative [1] or spectral [2] (user's choice).
  6. **Mosaic rendering** — GPU-accelerated warpPerspective when available.

Usage
-----
    $ python mosaic.py

This will load images from the configured dataset path, run the selected
synchronization method, and display the resulting mosaic.

References
----------
[1] R. Madhavan, A. Fusiello, F. Arrigoni, "Synchronization of Projective
    Transformations", 2024.
[2] P. Schroeder, A. Bartoli, P. Georgel, N. Navab, "Closed-Form Solutions
    to Multiple-View Homography Estimation", IEEE WACV 2011.
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from graph import Graph

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MIN_MATCH_COUNT = 10      # Minimum inliers to accept a homography
LOWE_RATIO      = 0.7     # Lowe's ratio test threshold (lower = stricter)
RANSAC_THRESH   = 5.0     # RANSAC inlier pixel tolerance


# ---------------------------------------------------------------------------
# CUDA availability check
# ---------------------------------------------------------------------------
CUDA_AVAILABLE = cv.cuda.getCudaEnabledDeviceCount() > 0
print(f"[Init] CUDA available: {CUDA_AVAILABLE}")


# ---------------------------------------------------------------------------
# SIFT detector — created once and reused for every image
# ---------------------------------------------------------------------------
_sift = cv.SIFT_create()


# ---------------------------------------------------------------------------
# GPU / CPU BFMatcher — created once and reused across all pair comparisons
#
# BFMatcher (Brute Force) on GPU with NORM_L2 is both faster and more
# accurate than FLANN for SIFT's 128-dim float descriptors: FLANN uses
# approximate nearest-neighbour search, while BFMatcher is exact but
# GPU-parallelised, so it ends up faster at scale.
# ---------------------------------------------------------------------------
_matcher_gpu = (
    cv.cuda.DescriptorMatcher_createBFMatcher(cv.NORM_L2)
    if CUDA_AVAILABLE else None
)
_matcher_cpu = cv.BFMatcher(cv.NORM_L2) if not CUDA_AVAILABLE else None


# ===================================================================
# Feature extraction
# ===================================================================

def extract_features(img_gray: np.ndarray) -> Tuple[tuple, np.ndarray]:
    """
    Extract SIFT keypoints and descriptors from a grayscale image.

    SIFT runs on CPU — the CUDA contrib module exists but is poorly
    supported, and the gains are marginal compared to the matching step
    which dominates runtime.

    Parameters
    ----------
    img_gray : np.ndarray
        Single-channel uint8 grayscale image.

    Returns
    -------
    kp : tuple of cv2.KeyPoint
        Detected keypoints.
    des : np.ndarray
        (N, 128) float32 descriptor matrix.
    """
    kp, des = _sift.detectAndCompute(img_gray, None)
    return kp, des


# ===================================================================
# Descriptor matching
# ===================================================================

def match_descriptors(des1: np.ndarray, des2: np.ndarray):
    """
    Match two sets of SIFT descriptors using the GPU BFMatcher if
    available, falling back to CPU otherwise.

    Returns raw knnMatch results (k=2) for subsequent Lowe's ratio test.
    Descriptors are cast to float32 defensively — GPU matchers are strict
    about dtype.
    """
    if CUDA_AVAILABLE:
        # Upload descriptor matrices to GPU memory.
        des1_gpu = cv.cuda_GpuMat()
        des2_gpu = cv.cuda_GpuMat()
        des1_gpu.upload(des1.astype(np.float32))
        des2_gpu.upload(des2.astype(np.float32))
        return _matcher_gpu.knnMatch(des1_gpu, des2_gpu, k=2)
    else:
        return _matcher_cpu.knnMatch(
            des1.astype(np.float32), des2.astype(np.float32), k=2
        )


# ===================================================================
# Pairwise homography from precomputed features
# ===================================================================

def compute_homography_from_features(
    kp1, des1, kp2, des2
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute the relative homography between two images from their
    precomputed SIFT features.

    Pipeline:
      1. kNN matching (k=2) via GPU BFMatcher.
      2. Lowe's ratio test to discard ambiguous matches.
      3. RANSAC to robustly fit a homography and reject outliers.

    Parameters
    ----------
    kp1, kp2 : tuple of cv2.KeyPoint
        Keypoints from images 1 and 2.
    des1, des2 : np.ndarray
        Corresponding descriptor matrices.

    Returns
    -------
    H : np.ndarray or None
        3×3 homography mapping points from image 1 to image 2.
    inlier_pts1, inlier_pts2 : np.ndarray or None
        Matched inlier point coordinates after RANSAC.
    """
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return None, None, None

    matches = match_descriptors(des1, des2)

    # Lowe's ratio test: keep only matches where the best match is
    # significantly closer than the second-best, filtering false
    # positives from repetitive patterns.
    good = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]

    if len(good) <= MIN_MATCH_COUNT:
        return None, None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # RANSAC: find the homography maximising the inlier count.
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, RANSAC_THRESH)

    if M is None:
        return None, None, None

    inlier_mask = mask.ravel() == 1
    return M, src_pts[inlier_mask], dst_pts[inlier_mask]


# ===================================================================
# Graph construction from real images
# ===================================================================

def create_graph_from_images(
    dataset_paths: List[str],
) -> Tuple[Graph, Dict, Dict]:
    """
    Build a synchronization graph from a list of image file paths.

    Each image becomes a vertex; an edge (with its relative homography)
    is added for every pair of images that can be matched with enough
    inliers.  Features are extracted once per image — O(N) — rather
    than re-extracted inside the O(N²) matching loop.

    Parameters
    ----------
    dataset_paths : list of str
        File paths to the input images.

    Returns
    -------
    graph : Graph
        The constructed homography graph (vertices initialised to I).
    matches_data : dict[(int,int), (ndarray, ndarray)]
        Inlier point correspondences for each edge, used later to
        compute the reprojection error.
    precomputed : dict[int, (tuple, ndarray)]
        Keypoints and descriptors for every image index.
    """
    graph = Graph()
    matches_data = {}

    n = len(dataset_paths)
    print(f"[Step 1/2] Extracting SIFT features for {n} images...")

    # --- Extract features once per image (O(N)) ---
    precomputed = {}
    for i, path in enumerate(dataset_paths):
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if img is not None:
            kp, des = extract_features(img)
            precomputed[i] = (kp, des)
        else:
            print(f"  Warning: could not load {path}")
            precomputed[i] = (None, None)

    total_pairs = n * (n - 1) // 2
    print(f"[Step 2/2] Matching {total_pairs} pairs "
          f"(GPU={'yes' if CUDA_AVAILABLE else 'no'})...")

    # --- Pairwise matching (O(N²)) ---
    edges_found = 0
    for i in range(n):
        kp1, des1 = precomputed[i]
        if des1 is None:
            continue
        for j in range(i + 1, n):
            kp2, des2 = precomputed[j]
            if des2 is None:
                continue

            M, pts1, pts2 = compute_homography_from_features(kp1, des1, kp2, des2)

            if M is not None:
                # M maps image i → image j.
                # In our convention edges[(j, i)] = Z_{ji} = X_j · X_i^{-1} = M.
                graph.add_edge(j, i, M)
                matches_data[(i, j)] = (pts1, pts2)
                edges_found += 1

    print(f"  Done — {edges_found} edges from {total_pairs} pairs")
    return graph, matches_data, precomputed


# ===================================================================
# Reprojection error
# ===================================================================

def calculate_reprojection_error(graph: Graph, matches_data: Dict) -> float:
    """
    Measure synchronization quality via reprojection error.

    For every edge (i, j) with matched inlier points, warp the points
    from image i into image j using the *synchronised* absolute
    homographies and measure the pixel distance to the actual matched
    keypoint in j.

    Lower is better.  Typical good values are < 5 px.

    Parameters
    ----------
    graph : Graph
        Graph after synchronization (vertices carry absolute homographies).
    matches_data : dict
        Inlier correspondences keyed by (i, j).

    Returns
    -------
    float
        Mean reprojection error in pixels.
    """
    total_error = 0.0
    total_points = 0

    for (i, j), (pts_i, pts_j) in matches_data.items():
        X_i = graph.get_vertex_proj(i)
        X_j = graph.get_vertex_proj(j)
        if X_i is None or X_j is None:
            continue

        try:
            # Reconstruct relative transform from synchronised absolutes:
            # Z_ji_sync ≈ X_j · X_i^{-1}
            Z_ji_sync = X_j @ np.linalg.inv(X_i)
        except np.linalg.LinAlgError:
            continue

        pred_pts_j = cv.perspectiveTransform(pts_i, Z_ji_sync)
        if pred_pts_j is not None:
            errors = np.linalg.norm(
                pred_pts_j.squeeze() - pts_j.squeeze(), axis=1
            )
            total_error += np.sum(errors)
            total_points += len(errors)

    return total_error / total_points if total_points > 0 else float("inf")


# ===================================================================
# Mosaic creation
# ===================================================================

def create_mosaic(
    dataset_paths: List[str],
    graph: Graph,
    reference_idx: int = 0,
) -> Optional[np.ndarray]:
    """
    Stitch all images onto a single canvas using synchronised homographies.

    The reference image defines the output coordinate frame.  Every other
    image is warped into that frame via:
        H_final = T · X_ref · X_i^{-1}
    where T is a translation that shifts coordinates so nothing is negative.

    When CUDA is available, ``cv.cuda.warpPerspective`` is used — warping
    is embarrassingly parallel (every output pixel is independent) and maps
    perfectly to GPU execution.

    Parameters
    ----------
    dataset_paths : list of str
        File paths to the input images.
    graph : Graph
        Synchronised graph.
    reference_idx : int
        Index of the reference image (anchor).

    Returns
    -------
    mosaic : np.ndarray or None
        The stitched BGR mosaic, or None on failure.
    """
    print(f"\n[Mosaic] Anchoring to image {reference_idx}...")

    images = [cv.imread(p) for p in dataset_paths]

    X_ref = graph.get_vertex_proj(reference_idx)
    if X_ref is None:
        print(f"Error: reference image {reference_idx} not in graph.")
        return None

    # --- Pass 1: compute canvas bounds ---
    # Warp each image's corners to determine the total extent.
    homographies = {}
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    for i, img in enumerate(images):
        if img is None:
            continue
        X_i = graph.get_vertex_proj(i)
        if X_i is None:
            continue

        H = X_ref @ np.linalg.inv(X_i)
        H = H / H[2, 2]   # normalise so H[2,2] == 1
        homographies[i] = H

        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        wc = cv.perspectiveTransform(corners, H)

        min_x = min(min_x, wc[:, 0, 0].min())
        min_y = min(min_y, wc[:, 0, 1].min())
        max_x = max(max_x, wc[:, 0, 0].max())
        max_y = max(max_y, wc[:, 0, 1].max())

    # Translation so that all coordinates are non-negative.
    T = np.array([[1, 0, -min_x],
                  [0, 1, -min_y],
                  [0, 0,      1]])

    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))

    if canvas_w > 15000 or canvas_h > 15000:
        print(f"Canvas too large ({canvas_w}×{canvas_h}) "
              "— synchronization likely diverged.")
        return None

    # --- Pass 2: warp and composite ---
    mosaic = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        if i not in homographies or img is None:
            continue

        H_final = T @ homographies[i]

        if CUDA_AVAILABLE:
            img_gpu = cv.cuda_GpuMat()
            img_gpu.upload(img)
            warped_gpu = cv.cuda.warpPerspective(img_gpu, H_final,
                                                  (canvas_w, canvas_h))
            warped = warped_gpu.download()
        else:
            warped = cv.warpPerspective(img, H_final, (canvas_w, canvas_h))

        # Mask-based composite: only overwrite pixels where this image has content.
        gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        inv_mask = cv.bitwise_not(mask)
        mosaic = cv.add(cv.bitwise_and(mosaic, mosaic, mask=inv_mask), warped)

    return mosaic


# ===================================================================
# Diagnostics
# ===================================================================

def diagnose_graph(
    dataset_paths: List[str],
    graph: Graph,
    matches_data: Dict,
    precomputed: Dict,
) -> None:
    """
    Print a diagnostic report about the homography graph: feature counts,
    edge inlier counts, connectivity, etc.  Useful for debugging sparse
    or disconnected graphs.
    """
    n = len(dataset_paths)
    total_pairs = n * (n - 1) // 2
    edges = len(matches_data)

    print(f"\n{'='*50}")
    print("GRAPH DIAGNOSTICS")
    print(f"{'='*50}")
    print(f"Images:        {n}")
    print(f"Total pairs:   {total_pairs}")
    print(f"Edges found:   {edges} ({100*edges/total_pairs:.1f}% of pairs matched)")
    print(f"{'='*50}")

    print("\nFeatures per image:")
    for i, path in enumerate(dataset_paths):
        kp, des = precomputed[i]
        name = path.split("/")[-1]
        count = len(kp) if kp is not None else 0
        in_graph = graph.get_vertex_proj(i) is not None
        status = "IN GRAPH" if in_graph else "NOT IN GRAPH"
        print(f"  [{i}] {name}: {count} keypoints — {status}")

    print("\nEdge inlier counts:")
    for (i, j), (pts_i, _) in matches_data.items():
        print(f"  ({i},{j}): {len(pts_i)} inliers")

    print("\nConnectivity (adjacency):")
    for i in range(n):
        neighbours = [
            j
            for (a, b) in matches_data.keys()
            for j in ([b] if a == i else [a] if b == i else [])
        ]
        print(f"  [{i}] connects to: {neighbours if neighbours else 'NONE'}")
    print(f"{'='*50}\n")


# ===================================================================
# MAIN — run the full mosaicking pipeline
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Image mosaicking via homography synchronization."
    )
    parser.add_argument(
        "--dataset", type=str, default="Alcatraz_courtyard",
        help="Name of the dataset folder (default: Alcatraz_courtyard)."
    )
    parser.add_argument(
        "--start", type=int, default=2313,
        help="Starting image number (default: 2313)."
    )
    parser.add_argument(
        "--count", type=int, default=20,
        help="Number of images to use (default: 20)."
    )
    parser.add_argument(
        "--method", type=str, default="iterative",
        choices=["iterative", "spectral_lsh", "spectral_gsh", "tree"],
        help="Synchronization method (default: iterative)."
    )
    parser.add_argument(
        "--avg", type=str, default="sphere",
        choices=["euclidean", "direction", "sphere"],
        help="Averaging method for the iterative approach (default: sphere)."
    )
    parser.add_argument(
        "--ref", type=int, default=0,
        help="Reference image index for the mosaic (default: 0)."
    )
    args = parser.parse_args()

    # Build image paths.
    dataset = [
        f"../{args.dataset}/San_Francisco_{i}.jpg"
        for i in range(args.start, args.start + args.count)
    ]

    # 1. Build the homography graph from image features.
    g, point_matches, precomputed_features = create_graph_from_images(dataset)

    # Optional diagnostics.
    # diagnose_graph(dataset, g, point_matches, precomputed_features)

    # 2. Normalise all matrices into SL(3).
    g.normalize()

    # 3. Synchronize using the selected method.
    if args.method == "iterative":
        print(f"\n[Sync] Running iterative synchronization (avg={args.avg})...")
        g.synchronize_iterative(avg_method=args.avg, max_iters=20)
    elif args.method == "spectral_lsh":
        print("\n[Sync] Running spectral synchronization (LSH)...")
        g.synchronize_spectral(method="lsh")
    elif args.method == "spectral_gsh":
        print("\n[Sync] Running spectral synchronization (GSH)...")
        g.synchronize_spectral(method="gsh")
    elif args.method == "tree":
        print("\n[Sync] Running spanning-tree synchronization...")
        g.synchronize_tree()

    # 4. Evaluate.
    error = calculate_reprojection_error(g, point_matches)
    print(f"[Eval] Mean reprojection error: {error:.2f} px")

    # 5. Render the mosaic.
    mosaic = create_mosaic(dataset, g, reference_idx=args.ref)

    if mosaic is not None:
        mosaic_rgb = cv.cvtColor(mosaic, cv.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.title(f"Mosaic — method: {args.method}")
        plt.imshow(mosaic_rgb)
        plt.axis("off")
        plt.show()
