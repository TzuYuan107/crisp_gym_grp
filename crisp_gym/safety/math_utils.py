"""CBF math and ellipsoid-fitting utilities, ported from aegis's ``utils.py``.

Everything here is pure numpy/scipy/cvxpy/open3d math and does not depend on any
specific environment (LIBERO, robosuite, crisp_gym, ...), so it is carried over from
aegis unchanged apart from ``filtering_points``, which took a LIBERO ``task_suite_name``
string to pick a hardcoded region of interest. crisp_gym has no such concept, so it now
takes explicit ``xyz_bounds`` instead -- callers (e.g. `crisp_gym.safety.SafetyFilter`)
are responsible for supplying bounds that make sense for their own robot workspace.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def vector_hat(v: np.ndarray) -> np.ndarray:
    """Hat operator: R^3 -> so(3)."""
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def project_matrix(z: np.ndarray) -> np.ndarray:
    """Compute (I - z z^T) for a (near) unit vector z."""
    z = z / (np.linalg.norm(z) + 1e-12)
    return np.eye(3) - np.outer(z, z)


def compute_h_ij(
    p_i: np.ndarray,
    Q_i_diag: np.ndarray,
    R_i: np.ndarray,
    p_j: np.ndarray,
    Q_j_diag: np.ndarray,
    R_j: np.ndarray,
    z_ij: np.ndarray,
) -> float:
    """Value of the ellipsoid-ellipsoid CBF h for bodies i (ego) and j (obstacle)."""
    Q_i = np.diag(Q_i_diag)
    Q_j = np.diag(Q_j_diag)

    Qbar_i = R_i @ Q_i @ R_i.T
    Qbar_j = R_j @ Q_j @ R_j.T

    Qbar_i_inv = np.linalg.inv(Qbar_i)

    z = z_ij / np.linalg.norm(z_ij)

    term1 = np.linalg.norm(Qbar_j @ Qbar_i_inv @ z)
    term2 = (p_j - p_i).T @ Qbar_i_inv @ z
    denom = np.linalg.norm(Qbar_i_inv @ z)

    return (-term1 + term2 - 1.0) / denom


def compute_h_coeffs_3d(
    p_i: np.ndarray,
    Q_i_diag: np.ndarray,
    R_i: np.ndarray,
    p_j: np.ndarray,
    Q_j_diag: np.ndarray,
    R_j: np.ndarray,
    z: np.ndarray,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Compute the CBF value and the coefficients of its derivative wrt (v_i, omega_i, z_ij).

    Args:
        p_i: World position of the ego ellipsoid center (e.g. end-effector proxy).
        Q_i_diag: Diagonal of the ego ellipsoid shape matrix, in the ego's local frame.
        R_i: World-frame rotation matrix of the ego ellipsoid.
        p_j: World position of the obstacle ellipsoid center.
        Q_j_diag: Diagonal of the obstacle ellipsoid shape matrix, in the obstacle's local frame.
        R_j: World-frame rotation matrix of the obstacle ellipsoid.
        z: Current unit vector defining the separating-plane direction between i and j.
        eps: Small constant to avoid division by zero.

    Returns:
        a_v: Coefficients of dh/d(body-frame linear velocity of i).
        a_omega: Coefficients of dh/d(body-frame angular velocity of i).
        a_uz: Coefficients of dh/d(tangential rate of change of z).
        h: Current value of the CBF.
        mu_row: dh/dz (also used as the nominal drift for z).
    """
    Q_i = np.diag(Q_i_diag)
    Q_j = np.diag(Q_j_diag)
    Qbar_i = R_i @ Q_i @ R_i.T
    Qbar_j = R_j @ Q_j @ R_j.T
    Qbar_i_inv = np.linalg.inv(Qbar_i)
    Qbar_i_inv2 = Qbar_i_inv @ Qbar_i_inv
    Qbar_j2 = Qbar_j @ Qbar_j

    z = z / (np.linalg.norm(z) + eps)
    a_vec = Qbar_i_inv @ z
    denom = np.linalg.norm(a_vec) + eps
    b_vec = Qbar_j @ a_vec
    term1 = np.linalg.norm(b_vec) + eps
    sigma = term1 * denom + eps
    rho = 1.0 - (p_j - p_i).T @ a_vec + term1

    eta_row = -(1.0 / denom) * (z.T @ Qbar_i_inv)
    term_mu_1 = (rho / (denom**3 + eps)) * (z.T @ Qbar_i_inv2)
    term_mu_2 = (1.0 / denom) * ((p_j - p_i).T @ Qbar_i_inv)
    term_mu_3 = (1.0 / sigma) * (z.T @ Qbar_i_inv @ Qbar_j2 @ Qbar_i_inv)
    mu_row = term_mu_1 + term_mu_2 - term_mu_3

    tmp1 = z.T @ Qbar_i_inv2 @ vector_hat(z)
    left_vec = z.T @ Qbar_i_inv @ Qbar_j2
    Ja_vec = vector_hat(a_vec)
    tmp2 = left_vec @ (Ja_vec - Qbar_i_inv @ vector_hat(z))
    part_a = (p_j - p_i).T @ Qbar_i_inv @ vector_hat(z)
    part_b = z.T @ Qbar_i_inv @ vector_hat(p_j - p_i)
    tmp3 = part_a + part_b
    zeta_tilde = rho * (1.0 / (denom**3 + eps)) * tmp1 + (1.0 / sigma) * tmp2 + (1.0 / denom) * tmp3

    a_v = (eta_row @ R_i).ravel()
    a_omega = (zeta_tilde @ R_i).ravel()
    a_uz = (mu_row @ project_matrix(z)).ravel()

    h = compute_h_ij(p_i, Q_i_diag, R_i, p_j, Q_j_diag, R_j, z)

    return a_v, a_omega, a_uz, h, mu_row


def mvee_cvxpy(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the Minimum Volume Enclosing Ellipsoid (MVEE) around a point set.

    Args:
        P: (N, 3) array of points on/inside the desired ellipsoid (typically its convex hull).

    Returns:
        c: (3,) ellipsoid center.
        A: (3, 3) symmetric positive-definite matrix such that
            {x : (x - c)^T A (x - c) <= 1} is the fitted ellipsoid.
    """
    import cvxpy as cp

    N, d = P.shape

    M = cp.Variable((d, d), PSD=True)
    g = cp.Variable(d)

    objective = cp.Minimize(-cp.log_det(M))
    constraints = [cp.norm(M @ P[i] - g) <= 1 for i in range(N)]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except cp.SolverError:
        prob.solve(verbose=False)

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"MVEE convex optimization failed. Status: {prob.status}")

    M_opt = M.value
    g_opt = g.value

    c = np.linalg.solve(M_opt, g_opt)
    A = M_opt.T @ M_opt

    return c, A


def fit_ellipse(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an oriented ellipsoid (center, rotation, semi-axes) around a 3D point cloud."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]
    center, A = mvee_cvxpy(hull_pts)

    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.clip(eigvals, 1e-15, None)
    axes = 1.0 / np.sqrt(eigvals)

    sort_idx = np.argsort(axes)[::-1]
    axes = axes[sort_idx]
    R = eigvecs[:, sort_idx]

    return center, R, axes


def crop_to_bounds(
    pts: np.ndarray,
    xyz_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    """Keep only points inside an axis-aligned ``((xmin,xmax),(ymin,ymax),(zmin,zmax))`` box.

    Factored out of `filtering_points` so callers that want just the region-of-interest crop
    -- without its statistical-outlier/RANSAC-plane/DBSCAN steps, e.g.
    `fit_ellipse_from_height_profile`'s intended input -- don't have to duplicate this.
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = xyz_bounds
    keep = (
        (pts[:, 0] > xmin)
        & (pts[:, 0] < xmax)
        & (pts[:, 1] > ymin)
        & (pts[:, 1] < ymax)
        & (pts[:, 2] > zmin)
        & (pts[:, 2] < zmax)
    )
    return pts[keep]


def filtering_points(
    pts: np.ndarray,
    xyz_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    stat_nb_neighbors: int = 20,
    stat_std_ratio: float = 1.5,
    dbscan_eps: float = 0.015,
    dbscan_min_points: int = 20,
    plane_distance_threshold: float = 0.008,
    plane_min_fraction: float = 0.4,
) -> np.ndarray:
    """Filter a raw obstacle point cloud down to its largest, most central cluster.

    Args:
        pts: (N, 3) candidate obstacle points, in the same world frame as the robot.
        xyz_bounds: ((xmin, xmax), (ymin, ymax), (zmin, zmax)) region of interest to keep.
            aegis picked this per-LIBERO-task via hardcoded ``task_suite_name`` string
            matching; crisp_gym has no such concept, so callers must pass explicit bounds
            for their own workspace (see `SafetyFilterConfig.xyz_bounds`).
        stat_nb_neighbors: Neighbors examined per point for Open3D's statistical outlier
            removal. Unlike a fixed keep-fraction, this is a local-density check, so it
            actually rejects a sparse trailing streak (e.g. depth-camera "flying pixel"
            artifacts at the obstacle/table edge, worse on glass/reflective objects) instead
            of a percentage cut that a directional streak of noise can drag along with it.
        stat_std_ratio: How many std-devs above the mean neighbor distance counts as an
            outlier; lower is stricter.
        dbscan_eps: DBSCAN neighborhood radius (meters). aegis's original value (1e-4, i.e.
            0.1mm) was tuned for MuJoCo's noise-free ground-truth depth, where it worked
            because simulated points on the same flat surface land almost exactly on top of
            each other. A real depth camera's neighboring points are typically 1-10mm apart,
            so 0.1mm makes every point its own singleton cluster -- DBSCAN silently becomes a
            no-op and whatever noise passed the outlier-removal step above goes through
            unfiltered. 15mm is sized for a small tabletop object (see
            `SafetyFilterConfig.obstacle_shape_diag`).
        dbscan_min_points: DBSCAN minimum cluster size.
        plane_distance_threshold: RANSAC inlier distance (meters) for the table-removal plane
            fit below. Should be a couple of the depth camera's noise std-devs.
        plane_min_fraction: Only treat the RANSAC plane as "the table" and strip it if its
            inlier count is at least this fraction of the (outlier-removed) cloud. GroundingDINO's
            box is a rectangle around the obstacle, so it near-always includes some visible
            table around/behind it; that table is flat, large, and low-noise, so before this
            step it was routinely winning "largest DBSCAN cluster" over the actual (usually
            smaller) obstacle sitting on it. Below this fraction, assume there's no dominant
            background plane in view (e.g. a tight box) and don't risk eating a genuinely flat
            obstacle.

    Returns:
        Filtered (M, 3) point array. Empty if nothing survives the bounds/outlier filtering,
        or if no DBSCAN cluster reaches `dbscan_min_points` -- callers should treat that as
        "no obstacle detected this episode" (see `SafetyFilter.reset`'s `len(...) < 4` check)
        rather than trusting the pre-clustering points.
    """
    import open3d as o3d

    pts = crop_to_bounds(pts, xyz_bounds)
    if len(pts) == 0:
        return pts

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=stat_nb_neighbors, std_ratio=stat_std_ratio)
    if len(pcd.points) == 0:
        return np.zeros((0, 3))

    # Strip the dominant supporting/background plane (typically the table) before clustering,
    # so it can't out-vote the actual obstacle for "largest cluster" below.
    if len(pcd.points) >= 3:
        try:
            _, inliers = pcd.segment_plane(
                distance_threshold=plane_distance_threshold, ransac_n=3, num_iterations=1000
            )
        except RuntimeError:
            inliers = []
        if len(inliers) >= plane_min_fraction * len(pcd.points):
            pcd = pcd.select_by_index(inliers, invert=True)
    if len(pcd.points) == 0:
        return np.zeros((0, 3))

    labels = np.array(
        pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points, print_progress=False)
    )
    if labels.max() < 0:
        return np.zeros((0, 3))

    largest = np.bincount(labels[labels >= 0]).argmax()
    return np.asarray(pcd.points)[labels == largest]


def filter_points_by_color(
    colors: np.ndarray,
    reference_color: np.ndarray | None = None,
    color_distance_threshold: float = 40.0,
) -> np.ndarray:
    """Boolean mask of which points' colors are close enough to a reference color to keep.

    Independent of and composable with `filtering_points` -- this only looks at color, never
    position, so it's meant to run as an *additional* pass, e.g. on `filtering_points`'s own
    output, to prune any leftover points (a shadow, a reflection, a sliver of background the
    geometric filters didn't catch) whose color doesn't match the majority.

    Args:
        colors: (N, 3) RGB colors (uint8 or float), one per point, same order/length as
            whatever points array this mask will be applied to.
        reference_color: (3,) RGB to compare against. If None, uses the median color across
            `colors` itself -- e.g. call this after `filtering_points` has already picked out
            the obstacle's rough region, so the median approximates "the obstacle's real color."
        color_distance_threshold: Max Euclidean RGB distance (0..~441) from `reference_color`
            to keep a point. Lower is stricter.

    Returns:
        (N,) boolean mask, True where a point's color survived the threshold. All-False if
        `colors` is empty.
    """
    colors = np.asarray(colors, dtype=float)
    if len(colors) == 0:
        return np.zeros(0, dtype=bool)
    if reference_color is None:
        reference_color = np.median(colors, axis=0)
    distances = np.linalg.norm(colors - np.asarray(reference_color, dtype=float), axis=1)
    return distances <= color_distance_threshold


def _select_cluster_nearest_center(
    points: np.ndarray, labels: np.ndarray, reference_center: np.ndarray
) -> int:
    """Return the DBSCAN cluster label whose centroid is closest to `reference_center`.

    Used instead of picking the largest cluster: a competing structure (e.g. a background
    surface/streak) can have more points than the actual object of interest, especially when the
    object itself is sparse (distant, or only partially in view) -- point *count* alone is then
    an unreliable way to tell them apart. `reference_center` should be something expected to be
    near the true object regardless of that noise (e.g. the crop region's own center, since
    detection boxes are usually roughly centered on what they detect).
    """
    best_label, best_dist = -1, np.inf
    for label in np.unique(labels[labels >= 0]):
        centroid = points[labels == label].mean(axis=0)
        dist = float(np.linalg.norm(centroid - reference_center))
        if dist < best_dist:
            best_label, best_dist = label, dist
    return best_label


def fit_ellipse_from_height_profile(
    pts: np.ndarray,
    top_fraction: float = 0.15,
    center_fraction: float = 0.15,
    min_top_points: int = 10,
    radius_percentile: float = 100.0,
    height_percentile: float = 100.0,
    margin_fraction: float = 0.05,
    full_cluster_eps: float = 0.02,
    full_cluster_min_points: int = 10,
    top_cluster_eps: float = 0.02,
    top_cluster_min_points: int = 10,
    fixed_radius: float | None = None,
    center_x_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct an axis-aligned ellipsoid directly from height, instead of fitting an MVEE.

    Alternative to `fit_ellipse` for tall objects resting on a flat surface (e.g. a bottle or
    cylinder on a table), where the surface's own points are hard to reliably strip out (color
    drifts frame-to-frame with lighting/auto-exposure noise, RANSAC plane removal only triggers
    above a coverage threshold) but visually/geometrically only ever appear near the *bottom* of
    the cloud. Rather than trying to filter those points out (risking cutting into the object's
    own base, which sits at the same height as the table), this only uses the cloud's very top
    slice -- which nothing but the object itself can reach -- to measure the object's true
    horizontal extent, while still using the full height range (including the base) for the
    vertical extent, so nothing near the bottom is ever discarded.

    Assumes a roughly circular cross-section (same radius in x and y) -- reasonable for the
    cylindrical/bottle-shaped obstacles this is meant for, and much less sensitive to noise in a
    small top slice than fitting an oriented ellipse to it would be.

    Args:
        pts: (N, 3) points. Intended to be lightly filtered only (an xyz_bounds crop is enough)
            -- deliberately NOT run through Open3D's `remove_statistical_outlier` first: that's
            a local-density check, and the genuine top of a rounded object (its rim/cap) is
            naturally sparser than the rest (nothing "above" it to pack neighbors in), so it can
            get misclassified as noise and stripped -- which would silently cut the top off this
            function's own height measurement. NOT `filtering_points`'s full pipeline either --
            its plane-removal/DBSCAN steps are a different (and, for this use case, less
            reliable) strategy for the same table problem this function's top-slice trick
            already handles.
        top_fraction: Fraction of the z-range (from the top) treated as "uncontaminated by the
            table." Widened automatically (doubled, capped at 1.0) if it captures too few
            points -- see `min_top_points`. Only used for the horizontal *radius* -- see
            `center_fraction` for the x/y *center*.
        center_fraction: Fraction of the z-range, centered on the object's vertical middle, used
            for `center`'s x/y position instead of `top_fraction`'s slice (same widen-if-too-few-
            points behavior). With an imperfect `camera_to_world`, a rotation error's positional
            effect grows with distance from wherever it effectively pivots, so the *top* of a
            tall object accumulates more displacement than points nearer its middle -- sourcing
            the center from the top slice (as `radius` still does, deliberately unchanged) can
            then report a center measurably off from the object's actual location. Radius keeps
            using the top slice regardless of this setting.
        min_top_points: If the initial `top_fraction`/`center_fraction` slice has fewer points
            than this, widen it (see above) until it does, logging a warning when that happens.
        radius_percentile: Percentile of the top slice's distances from its own centroid used
            as the horizontal radius. Defaults to 100 (the true max) -- full, guaranteed
            coverage of every point in `pts` is preferred here over a tighter fit that might
            clip a real point; lower this (e.g. 95) if you'd rather trade some coverage for
            robustness against one stray point inflating the whole ellipsoid.
        height_percentile: Percentile of `pts`' z-values used for the vertical extent
            (`height_percentile` for the top, `100 - height_percentile` for the bottom).
            Defaults to 100 (the true min/max), for the same full-coverage reason as
            `radius_percentile` -- and because `remove_statistical_outlier`'s density-based
            outlier removal (deliberately not run on `pts` before this function, see above) was
            already the source of one such clipping bug, so this errs toward not clipping at all.
        margin_fraction: Extra padding applied after the above, as a further safety margin so the
            ellipsoid comfortably contains the cloud even under plotting/sampling discretization
            (`_ellipsoid_surface` only samples a finite grid of points on the true mathematical
            surface) rather than sitting exactly flush with it. Applied uniformly to all three
            semi-axes when `fixed_radius` is `None`. When `fixed_radius` is given, x is a fixed
            constant you already chose rather than something measured here, so it's left
            unpadded -- only y and z (both derived from percentile-trimmed point-cloud
            measurements) get the margin.
        full_cluster_eps: DBSCAN neighborhood radius (meters) used to cluster the *entire* input
            cloud before anything else -- only the cluster nearest `pts`' own bounding-box center
            is kept (see `_select_cluster_nearest_center`), before `z_max`/`z_min`/the top slice
            are even computed. Without this, a competing structure (e.g. a background surface/
            streak, or one that happens to reach higher in z than the real object) that has more
            points than the real object -- easy when the object itself is small/distant/sparse --
            would silently hijack `z_max`/`z_min`, which then hijacks which points the top slice
            even looks at -- at that point `top_cluster_eps`/`radius_percentile` below are
            clustering/trimming noise, not the object, because the object's real top was never in
            the slice to begin with. This runs first specifically to stop that upstream failure
            mode. Picking the nearest-to-center cluster rather than the largest one is what makes
            this robust even when the competing structure has more points than the real object.
        full_cluster_min_points: DBSCAN minimum cluster size for the above. If the input cloud has
            fewer points than this, clustering is skipped (falls back to using the whole cloud).
        top_cluster_eps: DBSCAN neighborhood radius (meters) used to cluster the top slice before
            measuring its radius -- only the cluster nearest `pts`' own bounding-box center
            (projected to xy) is kept, same selection rule as `full_cluster_eps` above. Without
            this, a disjoint noise streak (e.g. a flying-pixel depth artifact trailing away from
            the object, common on glass/reflective surfaces) that happens to overlap the top
            slice's z-range gets included in the centroid/radius calculation wholesale, not just
            as a handful of extreme outliers `radius_percentile` could trim -- it can be a large
            fraction of the slice while still being a physically separate, separable cluster from
            the real object's top. (Only catches streaks that share the object's actual z-range;
            see `full_cluster_eps` above for streaks that don't.)
        top_cluster_min_points: DBSCAN minimum cluster size for the above. If the top slice has
            fewer points than this, clustering is skipped (falls back to using the whole slice).
        fixed_radius: If given, used directly as the horizontal radius instead of measuring it
            from the top slice's spread (`radius_percentile` is then unused). `center`'s x/y
            position (see `center_fraction`) and z_min/z_max are still measured from `pts`
            exactly as usual -- only the radius itself becomes a constant. For a known,
            repeatedly-detected obstacle whose real size doesn't change, this avoids radius
            measurements from getting less precise at longer range, where GroundingDINO box slop
            and depth-sensor noise both get amplified by the pixel-to-metric unprojection (see
            `crisp_gym.safety.perception.get_point_cloud`) -- letting the ellipsoid balloon toward
            a sphere for a distant obstacle even though its true shape hasn't changed.
        center_x_offset: Subtracted from `center`'s x after it's otherwise fully derived (see
            `center_fraction`). An empirical correction for residual `camera_to_world` error that
            a `center_fraction` adjustment alone doesn't fully cancel out -- 0.0 (no-op) unless
            you've measured a specific offset for your setup.

    Returns:
        center: (3,) ellipsoid center.
        R: (3, 3) identity -- this construction has no meaningful orientation to report (circular
            cross-section, axis-aligned by definition).
        axes: (3,) semi-axis lengths in (x, y, z) order to match R=I -- NOT sorted by magnitude,
            unlike `fit_ellipse`.
    """
    if len(pts) == 0:
        raise ValueError("fit_ellipse_from_height_profile: empty point cloud.")

    # Reference point for cluster selection below: `pts`' own median position -- fixed once, from
    # the *original* input, so it doesn't drift if/when a cluster-selection step below narrows
    # `pts` down to a subset of itself. Deliberately the median, not the bounding-box midpoint
    # ((min+max)/2): the crop region often contains scattered background/noise (e.g. a glare
    # reflection) that can sit far from the real object along some axis, and a bounding-box
    # midpoint is dragged by even a single such outlier -- sometimes past the real object and
    # onto the noise -- whereas the median tracks wherever the bulk of the points actually are.
    reference_center = np.median(pts, axis=0)

    # Cluster the whole cropped cloud first and keep only the cluster nearest reference_center,
    # before z_max/z_min/the top slice are even computed -- see full_cluster_eps's docstring.
    if len(pts) >= full_cluster_min_points:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        labels = np.array(
            pcd.cluster_dbscan(eps=full_cluster_eps, min_points=full_cluster_min_points, print_progress=False)
        )
        if labels.max() >= 0:
            chosen = _select_cluster_nearest_center(pts, labels, reference_center)
            pts = pts[labels == chosen]

    z = pts[:, 2]
    z_max = float(np.percentile(z, height_percentile))
    z_min = float(np.percentile(z, 100.0 - height_percentile))
    z_range = z_max - z_min

    frac = top_fraction
    top_mask = z >= (z_max - frac * z_range) if z_range > 0 else np.ones_like(z, dtype=bool)
    while top_mask.sum() < min_top_points and frac < 1.0:
        frac = min(frac * 2, 1.0)
        top_mask = z >= (z_max - frac * z_range) if z_range > 0 else np.ones_like(z, dtype=bool)
    if frac > top_fraction:
        logger.warning(
            "fit_ellipse_from_height_profile: widened top_fraction %.3f -> %.3f to reach "
            "min_top_points=%d (only had %d points in the original top slice).",
            top_fraction, frac, min_top_points, int(top_mask.sum()),
        )

    top_xy = pts[top_mask, :2]

    # Keep only the top slice's cluster nearest reference_center before measuring radius, so a
    # disjoint noise streak that overlaps the top slice's z-range can't drag the radius out just
    # because it's physically present there (or has more points) -- see top_cluster_eps's
    # docstring above.
    if len(top_xy) >= top_cluster_min_points:
        import open3d as o3d

        top_xyz = np.hstack([top_xy, np.zeros((len(top_xy), 1))])  # DBSCAN here needs 3D input
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(top_xyz))
        labels = np.array(
            pcd.cluster_dbscan(eps=top_cluster_eps, min_points=top_cluster_min_points, print_progress=False)
        )
        if labels.max() >= 0:
            chosen = _select_cluster_nearest_center(top_xy, labels, reference_center[:2])
            top_xy = top_xy[labels == chosen]

    top_centroid_xy = top_xy.mean(axis=0)
    if fixed_radius is None:
        radial_dist = np.linalg.norm(top_xy - top_centroid_xy, axis=1)
        radius = float(np.percentile(radial_dist, radius_percentile)) if len(radial_dist) > 0 else 0.0

    # x/y center: a separate slice around the object's vertical MIDDLE rather than the top slice
    # above -- see center_fraction's docstring for why (radius deliberately keeps using the top
    # slice/top_centroid_xy regardless of this).
    mid_z = (z_min + z_max) / 2.0
    center_frac = center_fraction
    center_mask = (
        (z >= mid_z - 0.5 * center_frac * z_range) & (z <= mid_z + 0.5 * center_frac * z_range)
        if z_range > 0 else np.ones_like(z, dtype=bool)
    )
    while center_mask.sum() < min_top_points and center_frac < 1.0:
        center_frac = min(center_frac * 2, 1.0)
        center_mask = (
            (z >= mid_z - 0.5 * center_frac * z_range) & (z <= mid_z + 0.5 * center_frac * z_range)
            if z_range > 0 else np.ones_like(z, dtype=bool)
        )
    if center_frac > center_fraction:
        logger.warning(
            "fit_ellipse_from_height_profile: widened center_fraction %.3f -> %.3f to reach "
            "min_top_points=%d (only had %d points in the original middle slice).",
            center_fraction, center_frac, min_top_points, int(center_mask.sum()),
        )

    center_xy_source = pts[center_mask, :2]
    if len(center_xy_source) >= top_cluster_min_points:
        import open3d as o3d

        center_xyz = np.hstack([center_xy_source, np.zeros((len(center_xy_source), 1))])
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(center_xyz))
        labels = np.array(
            pcd.cluster_dbscan(eps=top_cluster_eps, min_points=top_cluster_min_points, print_progress=False)
        )
        if labels.max() >= 0:
            chosen = _select_cluster_nearest_center(center_xy_source, labels, reference_center[:2])
            center_xy_source = center_xy_source[labels == chosen]
    center_xy = center_xy_source.mean(axis=0) if len(center_xy_source) > 0 else top_centroid_xy

    if fixed_radius is not None:
        # x ("thick") is the fixed constant; y ("width") is measured as a percentile-trimmed
        # range (y_max - y_min) / 2 -- same style as z_range above -- over the SAME middle-slice
        # point set center_xy itself comes from (center_fraction), not the top slice, so width
        # and center are measured at the same z-region. x stays fixed because a single camera
        # view can't reliably measure depth into the object (only the near/front surface is ever
        # seen); y is the direction actually visible face-on, so it's measured.
        x_semi_axis = fixed_radius
        if len(center_xy_source) > 0:
            y_vals = center_xy_source[:, 1]
            y_max = float(np.percentile(y_vals, radius_percentile))
            y_min = float(np.percentile(y_vals, 100.0 - radius_percentile))
            y_semi_axis = (y_max - y_min) / 2.0
        else:
            y_semi_axis = x_semi_axis
    else:
        x_semi_axis = radius
        y_semi_axis = radius

    center = np.array([center_xy[0] - center_x_offset, center_xy[1], (z_min + z_max) / 2.0])
    if fixed_radius is not None:
        # x is a fixed constant you already chose, not something trimmed/measured here -- only
        # pad the two axes margin_fraction is actually meant to add safety margin to (y and z,
        # both derived from percentile-trimmed point-cloud measurements above).
        axes = np.array([x_semi_axis, y_semi_axis * (1.0 + margin_fraction), (z_range / 2.0) * (1.0 + margin_fraction)])
    else:
        axes = np.array([x_semi_axis, y_semi_axis, z_range / 2.0]) * (1.0 + margin_fraction)
    R = np.eye(3)
    return center, R, axes
