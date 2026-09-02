"""Per-episode debug artifacts for `SafetyFilter`: obstacle point clouds and a 3D plot
comparing the raw/unfiltered trajectory against the safety-filtered one actually executed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def save_point_cloud(points: np.ndarray, path: Path) -> None:
    """Save a point cloud to a .ply file and log a one-line summary (count/centroid/bbox)."""
    centroid = points.mean(axis=0)
    bbox_min, bbox_max = points.min(axis=0), points.max(axis=0)
    logger.info(
        "SafetyFilter: obstacle point cloud has %d points, centroid=%s, bbox=[%s, %s]",
        len(points),
        np.array2string(centroid, precision=3),
        np.array2string(bbox_min, precision=3),
        np.array2string(bbox_max, precision=3),
    )

    import open3d as o3d

    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(path), pcd)
    logger.info("SafetyFilter: obstacle point cloud saved to %s", path)


def _ellipsoid_surface(
    center: np.ndarray, rotation: np.ndarray, semi_axes: np.ndarray, n: int = 24
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample points on an oriented ellipsoid's surface for plotting."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    a, b, c = semi_axes
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones_like(u), np.cos(v))
    ellipsoid = np.stack([x, y, z], axis=-1) @ rotation.T + center
    return ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2]


def _ellipsoid_point_cloud(
    center: np.ndarray, rotation: np.ndarray, semi_axes: np.ndarray, n: int = 400
) -> np.ndarray:
    """Evenly scatter `n` points over an oriented ellipsoid's surface (Fibonacci sphere lattice).

    Used to render the CBF's end-effector proxy ellipsoid (`SafetyFilter.p1/R1/ee_shape_diag`)
    as a "gripper point cloud", visually consistent with the obstacle's real depth-camera points.
    """
    indices = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    unit_sphere = np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=-1
    )
    return (unit_sphere * np.asarray(semi_axes)) @ rotation.T + center


def _arrow_visibility_scale(
    trajectory_actual: np.ndarray, *delta_arrays: np.ndarray, target_fraction: float = 0.15
) -> float:
    """Compute one scale factor shared across all given arrow sets, for visibility.

    A one-step CBF correction is typically a few mm to a few cm, while the trajectory itself
    spans tens of cm -- rendered at true scale (matplotlib's `quiver` and plotly's `Cone` with
    `sizemode="scaled"` both do this by default), arrows are technically present but visually
    imperceptible specks. Returns a single multiplier, computed from the largest magnitude
    across *all* of `delta_arrays` combined, so applying it to each preserves relative
    magnitude both within and between arrow sets (e.g. "predicted" vs. "modified" arrows stay
    directly comparable to each other, not each independently normalized to its own max).
    """
    mags = [np.linalg.norm(d, axis=1) for d in delta_arrays if len(d) > 0]
    if not mags:
        return 1.0
    max_mag = float(np.concatenate(mags).max())
    extent = float(np.max(trajectory_actual.max(axis=0) - trajectory_actual.min(axis=0)))
    if extent <= 0 or max_mag <= 0:
        return 1.0
    return (target_fraction * extent) / max_mag


def _perpendicular_offset(*delta_arrays: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to the average direction of the given arrow deltas.

    Used to nudge two arrow sets (predicted vs. modified) apart when they'd otherwise sit
    exactly on top of each other -- which is the *common* case, since most steps have zero
    correction (predicted == modified). Without an offset, whichever trace is drawn second
    fully hides the other wherever they coincide.
    """
    all_deltas = np.concatenate([d for d in delta_arrays if len(d) > 0], axis=0)
    avg_dir = all_deltas.mean(axis=0)
    norm = np.linalg.norm(avg_dir)
    avg_dir = avg_dir / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    reference = np.array([0.0, 0.0, 1.0]) if abs(avg_dir[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    perp = np.cross(avg_dir, reference)
    perp_norm = np.linalg.norm(perp)
    return perp / perp_norm if perp_norm > 1e-9 else np.array([0.0, 1.0, 0.0])


def plot_episode_debug(
    obstacle_points: np.ndarray,
    ellipsoid_center: np.ndarray,
    ellipsoid_rotation: np.ndarray,
    ellipsoid_semi_axes: np.ndarray,
    trajectory_actual: np.ndarray,
    trajectory_reference: np.ndarray,
    save_path: Path,
    gripper_center: np.ndarray | None = None,
    gripper_rotation: np.ndarray | None = None,
    gripper_semi_axes: np.ndarray | None = None,
    trajectory_predicted: np.ndarray | None = None,
    trajectory_modified: np.ndarray | None = None,
    trajectory_comparison: np.ndarray | None = None,
    trajectory_comparison_collision: np.ndarray | None = None,
) -> None:
    """Save a 3D plot comparing the safety-filtered trajectory to the raw/unfiltered one.

    Which pose drives `h`/correction (`trajectory_actual`, blue) versus which is comparison-only
    (`trajectory_comparison`, magenta) is a *caller* choice made in `SafetyFilter.step()`'s call
    site -- both labels below are written generically since either line could be end-effector
    or target-setpoint data depending on that choice; check the call site for which is which.

    Args:
        obstacle_points: (N, 3) filtered obstacle point cloud used for this episode's fit.
        ellipsoid_center: (3,) fitted obstacle ellipsoid center.
        ellipsoid_rotation: (3, 3) fitted obstacle ellipsoid orientation.
        ellipsoid_semi_axes: (3,) fitted obstacle ellipsoid semi-axis lengths.
        trajectory_actual: (T, 3) positions of the pose that actually drives `h`/correction
            each step (whatever was passed as `SafetyFilter.step()`'s `ee_pose`) -- i.e. the
            safety-filtered/"modified" path *for that pose*.
        trajectory_reference: (T, 3) positions from naively summing the raw, *uncorrected*
            commanded deltas from the same start pose -- the "original" path the unfiltered
            policy's own outputs would have traced had the safety filter never intervened.
        save_path: PNG output path.
        gripper_center: (3,) end-effector proxy ellipsoid center at episode end
            (`SafetyFilter.p1`). If None, the gripper point cloud is omitted.
        gripper_rotation: (3, 3) end-effector proxy ellipsoid orientation (`SafetyFilter.R1`).
        gripper_semi_axes: (3,) end-effector proxy ellipsoid semi-axis lengths
            (`SafetyFilterConfig.ee_shape_diag`).
        trajectory_predicted: (T-1, 3) one-step-ahead raw/uncorrected prediction from each
            *actually visited* point (`trajectory_actual[i+1] -> trajectory_predicted[i]` is
            "if we hadn't corrected here, where would this one step have gone" -- unlike
            `trajectory_reference`, which drifts away from reality by cumulatively
            re-integrating from episode start, this is anchored at the real position each
            step, so it never compounds. Rendered as purple per-step arrows. If None, omitted.
        trajectory_modified: (T-1, 3) one-step-ahead *corrected* delta, anchored at the same
            points as `trajectory_predicted` -- "what the CBF actually commanded from here."
            Rendered as red per-step arrows, scaled to the same reference as the purple ones
            so the two are directly comparable. If None, omitted.
        trajectory_comparison: (T, 3) the *second* pose passed to `step()` -- comparison only,
            never used by any h/correction/QP computation regardless of what it shows (see
            `SafetyFilter.step`'s docstring). Shows how far it diverges from `trajectory_actual`.
            Rendered as a magenta line. If None, omitted.
        trajectory_comparison_collision: (T,) bool per point on `trajectory_comparison`, True
            where *that pose itself* -- independently, via its own `compute_h_ij` against the
            same obstacle ellipsoid, never reusing `trajectory_actual`'s `h` -- would be in
            violation. E.g. lets you mark where the real arm position would actually collide,
            even while a different pose is the one driving corrections. Rendered as star
            markers. If None, omitted.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    # TRIMMED: only the safety-filtered ("modified", blue) trajectory is plotted -- obstacle/
    # gripper point clouds, the ellipsoid surface, the unfiltered/comparison trajectories, and
    # the per-step predicted/modified delta arrows were all removed per request.
    if len(trajectory_actual) > 1:
        ax.plot(*trajectory_actual.T, color="tab:blue", linewidth=2, label="modified (safety-filtered, drives h)")
        ax.scatter(*trajectory_actual[0].T, color="tab:blue", marker="o", s=40)
        ax.scatter(*trajectory_actual[-1].T, color="tab:blue", marker="^", s=60)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left")

    # ADDED: matplotlib's 3D default camera (elev=30, azim=-60) put the trajectory running
    # mostly diagonally across the view, hard to read. Rotating azim by +90 swaps which
    # screen-direction the x/y axes point toward (x's on-screen position ~= where y used to be)
    # without touching any of the plotted data -- just the viewing angle.
    ax.view_init(elev=30, azim=-60)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("SafetyFilter: episode trajectory plot saved to %s", save_path)


def plot_obstacle_pointcloud_debug(
    obstacle_points: np.ndarray,
    ellipsoid_center: np.ndarray | None,
    ellipsoid_rotation: np.ndarray | None,
    ellipsoid_semi_axes: np.ndarray | None,
    save_path: Path,
) -> None:
    """Same figure layout/camera angle as `plot_episode_debug`, but for `debug_obstacle_pointcloud.py`
    (no live trajectory there) -- just the obstacle point cloud and its fitted ellipsoid, if any.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    if len(obstacle_points) > 0:
        ax.scatter(*obstacle_points.T, s=3, c="gray", alpha=0.6, label="obstacle point cloud")

    if ellipsoid_center is not None:
        ex, ey, ez = _ellipsoid_surface(ellipsoid_center, ellipsoid_rotation, ellipsoid_semi_axes)
        ax.plot_surface(ex, ey, ez, color="tab:blue", alpha=0.3, linewidth=0)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left")
    ax.view_init(elev=30, azim=-60)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Obstacle point cloud plot saved to %s", save_path)


def plot_episode_debug_interactive(
    obstacle_points: np.ndarray,
    ellipsoid_center: np.ndarray,
    ellipsoid_rotation: np.ndarray,
    ellipsoid_semi_axes: np.ndarray,
    trajectory_actual: np.ndarray,
    trajectory_reference: np.ndarray,
    save_path: Path,
    gripper_center: np.ndarray | None = None,
    gripper_rotation: np.ndarray | None = None,
    gripper_semi_axes: np.ndarray | None = None,
    trajectory_predicted: np.ndarray | None = None,
    trajectory_modified: np.ndarray | None = None,
    trajectory_comparison: np.ndarray | None = None,
    trajectory_comparison_collision: np.ndarray | None = None,
) -> None:
    """Same content as `plot_episode_debug`, but as a standalone HTML you can rotate/zoom/pan.

    matplotlib's 3D plots are rasterized at save time, so the PNG's viewing angle is fixed.
    This uses plotly instead, which ships its own interactive JS viewer embedded in the HTML
    file -- open it directly in a browser, no server needed.

    Which pose drives `h`/correction (`trajectory_actual`) versus which is comparison-only
    (`trajectory_comparison`) is a caller choice made at `SafetyFilter.step()`'s call site -- see
    `plot_episode_debug`'s docstring for the same note.

    Args:
        gripper_center: (3,) end-effector proxy ellipsoid center at episode end
            (`SafetyFilter.p1`). If None, the gripper point cloud is omitted.
        gripper_rotation: (3, 3) end-effector proxy ellipsoid orientation (`SafetyFilter.R1`).
        gripper_semi_axes: (3,) end-effector proxy ellipsoid semi-axis lengths
            (`SafetyFilterConfig.ee_shape_diag`).
        trajectory_predicted: (T-1, 3) one-step-ahead raw/uncorrected prediction anchored at
            each actually-visited point -- see `plot_episode_debug`'s docstring. Rendered as
            purple per-step arrows (`bases=trajectory_actual[1:]`, pointing toward
            `trajectory_predicted`). If None, omitted.
        trajectory_modified: (T-1, 3) one-step-ahead *corrected* delta, same anchor points as
            `trajectory_predicted`. Rendered as red per-step arrows, scaled to the same
            reference as the purple ones for direct comparability. If None, omitted.
        trajectory_comparison: (T, 3) the *second* pose passed to `step()` -- comparison only,
            never used by any h/correction/QP computation. Rendered as a magenta dashed line.
            If None, omitted.
        trajectory_comparison_collision: (T,) bool per point on `trajectory_comparison`, True
            where that pose's own, independently-computed `h` is negative -- see
            `plot_episode_debug`'s docstring for the full explanation. Rendered as star-standin
            (diamond) markers. If None, omitted.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    if len(obstacle_points) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=obstacle_points[:, 0],
                y=obstacle_points[:, 1],
                z=obstacle_points[:, 2],
                mode="markers",
                marker=dict(size=2, color="gray", opacity=0.4),
                name="obstacle point cloud",
            )
        )

    ex, ey, ez = _ellipsoid_surface(ellipsoid_center, ellipsoid_rotation, ellipsoid_semi_axes)
    fig.add_trace(
        go.Surface(
            x=ex,
            y=ey,
            z=ez,
            colorscale=[[0, "red"], [1, "red"]],
            showscale=False,
            opacity=0.3,
            name="obstacle ellipsoid",
        )
    )

    if gripper_center is not None:
        gripper_points = _ellipsoid_point_cloud(gripper_center, gripper_rotation, gripper_semi_axes)
        fig.add_trace(
            go.Scatter3d(
                x=gripper_points[:, 0],
                y=gripper_points[:, 1],
                z=gripper_points[:, 2],
                mode="markers",
                marker=dict(size=2, color="green", opacity=0.5),
                name="gripper point cloud",
            )
        )

    if len(trajectory_reference) > 1:
        fig.add_trace(
            go.Scatter3d(
                x=trajectory_reference[:, 0],
                y=trajectory_reference[:, 1],
                z=trajectory_reference[:, 2],
                mode="lines+markers",
                line=dict(color="green", width=5),
                marker=dict(size=3, color="green"),
                name="original (unfiltered, cumulative)",
            )
        )
    if len(trajectory_actual) > 1:
        fig.add_trace(
            go.Scatter3d(
                x=trajectory_actual[:, 0],
                y=trajectory_actual[:, 1],
                z=trajectory_actual[:, 2],
                mode="lines+markers",
                line=dict(color="blue", width=5),
                marker=dict(size=3, color="blue"),
                name="modified (safety-filtered, drives h)",
            )
        )
    if trajectory_comparison is not None and len(trajectory_comparison) > 1:
        fig.add_trace(
            go.Scatter3d(
                x=trajectory_comparison[:, 0],
                y=trajectory_comparison[:, 1],
                z=trajectory_comparison[:, 2],
                mode="lines",
                line=dict(color="magenta", width=3, dash="dash"),
                name="second pose (comparison only)",
            )
        )
        if trajectory_comparison_collision is not None and len(trajectory_comparison_collision) == len(trajectory_comparison):
            hit = np.asarray(trajectory_comparison_collision, dtype=bool)
            if hit.any():
                # plotly's Scatter3d marker symbols don't include an actual star (that's a
                # 2D-scatter-only shape) -- a large gold diamond is the closest high-contrast
                # stand-in; the matplotlib PNG uses a real star marker.
                fig.add_trace(
                    go.Scatter3d(
                        x=trajectory_comparison[hit, 0],
                        y=trajectory_comparison[hit, 1],
                        z=trajectory_comparison[hit, 2],
                        mode="markers",
                        marker=dict(size=7, color="gold", symbol="diamond", line=dict(color="black", width=1)),
                        name="comparison pose collides here",
                    )
                )

    have_predicted = trajectory_predicted is not None and len(trajectory_predicted) > 0
    have_modified = trajectory_modified is not None and len(trajectory_modified) > 0
    if have_predicted or have_modified:
        n = len(trajectory_actual) - 1
        if have_predicted:
            n = min(n, len(trajectory_predicted))
        if have_modified:
            n = min(n, len(trajectory_modified))
        bases = trajectory_actual[1 : 1 + n]
        pred_deltas = trajectory_predicted[:n] - bases if have_predicted else np.zeros((0, 3))
        mod_deltas = trajectory_modified[:n] - bases if have_modified else np.zeros((0, 3))
        scale = _arrow_visibility_scale(trajectory_actual, pred_deltas, mod_deltas)
        # Nudge the two arrow sets apart perpendicular to their average direction -- most steps
        # have zero correction (predicted == modified), so without this, whichever is drawn
        # second would fully hide the other wherever they coincide.
        extent = float(np.max(trajectory_actual.max(axis=0) - trajectory_actual.min(axis=0)))
        offset = _perpendicular_offset(pred_deltas, mod_deltas) * (0.04 * extent)

        # Explicit line segments (base -> tip -> None -> next base -> ...) instead of go.Cone:
        # Cone's sizing depends on sizemode/sizeref semantics that can't be verified without a
        # live browser to render in, and getting that guess wrong is exactly what made the
        # arrows invisible before. Line width + a marker at the tip has no such ambiguity.
        def _add_arrows(deltas: np.ndarray, base_offset: np.ndarray, color: str, name: str) -> None:
            arrow_bases = bases + base_offset
            tips = arrow_bases + deltas * scale
            xs, ys, zs = [], [], []
            for base, tip in zip(arrow_bases, tips):
                xs += [base[0], tip[0], None]
                ys += [base[1], tip[1], None]
                zs += [base[2], tip[2], None]
            fig.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    line=dict(color=color, width=4), opacity=0.9, name=name,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=tips[:, 0], y=tips[:, 1], z=tips[:, 2], mode="markers",
                    marker=dict(size=4, color=color, symbol="diamond"), showlegend=False,
                )
            )

        if have_predicted:
            _add_arrows(pred_deltas, offset, "purple", "predicted (unfiltered) one-step delta")
        if have_modified:
            _add_arrows(mod_deltas, -offset, "red", "modified (corrected) one-step delta")

    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))
    logger.info("SafetyFilter: interactive episode trajectory plot saved to %s", save_path)
