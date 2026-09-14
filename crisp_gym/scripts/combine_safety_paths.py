"""Combine multiple episodes' saved trajectories onto one shared obstacle/ellipsoid scene.

`SafetyFilter._finalize_episode_debug` (see `crisp_gym/safety/safety_filter.py`) saves, per
episode, alongside its `trajectory.png`:
  - `obstacle_points.npy` (N, 3) float64 -- the obstacle point cloud, in the robot base frame.
  - `obstacle_colors.npy` (N, 3) uint8 -- each point's real RGB color, sampled from the camera
    image (only present if perception captured it; older episodes recorded before this existed
    won't have it).
  - `ellipsoid.npz` -- the fitted obstacle ellipsoid (`center`, `rotation`, `semi_axes`).
  - `trajectory_actual.npy` (T, 3) -- the safety-filtered end-effector path actually driven.

This script builds one "base" scene (point cloud + ellipsoid) from a single episode folder,
then overlays any number of `trajectory_actual.npy` paths (from that same folder and/or other
episodes) on top of it -- e.g. to see how the executed path varies episode-to-episode against
the same obstacle.

Usage:
    pixi run -e jazzy-groot python3 crisp_gym/scripts/combine_safety_paths.py \\
        --base-dir outputs/safety/MMpath \\
        outputs/safety/MMpath/trajectory_actual.npy \\
        outputs/safety/episode_0002/trajectory_actual.npy \\
        --out outputs/safety/combined_trajectory.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from crisp_gym.safety.debug_plots import _ellipsoid_surface
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)


def _ellipsoid_radius_along(direction: np.ndarray, rotation: np.ndarray, semi_axes: np.ndarray) -> float:
    """Distance from an ellipsoid's center to its surface along a given direction.

    Used to cap the highlight circle's radius accurately -- capping by the ellipsoid's largest
    semi-axis (a single conservative bound applied in every direction) shrinks the circle far
    more than necessary whenever the highlight sits off the ellipsoid's long axis, which is
    exactly the common case here (highlight endpoints sit off to the side, not stacked along
    the ellipsoid's tallest axis).
    """
    local = rotation.T @ direction
    norm = float(np.linalg.norm(local))
    if norm < 1e-9:
        return float(np.min(semi_axes))
    u = local / norm
    denom = float(np.sqrt(np.sum((u / semi_axes) ** 2)))
    return 1.0 / denom if denom > 1e-9 else float(np.max(semi_axes))


def _default_label(traj_path: Path) -> str:
    """Best-effort human label for a trajectory file: its episode folder name if there is one
    (e.g. `.../episode_0004/trajectory_actual.npy` -> "episode_0004"), else the filename stem.
    """
    parent_name = traj_path.resolve().parent.name
    return parent_name if parent_name else traj_path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "trajectories",
        nargs="+",
        type=Path,
        help="One or more trajectory_actual.npy files to overlay on the base scene.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Episode debug folder containing ellipsoid.npz and obstacle_points.npy "
        "(and obstacle_colors.npy, if that episode captured color) -- the fixed backdrop "
        "every trajectory is drawn against.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <base-dir>/combined_trajectory.png.",
    )
    parser.add_argument(
        "--legend-out",
        type=Path,
        default=None,
        help="Legend PNG path, saved as a separate figure from --out (see --legend-loc's "
        "docstring for why). Defaults to <out-stem>_legend.png next to --out.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Legend label per trajectory, same order/count as the trajectories argument. "
        "Defaults to each trajectory's episode folder name.",
    )
    parser.add_argument(
        "--colors",
        nargs="*",
        default=None,
        help="Line color per trajectory (any matplotlib color spec, e.g. 'purple' or "
        "'tab:blue'), same order/count as the trajectories argument. Defaults to matplotlib's "
        "automatic color cycle.",
    )
    parser.add_argument(
        "--offset-x",
        type=float,
        default=0.0,
        help="Shift the obstacle point cloud and the ellipsoid's center by this amount on x "
        "(meters), e.g. to correct a known calibration offset. Trajectories are left as-is.",
    )
    parser.add_argument(
        "--offset-y",
        type=float,
        default=0.0,
        help="Same as --offset-x, but on y.",
    )
    parser.add_argument(
        "--ellipsoid-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the ellipsoid's semi-axes, e.g. 1.15 for 15%% larger. "
        "Center stays fixed; only the size grows/shrinks around it.",
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=None,
        help="Point cloud marker opacity (0=invisible, 1=opaque). Defaults to 0.8 when real "
        "colors are loaded, 0.6 for the flat-gray fallback.",
    )
    parser.add_argument(
        "--zoom-padding",
        type=float,
        default=1.6,
        help="View is cropped to the ellipsoid's (scaled) semi-axes times this factor, centered "
        "on it -- not to however far the trajectories happen to reach. Larger shows more of "
        "each trajectory but shrinks the obstacle; smaller zooms in on the obstacle/paths near "
        "it at the cost of cropping trajectory points that stray further away. Axes are kept to "
        "true proportions regardless (via set_box_aspect), so this never distorts the shapes.",
    )
    parser.add_argument(
        "--zoom-padding-x",
        type=float,
        default=None,
        help="Overrides --zoom-padding for the x half-range only (e.g. to stretch just x so a "
        "trajectory's far end stops getting cropped, without shrinking the obstacle on y/z). "
        "Defaults to --zoom-padding's value.",
    )
    parser.add_argument(
        "--zoom-padding-y",
        type=float,
        default=None,
        help="Same as --zoom-padding-x, but for y.",
    )
    parser.add_argument(
        "--zoom-padding-z",
        type=float,
        default=None,
        help="Same as --zoom-padding-x, but for z.",
    )
    parser.add_argument(
        "--highlight-endpoints",
        nargs="+",
        type=Path,
        default=None,
        help="Subset of the trajectory files (from the positional trajectories) whose final "
        "point gets circled by a small transparent green sphere -- e.g. to flag where two "
        "paths end up close together. Sized to just cover the given endpoints (plus "
        "--highlight-padding room), and automatically shrunk so it never reaches into the "
        "obstacle ellipsoid.",
    )
    parser.add_argument(
        "--highlight-padding",
        type=float,
        default=1.3,
        help="Multiplier applied to the minimal enclosing radius of --highlight-endpoints, for "
        "a little breathing room around the endpoints.",
    )
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Overrides the obstacle-centered x range/box-aspect entirely with an explicit "
        "[XMIN, XMAX] -- e.g. to fit a trajectory's far end without changing y/z. y/z stay "
        "obstacle-centered as usual.",
    )
    parser.add_argument(
        "--envelope-trajectories",
        nargs="+",
        type=Path,
        default=None,
        help="Subset of the trajectory files (from the positional trajectories) to draw with an "
        "extra thick, translucent line underneath their normal thin line -- a rough visual "
        "'envelope' around them (e.g. to mark a group of runs as one distribution). All given "
        "files share one --envelope-label legend entry.",
    )
    parser.add_argument(
        "--envelope-color",
        default="navajowhite",
        help="Color of the --envelope-trajectories envelope.",
    )
    parser.add_argument(
        "--envelope-linewidth",
        type=float,
        default=25.0,
        help="Line width (points) of the --envelope-trajectories envelope.",
    )
    parser.add_argument(
        "--envelope-alpha",
        type=float,
        default=0.35,
        help="Opacity of the --envelope-trajectories envelope.",
    )
    parser.add_argument(
        "--envelope-label",
        default="normal distribution",
        help="Legend label for the --envelope-trajectories envelope.",
    )
    parser.add_argument(
        "--deployment-region",
        action="store_true",
        help="Draw one large translucent sphere sized to fill nearly the whole view (behind "
        "everything else), e.g. to represent an overall 'safe deployment' coverage region.",
    )
    parser.add_argument(
        "--deployment-color",
        default="lightblue",
        help="Color of the --deployment-region sphere.",
    )
    parser.add_argument(
        "--deployment-alpha",
        type=float,
        default=0.15,
        help="Opacity of the --deployment-region sphere.",
    )
    parser.add_argument(
        "--deployment-scale",
        type=float,
        default=0.95,
        help="Multiplier on the view's half-range used for the --deployment-region sphere's "
        "size. 1.0 exactly touches the view box's faces; higher overflows/clips at the box "
        "edges for a fuller-looking sphere; lower shrinks it inward from the edges.",
    )
    parser.add_argument(
        "--deployment-label",
        default="safe deployment distribution",
        help="Legend label for the --deployment-region sphere.",
    )
    parser.add_argument(
        "--deployment-xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Overrides the --deployment-region sphere's x center/extent with an explicit "
        "[XMIN, XMAX] -- e.g. to span exactly from a trajectory's start point to the goal area, "
        "independent of the general view's x range (--xlim). y/z keep using the view's range.",
    )
    parser.add_argument(
        "--legend-loc",
        default="center",
        help="matplotlib legend `loc` string for the standalone legend figure (--legend-out) -- "
        "the main plot (--out) itself no longer has a legend drawn on it at all, so this and "
        "--legend-bbox no longer affect --out's layout, only how the legend sits within its own "
        "separate, tightly-cropped figure. 'center' (the default) is usually right since that "
        "figure contains nothing else to be positioned relative to.",
    )
    parser.add_argument(
        "--legend-bbox",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help="Fine-tunes the legend position beyond --legend-loc, via matplotlib's "
        "bbox_to_anchor, within the standalone legend figure (--legend-out) -- see "
        "--legend-loc's docstring.",
    )
    parser.add_argument(
        "--legend-order",
        nargs="+",
        default=None,
        help="Exact legend entry order, by label text (must match the labels actually in use -- "
        "each --labels entry, plus the fixed 'obstacle point cloud'/'obstacle area'/'goal area' "
        "entries where applicable). Any in-use label not listed here keeps its default (plot) "
        "order and is appended after the ones given.",
    )
    args = parser.parse_args()

    setup_logging()

    if args.labels is not None and len(args.labels) != len(args.trajectories):
        parser.error(f"--labels has {len(args.labels)} entries but {len(args.trajectories)} trajectories were given.")
    if args.colors is not None and len(args.colors) != len(args.trajectories):
        parser.error(f"--colors has {len(args.colors)} entries but {len(args.trajectories)} trajectories were given.")

    ellipsoid_path = args.base_dir / "ellipsoid.npz"
    points_path = args.base_dir / "obstacle_points.npy"
    colors_path = args.base_dir / "obstacle_colors.npy"

    offset = np.array([args.offset_x, args.offset_y, 0.0])

    if not points_path.exists():
        parser.error(f"{points_path} not found -- is --base-dir an episode folder with a saved point cloud?")
    obstacle_points = np.load(points_path)
    if offset.any():
        obstacle_points = obstacle_points + offset

    obstacle_colors = None
    if colors_path.exists():
        obstacle_colors = np.load(colors_path)
        logger.info("Loaded real point-cloud colors from %s.", colors_path)
    else:
        logger.info("No %s found -- plotting the point cloud in a flat gray instead.", colors_path)

    ellipsoid = None
    if ellipsoid_path.exists():
        data = np.load(ellipsoid_path)
        center = data["center"]
        if offset.any():
            center = center + offset
        ellipsoid = (center, data["rotation"], data["semi_axes"] * args.ellipsoid_scale)
    else:
        logger.warning("No %s found -- plotting without the ellipsoid.", ellipsoid_path)

    # View centering/sizing is computed up front (rather than after plotting, as originally)
    # since --deployment-region needs it too, to size its "fill nearly the whole view" sphere.
    view_center = None
    view_half_range = None
    if ellipsoid is not None:
        view_center = ellipsoid[0]
        padding_xyz = np.array([
            args.zoom_padding_x if args.zoom_padding_x is not None else args.zoom_padding,
            args.zoom_padding_y if args.zoom_padding_y is not None else args.zoom_padding,
            args.zoom_padding_z if args.zoom_padding_z is not None else args.zoom_padding,
        ])
        view_half_range = ellipsoid[2] * padding_xyz
        if args.xlim is not None:
            view_half_range[0] = (args.xlim[1] - args.xlim[0]) / 2.0

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    drew_deployment = False
    if args.deployment_region and view_center is not None:
        # Semi-axes matched to view_half_range (not a plain sphere) so it reads as round once
        # set_box_aspect below is applied with the same half_range -- an actual sphere here
        # would look squashed/stretched under an anisotropic box aspect.
        deployment_center = view_center.copy()
        deployment_semi_axes = view_half_range * args.deployment_scale
        if args.deployment_xlim is not None:
            dxmin, dxmax = args.deployment_xlim
            deployment_center[0] = (dxmin + dxmax) / 2.0
            deployment_semi_axes[0] = (dxmax - dxmin) / 2.0
        dx, dy, dz = _ellipsoid_surface(deployment_center, np.eye(3), deployment_semi_axes)
        ax.plot_surface(dx, dy, dz, color=args.deployment_color, alpha=args.deployment_alpha, linewidth=0, zorder=0)
        drew_deployment = True

    # zorder is set explicitly (rather than relying on mplot3d's default view-dependent depth
    # sorting between different artist types, which doesn't reliably layer a surface over a
    # scatter) so the ellipsoid always renders in front of the point cloud.
    if len(obstacle_points) > 0:
        if obstacle_colors is not None:
            face_colors = obstacle_colors.astype(np.float64) / 255.0
            point_alpha = args.point_alpha if args.point_alpha is not None else 0.8
            ax.scatter(*obstacle_points.T, s=3, c=face_colors, alpha=point_alpha, label="obstacle point cloud", zorder=1)
        else:
            point_alpha = args.point_alpha if args.point_alpha is not None else 0.6
            ax.scatter(*obstacle_points.T, s=3, c="gray", alpha=point_alpha, label="obstacle point cloud", zorder=1)

    if ellipsoid is not None:
        center, rotation, semi_axes = ellipsoid
        ex, ey, ez = _ellipsoid_surface(center, rotation, semi_axes)
        ax.plot_surface(ex, ey, ez, color="red", alpha=0.25, linewidth=0, zorder=2)

    drew_highlight = False
    drew_envelope = False
    envelope_targets = {p.resolve() for p in args.envelope_trajectories} if args.envelope_trajectories else set()

    traj_endpoints = {}
    for i, traj_path in enumerate(args.trajectories):
        if not traj_path.exists():
            logger.warning("Skipping %s -- file not found.", traj_path)
            continue
        traj = np.load(traj_path)
        if len(traj) < 2:
            logger.warning("Skipping %s -- fewer than 2 points, nothing to draw.", traj_path)
            continue
        if traj_path.resolve() in envelope_targets:
            ax.plot(
                *traj.T, linewidth=args.envelope_linewidth, color=args.envelope_color,
                alpha=args.envelope_alpha, solid_capstyle="round", zorder=2.5,
            )
            drew_envelope = True
        label = args.labels[i] if args.labels is not None else _default_label(traj_path)
        color = args.colors[i] if args.colors is not None else None
        (line,) = ax.plot(*traj.T, linewidth=2, label=label, color=color, zorder=3)
        ax.scatter(*traj[0].T, color=line.get_color(), marker="o", s=40, zorder=3)
        ax.scatter(*traj[-1].T, color=line.get_color(), marker="^", s=60, zorder=3)
        traj_endpoints[traj_path.resolve()] = traj[-1]

    if args.highlight_endpoints:
        targets = {p.resolve() for p in args.highlight_endpoints}
        endpoints = [pt for path, pt in traj_endpoints.items() if path in targets]
        missing = targets - set(traj_endpoints.keys())
        if missing:
            logger.warning(
                "--highlight-endpoints referenced files that weren't plotted (check the paths "
                "match entries in `trajectories`): %s", missing,
            )
        if endpoints:
            highlight_center = np.mean(endpoints, axis=0)
            highlight_radius = max(
                (float(np.linalg.norm(p - highlight_center)) for p in endpoints), default=0.0
            ) * args.highlight_padding
            highlight_radius = max(highlight_radius, 0.01)  # floor so coincident endpoints still show
            if ellipsoid is not None:
                ellipsoid_center, ellipsoid_rotation, ellipsoid_semi_axes = ellipsoid
                center_to_center = highlight_center - ellipsoid_center
                ellipsoid_extent = _ellipsoid_radius_along(center_to_center, ellipsoid_rotation, ellipsoid_semi_axes)
                gap = float(np.linalg.norm(center_to_center)) - ellipsoid_extent
                if gap > 0:
                    highlight_radius = min(highlight_radius, gap * 0.95)
                else:
                    logger.warning(
                        "Highlight center sits inside the ellipsoid's real surface -- "
                        "the green circle may still overlap it."
                    )
            hx, hy, hz = _ellipsoid_surface(
                highlight_center, np.eye(3), np.array([highlight_radius] * 3)
            )
            ax.plot_surface(hx, hy, hz, color="green", alpha=0.3, linewidth=0, zorder=4)
            drew_highlight = True

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # plot_surface's own legend entries render incorrectly in mplot3d (a known matplotlib
    # limitation for Poly3DCollection), so the ellipsoid/highlight get proxy Patch handles
    # instead, appended to whatever legend matplotlib already built from the scatter/line labels.
    from matplotlib.patches import Patch

    handles, legend_labels = ax.get_legend_handles_labels()
    if ellipsoid is not None:
        handles.append(Patch(facecolor="red", alpha=0.25, label="obstacle area"))
        legend_labels.append("obstacle area")
    if drew_highlight:
        handles.append(Patch(facecolor="green", alpha=0.3, label="goal area"))
        legend_labels.append("goal area")
    if drew_envelope:
        handles.append(Patch(facecolor=args.envelope_color, alpha=args.envelope_alpha, label=args.envelope_label))
        legend_labels.append(args.envelope_label)
    if drew_deployment:
        handles.append(Patch(facecolor=args.deployment_color, alpha=args.deployment_alpha, label=args.deployment_label))
        legend_labels.append(args.deployment_label)
    if args.legend_order:
        order_index = {label: i for i, label in enumerate(args.legend_order)}
        order = sorted(
            range(len(legend_labels)),
            key=lambda i: (order_index.get(legend_labels[i], len(order_index)), i),
        )
        handles = [handles[i] for i in order]
        legend_labels = [legend_labels[i] for i in order]
    # ADDED: the legend used to be drawn directly on `ax` here (`ax.legend(...)`) -- now saved
    # as a wholly separate figure (see below, after --out is written) instead, so --out is a
    # clean legend-free plot and the legend can be placed/sized independently of it.
    ax.view_init(elev=30, azim=-60)

    # Center the view on the obstacle ellipsoid instead of mplot3d's default auto-scaled box
    # (which centers on the combined bounding box of everything plotted, so the obstacle drifts
    # off-center whenever a trajectory reaches further in one direction than the others). Sized
    # off the ellipsoid's own (scaled) semi-axes, NOT the trajectories' full extent -- otherwise
    # a trajectory that strays far from the obstacle zooms the whole scene out and shrinks the
    # obstacle/near-obstacle path detail down to unreadable, which is what --zoom-padding fixes.
    #
    # set_box_aspect is required alongside the limits: mplot3d otherwise stretches x/y/z
    # independently to fill a cubic box regardless of each axis's actual data range, so an
    # asymmetric crop (e.g. a flatter z half_range than x/y) would visibly distort the ellipsoid
    # and paths -- matching the box aspect to the real half_range ratios keeps 1 meter the same
    # size on screen on every axis.
    if view_center is not None:
        if args.xlim is not None:
            ax.set_xlim3d(args.xlim[0], args.xlim[1])
        else:
            ax.set_xlim3d(view_center[0] - view_half_range[0], view_center[0] + view_half_range[0])
        ax.set_ylim3d(view_center[1] - view_half_range[1], view_center[1] + view_half_range[1])
        ax.set_zlim3d(view_center[2] - view_half_range[2], view_center[2] + view_half_range[2])
        ax.set_box_aspect(view_half_range)

    out_path = args.out if args.out is not None else args.base_dir / "combined_trajectory.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Combined plot saved to %s", out_path)

    # ADDED: the legend as its own separate figure -- a blank canvas with nothing on it but
    # `handles`/`legend_labels` (the exact same ones the main plot used to draw inline), saved
    # with bbox_inches="tight" so the output crops down to just the legend box itself, no
    # surrounding whitespace.
    legend_out_path = (
        args.legend_out if args.legend_out is not None else out_path.with_name(f"{out_path.stem}_legend{out_path.suffix}")
    )
    legend_bbox = tuple(args.legend_bbox) if args.legend_bbox is not None else None
    legend_fig = plt.figure()
    legend_fig.legend(handles, legend_labels, loc=args.legend_loc, bbox_to_anchor=legend_bbox)
    legend_out_path.parent.mkdir(parents=True, exist_ok=True)
    legend_fig.savefig(legend_out_path, dpi=150, bbox_inches="tight")
    plt.close(legend_fig)
    logger.info("Legend saved separately to %s", legend_out_path)


if __name__ == "__main__":
    main()
