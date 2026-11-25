"""Debug tool: visualize the safety filter's obstacle point cloud + fitted ellipsoid.

Grabs one live RGB-D frame and runs it through the *exact* perception/filtering functions the
CBF safety filter uses at deploy time -- `crisp_gym.safety.perception.get_point_cloud` (with
GroundingDINO detection using the same fixed/VLM obstacle-naming rule as
`SafetyFilter._localize_obstacle`) and `crisp_gym.safety.math_utils.filtering_points` (the
xyz_bounds crop -> statistical outlier removal -> RANSAC table-plane removal -> DBSCAN cluster
pipeline), then `fit_ellipse` for the red ellipsoid -- all identical to what
`SafetyFilter.reset()` computes. On top of that (not part of the deployed safety filter), this
script additionally runs `crisp_gym.safety.math_utils.filter_points_by_color` as a second pass
on `filtering_points`'s own output, to prune any leftover points whose color doesn't match the
majority (a shadow, a reflection, a sliver of background the geometric filters missed).

The obstacle point cloud is unprojected straight into the robot base frame, using whatever
`safety_camera.get_camera_to_world()` resolves at call time -- the config's
`dummy_camera_to_world` matrix if set, otherwise a live TF lookup.

The output plot shows the RAW (pre-filter) point cloud in real RGB color, split into three
traces so you can see exactly what each filtering stage removed:
  - discarded by geometry (faded) -- never survived `filtering_points`.
  - kept by geometry, rejected by color (diamond markers) -- survived `filtering_points` but
    didn't match the majority color of that result.
  - kept by both (solid, black outline) -- the final obstacle point cloud the (red) MVEE
    ellipsoid is fit to.

For comparison, it also plots a second (blue) ellipsoid from
`crisp_gym.safety.math_utils.fit_ellipse_from_height_profile` -- an experimental alternative
that doesn't try to filter the table out at all (avoiding the plane/DBSCAN/color instability
above entirely); instead it measures height from the full cloud and horizontal extent only from
the cloud's uncontaminated top slice. Neither ellipsoid is used by the deployed safety filter
yet -- this script is for comparing which one tracks the real object better before deciding.

Usage (needs the robot bringup + camera running, inside the jazzy-policy-aegis pixi env):
    pixi run -e jazzy-policy-aegis python3 crisp_gym/scripts/debug_obstacle_pointcloud.py

Opens the saved HTML in a browser to rotate/zoom/pan.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from crisp_gym.safety.debug_plots import _ellipsoid_surface, plot_obstacle_pointcloud_debug
from crisp_gym.safety.depth_camera import SafetyRGBDCamera, SafetyRGBDCameraConfig
from crisp_gym.safety.math_utils import (
    crop_to_bounds,
    filter_points_by_color,
    filtering_points,
    fit_ellipse,
    fit_ellipse_from_height_profile,
)
from crisp_gym.safety.perception import (
    get_point_cloud,
    get_point_clouds_per_box,
    obstacle_detection,
)
from crisp_gym.safety.safety_filter import SafetyFilter, SafetyFilterConfig
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)


def _rgb_strings(colors: np.ndarray) -> list[str]:
    """(N, 3) uint8 array -> plotly-compatible 'rgb(r,g,b)' strings, one per point."""
    return [f"rgb({r},{g},{b})" for r, g, b in colors]


def _geometry_kept_mask(raw_points: np.ndarray, geo_filtered_points: np.ndarray) -> np.ndarray:
    """Which rows of ``raw_points`` survived into ``geo_filtered_points``.

    `filtering_points` only ever subsets rows (crop -> outlier removal -> plane removal ->
    clustering) and never modifies coordinate values, so exact-match nearest-neighbor lookup
    reliably recovers which raw points were kept.
    """
    if len(geo_filtered_points) == 0:
        return np.zeros(len(raw_points), dtype=bool)
    from scipy.spatial import cKDTree

    tree = cKDTree(geo_filtered_points)
    dist, _ = tree.query(raw_points, k=1)
    return dist < 1e-6


def _show_annotated_detection(annotated_path: Path | None, obstacle_name: str) -> None:
    """Show the 2D GroundingDINO detection (all matched boxes).

    `get_point_cloud`/
    `get_point_clouds_per_box` already saved it to disk) so you can see *what* it detected
    without separately hunting down the file. Non-blocking -- stays open while the rest of the
    script computes the 3D plot; closed later via a final `cv2.waitKey(0)`.
    """
    if annotated_path is None or not annotated_path.exists():
        logger.warning("No annotated_detection.jpg found (debug_dir unset or detection failed) -- skipping display.")
        return

    import cv2

    annotated = cv2.imread(str(annotated_path))
    cv2.imshow(f"GroundingDINO detection: '{obstacle_name}' (press any key to close)", annotated)
    cv2.waitKey(1)  # force the window to actually render now
    logger.info("Showing detection box from %s (window stays open until the script finishes).", annotated_path)


def _run_multi_ellipsoid(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    obstacle_name: str,
    model: Any,
    fc: SafetyFilterConfig,
    args: argparse.Namespace,
    debug_dir: Path | None,
    annotated_path: Path | None,
    out_path: Path,
) -> None:
    """Build one point cloud and fitted ellipsoid per GroundingDINO box.

    `--multi-ellipsoid` path: one point cloud + one fitted ellipsoid per GroundingDINO box
    matching ``obstacle_name`` (e.g. two same-looking cans), instead of the single merged/best-
    box cloud the rest of this script builds.

    Builds each box's ellipsoid the same way the single-object path's own plotted (blue)
    ellipsoid is built -- `fit_ellipse_from_height_profile` on an `xyz_bounds`-only crop -- not
    `fit_ellipse`'s exact MVEE, which is a poor fit here: it must strictly contain every point,
    so a single stray outlier (a reflection, background bleed at the crop edge, a noisy depth
    return) fully dictates one axis length regardless of how tightly the rest of the cloud
    clusters, and one-outlier accidents get more likely, not less, once there are several boxes.
    """
    box_clouds = get_point_clouds_per_box(
        rgb,
        depth,
        intrinsics,
        camera_to_world,
        obstacle_name,
        model,
        box_threshold=fc.box_threshold,
        text_threshold=fc.text_threshold,
        device=fc.perception_device,
        debug_dir=debug_dir,
    )
    logger.info("Got %d separate point cloud(s) for '%s'.", len(box_clouds), obstacle_name)
    _show_annotated_detection(annotated_path, obstacle_name)

    import plotly.graph_objects as go

    palette = ["red", "blue", "green", "purple", "orange", "cyan"]
    fig = go.Figure()
    fitted = 0
    for i, (raw_points, raw_colors) in enumerate(box_clouds):
        cropped_points, cropped_colors = crop_to_bounds(raw_points, fc.xyz_bounds, colors=raw_colors)
        logger.info("box %d: %d raw -> %d in xyz_bounds", i, len(raw_points), len(cropped_points))
        if len(cropped_points) == 0:
            logger.warning("box %d: no points survived the xyz_bounds crop -- skipping its ellipsoid.", i)
            continue

        center, rotation, semi_axes = fit_ellipse_from_height_profile(
            cropped_points,
            top_fraction=args.top_fraction,
            center_fraction=args.center_fraction,
            radius_percentile=args.radius_percentile,
            height_percentile=args.height_percentile,
            full_cluster_eps=args.full_cluster_eps,
            full_cluster_min_points=args.full_cluster_min_points,
            top_cluster_eps=args.top_cluster_eps,
            top_cluster_min_points=args.top_cluster_min_points,
            fixed_radius=args.fixed_radius,
            center_x_offset=args.center_x_offset,
            margin_fraction=args.margin_fraction,
        )
        logger.info("box %d height-profile ellipsoid center=%s semi_axes=%s", i, center, semi_axes)
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter3d(
                x=cropped_points[:, 0], y=cropped_points[:, 1], z=cropped_points[:, 2],
                mode="markers",
                marker=dict(size=3, color=_rgb_strings(cropped_colors), opacity=1.0, line=dict(width=0.5, color=color)),
                name=f"obstacle {i} points ({len(cropped_points)} pts)",
            )
        )
        ex, ey, ez = _ellipsoid_surface(center, rotation, semi_axes)
        fig.add_trace(
            go.Surface(
                x=ex, y=ey, z=ez, colorscale=[[0, color], [1, color]], showscale=False, opacity=0.3,
                name=f"obstacle {i} ellipsoid",
            )
        )
        fitted += 1

    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.write_html(str(out_path))
    logger.info(
        "Saved interactive plot with %d fitted ellipsoid(s) to %s -- open it in a browser.",
        fitted, out_path.resolve(),
    )


def main() -> None:
    """Capture a live RGB-D frame and save obstacle and ellipsoid diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        default="crisp_gym/config/policy/vlsa_config.yaml",
        help="Path to the vlsa_config.yaml to read the safety/safety_camera blocks from.",
    )
    parser.add_argument(
        "--task",
        default="pick the cube",
        help="Task description -- only used if obstacle_preference_list has 2+ entries (VLM naming).",
    )
    parser.add_argument(
        "--out",
        default="outputs/safety/debug_pointcloud.html",
        help="Where to save the interactive plot (the annotated GroundingDINO detection crop "
        "is saved alongside it in the same folder).",
    )
    parser.add_argument(
        "--color-distance-threshold",
        type=float,
        default=40.0,
        help="Max Euclidean RGB distance from the geometry-filtered result's median color to "
        "keep a point in the second (color) filtering pass. Lower is stricter.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=None,
        help="Fraction of the height-profile ellipsoid's z-range (from the top) used to measure "
        "its horizontal extent -- see fit_ellipse_from_height_profile. Defaults to --config's "
        "safety.ellipsoid_top_fraction (class default 0.15) when not given.",
    )
    parser.add_argument(
        "--center-fraction",
        type=float,
        default=None,
        help="Fraction of the height-profile ellipsoid's z-range, centered on the object's "
        "vertical middle, used for the ellipsoid's x/y center instead of --top-fraction's slice "
        "(radius keeps using the top slice regardless) -- see fit_ellipse_from_height_profile's "
        "center_fraction docstring. Defaults to --config's safety.ellipsoid_center_fraction "
        "(class default 0.15) when not given.",
    )
    parser.add_argument(
        "--center-x-offset",
        type=float,
        default=None,
        help="Subtracted from the ellipsoid's x-center after --center-fraction has otherwise "
        "fully derived it -- an empirical correction for residual camera_to_world error. "
        "Defaults to --config's safety.ellipsoid_center_x_offset (class default 0.0, a no-op) "
        "when not given.",
    )
    parser.add_argument(
        "--radius-percentile",
        type=float,
        default=None,
        help="Percentile (not the true max) of the top slice's distances from its own centroid "
        "used as the height-profile ellipsoid's horizontal radius (when --fixed-radius isn't "
        "given). Also used (both ends: percentile and 100-percentile) to trim the y_range "
        "measurement when --fixed-radius IS given. Lower it if stray points are inflating the "
        "measurement; raise toward 100 if it's clipping real points. Defaults to --config's "
        "safety.ellipsoid_radius_percentile (class default 100.0, no trimming) when not given.",
    )
    parser.add_argument(
        "--fixed-radius",
        type=float,
        default=None,
        help="If given (or set via --config's safety.ellipsoid_fixed_radius), switches from a "
        "circular cross-section (--radius-percentile's top-slice measurement, shared by x and y) "
        "to two independent semi-axes: x ('thick') becomes this fixed constant, y ('width') gets "
        "measured as a percentile-trimmed range from the middle-slice point cloud "
        "(--radius-percentile trims both ends) instead. None (the class default) on both leaves "
        "it dynamic, measured fresh each run.",
    )
    parser.add_argument(
        "--margin-fraction",
        type=float,
        default=None,
        help="Extra padding applied after the above (0.15 = 15%% larger), so the ellipsoid "
        "comfortably contains the cloud rather than sitting exactly flush with it. Applied to "
        "all three semi-axes when --fixed-radius isn't given; when it is, x is left unpadded "
        "(it's a fixed constant you already chose, not something measured here) -- only y and z "
        "get the margin. Defaults to --config's safety.ellipsoid_margin_fraction (class default "
        "0.15) when not given.",
    )
    parser.add_argument(
        "--height-percentile",
        type=float,
        default=None,
        help="Same trimming, applied to the z-extent (height_percentile for the top, "
        "100-height_percentile for the bottom) instead of the radius. Defaults to --config's "
        "safety.ellipsoid_height_percentile (class default 100.0, no trimming) when not given.",
    )
    parser.add_argument(
        "--full-cluster-eps",
        type=float,
        default=None,
        help="DBSCAN neighborhood radius (meters) used to cluster the whole cropped cloud before "
        "anything else -- only the largest cluster is kept, before z_max/the top slice are even "
        "computed. Without this, a disjoint noise streak taller than the real object can hijack "
        "z_max and drag the top slice away from the object entirely. See "
        "fit_ellipse_from_height_profile's full_cluster_eps docstring. Defaults to --config's "
        "safety.ellipsoid_full_cluster_eps (class default 0.02) when not given.",
    )
    parser.add_argument(
        "--full-cluster-min-points",
        type=int,
        default=None,
        help="DBSCAN minimum cluster size for --full-cluster-eps. If the cropped cloud has fewer "
        "points than this, clustering is skipped. Defaults to --config's "
        "safety.ellipsoid_full_cluster_min_points (class default 10) when not given.",
    )
    parser.add_argument(
        "--top-cluster-eps",
        type=float,
        default=None,
        help="DBSCAN neighborhood radius (meters) used to cluster the top slice before measuring "
        "its radius -- only the top slice's largest cluster is kept. See "
        "fit_ellipse_from_height_profile's top_cluster_eps docstring. Defaults to --config's "
        "safety.ellipsoid_top_cluster_eps (class default 0.02) when not given.",
    )
    parser.add_argument(
        "--top-cluster-min-points",
        type=int,
        default=None,
        help="DBSCAN minimum cluster size for --top-cluster-eps. If the top slice has fewer "
        "points than this, clustering is skipped. Defaults to --config's "
        "safety.ellipsoid_top_cluster_min_points (class default 10) when not given.",
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Seconds to wait for the camera to become ready."
    )
    parser.add_argument(
        "--merge-boxes",
        action="store_true",
        help="Union points from every GroundingDINO box matching the obstacle prompt (e.g. two "
        "same-looking cans both matching 'yellow can') into one combined point cloud, instead "
        "of only the single highest-confidence box. Off by default, matching deployed behavior. "
        "Mutually exclusive with --multi-ellipsoid (which fits a separate ellipsoid per box "
        "instead of merging them).",
    )
    parser.add_argument(
        "--multi-ellipsoid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Instead of one point cloud/ellipsoid, keep every GroundingDINO box matching the "
        "obstacle prompt separate and fit its own height-profile ellipsoid (e.g. two same-"
        "looking cans both matching 'yellow can' each get their own ellipsoid) -- one colored "
        "point cloud + ellipsoid per box in the saved HTML plot, instead of the single-object "
        "path's comparison traces and PNG export. Defaults to --config's "
        "safety.debug_multi_ellipsoid (class default False) when neither this nor "
        "--no-multi-ellipsoid is passed.",
    )
    args = parser.parse_args()

    setup_logging()

    config = yaml.safe_load(Path(args.config).read_text())
    safety_cfg = config["safety"]
    camera_cfg = config["safety_camera"]

    safety_filter = SafetyFilter(SafetyFilterConfig(**safety_cfg))
    # Debug-folder naming only (`_episode_debug_dir` uses this); real episode bookkeeping is
    # irrelevant here since `reset()`/`step()` are never called.
    safety_filter._episode_index = 0
    fc = safety_filter.config

    # CLI flags left unset (None) fall back to the matching safety.ellipsoid_*/debug_* config
    # field instead of a hardcoded script default, so --config alone keeps this tool's fit in
    # sync with the deployed SafetyFilterConfig without having to retype every flag every run.
    if args.top_fraction is None:
        args.top_fraction = fc.ellipsoid_top_fraction
    if args.center_fraction is None:
        args.center_fraction = fc.ellipsoid_center_fraction
    if args.center_x_offset is None:
        args.center_x_offset = fc.ellipsoid_center_x_offset
    if args.radius_percentile is None:
        args.radius_percentile = fc.ellipsoid_radius_percentile
    if args.fixed_radius is None:
        args.fixed_radius = fc.ellipsoid_fixed_radius
    if args.margin_fraction is None:
        args.margin_fraction = fc.ellipsoid_margin_fraction
    if args.height_percentile is None:
        args.height_percentile = fc.ellipsoid_height_percentile
    if args.full_cluster_eps is None:
        args.full_cluster_eps = fc.ellipsoid_full_cluster_eps
    if args.full_cluster_min_points is None:
        args.full_cluster_min_points = fc.ellipsoid_full_cluster_min_points
    if args.top_cluster_eps is None:
        args.top_cluster_eps = fc.ellipsoid_top_cluster_eps
    if args.top_cluster_min_points is None:
        args.top_cluster_min_points = fc.ellipsoid_top_cluster_min_points
    if args.multi_ellipsoid is None:
        args.multi_ellipsoid = fc.debug_multi_ellipsoid

    depth_camera = None
    try:
        logger.info("Connecting to the safety RGB-D camera...")
        depth_camera = SafetyRGBDCamera(SafetyRGBDCameraConfig(**camera_cfg))
        depth_camera.wait_until_ready(timeout=args.timeout)

        rgb = depth_camera.current_color
        depth = depth_camera.current_depth
        intrinsics = depth_camera.intrinsics
        # Irrelevant when `dummy_camera_to_world` is set (short-circuits before this is used) --
        # this repo has no standalone env/robot config to source the real default from here.
        base_frame = camera_cfg.get("base_frame_override") or "base"
        camera_to_world = depth_camera.get_camera_to_world(base_frame)
        logger.info("camera_to_world:\n%s", camera_to_world)
    finally:
        # Shut rclpy down *before* the interpreter starts tearing down modules -- otherwise the
        # camera's background spin thread (`SafetyRGBDCamera._spin`, `while rclpy.ok(): ...`)
        # gets torn down mid-iteration by interpreter shutdown, which is what was raising an
        # exception from Thread-2 on exit.
        import rclpy

        if depth_camera is not None:
            depth_camera.node.destroy_node()
            depth_camera._tf_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        time.sleep(0.2)  # give the daemon spin thread a chance to notice and exit cleanly

    # Same fixed/VLM obstacle-naming rule as `SafetyFilter._localize_obstacle` -- called
    # directly here (rather than going through that method) because it only returns the
    # already-filtered points, and this script needs the raw ones too for the kept/discarded
    # comparison.
    if fc.obstacle_preference_list and len(fc.obstacle_preference_list) == 1:
        obstacle_name = fc.obstacle_preference_list[0]
        logger.info("Using fixed obstacle name '%s'.", obstacle_name)
    else:
        obstacle_name = obstacle_detection(
            rgb, args.task, preference_list=fc.obstacle_preference_list, api_key=fc.zhipu_api_key
        )
        logger.info("VLM identified likely obstacle as '%s'.", obstacle_name)

    model = safety_filter._ensure_groundingdino_model()
    debug_dir = safety_filter._episode_debug_dir()
    annotated_path = (debug_dir / "annotated_detection.jpg") if debug_dir else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.multi_ellipsoid:
        _run_multi_ellipsoid(
            rgb, depth, intrinsics, camera_to_world, obstacle_name, model, fc, args, debug_dir, annotated_path, out_path
        )
        if annotated_path is not None and annotated_path.exists():
            import cv2

            logger.info("Press any key in the detection-box window to close it and exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    raw_points, raw_colors = get_point_cloud(
        rgb,
        depth,
        intrinsics,
        camera_to_world,
        obstacle_name,
        model,
        box_threshold=fc.box_threshold,
        text_threshold=fc.text_threshold,
        device=fc.perception_device,
        debug_dir=debug_dir,
        return_colors=True,
        merge_boxes=args.merge_boxes,
    )
    logger.info("raw points (inside detection box, valid depth): %d", len(raw_points))
    _show_annotated_detection(annotated_path, obstacle_name)

    # Height-profile ellipsoid (experimental, comparison only): xyz_bounds crop ONLY --
    # deliberately skips both filtering_points's plane-removal/DBSCAN AND statistical outlier
    # removal (that step's local-density check tends to misclassify the object's own sparse
    # top rim as noise and strip it, silently cutting the top off the height measurement --
    # fit_ellipse_from_height_profile guards against noise its own way, via height_percentile).
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = fc.xyz_bounds
    crop_mask = (
        (raw_points[:, 0] > xmin) & (raw_points[:, 0] < xmax)
        & (raw_points[:, 1] > ymin) & (raw_points[:, 1] < ymax)
        & (raw_points[:, 2] > zmin) & (raw_points[:, 2] < zmax)
    )
    cropped_points = raw_points[crop_mask]
    logger.info("height-profile input points (crop only): %d", len(cropped_points))
    height_center = height_rotation = height_axes = None
    if len(cropped_points) > 0:
        height_center, height_rotation, height_axes = fit_ellipse_from_height_profile(
            cropped_points,
            top_fraction=args.top_fraction,
            center_fraction=args.center_fraction,
            radius_percentile=args.radius_percentile,
            height_percentile=args.height_percentile,
            full_cluster_eps=args.full_cluster_eps,
            full_cluster_min_points=args.full_cluster_min_points,
            top_cluster_eps=args.top_cluster_eps,
            top_cluster_min_points=args.top_cluster_min_points,
            fixed_radius=args.fixed_radius,
            center_x_offset=args.center_x_offset,
            margin_fraction=args.margin_fraction,
        )
        logger.info(
            "height-profile ellipsoid center=%s semi_axes=%s", height_center, height_axes
        )

    # Stage 1: geometry (same as the deployed safety filter).
    geo_points = filtering_points(raw_points, fc.xyz_bounds)
    geo_kept = _geometry_kept_mask(raw_points, geo_points)
    geo_colors = raw_colors[geo_kept]
    logger.info("kept by geometry: %d / %d raw points", geo_kept.sum(), len(raw_points))

    # Third comparison ellipsoid (green): same MVEE fit, but on geo_points directly -- i.e.
    # skipping the color filter (stage 2 below) entirely, to see what the color pass changes.
    geo_only_center = geo_only_rotation = geo_only_axes = None
    if len(geo_points) >= 4:
        geo_only_center, geo_only_rotation, geo_only_axes = fit_ellipse(geo_points)
        logger.info(
            "geometry-only MVEE ellipsoid center=%s semi_axes=%s", geo_only_center, geo_only_axes
        )
    else:
        logger.warning(
            "Only %d points survived geometry filtering (need >= 4 for an ellipsoid fit) -- "
            "skipping the geometry-only MVEE ellipsoid.",
            len(geo_points),
        )

    # Stage 2: color, on top of stage 1's own result (reference_color=None -> median of
    # geo_points/geo_colors itself).
    color_mask_within_geo = filter_points_by_color(
        geo_colors, color_distance_threshold=args.color_distance_threshold
    )
    final_points = geo_points[color_mask_within_geo]
    logger.info(
        "kept by geometry + color: %d / %d geometry-kept points", color_mask_within_geo.sum(), len(geo_points)
    )

    # Map stage 2's mask (defined over geo_points) back onto the full raw_points index space,
    # so the plot can show a 3-way breakdown against the original detection-box point cloud.
    final_kept = np.zeros(len(raw_points), dtype=bool)
    final_kept[np.where(geo_kept)[0][color_mask_within_geo]] = True
    geo_only_kept = geo_kept & ~final_kept  # passed geometry, rejected by color
    discarded = ~geo_kept  # never passed geometry

    if len(final_points) < 4:
        logger.error(
            "Only %d points survived both filters (need >= 4 for an ellipsoid fit) -- plotting "
            "the colored cloud anyway so you can see what's there, but no ellipsoid.",
            len(final_points),
        )
        center = rotation = semi_axes = None
    else:
        center, rotation, semi_axes = fit_ellipse(final_points)
        logger.info("fitted ellipsoid center=%s semi_axes=%s", center, semi_axes)

    import plotly.graph_objects as go

    fig = go.Figure()
    if discarded.any():
        d, c = raw_points[discarded], raw_colors[discarded]
        fig.add_trace(
            go.Scatter3d(
                x=d[:, 0], y=d[:, 1], z=d[:, 2],
                mode="markers",
                marker=dict(size=2, color=_rgb_strings(c), opacity=0.25),
                name=f"discarded by geometry ({discarded.sum()} pts)",
            )
        )
    if geo_only_kept.any():
        d, c = raw_points[geo_only_kept], raw_colors[geo_only_kept]
        fig.add_trace(
            go.Scatter3d(
                x=d[:, 0], y=d[:, 1], z=d[:, 2],
                mode="markers",
                marker=dict(size=3, symbol="diamond", color=_rgb_strings(c), opacity=0.7),
                name=f"kept by geometry, rejected by color ({geo_only_kept.sum()} pts)",
            )
        )
    if final_kept.any():
        d, c = raw_points[final_kept], raw_colors[final_kept]
        fig.add_trace(
            go.Scatter3d(
                x=d[:, 0], y=d[:, 1], z=d[:, 2],
                mode="markers",
                marker=dict(size=3, color=_rgb_strings(c), opacity=1.0, line=dict(width=0.5, color="black")),
                name=f"kept by both ({final_kept.sum()} pts, final obstacle cloud)",
            )
        )
    # Red (MVEE geometry+color filtered) and green (MVEE geometry filtered only) ellipsoids are
    # still fitted and logged above -- just not drawn here, to keep the plot focused on the blue
    # height-profile ellipsoid against the raw point cloud.
    if height_center is not None:
        ex, ey, ez = _ellipsoid_surface(height_center, height_rotation, height_axes)
        fig.add_trace(
            go.Surface(x=ex, y=ey, z=ez, colorscale=[[0, "blue"], [1, "blue"]], showscale=False, opacity=0.3, name="height-profile ellipsoid (top-slice width)")
        )

    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.write_html(str(out_path))
    logger.info("Saved interactive plot to %s -- open it in a browser.", out_path.resolve())

    # ADDED: static PNG, same figure layout/camera angle as SafetyFilter's own trajectory.png
    # (plot_episode_debug) -- no trajectory here since this script never drives the robot, just
    # the final (kept-by-both) obstacle cloud and its blue height-profile ellipsoid.
    plot_obstacle_pointcloud_debug(
        final_points,
        height_center,
        height_rotation,
        height_axes,
        out_path.with_suffix(".png"),
    )

    if annotated_path is not None and annotated_path.exists():
        import cv2

        logger.info("Press any key in the detection-box window to close it and exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
