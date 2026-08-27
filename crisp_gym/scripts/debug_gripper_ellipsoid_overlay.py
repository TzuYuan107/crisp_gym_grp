"""Debug tool: stream the third-person camera feed with the CBF's gripper ellipsoid overlaid,
with live keyboard controls to nudge `camera_to_world` until it lines up with the real gripper.

Projects the CBF safety filter's end-effector proxy ellipsoid onto the third-person camera's
live image, every frame, using the exact same computation the deployed safety filter uses:
`SafetyFilter._ee_ball_pose` (built from the robot's live `end_effector_pose` plus
`ee_offset_local`) for the ellipsoid's center/orientation, `ee_shape_diag` for its semi-axes,
and `crisp_gym.safety.debug_plots._ellipsoid_point_cloud` (the same helper already used to
render this ellipsoid as a "gripper point cloud" in the 3D trajectory plots) to sample points
on its surface. Those 3D points get projected into the image via the camera's intrinsics and
`camera_to_world` (the config's `dummy_camera_to_world` if set, otherwise a live TF lookup) --
the same extrinsics the obstacle perception pipeline uses, just applied in the opposite
(world -> camera) direction.

The red center dot comes *only* from the robot's own live proprioceptive pose plus a fixed
local offset -- it never touches the camera or depth data. So if it doesn't land on the real
gripper in the image, that's isolated, unambiguous evidence of residual error in
`camera_to_world` specifically (not `ee_offset_local`, not depth noise, not perception). This
script lets you correct that live: nudge the extrinsic with the keyboard while watching the red
dot/green ellipsoid move, converge by eye, then read the corrected matrix off the console.

Pose comes from a plain, read-only `PoseStamped` subscription (default topic
`/left/current_pose`, same convention `handeye_calibration.py` uses) rather than a full
`crisp_py.robot.Robot` client -- `Robot` also tracks target poses and participates in the
control graph, so instantiating a second one alongside whatever's already driving the arm
risks conflicting with it. This script only ever reads, never publishes/controls anything.

Usage (needs the robot bringup, robot controller, and camera all running, inside the
jazzy-pi05-aegis pixi env):
    pixi run -e jazzy-pi05-aegis python3 crisp_gym/scripts/debug_gripper_ellipsoid_overlay.py \\
        --pose-topic /left/current_pose

Keyboard controls (rotation nudges are in the camera's own local frame; translation nudges are
in the TCP's *current* local frame -- i.e. w/s/a/d/r/f move the camera along whichever direction
the gripper is presently pointing/left/up, not fixed base-frame axes, since that's easier to
reason about by eye while watching the gripper than raw base x/y/z. The printed/final matrix is
always plain base-frame camera_to_world regardless -- the TCP frame is only used to interpret
each keypress before it's converted back to a base-frame delta):
    w/s   translate camera +/- along TCP local x     i/k   pitch camera +/-
    a/d   translate camera +/- along TCP local y     j/l   yaw camera +/-
    r/f   translate camera +/- along TCP local z     u/o   roll camera +/-

i/k/j/l/u/o still only ever accumulate into `camera_to_world`'s *rotation* -- plain, simple,
independent of p1/translation, so it can never end up in a bad state regardless of how the robot
moves between keypresses. Separately, purely for *display*, the main loop recomputes an "orbited"
camera position fresh every single frame (never fed back into `camera_to_world`) that keeps p1
pixel-fixed regardless of how much rotation is currently dialed in -- so only the geometry around
it (the ellipsoid's own great-ellipse curves) visibly rotates in place, letting you converge on
`camera_to_world`'s rotation by eye the same way w/s/a/d/r/f already let you converge on its
translation, without that visual aid ever corrupting the printed/final matrix.
    [ / ] halve / double the step size
    z     reset to the starting matrix
    p     print the current matrix (and equivalent static_transform_publisher command) now
    q     quit (also prints the final matrix)
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import rclpy.executors
import yaml
from crisp_py.utils.geometry import Pose
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation

from crisp_gym.safety.debug_plots import _ellipsoid_point_cloud
from crisp_gym.safety.depth_camera import SafetyRGBDCamera, SafetyRGBDCameraConfig
from crisp_gym.safety.safety_filter import SafetyFilter, SafetyFilterConfig
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)

BASE_TRANSLATION_STEP = 0.005  # meters
BASE_ROTATION_STEP = 0.5  # degrees


class PoseTopicSubscriber:
    """Minimal read-only `PoseStamped` subscriber -- no `Robot` control machinery, no publishing.

    Mirrors `handeye_calibration.py`'s `HandEyeCollector.gripper_pose()` pattern: just listens,
    never sends anything, so it can't conflict with whatever's already controlling the robot.
    """

    def __init__(self, pose_topic: str):
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("debug_gripper_overlay_pose_sub")
        self._lock = threading.Lock()
        self._pose: Pose | None = None
        self.node.create_subscription(PoseStamped, pose_topic, self._on_pose, 10)
        threading.Thread(target=self._spin, daemon=True).start()

    def _spin(self) -> None:
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self.node)
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)

    def _on_pose(self, msg: PoseStamped) -> None:
        p = msg.pose
        position = np.array([p.position.x, p.position.y, p.position.z])
        orientation = Rotation.from_quat(
            [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        )
        with self._lock:
            self._pose = Pose(position=position, orientation=orientation)

    @property
    def latest_pose(self) -> Pose | None:
        with self._lock:
            return self._pose

    def wait_until_ready(self, timeout: float = 10.0) -> None:
        start = time.time()
        while self.latest_pose is None:
            if time.time() - start > timeout:
                raise TimeoutError(f"No pose received on the subscribed topic within {timeout}s.")
            time.sleep(0.05)


def _project_to_image(
    points_world: np.ndarray, world_to_camera: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    """(N, 3) base-frame points -> (M, 2) pixel coords, dropping points behind the camera."""
    points_homo = np.hstack([points_world, np.ones((len(points_world), 1))])
    points_cam = (world_to_camera @ points_homo.T).T[:, :3]
    in_front = points_cam[:, 2] > 1e-3
    pixels_homo = (intrinsics @ points_cam[in_front].T).T
    return pixels_homo[:, :2] / pixels_homo[:, 2:3]


def _draw_closed_curve(
    img: np.ndarray, points_world: np.ndarray, world_to_camera: np.ndarray, intrinsics: np.ndarray,
    color: tuple[int, int, int], thickness: int = 2,
) -> None:
    """Draw a closed polyline (e.g. one of the ellipsoid's great-ellipse surface curves) by
    connecting consecutive `points_world` rows, wrapping the last point back to the first.
    Segments with either endpoint behind the camera are skipped individually, rather than
    dropping the whole curve, so the part of it that's actually visible still gets drawn."""
    points_homo = np.hstack([points_world, np.ones((len(points_world), 1))])
    points_cam = (world_to_camera @ points_homo.T).T[:, :3]
    in_front = points_cam[:, 2] > 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels_homo = (intrinsics @ points_cam.T).T
        pixels = pixels_homo[:, :2] / pixels_homo[:, 2:3]
    n = len(points_world)
    for i in range(n):
        j = (i + 1) % n
        if in_front[i] and in_front[j]:
            p_start, p_end = tuple(pixels[i].astype(int)), tuple(pixels[j].astype(int))
            cv2.line(img, p_start, p_end, color, thickness, cv2.LINE_AA)


def _print_matrix(camera_to_world: np.ndarray, base_frame: str, camera_frame: str) -> None:
    t = camera_to_world[:3, 3]
    quat = Rotation.from_matrix(camera_to_world[:3, :3]).as_quat()  # x, y, z, w
    logger.info("Current camera_to_world:\n%s", camera_to_world)
    logger.info(
        "As dummy_camera_to_world (paste into vlsa_config.yaml):\n"
        "  dummy_camera_to_world:\n"
        "    - [%.6f, %.6f, %.6f, %.6f]\n"
        "    - [%.6f, %.6f, %.6f, %.6f]\n"
        "    - [%.6f, %.6f, %.6f, %.6f]\n"
        "    - [0.0, 0.0, 0.0, 1.0]",
        *camera_to_world[0],
        *camera_to_world[1],
        *camera_to_world[2],
    )
    logger.info(
        "Equivalent static_transform_publisher command:\n"
        "  ros2 run tf2_ros static_transform_publisher --x %.6f --y %.6f --z %.6f "
        "--qx %.6f --qy %.6f --qz %.6f --qw %.6f --frame-id %s --child-frame-id %s",
        t[0], t[1], t[2], quat[0], quat[1], quat[2], quat[3], base_frame, camera_frame,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        default="crisp_gym/config/policy/vlsa_config.yaml",
        help="Path to vlsa_config.yaml (reads safety.ee_offset_local/ee_shape_diag and safety_camera).",
    )
    parser.add_argument(
        "--pose-topic",
        default="/left/current_pose",
        help="PoseStamped topic publishing the gripper's live pose in the robot base frame "
        "(crisp_controllers PoseBroadcaster, same convention as handeye_calibration.py).",
    )
    parser.add_argument(
        "--n-points", type=int, default=300, help="How many points to sample on the ellipsoid surface."
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Seconds to wait for the camera/robot to become ready."
    )
    args = parser.parse_args()

    setup_logging()

    config = yaml.safe_load(Path(args.config).read_text())
    safety_filter = SafetyFilter(SafetyFilterConfig(**config["safety"]))
    camera_cfg = config["safety_camera"]
    ee_shape_diag = np.array(safety_filter.config.ee_shape_diag)

    if not rclpy.ok():
        rclpy.init()

    depth_camera = None
    pose_sub = None
    try:
        logger.info("Connecting to the safety RGB-D camera...")
        depth_camera = SafetyRGBDCamera(SafetyRGBDCameraConfig(**camera_cfg))
        depth_camera.wait_until_ready(timeout=args.timeout)

        logger.info("Subscribing to gripper pose on '%s' (read-only)...", args.pose_topic)
        pose_sub = PoseTopicSubscriber(args.pose_topic)
        pose_sub.wait_until_ready(timeout=args.timeout)

        intrinsics = depth_camera.intrinsics
        base_frame = camera_cfg.get("base_frame_override") or "base"
        camera_frame = camera_cfg["camera_frame"]
        original_camera_to_world = depth_camera.get_camera_to_world(base_frame)
        camera_to_world = original_camera_to_world.copy()
        logger.info("Starting camera_to_world:\n%s", camera_to_world)
        logger.info("ee_offset_local=%s ee_shape_diag=%s", safety_filter.config.ee_offset_local, ee_shape_diag)

        step_multiplier = 1.0
        window = "gripper ellipsoid overlay -- w/a/s/d/r/f translate, i/j/k/l/u/o rotate, [/] step, z reset, p print, q quit"
        logger.info("Streaming. Keyboard controls are listed in this script's module docstring.")
        while True:
            pose = pose_sub.latest_pose
            p1, R1 = safety_filter._ee_ball_pose(pose)
            ellipsoid_points = _ellipsoid_point_cloud(p1, R1, ee_shape_diag, n=args.n_points)

            # Render-only "orbit" transform: reposition the *displayed* camera so p1 stays
            # pixel-fixed regardless of how much rotation has been dialed in via i/j/k/l/u/o,
            # without ever writing this back into `camera_to_world` itself (that stays a plain,
            # simple accumulation of the raw keypress deltas -- see those handlers below -- which
            # is what actually gets printed/returned). Recomputed from scratch every frame from
            # only the *current* camera_to_world, the fixed starting orientation, and the
            # *current* live p1 -- no dependency on any previous frame's p1 or on this same
            # formula's own prior output, so unlike an earlier version of this trick, it cannot
            # compound into a runaway result even if p1 jumps a lot between frames (e.g. because
            # the robot moved) or many rotation keys get pressed in a row.
            rotation_offset = camera_to_world[:3, :3] @ original_camera_to_world[:3, :3].T
            display_camera_to_world = camera_to_world.copy()
            display_camera_to_world[:3, 3] = p1 + rotation_offset @ (camera_to_world[:3, 3] - p1)
            world_to_camera = np.linalg.inv(display_camera_to_world)
            frame_bgr = cv2.cvtColor(depth_camera.current_color, cv2.COLOR_RGB2BGR)
            h, w = frame_bgr.shape[:2]

            # Translucent green fill over the convex hull of the projected ellipsoid point cloud,
            # instead of drawing each sampled point individually.
            ellipsoid_pixels = _project_to_image(ellipsoid_points, world_to_camera, intrinsics)
            if len(ellipsoid_pixels) >= 3:
                hull = cv2.convexHull(ellipsoid_pixels.astype(np.float32)).astype(np.int32)
                overlay = frame_bgr.copy()
                cv2.fillConvexPoly(overlay, hull, (0, 255, 0))
                cv2.addWeighted(overlay, 0.35, frame_bgr, 0.65, 0, frame_bgr)
                cv2.polylines(frame_bgr, [hull], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            for u, v in _project_to_image(p1[None, :], world_to_camera, intrinsics):
                if 0 <= u < w and 0 <= v < h:
                    cv2.circle(frame_bgr, (int(u), int(v)), 6, (0, 0, 255), -1)

            # Three orthogonal "great ellipse" curves lying on the ellipsoid's own surface -- its
            # cross-sections with the local xy/yz/xz planes, like a globe's equator plus two
            # meridians. Shows the ellipsoid's rotation directly, unlike the point cloud alone.
            # Colors deliberately avoid green (the point cloud's own color).
            curve_t = np.linspace(0, 2 * np.pi, 60)
            a, b, c = ee_shape_diag
            xy_local = np.stack([a * np.cos(curve_t), b * np.sin(curve_t), np.zeros_like(curve_t)], axis=-1)
            yz_local = np.stack([np.zeros_like(curve_t), b * np.cos(curve_t), c * np.sin(curve_t)], axis=-1)
            xz_local = np.stack([a * np.cos(curve_t), np.zeros_like(curve_t), c * np.sin(curve_t)], axis=-1)
            for local_curve, color in (
                (xy_local, (0, 0, 255)),  # red
                (yz_local, (0, 255, 255)),  # yellow
                (xz_local, (255, 255, 0)),  # cyan
            ):
                world_curve = local_curve @ R1.T + p1
                _draw_closed_curve(frame_bgr, world_curve, world_to_camera, intrinsics, color, thickness=2)

            t_delta = camera_to_world[:3, 3] - original_camera_to_world[:3, 3]
            trans_step = BASE_TRANSLATION_STEP * step_multiplier
            rot_step = BASE_ROTATION_STEP * step_multiplier
            overlay = [
                f"step: {trans_step * 1000:.1f}mm / {rot_step:.2f}deg  ([ ] to change)",
                f"translation delta from start: [{t_delta[0]:+.4f}, {t_delta[1]:+.4f}, {t_delta[2]:+.4f}]",
                "w/s/a/d/r/f=translate (TCP-local xyz)  i/k/j/l/u/o=rotate (camera-local)  z=reset  p=print  q=quit",
            ]
            for i, line in enumerate(overlay):
                cv2.putText(
                    frame_bgr, line, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                )

            cv2.imshow(window, frame_bgr)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("p"):
                _print_matrix(camera_to_world, base_frame, camera_frame)
            elif key == ord("z"):
                camera_to_world = original_camera_to_world.copy()
                step_multiplier = 1.0
                logger.info("Reset to starting camera_to_world.")
            elif key == ord("["):
                step_multiplier = max(step_multiplier / 2, 1.0 / 32)
            elif key == ord("]"):
                step_multiplier = min(step_multiplier * 2, 32.0)
            elif key in (ord("w"), ord("s"), ord("a"), ord("d"), ord("r"), ord("f")):
                delta_tcp = {
                    ord("w"): np.array([trans_step, 0, 0]),
                    ord("s"): np.array([-trans_step, 0, 0]),
                    ord("a"): np.array([0, trans_step, 0]),
                    ord("d"): np.array([0, -trans_step, 0]),
                    ord("r"): np.array([0, 0, trans_step]),
                    ord("f"): np.array([0, 0, -trans_step]),
                }[key]
                # R1 is the TCP's own current base-frame rotation (see `_ee_ball_pose` above) --
                # rotate the TCP-local delta into base frame before applying, so e.g. "w" always
                # means "along however the gripper is presently pointing", not fixed base x.
                camera_to_world[:3, 3] += R1 @ delta_tcp
            elif key in (ord("i"), ord("k"), ord("j"), ord("l"), ord("u"), ord("o")):
                euler_delta = {
                    ord("i"): [rot_step, 0, 0],
                    ord("k"): [-rot_step, 0, 0],
                    ord("j"): [0, rot_step, 0],
                    ord("l"): [0, -rot_step, 0],
                    ord("u"): [0, 0, rot_step],
                    ord("o"): [0, 0, -rot_step],
                }[key]
                nudge = Rotation.from_euler("xyz", euler_delta, degrees=True).as_matrix()
                # Plain accumulation, independent of p1/translation -- see the render-only orbit
                # transform above for how this stays visually pinned on the gripper without this
                # (the real, printed) matrix ever depending on the live, possibly-moving p1.
                camera_to_world[:3, :3] = camera_to_world[:3, :3] @ nudge

        logger.info("Final result:")
        _print_matrix(camera_to_world, base_frame, camera_frame)
    finally:
        cv2.destroyAllWindows()
        if pose_sub is not None:
            pose_sub.node.destroy_node()
        if depth_camera is not None:
            depth_camera.node.destroy_node()
            depth_camera._tf_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        time.sleep(0.2)  # give any background spin threads a chance to notice and exit cleanly


if __name__ == "__main__":
    main()
