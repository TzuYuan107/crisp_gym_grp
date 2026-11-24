"""Dedicated RGB-D ROS2 stream for the safety filter's obstacle perception.

`crisp_py.camera.Camera` (what `ManipulatorBaseEnv.cameras`/the VLA policy read) is
RGB-only by design: it exists to feed the policy, so it only subscribes to the color topic
and even throws away the color `CameraInfo` it does receive (see
`crisp_py/camera/camera.py:_callback_current_color_info`, which only reads height/width).
It also resizes/crops images to the policy's input resolution, which would desync the pixel
grid from a depth image and its intrinsics.

The safety filter needs metric depth registered 1:1 to a *known-resolution* color image plus
that resolution's intrinsics (K) and the camera's extrinsic transform into the robot base
frame -- none of which the VLA-facing `Camera` provides or should be made to provide. This
module subscribes to those topics independently (your camera's aligned-depth + camera-info
topics), so nothing about the existing RGB pipeline changes.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import rclpy
import rclpy.executors
import rclpy.time
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

if TYPE_CHECKING:
    from rclpy.node import Node

logger = logging.getLogger(__name__)



@dataclass
class SafetyRGBDCameraConfig:
    """Topics for the safety filter's own RGB-D perception stream.

    All three of ``color_image_topic``, ``depth_image_topic`` and ``depth_info_topic`` must
    refer to the *same* pixel grid (e.g. a RealSense depth stream aligned into the color
    frame: ``.../color/image_raw``, ``.../aligned_depth_to_color/image_raw`` and
    ``.../aligned_depth_to_color/camera_info``).
    """

    color_image_topic: str
    """Raw (uncompressed) topic name; this subscribes to f"{color_image_topic}/compressed",
    matching `crisp_py.camera.Camera`'s convention."""

    depth_image_topic: str
    """Metric (or mm-scale 16UC1) depth image topic, registered to color_image_topic's grid.
    Subscribed to directly -- compressedDepth transport is not supported, republish as raw
    sensor_msgs/Image if needed."""

    depth_info_topic: str
    """CameraInfo topic carrying the intrinsics matching the aligned color/depth grid."""

    camera_frame: str
    """TF optical frame the color/depth images are expressed in, used to look up extrinsics
    into the robot base frame."""

    base_frame_override: str | None = None
    """If set, used as the TF target frame instead of `env.robot.config.base_frame` -- for
    when your robot's real TF root link isn't named what crisp_py's RobotConfig assumes
    (e.g. Panda's default "base"). If a lookup ever fails, the raised error lists every frame
    actually known to the TF buffer at that moment, so you can read the real name off that."""

    dummy_camera_to_world: list[list[float]] | None = None
    """TESTING ONLY. A fixed 4x4 camera-to-base matrix to use instead of a real TF lookup --
    for exercising the rest of the pipeline (detection -> point cloud -> ellipsoid -> CBF)
    before the camera is actually calibrated against the robot, per the real aegis project's
    own `CAM_TO_BASE = np.eye(4)` placeholder in test_realsense_pointcloud.py. The obstacle
    position/orientation this produces will be physically meaningless -- do not run this near
    anything you don't want the arm to actually get close to. Unset this once you have a real
    calibration (see `base_frame_override` / `SafetyRGBDCamera.get_camera_to_world`)."""

    max_image_delay: float = 1.0


class SafetyRGBDCamera:
    """Subscribes to color + aligned depth + intrinsics, and resolves extrinsics via TF."""

    THREADS_REQUIRED = 3

    def __init__(
        self,
        config: SafetyRGBDCameraConfig,
        namespace: str = "",
        node: Node | None = None,
        spin_node: bool = True,
    ):
        """Initialize the perception stream.

        Args:
            config: Topic/frame configuration.
            namespace: ROS2 namespace to create the node in (e.g. the robot's namespace, so
                relative topic names resolve the same way as the rest of crisp_gym).
            node: Existing ROS2 node to reuse. If None, a new node is created.
            spin_node: Whether to spin the node in a background thread.
        """
        self.config = config

        if node is None:
            if not rclpy.ok():
                rclpy.init()
            self.node = rclpy.create_node("safety_rgbd_camera", namespace=namespace)
        else:
            self.node = node

        # tf2_ros.TransformListener subscribes to the *relative* topics "tf"/"tf_static".
        # /tf and /tf_static are conventionally global (unnamespaced) regardless of any
        # per-robot node namespace -- frames are disambiguated by frame_id, not by namespace.
        # If this listener were attached to `self.node` (namespaced above for the camera
        # topics), it would silently subscribe to e.g. "/left/tf" instead of "/tf" and never
        # see the robot's actual TF tree, so it gets its own unnamespaced node.
        import tf2_ros

        self._tf_node = rclpy.create_node("safety_rgbd_camera_tf")
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._tf_node)

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._color: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._intrinsics: np.ndarray | None = None

        self.node.create_subscription(
            CompressedImage,
            f"{config.color_image_topic}/compressed",
            self._on_color,
            qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )
        self.node.create_subscription(
            Image,
            config.depth_image_topic,
            self._on_depth,
            qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )
        self.node.create_subscription(
            CameraInfo,
            config.depth_info_topic,
            self._on_info,
            qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )

        if spin_node:
            threading.Thread(target=self._spin, daemon=True).start()

    def _spin(self):
        if not rclpy.ok():
            rclpy.init()
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=self.THREADS_REQUIRED)
        executor.add_node(self.node)
        executor.add_node(self._tf_node)
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)

    def _on_color(self, msg: CompressedImage):
        image = np.asarray(self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="rgb8"))
        with self._lock:
            self._color = image

    def _on_depth(self, msg: Image):
        depth = np.asarray(self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"))
        depth = depth.astype(np.float32)
        if msg.encoding in ("16UC1", "mono16"):
            depth = depth / 1000.0  # millimeters -> meters
        with self._lock:
            self._depth = depth

    def _on_info(self, msg: CameraInfo):
        with self._lock:
            self._intrinsics = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    @property
    def current_color(self) -> np.ndarray:
        """Latest RGB image, pixel-aligned with `current_depth`."""
        with self._lock:
            if self._color is None:
                raise RuntimeError(
                    "No color image received yet on "
                    f"{self.config.color_image_topic}/compressed. Call wait_until_ready() first."
                )
            return self._color

    @property
    def current_depth(self) -> np.ndarray:
        """Latest metric depth image (meters), pixel-aligned with `current_color`."""
        with self._lock:
            if self._depth is None:
                raise RuntimeError(
                    f"No depth image received yet on {self.config.depth_image_topic}. "
                    "Call wait_until_ready() first."
                )
            return self._depth

    @property
    def intrinsics(self) -> np.ndarray:
        """(3, 3) intrinsic matrix K matching the color/depth pixel grid."""
        with self._lock:
            if self._intrinsics is None:
                raise RuntimeError(
                    f"No CameraInfo received yet on {self.config.depth_info_topic}. "
                    "Call wait_until_ready() first."
                )
            return self._intrinsics

    def get_camera_to_world(self, base_frame: str, timeout: float = 1.0) -> np.ndarray:
        """Look up the (4, 4) homogeneous transform from the camera frame into ``base_frame``.

        Args:
            base_frame: Target frame, typically `env.robot.config.base_frame`. If that doesn't
                match a real frame in your TF tree (e.g. your URDF's root link isn't literally
                named "base"), pass the real frame name instead.
            timeout: Seconds to wait for the transform to become available.

        Returns:
            4x4 camera-to-world (base_frame) transform.
        """
        if self.config.dummy_camera_to_world is not None:
            logger.warning(
                "SafetyRGBDCamera: using dummy_camera_to_world instead of a real TF lookup -- "
                "obstacle position/orientation will NOT reflect physical reality. Testing only."
            )
            return np.array(self.config.dummy_camera_to_world, dtype=float)

        import tf2_ros
        from scipy.spatial.transform import Rotation

        try:
            transform = self._tf_buffer.lookup_transform(
                base_frame,
                self.config.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout),
            )
        except tf2_ros.TransformException as exc:
            raise tf2_ros.TransformException(
                f"{exc}\nKnown frames in this TF buffer right now:\n"
                f"{self._tf_buffer.all_frames_as_yaml()}"
            ) from exc
        translation = transform.transform.translation
        rotation = transform.transform.rotation

        camera_to_world = np.eye(4)
        camera_to_world[:3, :3] = Rotation.from_quat(
            [rotation.x, rotation.y, rotation.z, rotation.w]
        ).as_matrix()
        camera_to_world[:3, 3] = [translation.x, translation.y, translation.z]
        return camera_to_world

    def is_ready(self) -> bool:
        """Whether color, depth and intrinsics have all been received at least once."""
        with self._lock:
            return self._color is not None and self._depth is not None and self._intrinsics is not None

    def wait_until_ready(self, timeout: float = 10.0, check_frequency: float = 10.0):
        """Block until color, depth and intrinsics have all arrived, or raise `TimeoutError`."""
        rate = self.node.create_rate(check_frequency)
        while not self.is_ready():
            rate.sleep()
            timeout -= 1.0 / check_frequency
            if timeout <= 0:
                raise TimeoutError(
                    "Timeout waiting for safety RGB-D camera. Is something publishing "
                    f"{self.config.color_image_topic}/compressed, {self.config.depth_image_topic} "
                    f"and {self.config.depth_info_topic}?"
                )
