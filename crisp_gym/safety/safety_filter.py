"""CBF-QP safety filter, ported from the per-step control loop in aegis's ``main_aegis.py``.

aegis ran this loop directly against a LIBERO/robosuite MuJoCo simulation, where the action
convention, end-effector pose, and obstacle point cloud all came from `env.sim`. This module
carries the same CBF-QP math over to crisp_gym's real interfaces:

* Pose: aegis converted MuJoCo's ``[w, x, y, z]`` quaternion by hand (``quat_R``/``quat_euler``).
  crisp_gym's `crisp_py.robot.Robot.end_effector_pose` already returns a
  `crisp_py.utils.geometry.Pose` whose ``.orientation`` is a scipy `Rotation`, so that
  conversion is gone -- we just call ``.as_matrix()``.
* Action: aegis indexed the VLA action as ``action[:3]`` (world-frame translation delta),
  ``action[3:6]`` (world-frame rotation delta) and ``action[6]`` (gripper), which is exactly
  `crisp_gym.envs.manipulator_env.ManipulatorCartesianEnv`'s relative-cartesian action layout
  *when the environment's orientation representation is 3D* (Euler or angle-axis). crisp_gym
  also supports a 4D quaternion representation, for which a 3-vector angular-velocity-style CBF
  correction doesn't apply directly -- `step()` below raises a clear error in that case instead
  of silently misinterpreting the action, rather than guessing.
* Obstacle location: aegis re-detected the obstacle's point cloud from MuJoCo's ground-truth
  depth + camera calibration every episode -- VLM obstacle naming (`obstacle_detection`),
  GroundingDINO + depth unprojection (`get_point_cloud`), then ellipsoid fitting, all before
  the control loop starts. `SafetyFilter.reset()` runs the same pipeline: given the task
  instruction plus an RGB-D frame (`crisp_gym.safety.depth_camera.SafetyRGBDCamera`, since
  crisp_py's own cameras are RGB-only -- see that module's docstring), it calls
  `crisp_gym.safety.perception.obstacle_detection` to name the likely obstacle, then
  `get_point_cloud` + `crisp_gym.safety.math_utils.crop_to_bounds` +
  `crisp_gym.safety.math_utils.fit_ellipse_from_height_profile` to localize it -- a departure
  from `main_aegis.py`'s own MVEE-based fit, which needed a plane-removed/DBSCAN-clustered/
  color-filtered cloud to avoid fitting the table; that filtering proved unstable in practice
  (color drifts frame-to-frame with lighting, RANSAC plane removal only triggers above a
  coverage threshold), so this fits the ellipsoid directly from height/top-slice-width instead
  (see `SafetyFilterConfig.ellipsoid_top_fraction` and
  `fit_ellipse_from_height_profile`'s docstring). `mode="static"` (a fixed ellipsoid from
  config) is still available for when there's no depth camera at all.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from crisp_py.utils.geometry import Pose

from crisp_gym.safety.debug_plots import (
    plot_episode_debug,
    plot_episode_debug_interactive,
    save_point_cloud,
)
from crisp_gym.safety.math_utils import (
    compute_h_coeffs_3d,
    compute_h_ij,
    crop_to_bounds,
    fit_ellipse_from_height_profile,
)
from crisp_gym.safety.perception import get_point_cloud, obstacle_detection

logger = logging.getLogger(__name__)


@dataclass
class SafetyFilterConfig:
    """Configuration for `SafetyFilter`.

    All positions are expressed in the robot base frame, i.e. the same frame as
    `crisp_py.robot.Robot.end_effector_pose`.
    """

    enabled: bool = False

    mode: str = "static"
    """"static" uses a fixed obstacle ellipsoid from this config. "point_cloud" runs the full
    aegis perception pipeline each episode: VLM obstacle naming + GroundingDINO detection +
    depth unprojection + ellipsoid fit (see `reset`). Requires an RGB-D frame, which the caller
    (`VlsaLerobotPolicy`) sources from `crisp_gym.safety.depth_camera.SafetyRGBDCamera`."""

    # End-effector proxy ellipsoid (aegis's "ball" attached below the eef frame).
    ee_offset_local: tuple[float, float, float] = (0.0, 0.0, -0.08)
    ee_shape_diag: tuple[float, float, float] = (0.06, 0.12, 0.11)

    # Static obstacle ellipsoid (only used when mode == "static").
    obstacle_position: tuple[float, float, float] | None = None
    obstacle_shape_diag: tuple[float, float, float] = (0.06, 0.06, 0.06)

    # Region of interest for point-cloud filtering (only used when mode == "point_cloud").
    xyz_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None

    # Obstacle ellipsoid construction (only used when mode == "point_cloud"). Fitted directly
    # from height/top-slice-width (`fit_ellipse_from_height_profile`) rather than an MVEE fit
    # over a plane-removed/DBSCAN-clustered/color-filtered cloud: table points are hard to
    # reliably strip out (color drifts frame-to-frame with lighting/auto-exposure, RANSAC plane
    # removal only triggers above a coverage threshold) without risking cutting into the
    # object's own base, which sits at the same height as the table. This instead only crops to
    # xyz_bounds (no outlier/plane/DBSCAN filtering -- those risk clipping the object's own
    # sparse top rim, see `fit_ellipse_from_height_profile`'s docstring) and measures height from
    # the full cloud, horizontal extent from only its uncontaminated top slice. Validated by
    # comparing both approaches side by side in `crisp_gym/scripts/debug_obstacle_pointcloud.py`.
    ellipsoid_top_fraction: float = 0.15
    # Sources the ellipsoid's x/y *center* from a slice around the object's vertical middle
    # rather than ellipsoid_top_fraction's top slice (radius keeps using the top slice
    # regardless). With an imperfect camera_to_world, a rotation error's positional effect grows
    # with distance from wherever it effectively pivots, so the top of a tall object accumulates
    # more displacement than points nearer its middle -- see fit_ellipse_from_height_profile's
    # center_fraction docstring.
    ellipsoid_center_fraction: float = 0.15
    # Subtracted from the ellipsoid's x-center after ellipsoid_center_fraction has otherwise
    # fully derived it -- an empirical correction for residual camera_to_world error that
    # ellipsoid_center_fraction alone doesn't fully cancel out. See
    # fit_ellipse_from_height_profile's center_x_offset docstring. 0.0 (no-op) unless you've
    # measured a specific offset for your setup.
    ellipsoid_center_x_offset: float = 0.0
    ellipsoid_min_top_points: int = 10
    ellipsoid_radius_percentile: float = 100.0
    ellipsoid_height_percentile: float = 100.0
    ellipsoid_margin_fraction: float = 0.15
    # A disjoint noise streak (flying-pixel depth artifact, common on glass/reflective objects)
    # can be taller than the real object and hijack z_max before the top slice is even chosen --
    # this clusters the whole cropped cloud first and keeps only its largest cluster, so that
    # can't happen. See fit_ellipse_from_height_profile's full_cluster_eps docstring.
    ellipsoid_full_cluster_eps: float = 0.02
    ellipsoid_full_cluster_min_points: int = 10
    # A disjoint noise streak (flying-pixel depth artifact, common on glass/reflective objects)
    # overlapping the top slice's z-range can inflate the ellipsoid's width even past what
    # ellipsoid_radius_percentile alone trims, if it's more than a small minority of the slice --
    # see fit_ellipse_from_height_profile's top_cluster_eps docstring.
    ellipsoid_top_cluster_eps: float = 0.02
    ellipsoid_top_cluster_min_points: int = 10
    # If set, used directly as the ellipsoid's horizontal radius instead of measuring it from the
    # top slice's spread each episode (ellipsoid_radius_percentile then goes unused). Center
    # position and z-extent are still measured live as usual -- only the radius becomes a
    # constant. For a known, repeatedly-detected obstacle, this avoids the radius measurement
    # ballooning toward a sphere at longer range, where pixel-space detection/depth noise gets
    # amplified by the pixel-to-metric unprojection (see fit_ellipse_from_height_profile's
    # fixed_radius docstring) even though the obstacle's true size hasn't changed.
    ellipsoid_fixed_radius: float | None = None

    # Obstacle perception (only used when mode == "point_cloud").
    groundingdino_config_path: str | None = None
    groundingdino_checkpoint_path: str | None = None
    obstacle_preference_list: list[str] | None = None
    zhipu_api_key: str | None = None
    """Defaults to the ZHIPUAI_API_KEY environment variable if not set here."""
    perception_device: str = "cuda"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    debug_dir: str | None = "outputs/safety"
    """Every episode, saves (under debug_dir/episode_NNNN/): the annotated GroundingDINO
    detection crop, the filtered obstacle point cloud (obstacle_points.ply), and a 3D plot
    comparing the raw/unfiltered trajectory to the safety-filtered one (trajectory.png). A
    one-line point-cloud summary is always logged regardless of this setting. Relative paths
    are resolved against the current working directory (i.e. wherever you launched
    `crisp-deploy-vla` from) -- the default matches this repo's existing gitignored
    `outputs/` convention (see .gitignore). Set to None to disable saving files (the log
    summary still happens)."""

    # CBF-QP gains (defaults are aegis's originals).
    cbf_alpha: float = 10.0

    # ============================================================================
    # ADDED (not in aegis's original): safety-margin offset for the CBF constraint.
    # ============================================================================
    # The constraint below (`step()`) used to be `a·u + cbf_alpha*h >= 0` -- a HARD switch that
    # is exactly zero correction whenever h > 0, and only ever engages right as h reaches the
    # true boundary. That produced a sawtooth in practice: h coasts down unopposed for several
    # steps, dips slightly negative, then gets hit with one large corrective snap back to
    # positive, repeat -- the "sudden direction change" behavior. Shifting the constraint to use
    # `(h - h_margin)` instead of raw `h` moves that switch-on point earlier, to h == h_margin
    # instead of h == 0, so the correction gets runway to engage gradually while still
    # comfortably far from the real boundary, instead of only at the last instant. When the
    # constraint is active it's solved with equality, `a·u = cbf_alpha*(margin - h)` -- so the
    # correction is already exactly zero at h == h_margin and grows in proportion to how far h
    # has dropped *below* h_margin, with no other new logic required. Default 0.0 keeps this an
    # exact no-op (identical to the original hard-at-zero behavior) unless explicitly set.
    h_margin: float = 0.0
    # ============================================================================

    ref_gain: float = 5.0
    v_gain: float = 0.2
    omega_gain: float = 0.2
    z_gain: float = 10.0
    h_constraint: float = 0.0

    # Output safety net: caps the corrected per-tick translation/rotation delta so the QP can
    # never demand more motion in one control tick than the arm can actually execute -- neither
    # this QP's math nor aegis's own `MAX_VEL` constant (declared but never actually used in
    # `main_aegis.py`) enforce this on their own. A legitimate large correction (deep constraint
    # violation) or an ill-conditioned QP solve can otherwise exceed Franka's own Cartesian
    # velocity limit and trigger its reflex/safety stop -- this clamp is what's between "the CBF
    # asked for too much" and the arm actually being commanded to do it.
    max_translational_velocity: float = 1.7
    """Meters/second. Default matches libfranka's published Cartesian translational velocity
    limit for Panda/FR3; confirm against your actual robot/firmware if it's been reconfigured."""
    max_rotational_velocity: float = 2.5
    """Radians/second. Default matches libfranka's published Cartesian rotational velocity
    limit for Panda/FR3; confirm against your actual robot/firmware if it's been reconfigured."""

    qp_v_weight: float = 1.0 / 25
    qp_omega_weight: float = 1.0 / 25
    qp_uz_weight: float = 1.0

    # ============================================================================
    # ADDED (not in aegis's original): cap on the `dt` actually used for physics.
    # ============================================================================
    # `step()`'s caller passes the *real measured* elapsed time as `dt` (see
    # vlsa_lerobot_policy.py's `dt = now - last_step_time`), not a fixed nominal value -- so a
    # real stall (e.g. an action-chunking VLA policy's periodic re-inference call taking ~1s
    # instead of the usual ~33ms) inflates dt by ~30x for that one step. The velocity clamp
    # (`_clamp_to_speed`, `max_delta = max_speed * dt`) and the z_fixed integration
    # (`z_fixed += dz * dt`) both scale directly with dt, on the assumption that a "step" is a
    # small nominal time slice -- with dt=~1s instead, `max_delta` for a 1.7 m/s limit becomes
    # ~1.7m, i.e. the clamp stops meaningfully constraining anything, and a large real Cartesian
    # jump can get commanded in one step, which is a plausible trigger for a Franka reflex/
    # safety-stop (large realized joint velocity/force). This caps *only the dt used inside the
    # physics calculations* at `max_dt` -- `SafetyFilterStepInfo.dt` / step_log.csv still record
    # the real, uncapped dt, so stalls stay visible in the logs even though they can no longer
    # inflate the allowed per-step motion.
    max_dt: float = 0.1
    """Seconds. Real dt above this is clamped before use in the velocity clamp / z_fixed
    integration (see above) -- it does not affect what's logged for `dt`."""
    # ============================================================================

    def __post_init__(self):
        """Validate mode-dependent required fields."""
        if not self.enabled:
            return
        if self.mode == "static" and self.obstacle_position is None:
            raise ValueError(
                "SafetyFilterConfig(mode='static') requires obstacle_position "
                "(x, y, z in the robot base frame) -- there is no safe default."
            )
        if self.mode == "point_cloud":
            if self.xyz_bounds is None:
                raise ValueError(
                    "SafetyFilterConfig(mode='point_cloud') requires xyz_bounds "
                    "((xmin,xmax),(ymin,ymax),(zmin,zmax)) in the robot base frame."
                )
            if self.groundingdino_config_path is None or self.groundingdino_checkpoint_path is None:
                raise ValueError(
                    "SafetyFilterConfig(mode='point_cloud') requires groundingdino_config_path "
                    "and groundingdino_checkpoint_path."
                )
        if self.mode not in ("static", "point_cloud"):
            raise ValueError(f"Unknown SafetyFilterConfig.mode: {self.mode!r}")

        # `step()` builds its QP objective in a rescaled "u"-space (`u_v_ref = ref_gain * v_ref`)
        # then converts back to physical units on the way out (`R1 @ u_v * v_gain`) -- when the
        # constraint isn't binding, the QP's unconstrained optimum trivially satisfies it, so
        # the filter returns `ref_gain * v_gain * translation_ref` unmodified. That's an exact
        # passthrough ONLY if `ref_gain * v_gain == 1.0` (same for `omega_gain`); tuning one gain
        # without the other silently rescales the "no correction needed" action, not just active
        # corrections, which is an easy way to introduce a bug that only shows up as the arm
        # drifting even when nothing is nearby.
        for gain_name, gain in (("v_gain", self.v_gain), ("omega_gain", self.omega_gain)):
            product = self.ref_gain * gain
            if abs(product - 1.0) > 1e-6:
                logger.warning(
                    "SafetyFilterConfig: ref_gain (%.4g) * %s (%.4g) = %.4g, not 1.0 -- the "
                    "safety filter will scale the action even when no correction is needed "
                    "(i.e. even far from any obstacle). Set %s = 1.0 / ref_gain to preserve "
                    "exact passthrough when the CBF constraint isn't active.",
                    self.ref_gain,
                    gain_name,
                    gain,
                    product,
                    gain_name,
                )


@dataclass
class SafetyFilterStepInfo:
    """Debug info produced by the most recent `SafetyFilter.step()` call."""

    active: bool
    h: float | None = None
    feasible: bool | None = None
    correction_norm: float | None = None
    dt: float | None = None
    """Actual `dt` this step ran with (whatever the caller passed to `step()`) -- logged so a
    real measured control period can be told apart from a nominal `1/control_frequency` guess
    after the fact, e.g. to check the CBF's velocity clamp/z_fixed integration weren't fed a
    stale time assumption."""


class SafetyFilter:
    """Per-episode ellipsoid setup + per-step CBF-QP action correction."""

    def __init__(self, config: SafetyFilterConfig):
        """Initialize the filter (does not compute anything until `reset()` is called)."""
        self.config = config
        self._active = False
        self._groundingdino_model = None
        self.p1: np.ndarray | None = None
        self.R1: np.ndarray | None = None
        self.p2: np.ndarray | None = None
        self.R2: np.ndarray | None = None
        self.Q2_diag: np.ndarray | None = None
        self.z_fixed: np.ndarray | None = None
        self.obstacle_name: str | None = None
        self.last_info = SafetyFilterStepInfo(active=False)

        # Per-episode debug artifacts (see `_finalize_episode_debug`).
        self._episode_index = -1
        self._episode_points: np.ndarray | None = None
        self._trajectory_actual: list[np.ndarray] = []
        self._trajectory_reference: list[np.ndarray] = []
        self._trajectory_predicted: list[np.ndarray] = []
        self._trajectory_modified: list[np.ndarray] = []
        # Visualization/reporting only -- NEVER fed into h/correction/QP, regardless of what it
        # shows. Lets you compare `_trajectory_actual` (built from whichever pose the caller
        # passed as `step()`'s `ee_pose` -- the one that *does* drive h/correction and thus the
        # actual commanded action) against this second pose (whatever the caller passed as
        # `step()`'s `comparison_pose`). Populated only when the caller opts in via
        # `step(..., comparison_pose=...)`.
        self._trajectory_comparison: list[np.ndarray] = []
        # Per-point flag, aligned with `_trajectory_comparison`: whether *that pose* -- not the
        # one driving h/correction -- would itself be in violation (its own h against the same
        # obstacle ellipsoid, computed independently via `compute_h_ij`, read-only, never
        # affecting the QP). E.g. if `ee_pose=target_pose` drives corrections but
        # `comparison_pose=end_effector_pose`, this tells you whether the *real* arm position
        # would actually collide -- not just whether the commanded target would have.
        self._trajectory_comparison_collision: list[bool] = []
        self._step_index = -1

    @property
    def active(self) -> bool:
        """Whether the filter has a valid obstacle ellipsoid and will correct actions this episode."""
        return self._active

    def _ee_ball_pose(self, ee_pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        R1 = ee_pose.orientation.as_matrix()
        p1 = ee_pose.position + R1 @ np.array(self.config.ee_offset_local)
        return p1, R1

    @staticmethod
    def _clamp_to_speed(delta: np.ndarray, max_speed: float, dt: float) -> tuple[np.ndarray, bool]:
        """Scale `delta` down (preserving direction) so `||delta|| / dt` never exceeds `max_speed`.

        Returns the (possibly rescaled) delta and whether clamping actually triggered.
        """
        max_delta = max_speed * dt
        norm = np.linalg.norm(delta)
        if norm > max_delta and norm > 0:
            return delta * (max_delta / norm), True
        return delta, False

    def _ensure_groundingdino_model(self):
        if self._groundingdino_model is None:
            from groundingdino.util.inference import load_model

            self._groundingdino_model = load_model(
                self.config.groundingdino_config_path,
                self.config.groundingdino_checkpoint_path,
            )
        return self._groundingdino_model

    def _localize_obstacle(
        self,
        task_description: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
    ) -> np.ndarray:
        """Run the aegis perception pipeline: name obstacle -> GroundingDINO -> depth -> crop.

        If `config.obstacle_preference_list` has exactly one entry, that name is used directly
        as the GroundingDINO prompt and the VLM naming step (`obstacle_detection`, which needs
        a ZhipuAI API key) is skipped entirely -- same object every episode regardless of
        `task_description`, but no external API dependency.
        """
        if self.config.obstacle_preference_list and len(self.config.obstacle_preference_list) == 1:
            self.obstacle_name = self.config.obstacle_preference_list[0]
            logger.info(
                "SafetyFilter: using fixed obstacle name '%s' (obstacle_preference_list has a "
                "single entry, skipping the VLM naming step).",
                self.obstacle_name,
            )
        else:
            self.obstacle_name = obstacle_detection(
                rgb,
                task_description,
                preference_list=self.config.obstacle_preference_list,
                api_key=self.config.zhipu_api_key,
            )
            logger.info("SafetyFilter: VLM identified likely obstacle as '%s'.", self.obstacle_name)

        model = self._ensure_groundingdino_model()
        raw_points = get_point_cloud(
            rgb,
            depth,
            camera_intrinsics,
            camera_to_world,
            self.obstacle_name,
            model,
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            device=self.config.perception_device,
            debug_dir=self._episode_debug_dir(),
        )
        cropped = crop_to_bounds(raw_points, self.config.xyz_bounds)
        self._episode_points = cropped
        return cropped

    def _episode_debug_dir(self) -> Path | None:
        if not self.config.debug_dir:
            return None
        return Path(self.config.debug_dir).expanduser() / f"episode_{self._episode_index:04d}"

    @staticmethod
    def _action_column_names(n: int) -> list[str]:
        """Name each action component (`[dx,dy,dz,rx,ry,rz,*gripper]`) for its own CSV column."""
        names = ["dx", "dy", "dz", "rx", "ry", "rz"][: min(n, 6)]
        gripper_n = max(n - 6, 0)
        names += ["gripper"] if gripper_n == 1 else [f"gripper_{i}" for i in range(gripper_n)]
        return names

    def _log_step(self, raw_action: np.ndarray, safe_action: np.ndarray, info: SafetyFilterStepInfo) -> None:
        """Append one row to this episode's `step_log.csv` (original vs. modified action + CBF info).

        One CSV per episode folder (alongside `obstacle_points.ply`/`trajectory.html`) instead
        of a single un-keyed log spanning every episode of every run -- so a given episode's
        step-by-step behavior can be read on its own. Each action component gets its own column
        (rather than one packed string) so a plain CSV/spreadsheet viewer shows real numeric
        columns instead of a single opaque field.
        """
        debug_dir = self._episode_debug_dir()
        if debug_dir is None:
            return
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / "step_log.csv"
        is_new = not path.exists()
        raw = np.asarray(raw_action).reshape(-1)
        safe = np.asarray(safe_action).reshape(-1)
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                names = self._action_column_names(len(raw))
                writer.writerow(
                    ["step"]
                    + [f"original_{n}" for n in names]
                    + [f"modified_{n}" for n in names]
                    + ["h", "feasible", "correction_norm", "dt"]
                )
            writer.writerow(
                [self._step_index]
                + [f"{v:.6f}" for v in raw]
                + [f"{v:.6f}" for v in safe]
                + [
                    "" if info.h is None else f"{info.h:.6f}",
                    "" if info.feasible is None else info.feasible,
                    "" if info.correction_norm is None else f"{info.correction_norm:.6f}",
                    "" if info.dt is None else f"{info.dt:.6f}",
                ]
            )

    def _finalize_episode_debug(self) -> None:
        """Save/log the just-finished episode's obstacle point cloud + trajectory comparison.

        Handles both `mode="point_cloud"` (obstacle_points.ply + trajectory plots) and
        `mode="static"` (trajectory plots only -- there's no real point cloud to save, just the
        configured fixed ellipsoid, so `_episode_points` legitimately stays None all episode).
        Safe to call more than once for the same episode (e.g. once from an explicit
        `finalize()` at episode-end, once defensively at the top of the next `reset()`) --
        `_episode_points`/`_trajectory_actual` are cleared after a successful flush, so repeats
        are a no-op instead of silently re-saving/re-plotting identical files.
        """
        # Not just `is not None`: `_episode_points` gets set from crop_to_bounds's output, which
        # is legitimately a (0, 3) empty (but non-None) array whenever detection found nothing or
        # every point fell outside xyz_bounds that episode -- save_point_cloud's min/max would
        # crash on that.
        have_points = self._episode_points is not None and len(self._episode_points) > 0
        have_trajectory = self.p2 is not None and len(self._trajectory_actual) > 1
        if not have_points and not have_trajectory:
            return

        debug_dir = self._episode_debug_dir()
        if debug_dir is None:
            if have_points:
                centroid = self._episode_points.mean(axis=0)
                logger.info(
                    "SafetyFilter: episode %d obstacle point cloud: %d points, centroid=%s",
                    self._episode_index,
                    len(self._episode_points),
                    np.array2string(centroid, precision=3),
                )
        else:
            if have_points:
                save_point_cloud(self._episode_points, debug_dir / "obstacle_points.ply")
            if have_trajectory:
                obstacle_points = self._episode_points if have_points else np.zeros((0, 3))
                plot_episode_debug(
                    obstacle_points,
                    self.p2,
                    self.R2,
                    self.Q2_diag,
                    np.array(self._trajectory_actual),
                    np.array(self._trajectory_reference),
                    debug_dir / "trajectory.png",
                    gripper_center=self.p1,
                    gripper_rotation=self.R1,
                    gripper_semi_axes=np.array(self.config.ee_shape_diag),
                    trajectory_predicted=np.array(self._trajectory_predicted),
                    trajectory_modified=np.array(self._trajectory_modified),
                    trajectory_comparison=np.array(self._trajectory_comparison),
                    trajectory_comparison_collision=np.array(self._trajectory_comparison_collision),
                )
                plot_episode_debug_interactive(
                    obstacle_points,
                    self.p2,
                    self.R2,
                    self.Q2_diag,
                    np.array(self._trajectory_actual),
                    np.array(self._trajectory_reference),
                    debug_dir / "trajectory.html",
                    gripper_center=self.p1,
                    gripper_rotation=self.R1,
                    gripper_semi_axes=np.array(self.config.ee_shape_diag),
                    trajectory_predicted=np.array(self._trajectory_predicted),
                    trajectory_modified=np.array(self._trajectory_modified),
                    trajectory_comparison=np.array(self._trajectory_comparison),
                    trajectory_comparison_collision=np.array(self._trajectory_comparison_collision),
                )

        self._episode_points = None
        self._trajectory_actual = []
        self._trajectory_predicted = []
        self._trajectory_modified = []
        self._trajectory_comparison = []
        self._trajectory_comparison_collision = []

    def finalize(self) -> None:
        """Flush debug artifacts for the most recent episode. Call once, e.g. from `Policy.shutdown()`."""
        self._finalize_episode_debug()

    def reset(
        self,
        ee_pose: Pose,
        task_description: str | None = None,
        rgb: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        camera_intrinsics: np.ndarray | None = None,
        camera_to_world: np.ndarray | None = None,
    ) -> None:
        """(Re-)initialize the obstacle ellipsoid and end-effector proxy for a new episode.

        In `mode="point_cloud"`, this runs the whole aegis perception pipeline once per episode
        (VLM obstacle naming -> GroundingDINO -> depth unprojection -> filtering -> ellipsoid
        fit), exactly like `main_aegis.py` did before its control loop. It needs
        ``task_description``, ``rgb``, ``depth``, ``camera_intrinsics`` and ``camera_to_world``
        for that; typically sourced from `crisp_gym.safety.depth_camera.SafetyRGBDCamera`.

        Args:
            ee_pose: Current end-effector pose (`env.robot.end_effector_pose`).
            task_description: Natural-language task instruction (only used in "point_cloud" mode).
            rgb: (H, W, 3) uint8 RGB frame, pixel-aligned with ``depth`` (only "point_cloud" mode).
            depth: (H, W) metric depth frame (only "point_cloud" mode).
            camera_intrinsics: (3, 3) intrinsic matrix K matching ``rgb``/``depth`` (only "point_cloud" mode).
            camera_to_world: (4, 4) camera-to-base-frame transform (only "point_cloud" mode).
        """
        self._finalize_episode_debug()
        self._episode_index += 1
        self._episode_points = None
        self._trajectory_actual = []
        self._trajectory_reference = []
        self._trajectory_predicted = []
        self._trajectory_modified = []
        self._trajectory_comparison = []
        self._trajectory_comparison_collision = []
        self._step_index = -1

        self._active = False
        self.last_info = SafetyFilterStepInfo(active=False)
        if not self.config.enabled:
            return

        self.p1, self.R1 = self._ee_ball_pose(ee_pose)

        if self.config.mode == "static":
            self.p2 = np.array(self.config.obstacle_position, dtype=float)
            self.R2 = np.eye(3)
            self.Q2_diag = np.array(self.config.obstacle_shape_diag, dtype=float)
        else:
            missing = [
                name
                for name, value in (
                    ("task_description", task_description),
                    ("rgb", rgb),
                    ("depth", depth),
                    ("camera_intrinsics", camera_intrinsics),
                    ("camera_to_world", camera_to_world),
                )
                if value is None
            ]
            if missing:
                logger.warning(
                    "SafetyFilter(mode='point_cloud'): missing %s this episode. Safety filter "
                    "is disabled for this episode.",
                    missing,
                )
                return

            try:
                obstacle_points = self._localize_obstacle(
                    task_description, rgb, depth, camera_intrinsics, camera_to_world
                )
            except Exception:
                logger.exception(
                    "SafetyFilter(mode='point_cloud'): obstacle perception failed, disabling "
                    "for this episode."
                )
                return

            if len(obstacle_points) < 4:
                logger.warning(
                    "SafetyFilter(mode='point_cloud'): only %d obstacle points survived the "
                    "xyz_bounds crop (need >= 4 for an ellipsoid fit). Safety filter is "
                    "disabled for this episode.",
                    len(obstacle_points),
                )
                return
            try:
                self.p2, self.R2, self.Q2_diag = fit_ellipse_from_height_profile(
                    obstacle_points,
                    top_fraction=self.config.ellipsoid_top_fraction,
                    center_fraction=self.config.ellipsoid_center_fraction,
                    min_top_points=self.config.ellipsoid_min_top_points,
                    radius_percentile=self.config.ellipsoid_radius_percentile,
                    height_percentile=self.config.ellipsoid_height_percentile,
                    margin_fraction=self.config.ellipsoid_margin_fraction,
                    full_cluster_eps=self.config.ellipsoid_full_cluster_eps,
                    full_cluster_min_points=self.config.ellipsoid_full_cluster_min_points,
                    top_cluster_eps=self.config.ellipsoid_top_cluster_eps,
                    top_cluster_min_points=self.config.ellipsoid_top_cluster_min_points,
                    fixed_radius=self.config.ellipsoid_fixed_radius,
                    center_x_offset=self.config.ellipsoid_center_x_offset,
                )
            except Exception:
                logger.exception(
                    "SafetyFilter(mode='point_cloud'): ellipsoid fit failed, disabling for "
                    "this episode."
                )
                return

        z0 = self.p2 - self.p1
        norm = np.linalg.norm(z0)
        if norm < 1e-9:
            logger.warning("SafetyFilter: end-effector starts on top of the obstacle center; disabling.")
            return
        self.z_fixed = z0 / norm
        self._trajectory_actual = [self.p1.copy()]
        self._trajectory_reference = [self.p1.copy()]
        self._active = True

    def step(
        self,
        action: np.ndarray,
        ee_pose: Pose,
        rot_dim: int,
        dt: float,
        comparison_pose: Pose | None = None,
    ) -> np.ndarray:
        """Filter one `ManipulatorCartesianEnv` action through the CBF-QP.

        Args:
            action: Raw policy action, layout ``[dx, dy, dz, *d_rot, gripper]`` where ``d_rot``
                has ``rot_dim`` entries (world-frame deltas, matching
                `ManipulatorCartesianEnv.step`'s `use_relative_actions=True` convention).
            ee_pose: The pose `h`/the QP correction are computed from -- whatever pose the
                caller passes here *is* what counts as "where the ego ellipsoid is" for this
                step's safety decision, and *is* what the returned, actually-commanded action is
                based on. The caller chooses what that means: pass
                `env.robot.end_effector_pose` for a reactive filter (react only once the real,
                measured position is close to danger); pass `env.robot.target_pose` for a
                proactive one (react to where the arm is being commanded to go, before it gets
                there -- see `vlsa_lerobot_policy.py`'s call site for the current choice and
                its tradeoffs). This function does not know or care which one it's been given.
            rot_dim: Dimensionality of the environment's rotation representation
                (`env.get_rotation_dimension()`).
            dt: Control period in seconds (``1 / env.config.control_frequency``).
            comparison_pose: Optional *second* pose, purely for reporting -- NEVER used to pick
                the returned action, and never fed into the QP. Every step it's provided, its
                own `h` against the same obstacle ellipsoid is computed independently (read-only,
                via `compute_h_ij`) and recorded alongside it, so you can see e.g. whether the
                *real* arm position would actually collide even while a different pose (like
                `target_pose`) is the one driving corrections. Pass whichever pose you did *not*
                pass as `ee_pose`.

        Returns:
            A same-shape action with the translation and rotation components safety-corrected.
            The gripper component is always passed through unchanged. If the filter is
            disabled/inactive, ``action`` is returned unchanged.
        """
        self._step_index += 1

        # ADDED: cap the dt used for physics (velocity clamp, z_fixed integration) -- see
        # `SafetyFilterConfig.max_dt`'s docstring. `dt` itself (the real, uncapped value) is left
        # untouched below for `SafetyFilterStepInfo`/step_log.csv logging.
        dt_physics = dt
        if dt > self.config.max_dt:
            logger.warning(
                "SafetyFilter: real dt=%.3fs exceeds max_dt=%.3fs (likely a stall, e.g. an "
                "action-chunking policy's periodic re-inference) -- capping the dt used for the "
                "velocity clamp/z_fixed integration to max_dt so this can't translate into an "
                "oversized single-step motion. The real dt is still what's logged.",
                dt,
                self.config.max_dt,
            )
            dt_physics = self.config.max_dt

        comparison_p1 = comparison_R1 = None
        if comparison_pose is not None:
            comparison_p1, comparison_R1 = self._ee_ball_pose(comparison_pose)
            self._trajectory_comparison.append(comparison_p1)

        if not self.config.enabled or not self._active:
            self.last_info = SafetyFilterStepInfo(active=False, dt=dt)
            self._log_step(action, action, self.last_info)
            if comparison_pose is not None:
                # No obstacle ellipsoid (self.p2 etc.) exists yet this episode -- can't say.
                self._trajectory_comparison_collision.append(False)
            return action

        if rot_dim != 3:
            raise ValueError(
                "SafetyFilter only supports 3D rotation deltas (Euler or angle-axis "
                f"orientation_representation), got rot_dim={rot_dim}. Configure the environment "
                "with a 3D orientation representation, or disable the safety filter."
            )

        import cvxpy as cp

        p1, R1 = self._ee_ball_pose(ee_pose)
        self.p1, self.R1 = p1, R1
        self._trajectory_actual.append(p1.copy())

        translation_ref = action[:3]
        omega_ref = action[3:6]
        gripper = action[6:]

        self._trajectory_reference.append(self._trajectory_reference[-1] + translation_ref)
        # One-step prediction anchored at the REAL current position (unlike
        # `_trajectory_reference`'s cumulative re-integration from episode start, which drifts
        # away from reality over time) -- "if we hadn't corrected, where would this one raw
        # action have taken us from here." Rendered as per-step arrows in the debug plots.
        self._trajectory_predicted.append(p1 + translation_ref)

        v_ref = R1.T @ translation_ref
        u_v_ref = self.config.ref_gain * v_ref
        u_omega_ref = self.config.ref_gain * omega_ref

        a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(
            p1,
            np.array(self.config.ee_shape_diag),
            R1,
            self.p2,
            self.Q2_diag,
            self.R2,
            self.z_fixed,
        )
        a_u_v = self.config.v_gain * a_v
        a_u_omega = self.config.omega_gain * a_omega

        u_z_nom = self.config.z_gain * mu_row
        u = cp.Variable(9)
        weights = np.array(
            [self.config.qp_v_weight] * 3
            + [self.config.qp_omega_weight] * 3
            + [self.config.qp_uz_weight] * 3
        )
        W = np.diag(weights)
        u_ref_vec = np.hstack([u_v_ref, u_omega_ref, u_z_nom])
        objective = cp.Minimize(cp.quad_form(u - u_ref_vec, W))
        constraints = [
            # ADDED: was `... + self.config.cbf_alpha * h >= 0` -- see `h_margin`'s docstring
            # above (SafetyFilterConfig) for why. Default h_margin=0.0 makes this identical to
            # the original.
            a_u_v @ u[:3] + a_u_omega @ u[3:6] + a_uz @ u[6:]
            + self.config.cbf_alpha * (h - self.config.h_margin) >= self.config.h_constraint
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)

        feasible = u.value is not None
        if feasible:
            u_v, u_omega, u_z = u.value[:3], u.value[3:6], u.value[6:]
        else:
            # This QP is a single linear inequality on an otherwise-unbounded 9-dim variable,
            # which is essentially always solvable in principle -- `feasible=False` here means
            # OSQP itself failed (numerical ill-conditioning), not "no safe action exists".
            # Passing the raw, CBF-uncertified action through would be the riskier choice
            # precisely when the filter couldn't verify anything; freeze instead.
            logger.warning("SafetyFilter: CBF-QP infeasible, freezing translation/rotation this step.")
            u_v = np.zeros(3)
            u_omega = np.zeros(3)
            u_z = np.zeros(3)

        Id = np.eye(3)
        dz = (Id - np.outer(self.z_fixed, self.z_fixed)) @ u_z
        self.z_fixed = self.z_fixed + dz * dt_physics
        self.z_fixed = self.z_fixed / np.linalg.norm(self.z_fixed)

        # aegis's `action_input = np.zeros(7); action_input[:3] = ...; action_input[3:6] = ...;
        # action_input[6] = ...` -- generalized to an arbitrary gripper width via `len(gripper)`
        # instead of the hardcoded 7, since crisp_gym's gripper action isn't always 1D.
        translation_delta, translation_clamped = self._clamp_to_speed(
            self.config.v_gain * (R1 @ u_v), self.config.max_translational_velocity, dt_physics
        )
        rotation_delta, rotation_clamped = self._clamp_to_speed(
            self.config.omega_gain * u_omega, self.config.max_rotational_velocity, dt_physics
        )
        if translation_clamped or rotation_clamped:
            logger.warning(
                "SafetyFilter: corrected action exceeded the velocity limit this step "
                "(translation_clamped=%s, rotation_clamped=%s) -- capped to avoid a Franka "
                "reflex/safety stop.",
                translation_clamped,
                rotation_clamped,
            )

        action_input = np.zeros(6 + len(gripper), dtype=action.dtype)
        action_input[:3] = translation_delta
        action_input[3:6] = rotation_delta
        action_input[6:] = gripper

        # Same anchor point (real current position p1) as `_trajectory_predicted`, so the two
        # are directly comparable arrows: "what the raw policy wanted" vs. "what the CBF
        # actually commanded" from the exact same spot.
        self._trajectory_modified.append(p1 + translation_delta)

        self.last_info = SafetyFilterStepInfo(
            active=True,
            h=float(h),
            feasible=feasible,
            correction_norm=float(np.linalg.norm(action_input[:3] - translation_ref)),
            dt=dt,
        )
        if comparison_pose is not None:
            # Independent, read-only check: does comparison_pose's OWN position collide with
            # the obstacle, regardless of what h/correction (computed from ee_pose above) says.
            # Deliberately not reusing `h` -- that's ee_pose's violation status, not this pose's.
            comparison_h = compute_h_ij(
                comparison_p1,
                np.array(self.config.ee_shape_diag),
                comparison_R1,
                self.p2,
                self.Q2_diag,
                self.R2,
                self.z_fixed,
            )
            self._trajectory_comparison_collision.append(comparison_h < 0)

        self._log_step(action, action_input, self.last_info)
        return action_input
