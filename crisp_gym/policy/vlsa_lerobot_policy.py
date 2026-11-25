"""Interface for VLA policy inference in CRISP, with an aegis CBF safety layer.

This is `crisp_gym.policy.vla_lerobot_policy.VlaLerobotPolicy` plus the per-step CBF-QP
safety filter from aegis's ``main_aegis.py``. The VLA inference (child process, Pipe
communication) is unchanged from `VlaLerobotPolicy`; the safety filter runs in the *parent*
process instead, because it needs `self.env` (robot pose, cameras) directly and, unlike VLA
inference, is cheap enough to run every control tick without its own process -- exactly how
aegis ran it inline in its own control loop rather than behind the policy-server RPC.

See `crisp_gym.safety.safety_filter` for what had to change to make aegis's action/pose
assumptions compatible with crisp_gym. In `mode="point_cloud"`, this class also owns a
`crisp_gym.safety.depth_camera.SafetyRGBDCamera` -- crisp_py's own `Camera` (what the VLA
reads) is RGB-only, so obstacle perception subscribes to your camera's aligned-depth +
camera-info topics independently, and hands the frame to `SafetyFilter.reset()` once per
episode to run the full VLM + GroundingDINO + point-cloud pipeline.
"""

import json
import logging
import multiprocessing as mp
import os
import time
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
import torch
import torch._inductor.config as inductor_config
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.factory import (
    get_policy_class,
    make_pre_post_processors,
)
from typing_extensions import override

from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv
from crisp_gym.policy.policy import Action, Observation, Policy, register_policy
from crisp_gym.safety.depth_camera import SafetyRGBDCamera, SafetyRGBDCameraConfig
from crisp_gym.safety.safety_filter import SafetyFilter, SafetyFilterConfig
from crisp_gym.util.lerobot_features import concatenate_state_features, numpy_obs_to_torch
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)
inductor_config.triton.cudagraphs = False

DEBUG_LOG_DIR = "outputs/logs"
"""Relative to wherever you launch `crisp-deploy-vla` from, matching the
`SafetyFilterConfig.debug_dir` convention (default `outputs/safety`) instead of a fixed
home-directory path -- so all of a run's debug output lands under the same `outputs/` tree."""


@register_policy("vlsa_lerobot_policy")
class VlsaLerobotPolicy(Policy):
    """VLA policy inference wrapped with an aegis-style CBF safety filter.

    Runs LeRobot policy inference in a separate process (same as `VlaLerobotPolicy`) and, in
    the parent process, passes every predicted action through a `SafetyFilter` before stepping
    the environment -- so the environment always sees a collision-avoidance-corrected action,
    never the raw VLA output.
    """

    def __init__(
        self,
        pretrained_path: str,
        env: ManipulatorBaseEnv,
        overrides: dict | None = None,
        safety: dict | None = None,
        safety_camera: dict | None = None,
    ):
        """Initialize the Vlsa policy.

        Args:
            pretrained_path (str): Path to the pretrained policy model.
            env (ManipulatorBaseEnv): The environment in which the policy will be applied.
            overrides (dict | None): Optional overrides for the policy configuration.
            safety (dict | None): Optional `SafetyFilterConfig` kwargs. If omitted, the safety
                filter is disabled and the raw VLA action is used unfiltered.
            safety_camera (dict | None): `SafetyRGBDCameraConfig` kwargs (color/depth/info
                topics + camera_frame). Required when ``safety["mode"] == "point_cloud"``;
                unused otherwise.
        """
        self.parent_conn, self.child_conn = Pipe()
        self.env = env
        self.overrides = overrides if overrides is not None else {}

        self.safety_filter = SafetyFilter(SafetyFilterConfig(**(safety or {})))

        self.depth_camera: SafetyRGBDCamera | None = None
        if self.safety_filter.config.enabled and self.safety_filter.config.mode == "point_cloud":
            if not safety_camera:
                raise ValueError(
                    "safety.mode='point_cloud' requires a safety_camera config with "
                    "color_image_topic, depth_image_topic, depth_info_topic and camera_frame."
                )
            self.depth_camera = SafetyRGBDCamera(
                SafetyRGBDCameraConfig(**safety_camera),
                namespace=env.robot.node.get_namespace(),
            )
            logger.info("Waiting for the safety RGB-D camera to become ready...")
            self.depth_camera.wait_until_ready(timeout=10.0)

        # Extract serializable env data for spawn multiprocessing
        env_metadata = env.get_metadata() if hasattr(env, 'get_metadata') else {}

        context = mp.get_context('spawn')
        self.inf_proc = context.Process(
            target=inference_worker,
            kwargs={
                "conn": self.child_conn,
                "pretrained_path": pretrained_path,
                "env_metadata": env_metadata,
                "observation_space": env.observation_space,
                "overrides": self.overrides,
            },
            daemon=True,
        )
        self.inf_proc.start()

    @override
    def make_data_fn(self, task_description: str) -> Callable[[], Tuple[Observation, Action]]:  # noqa: ANN002, ANN003
        """Generate observation and action by communicating with the inference worker."""
        logger.info(f"Syncing task to worker: {task_description}")
        self.parent_conn.send(task_description)

        rot_dim = self.env.get_rotation_dimension()
        if self.safety_filter.config.enabled and rot_dim != 3:
            raise ValueError(
                "VlsaLerobotPolicy's safety filter requires a 3D orientation representation "
                "(euler or angle_axis) on the environment, "
                f"got rot_dim={rot_dim}. Use a 3D representation or set safety.enabled=False."
            )

        rgb = depth = camera_intrinsics = camera_to_world = None
        if self.depth_camera is not None:
            rgb = self.depth_camera.current_color
            depth = self.depth_camera.current_depth
            camera_intrinsics = self.depth_camera.intrinsics
            base_frame = self.depth_camera.config.base_frame_override or self.env.robot.config.base_frame
            camera_to_world = self.depth_camera.get_camera_to_world(base_frame)

        self.safety_filter.reset(
            self.env.robot.end_effector_pose,
            task_description=task_description,
            rgb=rgb,
            depth=depth,
            camera_intrinsics=camera_intrinsics,
            camera_to_world=camera_to_world,
        )
        logger.info(
            f"Safety filter active={self.safety_filter.active}, "
            f"obstacle={self.safety_filter.obstacle_name}"
        )

        # Real elapsed wall-clock time between control ticks, not the nominal
        # 1/control_frequency -- the loop below blocks on inference (a separate process, over
        # a Pipe) and steps the env non-blocking (`block=False`; `block=True` is documented as
        # broken in ManipulatorCartesianEnv.step, see its FIXME), so nothing enforces the
        # nominal rate. Feeding the CBF's velocity clamp and z_fixed integration a `dt` that's
        # smaller than what actually elapsed understates how far the arm could really have
        # moved. None on the first call of a fresh episode (nothing to measure yet); falls back
        # to the nominal rate for that one tick only.
        last_step_time: float | None = None

        def _fn() -> tuple:
            """Function to apply the policy in the environment.

            This function observes the current state of the environment, sends the observation
            to the inference worker, receives the raw action, safety-filters it through the CBF
            QP, and steps the environment with the corrected action.

            Returns:
                tuple: A tuple containing the observation from the environment and the
                    safety-filtered action actually applied.
            """
            # ADDED: per-phase timing, purely diagnostic -- we don't yet know which part of this
            # function is responsible for the ~150-250ms/step slowdown seen in practice (VLA
            # chunk-caching itself was checked and ruled out: predict_action_chunk only runs once
            # every n_action_steps calls, same as intended). Logs a breakdown instead of guessing,
            # whenever a step's total exceeds `slow_step_threshold_s`.
            t_start = time.time()

            logger.debug("Requesting action from Vlsa policy...")
            obs_raw: Observation = self.env.get_obs()
            t_get_obs = time.time()

            obs_raw["observation.state"] = concatenate_state_features(obs_raw)

            os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
            with open(os.path.join(DEBUG_LOG_DIR, "obs_log.txt"), "a") as f:
                obs_str = ",".join(map(str, obs_raw["observation.state"].flatten()))
                f.write(f"{obs_str}\n")
                f.flush()
            t_debug_log = time.time()

            self.parent_conn.send(obs_raw)
            action: Action = self.parent_conn.recv().squeeze(0).to("cpu").numpy()
            t_ipc = time.time()

            nonlocal last_step_time
            now = time.time()
            dt = (now - last_step_time) if last_step_time is not None else (
                1.0 / self.env.config.control_frequency
            )
            last_step_time = now

            # original action, modified action, h, feasible, correction, dt: logged per-step to
            # this episode's outputs/safety/episode_NNNN/step_log.csv by SafetyFilter.step()
            # itself (it already owns the episode index and debug_dir).
            #
            # DELIBERATE CHOICE: target_pose (not end_effector_pose) drives h/correction here --
            # proactive avoidance, reacting to where the arm is being commanded to go next
            # rather than only where it currently, physically is. target_pose leads the real
            # position (see manipulator_env.py:662-663 -- each new target is built by adding the
            # action onto the *previous target*, not onto the real measured position), so this
            # gives the filter a head start over a purely reactive check.
            # Known tradeoff, accepted: end_effector_pose never drives the returned action or
            # h/correction -- it's passed as `comparison_pose=` purely for reporting: the debug
            # plot's second/comparison line (magenta) shows the real path alongside the
            # commanded one, and its own, independently-computed h flags (star markers) whether
            # the *real* arm position would actually collide, even while target_pose is what's
            # driving corrections. See SafetyFilter.step's docstring for the generic
            # (caller-agnostic) version of this same explanation.
            safe_action = self.safety_filter.step(
                action,
                self.env.robot.target_pose,
                rot_dim=rot_dim,
                dt=dt,
                comparison_pose=self.env.robot.end_effector_pose,
                joint_values=self.env.robot.joint_values,
            )
            t_safety = time.time()

            try:
                self.env.step(safe_action, block=False)
            except Exception as e:
                logger.exception(f"Error during environment step: {e}")
            t_env_step = time.time()

            # ADDED: see the timing comment above -- only logs when this step was actually slow,
            # so normal fast steps don't get spammed.
            total = t_env_step - t_start
            slow_step_threshold_s = 0.1
            if total > slow_step_threshold_s:
                logger.warning(
                    "Slow step breakdown: total=%.3fs  get_obs=%.3fs  debug_log_write=%.3fs  "
                    "ipc_send_recv(inference worker)=%.3fs  safety_filter.step=%.3fs  env.step=%.3fs",
                    total,
                    t_get_obs - t_start,
                    t_debug_log - t_get_obs,
                    t_ipc - t_debug_log,
                    t_safety - t_ipc,
                    t_env_step - t_safety,
                )

            return obs_raw, safe_action

        return _fn

    @override
    def reset(self):
        """Reset the policy state."""
        self.parent_conn.send("reset")

    @override
    def shutdown(self):
        """Shutdown the policy and release resources."""
        self.safety_filter.finalize()
        self.parent_conn.send(None)
        self.inf_proc.join()


def inference_worker(
    conn: Connection,
    pretrained_path: str,
    env_metadata: dict,
    observation_space: Any,
    overrides: dict | None = None,
):  # noqa: ANN001
    """Policy inference process: loads policy, receives observations via conn, returns actions, and exits on None.

    Args:
        conn (Connection): The connection to the parent process for sending and receiving data.
        pretrained_path (str): Path to the pretrained policy model.
        env_metadata (dict): Environment metadata for validation.
        observation_space (Any): Observation space from the environment.
        overrides (dict | None): Optional overrides for the policy configuration.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
    except ImportError:
        logger.warning(
            "[Vlsa Inference] Could not import third-party plugins for LeRobot. Continuing without them."
        )
    logger.info("[Vlsa Inference] Starting Vlsa inference worker...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Vlsa Inference] Using device: {device}")

        logger.info(f"[Vlsa Inference] Loading training config from {pretrained_path}...")

        train_config = TrainPipelineConfig.from_pretrained(pretrained_path)
        _check_dataset_metadata(train_config, env_metadata, logger)
        logger.info("[Vlsa Inference] Loaded training config.")
        logger.debug(f"[Vlsa Inference] Train config: {train_config}")
        if train_config.policy is None:
            raise ValueError(
                f"Policy configuration is missing in the pretrained path: {pretrained_path}. "
                "Please ensure the policy is correctly configured."
            )

        logger.info("[Vlsa Inference] Loading policy...")
        policy_cls = get_policy_class(train_config.policy.type)
        logger.info("Step 1: Loading Base Model to RAM...")

        policy = policy_cls.from_pretrained(
            pretrained_path,  # We load the base model from the local path where we downloaded it, not from the pretrained_path which is where the adapter is
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

        if train_config.policy.type == "groot":
            policy._groot_model.action_head.num_inference_timesteps = 30
            logger.info("[Vlsa Inference] Overriding GROOT num_inference_timesteps to 30.")

        for override_key, override_value in (overrides or {}).items():
            logger.warning(
                f"[Vlsa Inference] Overriding policy config: {override_key} = {getattr(policy.config, override_key)} -> {override_value}"
            )
            setattr(policy.config, override_key, override_value)

        policy.reset()
        policy.to(device).eval()

        # Build processors so obs/action scaling follows policy.config.normalization_mapping
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=pretrained_path,
            preprocessor_overrides={"device_processor": {"device": device.type}},
        )

        warmup_obs_raw = observation_space.sample()
        warmup_obs_raw["observation.state"] = concatenate_state_features(warmup_obs_raw)
        default_text = "complete the task"
        warmup_obs = numpy_obs_to_torch(warmup_obs_raw)
        warmup_obs["task"] = default_text
        warmup_obs = preprocessor(warmup_obs)

        logger.info("[Vlsa Inference] Warming up policy...")
        elapsed_list = []
        with torch.inference_mode():
            import time

            for _ in range(100):
                start = time.time()
                _ = policy.select_action(warmup_obs)
                end = time.time()
                elapsed_list.append(end - start)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        avg_elapsed = sum(elapsed_list) / len(elapsed_list)
        std_elapsed = np.std(elapsed_list)
        max_elapsed = max(elapsed_list)
        min_elapsed = min(elapsed_list)
        logger.info(
            f"[Vlsa Inference] Warm-up timing over 100 runs: "
            f"avg={avg_elapsed * 1000:.2f}ms, std={std_elapsed * 1000:.2f}ms, max={max_elapsed * 1000:.2f}ms, min={min_elapsed * 1000:.2f}ms"
        )

        logger.info("[Vlsa Inference] Warm-up complete")
        current_task = default_text
        while True:
            # ADDED: per-phase timing inside the worker itself, purely diagnostic -- the parent
            # process's own breakdown (vlsa_lerobot_policy.py's `_fn`) showed the whole
            # send+recv round trip costing ~0.3s on nearly every step, too consistent to be
            # `n_action_steps=25` chunk-caching's occasional expensive `predict_action_chunk`
            # call (that should look bursty: mostly near-zero, occasional real spikes). This
            # narrows down whether that ~0.3s is genuine model compute (`select_action`), pipe
            # transport (`recv`/`send`), or something else -- and `queue_len` directly shows
            # cache hit vs miss: `n_action_steps - 1` (24) right after a fresh
            # predict_action_chunk call, decreasing by 1 each subsequent cache-hit call until it
            # empties and regenerates.
            msg = conn.recv()
            if msg is None:
                break
            if msg == "reset":
                logger.info("[Vlsa Inference] Resetting policy")
                policy.reset()
                continue

            if isinstance(msg, str):
                current_task = msg
                logger.info(f"[Vlsa Inference] Task updated to: '{current_task}'")
                # We don't run inference yet, we just wait for the first observation
                continue

            with torch.inference_mode():
                obs_with_task = numpy_obs_to_torch(msg)
                obs_with_task["task"] = current_task

                obs = preprocessor(obs_with_task)
                action = policy.select_action(obs)
                if torch.cuda.is_available():
                    # select_action's own CUDA ops are asynchronous by default -- without this,
                    # wall-clock time here would just measure kernel-launch/dispatch, not actual
                    # GPU compute completion, understating the real cost.
                    torch.cuda.synchronize()
                action = postprocessor(action)
            conn.send(action)
    except Exception as e:
        logger.exception(f"[Vlsa Inference] Exception in inference worker: {e}")

    conn.close()
    logger.info("[Vlsa Inference] Worker shutting down")


def _check_dataset_metadata(
    train_config: TrainPipelineConfig,
    env_metadata: dict,
    logger: logging.Logger,
    keys_to_skip: list[str] | None = None,
):
    """Check if the dataset metadata matches the environment configuration.

    Args:
        train_config (TrainPipelineConfig): The training pipeline configuration.
        env_metadata (dict): The environment metadata dictionary to compare against.
        logger (logging.Logger): Logger for logging information.
        keys_to_skip (list[str] | None): List of metadata keys to skip during comparison.
    """
    if keys_to_skip is None:
        keys_to_skip = []

    def _warn_if_not_equal(key: str, env_val: Any, policy_val: Any):
        if env_val != policy_val:
            logger.warning(
                f"[Vlsa Inference] Mismatch in metadata for key '{key}': "
                f"env has '{env_val}', policy has '{policy_val}'."
            )

    def _warn_if_missing(key: str):
        logger.warning(f"[Vlsa Inference] Key '{key}' not found in environment metadata.")

    try:
        metadata = LeRobotDatasetMetadata(repo_id=train_config.dataset.repo_id)
        logger.debug(f"[Vlsa Inference] Loaded dataset metadata: {metadata}")

        path_to_metadata = Path(metadata.root / "meta" / "crisp_meta.json")
        if path_to_metadata.exists():
            logger.info(
                "[Vlsa Inference] Found crisp_meta.json in dataset, comparing environment and policy configs..."
            )
            with open(path_to_metadata, "r") as f:
                dataset_metadata = json.load(f)
            for key, value in dataset_metadata.items():
                if key in keys_to_skip:
                    continue
                if isinstance(value, dict):
                    if key not in env_metadata:
                        _warn_if_missing(key)
                        continue
                    for subkey, subvalue in value.items():
                        if subkey not in env_metadata[key]:
                            _warn_if_missing(f"{key}.{subkey}")
                            continue
                        _warn_if_not_equal(
                            f"{key}.{subkey}",
                            env_metadata[key].get(subkey),
                            subvalue,
                        )
                else:
                    if key not in env_metadata:
                        _warn_if_missing(key)
                    _warn_if_not_equal(key, env_metadata.get(key), value)

    except Exception as e:
        logger.warning(f"[Vlsa Inference] Could not load dataset metadata: {e}")
        logger.info("[Vlsa Inference] Skipping metadata comparison.")
