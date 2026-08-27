"""Obstacle perception utilities, ported from aegis's ``utils.py``/``main_aegis.py``.

aegis originally read images from a locally-attached robosuite/MuJoCo simulated camera and
used the simulator's *ground-truth* depth map + camera intrinsics/extrinsics
(`robosuite.utils.camera_utils`) to unproject the detected obstacle into a 3D point cloud.

In crisp_gym, RGB images already arrive the same way regardless of where the camera
physically lives: `crisp_py.camera.Camera` subscribes to a ROS2 topic and exposes a plain
``(H, W, 3)`` uint8 numpy array via ``camera.current_image`` -- it makes no difference to this
module whether that topic is published by a camera on this machine or streamed in from another
PC, as is the case in this deployment. So ``get_point_cloud`` below takes a plain ``rgb``
array exactly like it would from ``env.get_obs()["observation.images.<camera_name>"]``.

What *cannot* be carried over as-is is the depth + calibration: `crisp_py.camera.Camera` has
no depth subscription and does not store the camera intrinsic matrix K or the
camera-to-base extrinsic transform anywhere (see `crisp_py/camera/camera.py`), whereas aegis
relied on MuJoCo handing both to it for free. Real 3D obstacle localization therefore needs
those to be supplied by the caller (e.g. from a depth camera driver + a TF/static calibration
lookup) via the ``depth``/``camera_intrinsics``/``camera_to_world`` arguments -- this module
does not fabricate them.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BOX_THRESHOLD = 0.35
DEFAULT_TEXT_THRESHOLD = 0.25


def get_point_cloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    text_prompt: str,
    model: Any,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    device: str = "cuda",
    debug_dir: Path | None = None,
    return_colors: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Detect ``text_prompt`` in ``rgb`` with GroundingDINO and unproject its bounding box to 3D.

    Args:
        rgb: (H, W, 3) uint8 RGB image, right-side-up (e.g. straight from
            `crisp_py.camera.Camera.current_image` / `env.get_obs()`).
        depth: (H, W) metric depth map registered to ``rgb`` (meters).
        camera_intrinsics: (3, 3) camera intrinsic matrix K.
        camera_to_world: (4, 4) homogeneous transform from camera frame to the same world/base
            frame as `crisp_py.robot.Robot.end_effector_pose`.
        text_prompt: Open-vocabulary text prompt describing the obstacle to localize.
        model: A loaded GroundingDINO model (see `groundingdino.util.inference.load_model`).
        box_threshold: GroundingDINO box confidence threshold.
        text_threshold: GroundingDINO text confidence threshold.
        device: Torch device for GroundingDINO inference.
        debug_dir: If given, save the annotated detection crop here for debugging.
        return_colors: If True, also return each point's (uint8) RGB color, sampled from
            ``rgb`` at its source pixel -- for visualization and `filter_points_by_color`
            (`crisp_gym.safety.math_utils`); the safety filter itself never needs color, so
            existing callers are unaffected by this being off by default.

    Returns:
        (N, 3) world-frame points inside the detected bounding box, empty if nothing detected.
        If ``return_colors``, a ``(points, colors)`` tuple instead, ``colors`` being (N, 3) uint8
        and empty (shape (0, 3)) alongside an empty ``points`` on every early-exit path.
    """
    import cv2
    import torch
    from groundingdino.util.box_ops import box_cxcywh_to_xyxy
    from groundingdino.util.inference import annotate, load_image, predict

    def _empty():
        return (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)) if return_colors else np.zeros((0, 3))

    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = os.path.join(tmp_dir, "obstacle_detection_input.png")
        cv2.imwrite(image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        image_source, image_cv = load_image(image_path)

        boxes, logits, phrases = predict(
            model=model,
            image=image_cv,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )

        if debug_dir is not None:
            debug_dir = Path(debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            annotated = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            cv2.imwrite(str(debug_dir / "annotated_detection.jpg"), annotated)

    if boxes.shape[0] == 0:
        logger.warning("GroundingDINO detected no object matching '%s'.", text_prompt)
        return _empty()

    # GroundingDINO's `predict()` returns boxes in whatever raw order its internal queries
    # happen to produce -- NOT sorted by confidence -- so when a prompt matches multiple
    # candidates (e.g. several visually-similar cans), picking a fixed index picks an
    # essentially arbitrary one of them, regardless of which actually scored highest. Explicitly
    # take the highest-logit box instead.
    best_idx = int(logits.argmax())
    if boxes.shape[0] > 1:
        logger.info(
            "GroundingDINO matched %d candidates for '%s'; picked '%s' (logit=%.3f) over %s.",
            boxes.shape[0], text_prompt, phrases[best_idx], float(logits[best_idx]),
            [f"{p} ({float(l):.3f})" for i, (p, l) in enumerate(zip(phrases, logits)) if i != best_idx],
        )

    h, w, _ = rgb.shape
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    size_tensor = torch.tensor([w, h, w, h], device=boxes.device)
    pixel_boxes = boxes_xyxy * size_tensor
    xmin, ymin, xmax, ymax = pixel_boxes[best_idx].cpu().numpy().astype(int)

    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(w, xmax), min(h, ymax)

    cropped_rgb = rgb[ymin:ymax, xmin:xmax]
    cropped_depth = depth[ymin:ymax, xmin:xmax]
    if cropped_rgb.size == 0:
        logger.warning("Invalid bounding box for '%s': [%d, %d, %d, %d]", text_prompt, xmin, ymin, xmax, ymax)
        return _empty()

    v_full, u_full = np.indices((h, w))
    cropped_u = u_full[ymin:ymax, xmin:xmax].flatten()
    cropped_v = v_full[ymin:ymax, xmin:xmax].flatten()
    depth_flat = cropped_depth.flatten()
    # Row-major flatten of the same (ymax-ymin, xmax-xmin) crop as cropped_u/cropped_v above,
    # so index i here is the same source pixel as cropped_u[i]/cropped_v[i].
    colors_flat = cropped_rgb.reshape(-1, 3)

    # A zero/NaN depth reading (no valid return -- common on reflective/transparent surfaces
    # like glass) would otherwise unproject to exactly the camera's own optical center
    # (`points_cam = K_inv @ pixel * 0 = [0,0,0]`), a fake point that has nothing to do with
    # the detected object and can distort the downstream MVEE ellipsoid fit.
    valid = np.isfinite(depth_flat) & (depth_flat > 0.05)
    if not np.any(valid):
        logger.warning("No valid depth returns for '%s' inside its detection box.", text_prompt)
        return _empty()
    cropped_u = cropped_u[valid]
    cropped_v = cropped_v[valid]
    depth_flat = depth_flat[valid]
    colors_flat = colors_flat[valid]

    K_inv = np.linalg.inv(camera_intrinsics)
    pixels_homo = np.stack([cropped_u, cropped_v, np.ones_like(cropped_u)], axis=0)
    points_cam = K_inv @ pixels_homo * depth_flat

    points_cam_homo = np.vstack([points_cam, np.ones_like(depth_flat)])
    points_world_homo = camera_to_world @ points_cam_homo
    points = points_world_homo[:3, :].T
    return (points, colors_flat) if return_colors else points


def obstacle_detection(
    image: np.ndarray,
    instruction: str,
    preference_list: list[str] | None = None,
    api_key: str | None = None,
    model: str = "glm-4.5v",
) -> str:
    """Ask a VLM which single object is most likely to obstruct ``instruction``'s execution.

    Args:
        image: (H, W, 3) uint8 RGB image of the scene.
        instruction: Natural-language task instruction the robot is about to execute.
        preference_list: Optional list of known object names to bias the VLM's answer towards.
        api_key: ZhipuAI API key. Defaults to the ``ZHIPUAI_API_KEY`` environment variable.
        model: ZhipuAI model name.

    Returns:
        A short object name (color + type), suitable as a GroundingDINO text prompt.
    """
    from zai import ZhipuAiClient

    api_key = api_key or os.environ.get("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError(
            "obstacle_detection requires a ZhipuAI API key. Pass api_key= or set ZHIPUAI_API_KEY."
        )
    client = ZhipuAiClient(api_key=api_key)

    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = os.path.join(tmp_dir, "obstacle_detection.png")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.imsave(image_path, image)
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    prompt_text = (
        f"The robot must follow this instruction: {instruction}. Based on both the "
        "instruction and the image, identify exactly one non-robot object that is most "
        "likely to obstruct the robot's motion during task execution. You must output a "
        "uniquely identifiable obstacle name including both color and object type"
    )
    if preference_list:
        prompt_text += f", preferably from this list when applicable: {preference_list}"
    prompt_text += ". Output only the object name, with no additional words."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        temperature=0.1,
        top_p=0.1,
        thinking={"type": "enabled"},
    )
    return (
        response.choices[0]
        .message.content.replace("<|begin_of_box|>", "")
        .replace("<|end_of_box|>", "")
        .strip()
    )
