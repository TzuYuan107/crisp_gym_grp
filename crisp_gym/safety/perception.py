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


def _detect_boxes(
    rgb: np.ndarray,
    text_prompt: str,
    model: Any,
    box_threshold: float,
    text_threshold: float,
    device: str,
    debug_dir: Path | None,
) -> tuple[Any, Any, list[str]]:
    """Run GroundingDINO on ``rgb`` for ``text_prompt``.

    Save the annotated detection to ``debug_dir`` if given. Return the raw
    ``(boxes, logits, phrases)`` from
    ``groundingdino.util.inference.predict`` -- ``boxes`` may be empty (shape (0, 4)).
    """
    import cv2
    from groundingdino.util.inference import annotate, load_image, predict

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

    return boxes, logits, phrases


def _to_pixel_boxes(boxes: Any, h: int, w: int) -> np.ndarray:
    """GroundingDINO's normalized cxcywh boxes -> pixel-space xyxy int array, shape (N, 4)."""
    import torch
    from groundingdino.util.box_ops import box_cxcywh_to_xyxy

    boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    size_tensor = torch.tensor([w, h, w, h], device=boxes.device)
    return (boxes_xyxy * size_tensor).cpu().numpy().astype(int)


def _unproject_pixel_box(
    rgb: np.ndarray,
    depth: np.ndarray,
    camera_to_world: np.ndarray,
    K_inv: np.ndarray,
    u_full: np.ndarray,
    v_full: np.ndarray,
    pixel_box: np.ndarray,
    text_prompt: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Unproject one pixel-space xyxy box's valid-depth pixels into world-frame (points, colors).

    Returns None if the box is degenerate (empty crop) or has no valid depth returns.
    """
    h, w, _ = rgb.shape
    xmin, ymin, xmax, ymax = pixel_box
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(w, xmax), min(h, ymax)

    cropped_rgb = rgb[ymin:ymax, xmin:xmax]
    cropped_depth = depth[ymin:ymax, xmin:xmax]
    if cropped_rgb.size == 0:
        logger.warning("Invalid bounding box for '%s': [%d, %d, %d, %d]", text_prompt, xmin, ymin, xmax, ymax)
        return None

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
        return None
    cropped_u = cropped_u[valid]
    cropped_v = cropped_v[valid]
    depth_flat = depth_flat[valid]
    colors_flat = colors_flat[valid]

    pixels_homo = np.stack([cropped_u, cropped_v, np.ones_like(cropped_u)], axis=0)
    points_cam = K_inv @ pixels_homo * depth_flat
    points_cam_homo = np.vstack([points_cam, np.ones_like(depth_flat)])
    points_world_homo = camera_to_world @ points_cam_homo
    return points_world_homo[:3, :].T, colors_flat


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
    merge_boxes: bool = False,
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
        merge_boxes: If True, union the unprojected points from *every* box GroundingDINO
            matched to ``text_prompt`` (e.g. two same-looking cans both matching "yellow can")
            into one combined point cloud, instead of only the single highest-logit box.
            Off by default so existing single-obstacle callers are unaffected. See
            `get_point_clouds_per_box` instead if you want a separate point cloud (and
            eventually a separate ellipsoid) per matched box rather than one merged cloud.

    Returns:
        (N, 3) world-frame points inside the detected bounding box(es), empty if nothing
        detected. If ``return_colors``, a ``(points, colors)`` tuple instead, ``colors`` being
        (N, 3) uint8 and empty (shape (0, 3)) alongside an empty ``points`` on every early-exit
        path.
    """

    def _empty() -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        return (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)) if return_colors else np.zeros((0, 3))

    boxes, logits, phrases = _detect_boxes(rgb, text_prompt, model, box_threshold, text_threshold, device, debug_dir)
    if boxes.shape[0] == 0:
        logger.warning("GroundingDINO detected no object matching '%s'.", text_prompt)
        return _empty()

    # GroundingDINO's `predict()` returns boxes in whatever raw order its internal queries
    # happen to produce -- NOT sorted by confidence -- so when a prompt matches multiple
    # candidates (e.g. several visually-similar cans), picking a fixed index picks an
    # essentially arbitrary one of them, regardless of which actually scored highest. Explicitly
    # take the highest-logit box instead, unless merge_boxes is set, in which case every matched
    # box contributes to one combined point cloud.
    best_idx = int(logits.argmax())
    if boxes.shape[0] > 1:
        if merge_boxes:
            logger.info(
                "GroundingDINO matched %d candidates for '%s'; merging all of them: %s.",
                boxes.shape[0], text_prompt,
                [f"{p} ({float(logit):.3f})" for p, logit in zip(phrases, logits)],
            )
        else:
            logger.info(
                "GroundingDINO matched %d candidates for '%s'; picked '%s' (logit=%.3f) over %s.",
                boxes.shape[0], text_prompt, phrases[best_idx], float(logits[best_idx]),
                [f"{p} ({float(logit):.3f})" for i, (p, logit) in enumerate(zip(phrases, logits)) if i != best_idx],
            )
    selected_indices = range(boxes.shape[0]) if merge_boxes else [best_idx]

    h, w, _ = rgb.shape
    pixel_boxes = _to_pixel_boxes(boxes, h, w)
    K_inv = np.linalg.inv(camera_intrinsics)
    v_full, u_full = np.indices((h, w))

    all_points = []
    all_colors = []
    for idx in selected_indices:
        result = _unproject_pixel_box(rgb, depth, camera_to_world, K_inv, u_full, v_full, pixel_boxes[idx], text_prompt)
        if result is None:
            continue
        pts, cols = result
        all_points.append(pts)
        all_colors.append(cols)

    if not all_points:
        return _empty()

    points = np.concatenate(all_points, axis=0)
    colors_flat = np.concatenate(all_colors, axis=0)
    return (points, colors_flat) if return_colors else points


def _mask_out_boxes(rgb: np.ndarray, pixel_boxes: np.ndarray) -> None:
    """Gray-fill every pixel-space xyxy box's region in `rgb`, in place.

    Used between passes of `get_point_clouds_per_box`'s masked-reattempt loop so a re-run of
    GroundingDINO on the same prompt can no longer re-detect what a previous pass already found.
    """
    h, w, _ = rgb.shape
    for xmin, ymin, xmax, ymax in pixel_boxes:
        xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
        xmax, ymax = min(w, int(xmax)), min(h, int(ymax))
        rgb[ymin:ymax, xmin:xmax] = 128


def get_point_clouds_per_box(
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
    max_passes: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return separate point clouds for each matched GroundingDINO box.

    Like `get_point_cloud`, but keeps every matched GroundingDINO box separate instead of
    merging or picking one -- one (points, colors) pair per detected instance (e.g. two
    same-looking cans both matching "yellow can"), so each can get its own fitted ellipsoid.

    GroundingDINO's `predict()` applies no top-K/NMS cap of its own -- every decoder query above
    `box_threshold` is kept independently -- but the underlying model's queries still compete
    with each other during inference, and it's trained mostly on one-salient-target-per-phrase
    data. In practice this means a strong, close match can push a weaker, farther match's own
    confidence below `box_threshold` even though nothing in this codebase capped it. To work
    around that, this runs detection in up to `max_passes` rounds: after each round, every box
    just found gets gray-filled out of a *working copy* of `rgb` (`_mask_out_boxes` -- never the
    original, which is still what the actual point-cloud unprojection below reads from) before
    the next round runs the same prompt again, so an already-found object can no longer dominate
    the query competition and suppress a previously-missed one. Stops as soon as a round finds
    nothing new.

    Args, other than the ones below, are the same as `get_point_cloud`. There is no
    `merge_boxes`/`return_colors` choice here -- colors are always included, one array per box.

    Args:
        rgb: (H, W, 3) uint8 RGB source image.
        depth: (H, W) registered depth in meters.
        camera_intrinsics: (3, 3) intrinsic matrix.
        camera_to_world: (4, 4) camera-to-base transform.
        text_prompt: Text identifying the obstacle to detect.
        model: Loaded GroundingDINO model.
        box_threshold: Confidence threshold for box detection.
        text_threshold: Confidence threshold for text matching.
        device: Torch device for inference.
        debug_dir: Optional directory for the annotated detection image.
        max_passes: Maximum masked-reattempt rounds (>= 1). Higher catches more distinct
            instances at the cost of one extra GroundingDINO forward pass each round that still
            finds something new.

    Returns:
        A list of (points, colors) pairs, one per box GroundingDINO matched to ``text_prompt``
        across every pass, that had at least one valid depth return (boxes without one are
        skipped). Empty list if nothing matched.
    """
    h, w, _ = rgb.shape
    working_rgb = rgb.copy()
    all_boxes, all_logits, all_phrases = [], [], []

    for pass_idx in range(max_passes):
        boxes, logits, phrases = _detect_boxes(
            working_rgb, text_prompt, model, box_threshold, text_threshold, device, debug_dir=None
        )
        if boxes.shape[0] == 0:
            break
        logger.info(
            "get_point_clouds_per_box: pass %d matched %d new candidate(s) for '%s': %s.",
            pass_idx, boxes.shape[0], text_prompt,
            [f"{p} ({float(logit):.3f})" for p, logit in zip(phrases, logits)],
        )
        all_boxes.append(boxes)
        all_logits.append(logits)
        all_phrases.extend(phrases)
        if pass_idx == max_passes - 1:
            break
        _mask_out_boxes(working_rgb, _to_pixel_boxes(boxes, h, w))

    if not all_boxes:
        logger.warning("GroundingDINO detected no object matching '%s'.", text_prompt)
        return []

    import torch

    boxes = torch.cat(all_boxes, dim=0)
    logits = torch.cat(all_logits, dim=0)
    phrases = all_phrases
    logger.info(
        "GroundingDINO matched %d total candidate(s) for '%s' across %d masked-reattempt pass(es).",
        boxes.shape[0], text_prompt, len(all_boxes),
    )

    if debug_dir is not None:
        import cv2
        from groundingdino.util.inference import annotate

        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        # Annotated over the ORIGINAL (unmasked) rgb, not any pass's gray-filled working copy --
        # this is purely a detection-avoidance trick, not something that should show up visually.
        annotated = annotate(image_source=rgb, boxes=boxes, logits=logits, phrases=phrases)
        cv2.imwrite(str(debug_dir / "annotated_detection.jpg"), annotated)

    pixel_boxes = _to_pixel_boxes(boxes, h, w)
    K_inv = np.linalg.inv(camera_intrinsics)
    v_full, u_full = np.indices((h, w))

    results = []
    for idx in range(boxes.shape[0]):
        result = _unproject_pixel_box(rgb, depth, camera_to_world, K_inv, u_full, v_full, pixel_boxes[idx], text_prompt)
        if result is not None:
            results.append(result)
    return results


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
