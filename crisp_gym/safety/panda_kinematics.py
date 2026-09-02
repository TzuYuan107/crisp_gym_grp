"""Panda analytic forward kinematics/Jacobian, and a joint-velocity clamp built on it.

Self-contained and optional -- nothing else in `crisp_gym` imports this unless
`SafetyFilterConfig.enable_joint_velocity_limit` is turned on in `safety_filter.py`. To remove
this feature entirely: delete this file, then remove `enable_joint_velocity_limit` and
`joint_velocity_margin_fraction` from `SafetyFilterConfig`, the `joint_values` parameter from
`SafetyFilter.step()`, and the single `if self.config.enable_joint_velocity_limit:` block inside
it (see that block's own comment markers).

Why this exists: the CBF-QP and its velocity/jerk clamps in `safety_filter.py` only ever reason
in Cartesian task space -- they have no notion of the arm's joint angles. Panda has 7 joints
driving a 6-DOF Cartesian target, so a Cartesian correction that's well within the Cartesian
velocity limit can still demand an excessive *joint* velocity, either through the redundant
(null-space) degree of freedom or through poor Jacobian conditioning near certain
configurations. `clamp_to_joint_velocity_limits` catches that by predicting the joint velocity a
correction would require (via the analytic Jacobian below) and scaling the whole correction down
if any joint would exceed its limit.

Caveat: this uses the Moore-Penrose pseudoinverse to *estimate* the minimum-norm joint velocity
a Cartesian command would produce. The real Cartesian impedance controller has its own null-space
term (see `nullspace.stiffness` in the control-param yaml configs), so its actual resulting joint
velocity can differ from this estimate -- treat this as a conservative safety margin, not an
exact prediction of what the low-level controller will do.

DH parameters (modified/Craig convention) are Franka's commonly published values for the Panda
arm; the flange offset (0.107m) is Franka's default flange, not whatever gripper is actually
mounted (e.g. an aloha4franka adapter) -- so the estimated end-effector position/Jacobian is
accurate to within the unmodeled gripper offset. That's fine for this use (a velocity-direction
estimate), but don't reuse this module anywhere precise Cartesian position matters.
"""

import numpy as np

# (a, alpha, d) per joint, modified DH convention. theta_i = q[i] is the variable.
_PANDA_DH = (
    (0.0, 0.0, 0.333),
    (0.0, -np.pi / 2, 0.0),
    (0.0, np.pi / 2, 0.316),
    (0.0825, np.pi / 2, 0.0),
    (-0.0825, -np.pi / 2, 0.384),
    (0.0, np.pi / 2, 0.0),
    (0.088, np.pi / 2, 0.0),
)
# Fixed flange offset appended after joint 7 (theta=0, not a joint).
_PANDA_FLANGE = (0.0, 0.0, 0.107)

# libfranka's published per-joint velocity limits (rad/s).
PANDA_JOINT_VELOCITY_LIMITS = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])


def _dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Single modified-DH homogeneous transform (Craig's convention)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -sa * d],
            [st * sa, ct * sa, ca, ca * d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def panda_jacobian(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analytic forward kinematics + geometric Jacobian for Panda's 7 joints.

    Args:
        q: (7,) joint angles, radians.

    Returns:
        (ee_position, J): `ee_position` is the (3,) flange position in the base frame.
        `J` is the (6, 7) geometric Jacobian (rows: vx,vy,vz,wx,wy,wz; columns: joints 1-7),
        such that `[ee_linear_velocity; ee_angular_velocity] = J @ q_dot`.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    T = np.eye(4)
    origins = []
    z_axes = []
    ds = []
    for i, (a, alpha, d) in enumerate(_PANDA_DH):
        T = T @ _dh_transform(a, alpha, d, q[i])
        origins.append(T[:3, 3].copy())
        z_axes.append(T[:3, 2].copy())
        ds.append(d)

    T = T @ _dh_transform(*_PANDA_FLANGE, 0.0)
    ee_position = T[:3, 3].copy()

    J = np.zeros((6, 7))
    for i in range(7):
        # Origin of the point joint i actually rotates about -- frame i's origin minus the
        # d_i offset along its own z-axis, since Rz(theta_i)/Dz(d_i) never change that axis's
        # direction (only Rx(alpha_{i-1}) does, and that's already baked into z_axes[i]).
        pivot = origins[i] - z_axes[i] * ds[i]
        J[:3, i] = np.cross(z_axes[i], ee_position - pivot)
        J[3:, i] = z_axes[i]
    return ee_position, J


def clamp_to_joint_velocity_limits(
    translation_delta: np.ndarray,
    rotation_delta: np.ndarray,
    joint_values: np.ndarray,
    dt: float,
    joint_velocity_limits: np.ndarray = PANDA_JOINT_VELOCITY_LIMITS,
    margin_fraction: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Scale down a Cartesian correction if it would drive any Panda joint over its velocity limit.

    Estimates the joint velocity `q_dot` the given Cartesian velocity would require (via the
    Jacobian pseudoinverse at the current `joint_values`), and if any joint's estimated
    `|q_dot_i|` exceeds `margin_fraction * joint_velocity_limits[i]`, scales BOTH
    `translation_delta` and `rotation_delta` down together (preserving their relative direction)
    by whatever factor brings every joint back within its margin.

    Args:
        translation_delta: (3,) Cartesian translation delta for this step, meters.
        rotation_delta: (3,) Cartesian rotation delta for this step, radians (angle-axis/Euler).
        joint_values: (7,) current joint angles, radians.
        dt: seconds this delta is meant to be executed over.
        joint_velocity_limits: (7,) per-joint velocity limit, rad/s.
        margin_fraction: fraction of each limit to actually clamp to.

    Returns:
        (translation_delta, rotation_delta, clamped) -- clamped is True if scaling was applied.
    """
    if dt <= 0:
        return translation_delta, rotation_delta, False

    _, J = panda_jacobian(np.asarray(joint_values)[:7])
    cartesian_velocity = np.concatenate([translation_delta, rotation_delta]) / dt
    q_dot = np.linalg.pinv(J) @ cartesian_velocity

    limits = np.asarray(joint_velocity_limits) * margin_fraction
    ratios = np.abs(q_dot) / limits
    max_ratio = float(np.max(ratios))
    if max_ratio > 1.0:
        scale = 1.0 / max_ratio
        return translation_delta * scale, rotation_delta * scale, True
    return translation_delta, rotation_delta, False
