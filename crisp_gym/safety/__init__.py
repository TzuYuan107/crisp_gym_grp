"""Control-Barrier-Function (CBF) based safety layer, ported from the 'aegis' project.

This package keeps the aegis math and perception utilities separate from
`crisp_gym.policy` so mismatches between aegis's original assumptions (LIBERO/robosuite
simulation, MuJoCo quaternion order, local camera capture) and crisp_gym's real interfaces
(ROS2-backed `crisp_py` cameras/robot, scipy `Rotation` poses) are isolated in one place.
"""

from crisp_gym.safety.safety_filter import SafetyFilter, SafetyFilterConfig

__all__ = ["SafetyFilter", "SafetyFilterConfig"]
