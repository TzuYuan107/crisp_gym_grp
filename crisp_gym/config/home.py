"""Contains some home configurations."""
# TODO: make the configs robot specific

from enum import Enum

home_close_to_table = [
    -1.73960110e-02,
    9.55319758e-02,
    8.09703053e-04,
    -1.94272034e00,
    -4.01435784e-03,
    2.06584183e00,
    7.97426445e-01,
]

home_front_up = [
    -0.02312892,
    -0.10664185,
    -0.0195703,
    -1.75644521,
    -0.00732298,
    1.68992915,
    0.8040582,
]

# home_touch_table = [
#     0.0,
#     0.066,
#     0.04,
#     -2.9,
#     0.0,
#     2.9,
#     0.785,
# ]

home_touch_table = [
    0.14719012847541557,
    0.06743427233534763,
    -0.14547530311439652,
    -2.980726624059626,
    0.176014767534203,
    3.0308150702052647,
    0.6691470556495721,
]

home_gopro = [
    1.7285,
    0.4966,
    -0.1122,
    -2.5958,
    -1.5106,
    1.6474,
    0.743,
]


home_panda = [
    0.0,
    -0.785,
    0.0,
    -2.355,
    0.0,
    1.57,
    0.785,
]


class HomeConfig(Enum):
    """Enum for different home configurations."""

    CLOSE_TO_TABLE = home_close_to_table
    TOUCH_TABLE = home_touch_table
    FRONT_UP = home_front_up
    PANDA = home_panda
    GOPRO = home_gopro

    def randomize(self, noise: float = 0.01) -> list:
        """Randomize the home configuration."""
        import numpy as np

        return (
            np.array(self.value) + np.random.uniform(-noise, noise, size=len(self.value))
        ).tolist()
