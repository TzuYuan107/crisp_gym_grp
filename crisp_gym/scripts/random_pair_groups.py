"""Randomly pair group1's letter codes with group2's fixed number+suffix tokens, one-to-one."""

import random

group1 = ["ST", "M", "CO", "R"]
group2 = ["19X8", "18X4", "3X12", "2X3"]


def random_pair_groups(group1: list, group2: list) -> str:
    """Randomly pair each element of group1 with a distinct element of group2, joined by '_'."""
    shuffled = group1.copy()
    random.shuffle(shuffled)
    return "_".join(f"{a}{b}" for a, b in zip(shuffled, group2))


if __name__ == "__main__":
    print(random_pair_groups(group1, group2))
