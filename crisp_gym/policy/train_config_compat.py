"""Load LeRobot train configs saved by older versions (e.g. 0.4.4) with LeRobot 0.7.0."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig

if TYPE_CHECKING:
    from os import PathLike

# Fields renamed in 0.7.0: old name -> new name.
_RENAMED_FIELDS = {"eval_freq": "env_eval_freq"}


def load_train_config(pretrained_path: str | PathLike) -> TrainPipelineConfig:
    """Load a `TrainPipelineConfig`, migrating legacy field names for local checkpoints.

    Args:
        pretrained_path: Local checkpoint directory or Hub repo id.

    Returns:
        The parsed training configuration.
    """
    config_file = Path(str(pretrained_path)) / TRAIN_CONFIG_NAME
    if not config_file.is_file():
        return TrainPipelineConfig.from_pretrained(pretrained_path)

    with open(config_file) as f:
        config = json.load(f)

    changed = False
    for old, new in _RENAMED_FIELDS.items():
        if old in config:
            value = config.pop(old)
            config.setdefault(new, value)
            changed = True

    if not changed:
        return TrainPipelineConfig.from_pretrained(pretrained_path)

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
        json.dump(config, f)
    try:
        return TrainPipelineConfig.from_pretrained(f.name)
    finally:
        Path(f.name).unlink(missing_ok=True)
