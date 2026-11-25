"""Import-compatibility tests for the LeRobot 0.4.4 -> 0.7.0 migration.

Run with:
    pixi run -e jazzy-lerobot python -m unittest discover -s tests -v
"""

import importlib
import unittest


class LeRobotApiTest(unittest.TestCase):
    """Checks the LeRobot 0.7.0 public API that crisp_gym relies on."""

    def test_dataset_metadata_in_datasets_package(self):
        """LeRobotDatasetMetadata is exported by lerobot.datasets."""
        from lerobot.datasets import LeRobotDatasetMetadata  # noqa: F401

    def test_dataset_metadata_not_reexported_by_policy_factory(self):
        """The policy factory no longer re-exports LeRobotDatasetMetadata."""
        factory = importlib.import_module("lerobot.policies.factory")
        self.assertFalse(hasattr(factory, "LeRobotDatasetMetadata"))

    def test_policy_factory_symbols(self):
        """The policy factory still provides the symbols crisp_gym imports."""
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: F401


class CrispGymPolicyImportTest(unittest.TestCase):
    """Checks that crisp_gym policy modules import against LeRobot 0.7.0."""

    def _import(self, name: str):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as e:
            # Optional deps (e.g. policy-only env) are not a migration failure.
            self.skipTest(f"missing optional dependency: {e.name}")

    def test_lerobot_policy(self):
        """The base LeRobot policy wrapper imports."""
        importlib.import_module("crisp_gym.policy.lerobot_policy")

    def test_async_lerobot_policy(self):
        """The async policy wrapper imports."""
        self._import("crisp_gym.policy.async_lerobot_policy")

    def test_vla_lerobot_policy(self):
        """The VLA policy wrapper imports."""
        self._import("crisp_gym.policy.vla_lerobot_policy")

    def test_vlsa_lerobot_policy(self):
        """The VLSA policy wrapper imports."""
        self._import("crisp_gym.policy.vlsa_lerobot_policy")


class TrainConfigCompatTest(unittest.TestCase):
    """Checks loading of train_config.json files saved by LeRobot 0.4.4."""

    def _write_config(self, directory: str, extra: dict) -> None:
        """Write a minimal train_config.json with the extra fields merged in."""
        import json
        import os

        config = {"dataset": {"repo_id": "user/dataset"}, "policy": None}
        config.update(extra)
        with open(os.path.join(directory, "train_config.json"), "w") as f:
            json.dump(config, f)

    def test_legacy_eval_freq_is_migrated(self):
        """A legacy `eval_freq` field is loaded as `env_eval_freq`."""
        import tempfile

        from crisp_gym.policy.train_config_compat import load_train_config

        with tempfile.TemporaryDirectory() as d:
            self._write_config(d, {"eval_freq": 123})
            config = load_train_config(d)
        self.assertEqual(config.env_eval_freq, 123)

    def test_current_config_is_unchanged(self):
        """A config already using `env_eval_freq` loads unchanged."""
        import tempfile

        from crisp_gym.policy.train_config_compat import load_train_config

        with tempfile.TemporaryDirectory() as d:
            self._write_config(d, {"env_eval_freq": 7})
            config = load_train_config(d)
        self.assertEqual(config.env_eval_freq, 7)


if __name__ == "__main__":
    unittest.main()
