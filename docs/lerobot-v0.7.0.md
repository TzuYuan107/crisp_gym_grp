# LeRobot 0.4.4 → 0.7.0 migration

## Decision and installation

**Merge the policy environments, not the recording and policy environments.** In [v0.4.4](https://github.com/huggingface/lerobot/blob/v0.4.4/pyproject.toml) `pi` used a Transformers Git fork and explicitly conflicted with `groot`. In the [v0.7.0 manifest](https://github.com/huggingface/lerobot/blob/v0.7.0/pyproject.toml), both use `transformers-dep` (`>=5.22,<5.23`). Both policy classes actually load together. Recording needs dataset/video support but not those models; point-cloud deployment additionally needs Aegis.

| Previous environment | Replacement | Purpose |
|---|---|---|
| `jazzy-lerobot` | `jazzy-lerobot` | Recording, dataset tools; `lerobot[dataset]` |
| `jazzy-pi05`, `jazzy-groot` | `jazzy-policy` | Training and deployment; `lerobot[training,pi,groot]` |
| `jazzy-pi05-aegis`, `jazzy-groot-aegis` | `jazzy-policy-aegis` | Either policy with point-cloud safety dependencies |

The plain `jazzy` and `dev` environments remain. Use Pixi from the repo root:

```sh
pixi install --locked -e jazzy-lerobot
pixi install --locked -e jazzy-policy
pixi install --locked -e jazzy-policy-aegis
```

All three environments installed on Linux x86-64 with Pixi 0.84.2. [pixi.lock](../pixi.lock) is the new reproducible solve; `pixi.lock.bak` is **not** an input lock. Post-link scripts for Graphviz/GTK were skipped by Pixi's default policy; no system configuration was changed.

### Dependency changes

| Area | 0.7.0 decision and evidence |
|---|---|
| Python / numeric stack | Upstream now requires Python ≥3.12; Jazzy already uses 3.12. Keep NumPy ≥2,<2.3. The local SciPy ceiling overlaps the `pi` minimum (≥1.14). |
| Distribution | Use the [published 0.7.0 wheel](https://pypi.org/project/lerobot/0.7.0/). Datasets and training are now extras, not implicit core dependencies. `groot` already brings PEFT. |
| GPU / video | Remove local torch/torchvision/torchcodec and Transformers-fork pins; GR00T no longer declares flash-attn, so remove its build-isolation exception, wheel, CUDA toolkit and NVIDIA channel. The installed stack has torch 2.15.0, torchvision 0.29.2, torchcodec 0.17.0, Transformers 5.22.2, PEFT 0.22.0 and datasets 5.2.0. No custom flash-attn build or N1.5 tokenizer-cache patch. |
| ROS / Python overlap | Align conda's OpenCV (<4.14), Pillow (<14), packaging (<27), setuptools (<85) and CMake (<4.2) with the upstream ranges. ROS cv-bridge and the recorder use OpenCV 4.13.0. Keep conda `pynput` for the local recording manager. |
| Training telemetry | `training` includes Accelerate and Trackio, not W&B. Host scripts disable logging; container scripts use Trackio. |

### Training telemetry: W&B → Trackio

The [v0.7.0 release notes](https://github.com/huggingface/lerobot/releases/tag/v0.7.0) list this as a breaking training change: Weights & Biases is replaced by Trackio, and the `training` extra installs `trackio`.

| 0.4.4 | 0.7.0 |
|---|---|
| `--wandb.enable=true` | `--trackio.enable=true` |
| `--wandb.disable_artifact=...` | `--trackio.disable_artifact=...` |
| `--wandb.entity`, `--wandb.mode` | No equivalent |

- Remote jobs with tracking need `--trackio.space_id`.
- Checkpoints are no longer uploaded as artifacts by default.
- `docker/docker_train_pi05.sh` and `docker/docker_train_groot.sh` already use `--trackio.*`.
- **Do not mix interfaces.** An image or environment that still has the pre-0.7.0 training CLI rejects `--trackio.*`.

## Validation evidence

- **Binary stack:** `rclpy`, OpenCV, torch and torchcodec imported in the recording environment. A six-frame synthetic camera/action episode was written, finalized, reopened, and decoded (`[3,48,64]`, expected action and task). A CUDA torchvision NMS kernel ran on the RTX 3090 Ti (driver 595.99.02).
- **Shared policy:** both `get_policy_class("pi05")` and `get_policy_class("groot")` loaded in the same interpreter. This tests import/registration, not checkpoint compatibility or model quality.
- **Metadata warnings:** `pip check` reports missing `colcon-notification` in all three environments. The conda `colcon-common-extensions` recipe does not require it, but the distribution's Python metadata does. Pip also reports decord 0.6.0 as unsupported in policy environments: its internal WHEEL tag is `cp36-cp36m` on Python 3.12; `import decord; decord.cpu(0)` works here. Do not treat these as a clean `pip check`.
- **Auxiliary tools:** the relabeler uses upstream `modify_tasks` to update every episode shard and frame task index on a staged copy. Two synthetic tasks in two episodes (six frames) were changed to one task and read back. Four dataset conversion scripts now use `add_frame(frame)` (`task` inside the frame) and explicit finalization. Their syntax and auxiliary LeRobot imports were checked; those four data-specific conversions were not run or published.

## Checkpoint and dataset decisions

- A LeRobot **package version** is not a **dataset format version**. The tagged [metadata class](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_metadata.py) still declares `CODEBASE_VERSION = "v3.0"`. Don't bulk-convert existing v3.0 data just to upgrade the package. Back up a representative recording and test camera decode, state/action names, task strings and episode counts before resuming on the original.
- **GR00T N1.5 checkpoints cannot be loaded by 0.7.0.** The [0.7.0 GR00T configuration](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/groot/configuration_groot.py) explicitly rejects N1.5 and names 0.5.1 as the last supporting release. Keep the former environment/image for N1.5 inference or fine-tune an N1.7 base (`nvidia/GR00T-N1.7-3B`). Renaming an old checkpoint is not a migration.

## On-site testing log

### `pixi run -e jazzy-lerobot crisp-record-leader-follower`

- **Error:** `ImportError: cannot import name 'LeRobotDatasetMetadata' from 'lerobot.policies.factory'` raised from `crisp_gym/policy/lerobot_policy.py` (imported via `crisp_gym/policy/__init__.py`).
- **Cause:** 0.7.0's `lerobot.policies.factory` no longer re-exports `LeRobotDatasetMetadata`; it lives in `lerobot.datasets` (`dataset_metadata.py`).
- **Fix:** import it with `from lerobot.datasets import LeRobotDatasetMetadata` in:
  - `crisp_gym/policy/lerobot_policy.py`
  - `crisp_gym/policy/vla_lerobot_policy.py`
  - `crisp_gym/policy/vlsa_lerobot_policy.py`
  
  `get_policy_class` and `make_pre_post_processors` are still imported from `lerobot.policies.factory`. `async_lerobot_policy.py` needed no change.
- **Verified:** `pixi run -e jazzy-lerobot python -c "import crisp_gym.policy.lerobot_policy"` succeeds. The vla/vlsa modules were edited but not import-tested in this environment (they need the policy environment). The full recording run has not yet been re-run.

### `sbatch sbatch.sh` (pi05 Apptainer training)

- **Error:** `lerobot-train: error: unrecognized arguments: --trackio.enable=true --trackio.disable_artifact=false` (`.logs/job_99079.err`). The usage text in that log lists `--wandb.*` options and no `--trackio.*` options.
- **Cause:** the launcher uses the 0.7.0 flags, but the `lerobot-train` executable inside `docker/pi05.sif` exposes the older W&B interface. The installed LeRobot version in the image has not been checked. [pi05.def](../docker/pi05.def) requests `lerobot[pi,training]==0.7.0`, so the `.sif` is probably older than the definition or was built with a different version; rebuilding changes the image, not the launcher.
- **Next step:** run `apptainer exec docker/pi05.sif python -c "import lerobot; print(lerobot.__version__)"` and `apptainer exec docker/pi05.sif lerobot-train --help | grep -E "trackio|wandb"`, then rebuild the image if it is not 0.7.0. Keep `--trackio.*`; do not revert to `--wandb.*`.
