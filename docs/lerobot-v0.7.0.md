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

## Policy Apptainer image (PI0.5 and GR00T N1.7)

For Apptainer training, build the image before submitting the Slurm job. From
the repository root:

```bash
# Optional (cluster): choose locations with enough space for the build and downloads.
export APPTAINER_TMPDIR=/path/to/large/tmp
export APPTAINER_CACHEDIR=/path/to/large/cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
./docker/build_sif.sh
```

[build_sif.sh](../docker/build_sif.sh) is interactive. It lists every
`docker/<name>.def` and asks which to build: [policy.def](../docker/policy.def) gives
`docker/policy.sif`, the single image for both PI0.5 and GR00T N1.7. A new `.def` appears
in the menu with no script changes. It then asks
whether to run Apptainer on this machine or through Docker. If `docker/<name>.sif`
already exists you must confirm the replacement, and the existing image is
retained if the build fails.

- **Apptainer (cluster):** uses `--ignore-fakeroot-command` to avoid injecting a host
  fakeroot library that can require a newer glibc than the Ubuntu 22.04 image. This
  does not require sudo if the cluster permits the necessary user namespaces, but may
  fail if its fakeroot configuration is insufficient; ask the cluster administrator
  or use the Docker option.
- **Docker (local):** Docker must be installed and able to run privileged containers.
  The build runs the `kaczmarj/apptainer:latest` builder with `--privileged` and
  mounts the repository; only use it if you trust that image. It needs internet
  access and substantial free space in Docker's storage and the repository
  filesystem. A local `APPTAINER_TMPDIR` or `APPTAINER_CACHEDIR` is not passed into
  the Docker builder.

### One image for both policies

[policy.def](../docker/policy.def) and [Dockerfile.policy](../docker/Dockerfile.policy) install
`lerobot[pi,groot,training]==0.7.0` under [training-constraints.txt](../docker/training-constraints.txt)
(ffmpeg 8 PPA), mirroring the `jazzy-policy` Pixi environment. The v0.4.4 separate images were needed
because `pi` and `groot` conflicted; in 0.7.0 both share `transformers-dep`, so the Dockerfiles
differed only in the extra. Policy-specific settings live in
`docker/docker_train_<policy>.sh`. [sbatch_policy.sh](../sbatch_policy.sh) selects the policy with
`POLICY` (default `pi05`) and runs `docker/docker_train_<policy>.sh` in `docker/<policy>.sif` if it
exists, else in `docker/policy.sif`: `POLICY=groot sbatch --job-name=groot_training sbatch_policy.sh`.

**Adding a policy:** with the same dependencies, only add `docker/docker_train_<policy>.sh` and list
it in the `COPY` line of `Dockerfile.policy`. For a new compatible extra, add it to the install line
in `Dockerfile.policy` and `policy.def` and rebuild. Only for conflicting dependencies add
`Dockerfile.<policy>` and `<policy>.def`; the build scripts and `sbatch_policy.sh` find them
automatically. See the README section "Adding a new policy".

### Transferring the image through a registry (ORAS)

If the cluster cannot receive the file directly, push the locally built image to
an OCI registry with [push_sif.sh](../docker/push_sif.sh) and pull it on the
cluster. The `oras://` protocol needs only Apptainer on the cluster, not Docker.
The script asks which built image to push, the registry account, registry host
(default `ghcr.io`), tag (default `0.7.0`) and whether to use Apptainer or Docker.
`REGISTRY_ACCOUNT`, `REGISTRY_HOST` and `SIF_TAG` set the prompt defaults.

```bash
# Local machine
./docker/push_sif.sh

# Cluster, from the repository root (the script prints the exact command)
apptainer pull docker/policy.sif oras://ghcr.io/<your_account>/policy:0.7.0
# example: apptainer pull docker/policy.sif oras://ghcr.io/tzuyuan107/policy:0.7.0
```

- `<your_account>` is your **GitHub username**.
- **Create a personal access token first:** GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token, with the `write:packages` scope. A token with `read:packages` is enough to pull a private image on the cluster.
- When the script prompts for a password, **paste the token**, not your GitHub password. With the Docker option it is stored only in a throwaway container.
- The registry must support OCI artifacts. GHCR does; Docker Hub support for ORAS artifacts is limited, so GHCR is the recommended default.
- A GHCR package is private by default, and GitHub provides no API to change this from the push script. After the first push of each image (`policy`, ...), make it public once in the web UI (package → Package settings → Danger Zone → Change visibility); the script prints the link. Later pushes keep the setting. Otherwise log in on the cluster first with `apptainer registry login --username <your_account> oras://ghcr.io`.
- Add `--force` to `apptainer pull` to replace a stale `docker/<name>.sif`.
- Pushing sends a multi-GB image, and the pulled file is the same size, so allow for the space and time.

On the cluster, verify the copied or built image before training:

```bash
apptainer exec docker/policy.sif python -c "import lerobot; print(lerobot.__version__)"
apptainer exec docker/policy.sif lerobot-train --help | grep trackio
mkdir -p .logs
sbatch sbatch_policy.sh
```

The version should be `0.7.0`, and the help output should include Trackio
options. Configure the dataset and policy IDs in
[docker_train_pi05.sh](../docker/docker_train_pi05.sh) before submitting.

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

### `sbatch sbatch_policy.sh` (pi05 Apptainer training)

- **Error:** `lerobot-train: error: unrecognized arguments: --trackio.enable=true --trackio.disable_artifact=false` (`.logs/job_99079.err`). The usage text in that log lists `--wandb.*` options and no `--trackio.*` options.
- **Cause:** the launcher uses the 0.7.0 flags, but the `lerobot-train` executable inside `docker/pi05.sif` exposes the older W&B interface. Checked inside the image: `lerobot.__version__` is `0.4.4`, and `lerobot-train --help` shows `--wandb.*` options but no `--trackio.*` options. The former `pi05.def` requests `lerobot[pi,training]==0.7.0`, so `docker/pi05.sif` is stale and was built before the definition changed.
- **Fix:** rebuild `docker/pi05.sif` from the current `policy.def`, then repeat the two checks above (`python -c "import lerobot; print(lerobot.__version__)"` should print `0.7.0`; `lerobot-train --help | grep trackio` should match). Keep `--trackio.*`; do not revert to `--wandb.*`.

### `pixi run -e jazzy-policy crisp-deploy-vla` (pi05 checkpoint)

- **Error:** `DecodingError: The fields `eval_freq` are not valid for TrainPipelineConfig`.
- **Cause:** a checkpoint trained with 0.4.4 has `eval_freq` in `train_config.json`; 0.7.0 renamed it to `env_eval_freq`.
- **Fix:** new `crisp_gym/policy/train_config_compat.py` (`load_train_config`) renames the field before parsing; used in `lerobot_policy.py`, `vla_lerobot_policy.py` and `vlsa_lerobot_policy.py`. Not yet re-run on the robot.
