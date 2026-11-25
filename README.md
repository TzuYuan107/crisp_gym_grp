# CRISP_GYM

**CRISP_GYM_ITR** is based on **CRISP_GYM** and aims to make fine-tuning and deploying VLAs easy and fast.

## Features

- **Full Support:** Fully supports the CRISP ecosystem and the `CRISP_PY` API.
- **Data Recording:** Records manipulation data in the `LeRobotDataset v3.0` format.
- **Environment & Deployment:** PI0.5 and GR00T N1.7 share one policy environment.

## **Prerequisites**

```jsx
#install pixi 
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc
```

## Quick Start

```jsx
git clone [https://github.com/brianchang1025/crisp_gym_ITR](https://github.com/brianchang1025/crisp_gym_ITR).git
cd crisp_gym_ITR

export GIT_LFS_SKIP_SMUDGE=1
```

Create a file named `set_env.sh` and paste the code below into it. You can also add your own custom configuration paths to this file to suit your local environment.

```bash
export GIT_LFS_SKIP_SMUDGE=1

# Enable SVT logging (set to 1 to turn on; used by some tools/libraries)
export SVT_LOG=1

# ROS2 DDS domain ID — isolates ROS2 traffic between networks/processes
export ROS_DOMAIN_ID=100
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# CRISP config search path (colon-separated list of directories
# Replace the example paths with the actual config folders you want to use.
#export CRISP_CONFIG_PATH="/path/to/config1:/path/to/config2"
```

LeRobot 0.7.0 uses one environment for **PI0.5 and NVIDIA GR00T N1.7**. Use the smaller dataset environment on recording machines. See the [migration and validation notes](docs/lerobot-v0.7.0.md) before reusing a checkpoint.

```bash
# lerobot dataset record env
pixi install -e jazzy-lerobot
pixi run -e jazzy-lerobot python -c "import crisp_gym"

# training and deployment (either policy)
pixi install -e jazzy-policy

# deployment with Aegis point-cloud dependencies
pixi install -e jazzy-policy-aegis
```

You can verify if your configuration has loaded correctly by running the following command:

```bash
pixi run -e jazzy-lerobot crisp-check-config
```

if this work we can move to next step 

## Dataset Recording

You can start recording your dataset by running the following command:

```bash
# login huggingface, you only do this once
pixi run -e jazzy-lerobot hf auth login 

pixi run -e jazzy-lerobot crisp-record-leader-follower \
   --repo-id huggingface_username/dataset_repo_name --task "the task description" \
   --num-episodes int 
```

Once the recording interface opens, you can select your hardware settings for the teleoperation **leader** and **follower**. For detailed instructions on how to set up your own leader and follower configurations, please check the [placeholder].

You can use the `--resume` flag to continue recording to an existing dataset. Additionally, use the `--recording-manager-type` argument to choose between `keyboard` or `ros` control. If you select `ros`, you can manage the recording process directly using the buttons on the leader arm.

```bash
pixi run -e jazzy-lerobot crisp-record-leader-follower \
   --repo-id <your_huggingface_account>/<repo_name> --task <the task description> \
   --num-episodes <int> --recording-manager-type "ros"
   
```

When using the ROS manager, you can control the recording using the buttons on the leader arm:

- Press the **Circle** button to start or stop recording an episode.
- Press the **Check** button to save the current recording.
- Press the **Cross** button to delete the current recording.-

![image.png](CRISP_GYM/image.png)

---

## Fine-tuning

Set your source dataset and destination policy Hub IDs, then run one launcher:

```bash
export DATASET_REPO=your-account/your-dataset
export POLICY_REPO=your-account/your-policy
./scripts/train_pi05.sh  # or ./scripts/train_groot.sh
```

The `docker/` folder also provides container training launchers.

To simplify the setup, use the provided automation script to build your image and launch the container:

```bash
# automation script to build your image and launch the container:
./docker/build_and_run_docker.sh
```

Inside the container, choose one launcher. The container scripts log to Trackio rather than W&B.
Edit `DATASET_REPO` and `POLICY_REPO` directly in `docker/docker_train_pi05.sh` or
`docker/docker_train_groot.sh`; exported repository IDs do not override these values.
Both launchers print the configured IDs before starting training. When run through Slurm,
these messages appear in the job's `.out` log.

Training downloads the dataset and base model from, and pushes the trained policy to,
the Hugging Face Hub, so it needs a Hugging Face login with write access. Both launchers
check this by default: if you are not logged in (no saved login and no `HF_TOKEN`), they run
`hf auth login` for you when started from a terminal, or stop with an error when there is
no terminal (e.g. a Slurm job). You can also log in yourself beforehand; note that a
login inside a container is lost when the container is removed. Run one launcher:

```bash
# train script for pi05
./docker/docker_train_pi05.sh
# train script for groot
./docker/docker_train_groot.sh
```

For Slurm training (Apptainer), there is no terminal for the login prompt, so the
launcher would stop. Export your Hugging Face token before submitting; `sbatch.sh` forwards `HF_TOKEN`
into the container. Use the IDs configured in the policy's `docker/docker_train_<policy>.sh`.
`sbatch.sh` picks the policy from the `POLICY` environment variable (default `pi05`) and
runs `docker/<policy>.sif` with `docker/docker_train_<policy>.sh`:

```bash
export HF_TOKEN=<your Hugging Face token>
# On the cluster, build or pull the LeRobot 0.7.0 Apptainer image first (see below).
mkdir -p .logs
sbatch sbatch.sh                                              # pi05
POLICY=groot sbatch --job-name=groot_training sbatch.sh      # GR00T N1.7
```

### Apptainer images (`docker/*.sif`)

`./docker/build_sif.sh` builds an image from a definition file in `docker/`. It asks
which image to build (every `docker/<name>.def`, currently
[pi05.def](docker/pi05.def) and [groot.def](docker/groot.def) for GR00T N1.7) and
whether to run Apptainer directly or through Docker. To support a new policy, add
`docker/<name>.def`; it appears in the menu automatically. The result is
`docker/<name>.sif`, and an existing image is only replaced if you confirm.

- **On the cluster:** choose *Apptainer*. The build uses `--ignore-fakeroot-command`
  to avoid a host/container glibc incompatibility; it still needs suitable
  user-namespace permissions, internet access, and enough temporary and cache space.
- **On a local machine without Apptainer:** choose *Docker*. This runs a
  **privileged** container, so use it only if you trust the builder image.

If the cluster cannot receive file transfers, push the image to an OCI registry
(default GHCR) and pull it on the cluster, which needs only Apptainer.
`./docker/push_sif.sh` asks which built image to push, your registry account, the
registry host (default `ghcr.io`), the tag (default `0.7.0`), and whether to use
Apptainer or Docker.

`<your_account>` is your **GitHub username**. Before pushing, create a GitHub
**personal access token (classic)** at GitHub → Settings → Developer settings →
Personal access tokens → Tokens (classic), with the `write:packages` scope
(`read:packages` is enough for pulling a private image on the cluster). When the
script prompts for a password, paste the token (not your GitHub password).

```bash
# Local machine: answer the prompts, then paste the token when asked for a password
./docker/push_sif.sh

# Cluster: pull into the path used by sbatch.sh (the script prints the exact command)
apptainer pull docker/pi05.sif oras://ghcr.io/<your_account>/pi05:0.7.0
# example: apptainer pull docker/pi05.sif oras://ghcr.io/tzuyuan107/pi05:0.7.0
# GR00T N1.7: apptainer pull docker/groot.sif oras://ghcr.io/<your_account>/groot:0.7.0
```

A new GHCR package is **private** and GitHub offers no way to change this from the
script, so after the first push of each image make it public once on GitHub
(package → Package settings → Danger Zone → Change visibility; the script prints
the link). Later pushes keep that setting. Otherwise, for a private package, run
`apptainer registry login --username <your_account> oras://ghcr.io` on the cluster first.

See the [migration notes](docs/lerobot-v0.7.0.md#pi05-apptainer-image) for build
prerequisites and image verification.

### Adding a new policy

Replace `<policy>` with your policy name (e.g. `smolvla`). Use `docker/pi05.def`,
`docker/Dockerfile.pi05` and `docker/docker_train_pi05.sh` as templates.

**Apptainer (cluster)**

1. Add `docker/<policy>.def` (change the extras in `lerobot[<extra>,training]==0.7.0`).
2. Add `docker/docker_train_<policy>.sh` (dataset/policy IDs and `lerobot-train` arguments).
3. Nothing to edit in `build_sif.sh` and `push_sif.sh`: they list every `docker/<name>.def`
   and every built `docker/<name>.sif`, so the new policy appears in their menus automatically.
4. Nothing to edit in `sbatch.sh` either: run `POLICY=<policy> sbatch sbatch.sh`.

**Docker**

1. Add `docker/Dockerfile.<policy>`.
2. Add `docker/docker_train_<policy>.sh` (the Dockerfile copies it into the image).
3. Add `"Dockerfile.<policy>"` to the `DOCKERFILES` list in `docker/build_and_run_docker.sh`.

Also check:
- Both launchers must use `--trackio.*` (not `--wandb.*`) arguments and keep the Hugging Face
  login check from the existing launchers.
- If the policy needs different dependencies, update `docker/training-constraints.txt` (shared by
  all images) and the migration notes.
- Optionally add `scripts/train_<policy>.sh` for training outside a container.
- For deployment, check that the policy is covered by the `jazzy-policy` Pixi environment (`pixi.toml`).
- `.sif` files are git-ignored; share a new image through `push_sif.sh` and make its GHCR package public once.

---

## Inference

Once you have finished fine-tuning your model or have downloaded a pre-trained policy from the Hugging Face Hub, you can begin running inference to test your policy on the hardware.

```bash
#deploy pi05 policy
pixi run -e jazzy-policy crisp-deploy-vla --path <your policy path> --task <the task description> \
--repo-id <your_huggingface_account>/<repo_name>
```

## Setting Your Own Config

### Leader and Follower

You can set your own leader and follower configurations in `/crisp_gym/teleop/teleop_robot_config.py` and `/crisp_gym/envs/manipulator_env_config.py` separately.

```python
# Example for leader config
@dataclass
class RightPandaTeleopRobotConfig(TeleopRobotConfig):
    """Configuration for the right robot as a leader."""

    leader: RobotConfig = field(default_factory=lambda: PandaConfig())
    leader_gripper: GripperConfig | None = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(find_config("grippers/gripper_right.yaml")).resolve()
        )
    )

    gravity_compensation_controller: Path = field(
        default_factory=lambda: find_config("control/gravity_compensation.yaml")
    )

    leader_namespace = "right"
    leader_gripper_namespace = "right"
```

```python
#Example for follower config
@dataclass
class FrankaGripperPandaEnvConfig(PandaEnvConfig):
    """Custom Panda Gym Environment Configuration for Panda with an Franka gripper and cameras."""

    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(
                find_config("grippers/panda_gripper.yaml") or CRISP_CONFIG_PATH / "grippers" / "panda_gripper.yaml"
            ).resolve()
        )
    )
    camera_configs: List[CameraConfig] = field(
        default_factory=lambda: [
            CameraConfig(
                camera_name="primary",
                camera_frame="primary_link",
                resolution=[256, 256],
                camera_color_image_topic="/camera/third_person_camera/color/image_raw",
                camera_color_info_topic="/camera/third_person_camera/color/camera_info",
            ),
            CameraConfig(
                camera_name="wrist",
                camera_frame="wrist_link",
                resolution=[256, 256],
                camera_color_image_topic="/camera/wrist_camera/color/image_rect_raw",
                camera_color_info_topic="/camera/wrist_camera/color/camera_info",
            ),
        ]
    )

    max_episode_steps: int | None = 1000
```

After setting your own config, you need to register it in the `STRING_TO_CONFIG` section located at the end of each Python file. Only after you register it will the config show up in the record and deploy interface.

### CRISP_PY

`crisp_py` provides the Python API for controlling the robot, as well as configurations for controller parameters and equipment. Currently, the `crisp_py` version used in this repository is `crisp_py_ITR`. You can also import your own version by replacing the link in the `pixi.toml` file.

```xml
crisp-python = { git = "[https://github.com/brianchang1025/crisp_py_panda](https://github.com/brianchang1025/crisp_py_panda)"}
```

## Acknowledgements

This project is built heavily on `CRISP_GYM` and `CRISP_PY`. Check their official project website for more detailed information and instructions on how to modify your own CRISP version.

**CRISP :** https://learnsyslab.github.io/crisp_controllers/