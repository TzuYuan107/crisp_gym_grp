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

```bash
hf auth login

# train script for pi05
./docker/docker_train_pi05.sh
# train script for groot
./docker/docker_train_groot.sh
```

For Slurm pi05 training, use the same IDs configured in `docker/docker_train_pi05.sh`:

```bash
# On the cluster, build the LeRobot 0.7.0 Apptainer image first.
./docker/build_pi05_sif.sh
mkdir -p .logs
sbatch sbatch.sh
```

The build produces `docker/pi05.sif`, which `sbatch.sh` uses. If an older image
already exists, rebuild it with `./docker/build_pi05_sif.sh --force`. The
cluster build uses `--ignore-fakeroot-command` to avoid a host/container glibc
incompatibility; it still needs suitable user-namespace permissions. Build on a
node with internet access and sufficient temporary and cache space.

Alternatively, on a local machine with Docker, run
`./docker/build_pi05_sif.sh --docker` (add `--force` to replace an existing
image). This uses a **privileged** Docker container, so use it only if you trust
the builder image.

If the cluster cannot receive file transfers, push the image to an OCI registry
(default GHCR) and pull it on the cluster, which needs only Apptainer:

`<your_account>` is your **GitHub username**. Before pushing, create a GitHub
**personal access token (classic)** at GitHub → Settings → Developer settings →
Personal access tokens → Tokens (classic), with the `write:packages` scope
(`read:packages` is enough for pulling a private image on the cluster). When the
script prompts for a password, paste the token (not your GitHub password).

```bash
# Local machine: you are prompted for the token
./docker/push_pi05_sif.sh --account <your_account> --docker   # drop --docker if Apptainer is installed

# Cluster: pull into the path used by sbatch.sh (use the command the script prints)
apptainer pull docker/pi05.sif oras://ghcr.io/<your_account>/pi05:0.7.0
# example: apptainer pull docker/pi05.sif oras://ghcr.io/tzuyuan107/pi05:0.7.0
```

Use `--registry` and `--tag` to change the registry or tag. A new GHCR package is
**private** and GitHub offers no way to change this from the script, so after the
first push make it public once on GitHub (`pi05` package → Package settings →
Danger Zone → Change visibility; the script prints the link). Later pushes keep
that setting. Otherwise, for a private
package, run `apptainer registry login --username <your_account> oras://ghcr.io`
on the cluster first. See the [migration notes](docs/lerobot-v0.7.0.md#pi05-apptainer-image) for
build prerequisites and image verification.

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