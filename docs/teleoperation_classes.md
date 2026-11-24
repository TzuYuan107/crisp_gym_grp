# Leader–follower teleoperation: classes, hardware API, and recording

The leader is a `TeleopRobot`; the Cartesian follower is a
`ManipulatorCartesianEnv`. Both **contain** `crisp_py` hardware interface objects;
neither inherits from `Robot` or `Gripper`.

**Scope:** the hardware members below are verified from this project's call
sites, not from an installed `crisp_py` implementation. They are the relevant
API subset, not an exhaustive library reference. Signatures are abbreviated;
internal ROS subscriptions, buffering, units of raw gripper values, and
timestamp semantics are not inferred here.

## 1. Class relationships and hardware API

Members **without parentheses** are accessed as attributes/properties. Members
**with parentheses** are methods. `+` denotes public access.

```mermaid
classDiagram
    direction LR

    class TeleopRobot {
        +Robot robot
        +Gripper gripper
        +wait_until_ready()
        +prepare_for_teleop(home, blocking)
    }
    class ManipulatorBaseEnv {
        +Robot robot
        +Gripper gripper
        +List~Camera~ cameras
        +wait_until_ready()
        +get_obs() dict
        +reset(seed, options)
        +home(home_config, blocking)
        +close()
    }
    class ManipulatorCartesianEnv {
        +step(action, block)
        -_get_obs() dict
    }

    class Robot {
        <<crisp_py>>
        +end_effector_pose
        +joint_values
        +target_pose
        +target_joint
        +cartesian_controller_parameters_client
        +joint_controller_parameters_client
        +controller_switcher_client
        +wait_until_ready(timeout)
        +set_target(pose)
        +set_target_joint(target_joint)
        +reset_targets()
        +home(home_config, blocking)
        +move_to(position, pose, speed)
        +robot_mode_monitor()
        +shutdown()
    }
    class Pose {
        <<crisp_py>>
        +position
        +orientation
        +to_array(representation)
    }
    class Gripper {
        <<crisp_py>>
        +raw_value
        +value
        +current_status
        +wait_until_ready(timeout)
        +is_open(open_threshold)
        +set_target_status(status)
        +set_target(value)
        +open()
        +close()
        +enable_torque()
        +disable_torque()
    }
    class Camera {
        <<crisp_py>>
        +current_image
        +config
        +wait_until_ready(timeout)
    }
    class CameraConfig {
        +camera_name
        +resolution
    }

    ManipulatorBaseEnv <|-- ManipulatorCartesianEnv
    TeleopRobot "1" *-- "1" Robot : leader.robot
    TeleopRobot "1" *-- "0..1" Gripper : leader.gripper
    ManipulatorBaseEnv "1" *-- "1" Robot : env.robot
    ManipulatorBaseEnv "1" *-- "1" Gripper : env.gripper
    ManipulatorBaseEnv "1" *-- "0..*" Camera : env.cameras
    Robot --> Pose : end_effector_pose and target_pose
    Camera --> CameraConfig : config
```

### Signal flow: recording loop → teleoperation → follower

```mermaid
sequenceDiagram
    participant Recording as Recording loop
    participant Fn as Closure from make_teleop_fn
    participant Leader as leader: TeleopRobot
    participant Env as env: ManipulatorCartesianEnv
    participant Robot as env.robot: Robot
    participant Gripper as env.gripper: Gripper

    Note over Fn,Env: First call initializes previous leader state and returns (None, None)
    Recording->>Fn: Call on a subsequent frame
    Fn->>Leader: Read robot pose, joints, and optional gripper status
    Fn->>Env: Read config and follower gripper status
    Note over Fn: Build relative or absolute Cartesian action plus gripper action
    Fn->>Env: step(action, block=False)
    Env->>Robot: set_target(pose=target_pose)
    opt Gripper mode is not NONE
        Env->>Gripper: Apply command through _set_gripper_action
    end
    Note over Env: Collect robot, gripper, camera, and sensor observations
    Env-->>Fn: observation, reward, terminated, truncated, info
    Fn-->>Recording: observation, action
```


### Responsibilities and accessible operations

| Interface | Read data | Send commands / manage operation |
|---|---|---|
| `Robot` | `end_effector_pose`, `joint_values`; `target_pose` and `target_joint` describe targets, not measured state | `set_target()`, `set_target_joint()`, `home()`, `move_to()`, `reset_targets()`, `wait_until_ready()`, `shutdown()` |
| `Gripper` | `current_status`, `raw_value`, `value`, `is_open()` | `set_target_status()`, `set_target()`, `open()`, `close()`, torque enable/disable, readiness wait |
| `Camera` | `current_image`; `config.camera_name` identifies the image stream | `wait_until_ready()`; this recording path uses no explicit capture/start-recording method |
| `Pose` | `position`, `orientation`; `to_array(representation=...)` converts a pose to a numeric vector | Used to construct targets and compute leader pose differences |

Controller configuration is accessed through nested robot clients, for example
`robot.cartesian_controller_parameters_client.load_param_config(...)` and
`robot.controller_switcher_client.switch_controller(...)`. These configure
control; they are not observation-reading methods.

**Reading data does not require sending a new command.** Use `env.get_obs()` to
collect the configured follower observations without calling `env.step()`.
Conversely, `step()` commands the follower and then collects observations.

## 2. Which readings become recorded data?

**Recorded frame = follower observations + leader-derived action + task label.**
Leader readings generate commands; they are not saved as separate observations.

`make_teleop_fn(env, leader)` connects both objects through a closure, not another
class or a direct reference between them.

### Leader readings → recorded action

To tell the follower what to do, the closure:

- Reads `leader.robot.end_effector_pose` to form a Cartesian action: the
    difference from the previous pose for relative actions, or the current pose
    for absolute actions.
- Reads `leader.robot.joint_values`; these contribute to the arm action only
    in joint-control mode.
- Reads `leader.gripper.current_status` to form the gripper command, using
    `False` when no leader gripper is present. It also reads
    `env.gripper.current_status` for the gripper-action conversion, particularly
    in relative modes.
- Combines the arm and gripper commands into `action` and calls
    `env.step(action, block=False)`. It returns the follower's observations and
    this action to the recording loop.

The Cartesian action has 7 elements for Euler/angle-axis orientation, or 8 for
quaternion orientation at the environment level. Saving this `action` preserves
the command derived from the leader, **not a separate recording of the leader's
measured state**. It is also not a measurement of the follower's resulting motion.

### Follower readings → recorded observations

| Source / access | Dataset key | Shape / meaning |
|---|---|---|
| Leader readings → `make_teleop_fn(env, leader)` → `env.step(action)` | `action` | Leader-derived command, **not follower motion feedback**; saved as `float32` before environment target clipping. Relative gripper commands also use follower gripper status. |
| `env.robot.end_effector_pose.to_array(representation=env.config.orientation_representation)` | `observation.state.cartesian` | `(6,)` for Euler/angle-axis or `(7,)` for quaternion; pose feedback, with the environment's representation handling applied |
| `env.robot.joint_values` | `observation.state.joints` | `(N,)`, where `N` is the robot's joint count |
| `env.gripper.raw_value` | `observation.state.gripper` | `(1,)`, used for `ABSOLUTE_CONTINUOUS` mode |
| `env.gripper.current_status` | `observation.state.gripper` | `(1,)`, converted to float in other enabled gripper modes |
| `env.robot.target_pose.to_array(representation=env.config.orientation_representation)` | `observation.state.target` | Optional Cartesian target pose; not the measured pose |
| `camera.current_image`, for each `camera` in `env.cameras` | `observation.images.<camera_name>` | Declared as `(*camera.config.resolution, 3)`, `uint8` |
| `sensor.value`, for each `sensor` in `env.sensors` | `observation.state.sensors_<sensor_name>` | Optional extra sensor data, shaped by sensor config |

Cartesian, joint, gripper, and target state fields are selected through
`env.config.observations_to_include_to_state`. Configured cameras and sensors
are included by `_get_obs()` independently of that state selection. With
`gripper_mode=NONE`, a selected gripper observation is a zero placeholder.

The recorder also writes:

- `task`: the episode task passed to `record_episode()`. The writer uses this
    label, rather than relying on `env.task`.
- `observation.state`: the concatenation of individual follower state observations.
- Individual observation fields present in the generated dataset feature schema.

## 3. One recording frame: UML sequence

### Detailed view: hardware reads and dataset writing

The following expands the signal-flow diagram in Section 1 with camera reads,
frame pacing, and the separate dataset writer process.

```mermaid
sequenceDiagram
        participant Manager as RecordingManager
        participant Fn as Teleoperation closure
        participant Leader as Leader Robot and Gripper
        participant Env as ManipulatorCartesianEnv
        participant Hardware as Follower Robot and Gripper
        participant Camera as Each Camera
        participant Writer as Dataset writer process
        participant Dataset as LeRobotDataset

        Note over Manager,Dataset: Setup: get_features(env) defines schema, writer creates or resumes dataset
        Note over Manager,Fn: First closure call resets previous state and returns None, None, frame is skipped

        Manager->>Fn: data_fn() on a subsequent frame
        Fn->>Leader: Read pose, joint_values, current_status
        Leader-->>Fn: Leader readings
        Fn->>Hardware: Read follower gripper current_status
        Hardware-->>Fn: Follower gripper status
        Note over Fn: Convert leader readings to Cartesian action and gripper action
    Fn->>Env: step(action, block=False)
        Env->>Hardware: robot.set_target(pose=target_pose)
        opt Enabled gripper and applicable command
                Env->>Hardware: Gripper command via _set_gripper_action()
    end
        Note over Env,Hardware: No wait for target completion in this teleoperation path
        Env->>Hardware: Read pose, joints, gripper state, optional target
        Hardware-->>Env: Follower state
        loop Each configured camera
                Env->>Camera: Read current_image
                Camera-->>Env: Image array
        end
        Note over Env: Also read optional sensors and assemble observation dictionary
        Env-->>Fn: obs, reward, terminated, truncated, info
        Fn-->>Manager: obs, action
        Manager->>Writer: Queue FRAME with obs, action, task
        Note over Manager: Sleep for remaining frame budget when positive
        Note over Writer: Convert state/action to float32 and concatenate state
        Writer->>Dataset: add_frame(frame)

        opt Episode accepted for saving
                Manager->>Writer: Queue SAVE_EPISODE
                Writer->>Dataset: save_episode()
        end
```

The hardware interfaces provide values and accept commands; **the recording
manager and dataset writer perform persistence**, not `Robot`, `Gripper`, or
`Camera` themselves. `get_features(env)` declares camera fields as video by
default, while each collected observation supplies an image array.

## 4. Important recording details in the current implementation

1. **Command/read ordering is not synchronization.** Observations are collected
     after `set_target()`, but `block=False` does not wait for motion completion.
     This code also does not align camera and robot readings by timestamps or
     guarantee a new camera frame on every iteration.
2. **Recorded action differs from measured motion.** The stored action is the
     closure's input to `step()`, not the measured displacement. The follower may
     clip its target position to the safety box, so the recorded action is not
     necessarily the final clipped target command either.
3. **Gripper labels are mode-dependent.** The teleoperation action uses
     `current_status`, even in `ABSOLUTE_CONTINUOUS` mode. That mode records
     `raw_value` as the observation, but the current `_set_gripper_action()` uses
     the same 0.5-threshold status commands as `ABSOLUTE_BINARY`. Do not assume a
     continuous leader position command or infer raw-value units from the name.
4. **FPS is a requested loop rate.** This script defaults `--fps` to 15 and the
     manager sleeps for the remaining frame budget. Slow processing can reduce
     the achieved rate; sensor updates are not guaranteed to match it. Video
     feature metadata separately uses `env.config.control_frequency`, so check
     that it agrees with the intended recording rate.
5. **The feature helper currently assumes six Cartesian pose components.**
     Although the environment supports quaternion observations/actions,
     `get_features()` hard-codes Cartesian names and a seven-element action
     schema. Quaternion recording needs schema changes before use.
6. **Combined-state ordering deserves attention.** `concatenate_state_features()`
     follows observation dictionary order, while feature names are generated by
     iterating the observation space. Do not assume these orders match when
     interpreting the combined vector. In the current observation construction,
     enabled fields appear as Cartesian, gripper, joints, sensors, then target.

## Source references

- Leader and factory: [crisp_gym/teleop/teleop_robot.py](../crisp_gym/teleop/teleop_robot.py)
- Base environment, Cartesian follower, and factory: [crisp_gym/envs/manipulator_env.py](../crisp_gym/envs/manipulator_env.py)
- Leader configuration: [crisp_gym/teleop/teleop_robot_config.py](../crisp_gym/teleop/teleop_robot_config.py)
- Follower configuration: [crisp_gym/envs/manipulator_env_config.py](../crisp_gym/envs/manipulator_env_config.py)
- Teleoperation closure: [crisp_gym/record/record_functions.py](../crisp_gym/record/record_functions.py)
- Recording manager and writer: [crisp_gym/record/recording_manager.py](../crisp_gym/record/recording_manager.py)
- Dataset feature schema: [crisp_gym/util/lerobot_features.py](../crisp_gym/util/lerobot_features.py)
- Recording orchestration: [crisp_gym/scripts/record_lerobot_format_leader_follower.py](../crisp_gym/scripts/record_lerobot_format_leader_follower.py)

Open this document in a Mermaid-capable Markdown preview to render the diagrams.