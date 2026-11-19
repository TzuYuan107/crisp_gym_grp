#!/bin/bash
# Record leader-follower teleop episodes in LeRobot format.
# Edit the values below, then run:  bash crisp_gym/scripts/record_leader_follower.sh
# Leave a value empty ("") to use the default from record_lerobot_format_leader_follower.py.

# ============================ EDIT HERE ============================
REPO_ID="username/record"             # dataset repo id
TASK="close the teacup drawer"        # language instruction
NUM_EPISODES=50                       # number of episodes to record
FPS=15                                # recording frames per second

LEADER_CONFIG=""                      # leader robot config name (empty = prompt)
FOLLOWER_CONFIG=""                    # follower robot config name (empty = prompt)
LEADER_NAMESPACE=""                   # leader robot ROS namespace (empty = prompt)
FOLLOWER_NAMESPACE=""                 # follower robot ROS namespace (empty = prompt)
ROBOT_TYPE="panda"
RECORDING_MANAGER="ros"               # "keyboard" or "ros"
HOME_NOISE=0.0                        # noise on home configuration
LOG_LEVEL="INFO"                      # DEBUG | INFO | WARNING | ERROR | CRITICAL

# Flags: true / false
PUSH_TO_HUB=true
RESUME=false
JOINT_CONTROL=false
USE_STREAMED_TELEOP=false
# ===================================================================

ARGS=(
    --repo-id "$REPO_ID"
    --tasks "$TASK"
    --num-episodes "$NUM_EPISODES"
    --fps "$FPS"
    --robot-type "$ROBOT_TYPE"
    --recording-manager-type "$RECORDING_MANAGER"
    --home-config-noise "$HOME_NOISE"
    --log-level "$LOG_LEVEL"
)

[[ -n "$LEADER_CONFIG"      ]] && ARGS+=(--leader-config "$LEADER_CONFIG")
[[ -n "$FOLLOWER_CONFIG"    ]] && ARGS+=(--follower-config "$FOLLOWER_CONFIG")
[[ -n "$LEADER_NAMESPACE"   ]] && ARGS+=(--leader-namespace "$LEADER_NAMESPACE")
[[ -n "$FOLLOWER_NAMESPACE" ]] && ARGS+=(--follower-namespace "$FOLLOWER_NAMESPACE")

[[ "$PUSH_TO_HUB"         == true ]] && ARGS+=(--push-to-hub) || ARGS+=(--no-push-to-hub)
[[ "$RESUME"              == true ]] && ARGS+=(--resume)
[[ "$JOINT_CONTROL"       == true ]] && ARGS+=(--joint-control)
[[ "$USE_STREAMED_TELEOP" == true ]] && ARGS+=(--use-streamed-teleop)

echo "Running: python crisp_gym/scripts/record_lerobot_format_leader_follower.py ${ARGS[*]}"
python crisp_gym/scripts/record_lerobot_format_leader_follower.py "${ARGS[@]}"
