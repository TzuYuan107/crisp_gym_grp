#!/bin/bash
# Deploy a VLA policy and record episodes.
# Edit the values below, then run:  bash crisp_gym/scripts/deploy_vla.sh
# Leave a value empty ("") to use the default from deploy_vla.py.

# ============================ EDIT HERE ============================
REPO_ID="username/recordvideo12"      # dataset repo id
TASK="close the teacup drawer"        # language instruction
NUM_EPISODES=20                       # number of episodes to record
FPS=15                                # recording frames per second

POLICY_SOURCE=""                      # "hf" or "local" (empty = prompt)
POLICY_PATH=""                        # model path / HF repo (empty = prompt)
POLICY_CONFIG=""                      # custom policy YAML (optional)
ENV_CONFIG=""                         # follower robot config name (optional)
ENV_NAMESPACE=""                      # follower robot ROS namespace (optional)
ROBOT_TYPE="panda"
RECORDING_MANAGER="keyboard"          # "keyboard" or "ros"
HOME_NOISE=0.0                        # noise on home configuration
LOG_LEVEL="INFO"                      # DEBUG | INFO | WARNING | ERROR | CRITICAL

# Flags: true / false
PUSH_TO_HUB=true
RESUME=false
JOINT_CONTROL=false
EVALUATE=false
# ===================================================================

# Previous tasks (copy into TASK above):
#   "put the sponge into the basket"
#   deploy_task2_ID_PND1_epi_20

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

[[ -n "$POLICY_SOURCE" ]] && ARGS+=(--policy-source "$POLICY_SOURCE")
[[ -n "$POLICY_PATH"   ]] && ARGS+=(--path "$POLICY_PATH")
[[ -n "$POLICY_CONFIG" ]] && ARGS+=(--policy-config "$POLICY_CONFIG")
[[ -n "$ENV_CONFIG"    ]] && ARGS+=(--env-config "$ENV_CONFIG")
[[ -n "$ENV_NAMESPACE" ]] && ARGS+=(--env-namespace "$ENV_NAMESPACE")

[[ "$PUSH_TO_HUB"   == true ]] && ARGS+=(--push-to-hub) || ARGS+=(--no-push-to-hub)
[[ "$RESUME"        == true ]] && ARGS+=(--resume)
[[ "$JOINT_CONTROL" == true ]] && ARGS+=(--joint-control)
[[ "$EVALUATE"      == true ]] && ARGS+=(--evaluate)

echo "Running: python crisp_gym/scripts/deploy_vla.py ${ARGS[*]}"
python crisp_gym/scripts/deploy_vla.py "${ARGS[@]}"
