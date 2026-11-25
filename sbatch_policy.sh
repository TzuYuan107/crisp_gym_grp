#!/bin/bash

# ==========================================
# 1. Slurm Resource Requests (from your srun)
# ==========================================
#SBATCH --job-name=policy_training  # override with: sbatch --job-name=groot_training sbatch.sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00               # Adjust this time limit as needed (HH:MM:SS)
#SBATCH --output=/home_net/go68xoq/code/teleop/crisp_gym_grp/.logs/job_%j.out       # Saves terminal output to a logs folder
#SBATCH --error=/home_net/go68xoq/code/teleop/crisp_gym_grp/.logs/job_%j.err        # Saves errors to a logs folder

# Ensure the script starts in the directory where you ran sbatch
cd "$SLURM_SUBMIT_DIR" || exit 1

# Policy to train: POLICY=<name> sbatch sbatch.sh (default pi05).
# Image: docker/${IMAGE}.sif if IMAGE is set (e.g. IMAGE=policy-arm64 on GB200), else docker/<name>.sif
# if present, else the shared docker/policy.sif. Launcher: docker/docker_train_<name>.sh.
POLICY="${POLICY:-pi05}"
SIF="docker/${IMAGE:-$POLICY}.sif"
[ -n "${IMAGE:-}" ] || [ -f "$SIF" ] || SIF="docker/policy.sif"  # shared image unless the policy has its own
LAUNCHER="./docker/docker_train_${POLICY}.sh"
if [ ! -f "$SIF" ] || [ ! -f "$LAUNCHER" ]; then
  echo "Missing $SIF or $LAUNCHER for POLICY=$POLICY" >&2
  exit 1
fi

# ==========================================
# 2. Environment & Container Execution
# ==========================================

# Create writable host directories before mounting them into the container.
mkdir -p "$PWD/.cache/home" \
  "$PWD/.cache/huggingface/lerobot" \
  "$PWD/.cache/torch" \
  "$PWD/.cache/triton" || exit 1

mkdir -p "$PWD/.logs" || exit 1

# We change 'shell' to 'exec', and append the python command at the very end
apptainer exec --nv --cleanenv \
  --home "$PWD/.cache/home":/home/user_lerobot \
  --bind "$PWD":/workspace \
  --pwd /workspace \
  --env HF_HOME=/workspace/.cache/huggingface \
  --env HF_LEROBOT_HOME=/workspace/.cache/huggingface/lerobot \
  --env HF_TOKEN="${HF_TOKEN:-}" \
  --env TORCH_HOME=/workspace/.cache/torch \
  --env TRITON_CACHE_DIR=/workspace/.cache/triton \
  --env OUTPUT_DIR="${OUTPUT_DIR:-/workspace/apptainer_outputs/$POLICY}" \
  --env JOB_NAME="${JOB_NAME:-$POLICY}" \
  "$SIF" \
  "$LAUNCHER"