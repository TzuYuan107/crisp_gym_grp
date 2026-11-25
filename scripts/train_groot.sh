#!/bin/bash

set -eu
: "${DATASET_REPO:?Set DATASET_REPO to the training dataset Hub ID}"
: "${POLICY_REPO:?Set POLICY_REPO to the destination policy Hub ID}"
OUTPUT_DIR=${OUTPUT_DIR:-outputs/groot}
JOB_NAME=${JOB_NAME:-groot}

STEPS=100

exec pixi run -e jazzy-policy lerobot-train \
    --dataset.repo_id="$DATASET_REPO" \
    --policy.type=groot \
    --policy.base_model_path=nvidia/GR00T-N1.7-3B \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --policy.repo_id="$POLICY_REPO" \
    --trackio.enable=false \
    --trackio.disable_artifact=true \
    --policy.tune_diffusion_model=false \
    --steps=$STEPS \
    --policy.device=cuda:0 \
    --batch_size=1 \
    --save_checkpoint true \
    --save_freq 100 "$@"