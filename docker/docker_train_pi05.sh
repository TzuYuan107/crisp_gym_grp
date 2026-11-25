#!/bin/bash

set -eu
DATASET_REPO=IanHHH698/ian_telop_helloworld_v2
POLICY_REPO=IanHHH698/pi05_testv0.7.0_v1
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/apptainer_outputs/pi05}
JOB_NAME=${JOB_NAME:-pi05}
STEPS=20000

echo "DATASET_REPO=$DATASET_REPO"
echo "POLICY_REPO=$POLICY_REPO"

exec lerobot-train \
    --dataset.repo_id="$DATASET_REPO" \
    --policy.type=pi05 \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --policy.repo_id="$POLICY_REPO" \
    --policy.pretrained_path=lerobot/pi05_libero \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --trackio.enable=true \
    --trackio.disable_artifact=false \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=true \
    --steps=$STEPS \
    --policy.device=cuda:0 \
    --batch_size=32 \
    --save_checkpoint true \
    --save_freq 5000 "$@"