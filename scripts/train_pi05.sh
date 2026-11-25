#!/bin/bash

set -eu
: "${DATASET_REPO:?Set DATASET_REPO to the training dataset Hub ID}"
: "${POLICY_REPO:?Set POLICY_REPO to the destination policy Hub ID}"
OUTPUT_DIR=${OUTPUT_DIR:-outputs/pi05}
JOB_NAME=${JOB_NAME:-pi05}

exec pixi run -e jazzy-policy lerobot-train \
    --dataset.repo_id="$DATASET_REPO" \
    --policy.type=pi05 \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --policy.repo_id="$POLICY_REPO" \
    --policy.pretrained_path=lerobot/pi05_libero \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --trackio.enable=false \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=true \
    --steps=100 \
    --policy.device=cuda:0 \
    --batch_size=2 \
    --env_eval_freq 100 \
    --eval.n_episodes 50 \
    --save_checkpoint true \
    --save_freq 100 "$@"