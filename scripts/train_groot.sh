#!/bin/bash

# replace with your dataset repo e.g. cbrian/pi05_test_dataset
DATASET_REPO=
# replace with your output directory e.g. outputs/pi05_test_training
# NOTE: Keep "outputs/" prefix, only change the folder name after it
OUTPUT_DIR=outputs/training_pi05_task1_DND1_epi_200_step_20000_batch_32
# replace with your job name e.g. pi05_test_job
JOB_NAME=jobname
# replace with your policy repo e.g. cbrian/pi05_test_trained_policy
POLICY_REPO=cbrian/testgroot

STEPS=100

pixi run -e jazzy-groot python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="$DATASET_REPO" \
    --policy.type=groot \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --policy.repo_id="$POLICY_REPO" \
    --wandb.enable=false \
    --wandb.disable_artifact=true \
    --policy.tune_diffusion_model=false \
    --steps=$STEPS \
    --policy.device=cuda:0 \
    --batch_size=1 \
    --save_checkpoint true \
    --save_freq 100 \


# Use pixi run -e jazzy-pi05 python -m lerobot.scripts.lerobot_train --help to see all available options and their descriptions.