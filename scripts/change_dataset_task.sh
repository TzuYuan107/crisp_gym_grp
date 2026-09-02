#!/bin/bash
# Wrapper for scripts/change_dataset_task.py -- replaces the task description for every
# episode in a dataset, optionally pushing the result to the Hub.

# Source dataset repo id
REPO_ID=cbrian/task1_mode1_obst_SP10.5X7.5_item_B18X12_S10X13_dist_CO19X8_R18X4_M3X12_ST2X3_CA3X4_epi_10_1
# Output repo id (created locally under HF_LEROBOT_HOME; same as REPO_ID here to overwrite in place)
OUTPUT_REPO_ID=cbrian/task1_mode2_obst_SP13X6_item_B18X12_S10X13_dist_ST19X8_M18X4_CO3X12_R2X3_CA3X4_epi_10_1
# New task description to set for every episode
TASK="put the sponge into the basket"

# Pixi environment to use
PIXI_ENV="jazzy-lerobot"

python scripts/change_dataset_task.py \
    --repo-id "$REPO_ID" \
    --output-repo-id "$REPO_ID" \
    --task "$TASK" \
    --force-overwrite \
    --push-to-hub
