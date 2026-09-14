#!/bin/bash
# Wrapper for scripts/change_dataset_task.py -- replaces the task description for every
# episode in a dataset, optionally pushing the result to the Hub.

# Source dataset repo id
REPO_ID=
# Output repo id (created locally under HF_LEROBOT_HOME; same as REPO_ID here to overwrite in place)
OUTPUT_REPO_ID=
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
