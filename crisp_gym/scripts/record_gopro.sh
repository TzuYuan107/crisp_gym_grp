#!/bin/bash

# replace with your dataset repo id
REPO_ID=username/test_gopro_final0000111111111
# replace with how many episodes to record
NUM_EPISODES=10




python crisp_gym/scripts/record_gopro.py \
    --repo-id "$REPO_ID" \
    --tasks "put the sponge into the basket" \
    --num-episodes "$NUM_EPISODES" \
    --recording-manager-type "ros" \
    --fps 20
