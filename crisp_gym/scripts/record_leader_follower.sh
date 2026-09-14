#!/bin/bash

# replace with your dataset repo id
REPO_ID=username/record
NUM_EPISODES=50




python crisp_gym/scripts/record_lerobot_format_leader_follower.py \
    --repo-id "$REPO_ID" \
    --task "close the teacup drawer" \
    --num-episodes "$NUM_EPISODES" \
    --recording-manager-type "ros" 


