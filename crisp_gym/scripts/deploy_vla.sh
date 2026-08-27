#!/bin/bash

# replace with your dataset repo id
REPO_ID=cbrian/deploy_OOD-type_GDND1_epi_20


# replace with how many episodes to record
NUM_EPISODES=20



python crisp_gym/scripts/deploy_vla.py \
    --repo-id "$REPO_ID" \
    --task "put the sponge into the basket" \
    --num-episodes "$NUM_EPISODES" \
    --push-to-hub \
    --resume



    
    
    
    

    
~/.cache/huggingface/lerobot/cbrian/task1_mode2_obst_SP10.5X7.5_item_B18X12_S10X13_dist_CO19X8_ST18X4_M3X12_R2X3_CA3X4_epi_10_6
