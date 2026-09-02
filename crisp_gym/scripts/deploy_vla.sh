#!/bin/bash

# replace with your dataset repo id
REPO_ID=cbrian/debug12

#deploy_task2_ID_PND1_epi_20


# replace with how many episodes to record
NUM_EPISODES=24




python crisp_gym/scripts/deploy_vla.py \
    --repo-id "$REPO_ID" \
    --task "put the sponge into the basket" \
    --num-episodes "$NUM_EPISODES" \
    --push-to-hub #--resume
    




# python crisp_gym/scripts/deploy_vla.py \
#     --repo-id "$REPO_ID" \
#     --task "close the teacup drawer" \
#     --num-episodes "$NUM_EPISODES" \
#     --push-to-hub #-resume
   
    
    
    

    
~/.cache/huggingface/lerobot/cbrian/task1_mode2_obst_SP10.5X7.5_item_B18X12_S10X13_dist_CO19X8_ST18X4_M3X12_R2X3_CA3X4_epi_10_6
