#!/bin/bash

# replace with your dataset repo id
REPO_ID=username/recordvideo12




#deploy_task2_ID_PND1_epi_20


# replace with how many episodes to record
NUM_EPISODES=20




# python crisp_gym/scripts/deploy_vla.py \
#     --repo-id "$REPO_ID" \
#     --task "put the sponge into the basket" \
#     --num-episodes "$NUM_EPISODES" \
#     --push-to-hub --resume
    




python crisp_gym/scripts/deploy_vla.py \
    --repo-id "$REPO_ID" \
    --task "close the teacup drawer" \
    --num-episodes "$NUM_EPISODES" \
    --push-to-hub --resume
   
    
    
    


