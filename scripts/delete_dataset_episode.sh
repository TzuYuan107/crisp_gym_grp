#!/bin/bash
lerobot-edit-dataset \
  --repo_id cbrian/task1_mode2_obst_SP10.5X7.5_item_B18X12_S10X13_dist_CO19X8_ST18X4_M3X12_R2X3_CA3X4_epi_10_4 \
  --operation.type delete_episodes \
  --operation.episode_indices "[0]" \
  --push_to_hub true 
  
#--new_repo_id cbrian/dataset_merged_ID_MDWC15_numepi_100_ctrl_cartesian