#!/bin/bash
lerobot-edit-dataset \
  --repo_id  \
  --operation.type delete_episodes \
  --operation.episode_indices "[0]" \
  --push_to_hub true 
  
#--new_repo_id cbrian/dataset_merged_ID_MDWC15_numepi_100_ctrl_cartesian