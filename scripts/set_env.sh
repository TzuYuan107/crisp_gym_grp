#!/bin/bash

export GIT_LFS_SKIP_SMUDGE=1

# Enable SVT logging (set to 1 to turn on; used by some tools/libraries)
export SVT_LOG=1

# ROS2 DDS domain ID — isolates ROS2 traffic between networks/processes
export ROS_DOMAIN_ID=100
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# CRISP config search path (colon-separated list of directories)
# Replace the example paths with the actual config folders you want to use.
#export CRISP_CONFIG_PATH="/path/to/config1:/path/to/config2"
