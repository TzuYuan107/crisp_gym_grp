#!/bin/bash
# Script to merge multiple LeRobot datasets and optionally push to Hub

# ============================================================================
# CONFIGURATION - MODIFY THESE VARIABLES TO CHANGE DATASETS
# ============================================================================

# List of source dataset repo IDs to merge (space-separated)
DATASETS=(
    "cbrian/dataset_env_starbuckswallvegetabledistractor_task_pickcube_straight_epi_10_ctrl_cartesian"
    "cbrian/dataset_env_starbuckswallvegetabledistractor_task_pickcube_straight_epi_40_ctrl_cartesian"
    "cbrian/dataset_env_starbuckswallvegetabledistractor_task_pickcube_right_epi_10_ctrl_cartesian"
    "cbrian/dataset_env_starbuckswallvegetabledistractor_task_pickcube_right_epi_40_ctrl_cartesian"
    "cbrian/dataset_env_starbuckswall2vegetabledistractor_task_pickcube_right_epi_10_ctrl_cartesian"
    "cbrian/dataset_env_starbuckswall3vegetabledistractor_task_pickcube_right_epi_10_ctrl_cartesian"
    "cbrian/dataset_env_starbuckswall3vegetabledistractor_task_pickcube_right_epi_100_ctrl_cartesian"
    "cbrian/dataset_env_starbuckswallvegetabledistractor_task_pickcube_right_epi_100_ctrl_cartesian"
    "cbrian/dataset_env_starbuckswall2vegetabledistractor_task_pickcube_right_epi_100_ctrl_cartesian"
    "cbrian/dataset_env_starbuckswall2vegetabledistractor_task_pickcube_right_epi_40_ctrl_cartesian"
    "cbrian/dataset_env_starbuckswall3vegetabledistractor_task_pickcube_right_epi_40_ctrl_cartesian"
    
)

# Output dataset repo ID
OUTPUT_REPO="cbrian/dataset_merged_ID_MDWC17_numepi_500_ctrl_cartesian"

# Whether to make the output repo public (true/false)
PUBLIC="true"

# Optional: push to Hub after merge (true/false)
PUSH_TO_HUB="true"

# Optional: make output repo private (only if PUSH_TO_HUB=true)
PRIVATE="false"

# Pixi environment to use
PIXI_ENV="jazzy-lerobot"

# ============================================================================
# END OF CONFIGURATION
# ============================================================================

set -e  # Exit on error

echo "Merging datasets:"
for ds in "${DATASETS[@]}"; do
    echo "  - $ds"
done
echo "Output: $OUTPUT_REPO"
echo ""

# Build the datasets argument
DATASETS_ARG=$(printf " %s" "${DATASETS[@]}")

# Run the merge command
pixi run -e "$PIXI_ENV" python scripts/lerobot_dataset_merge.py \
    --datasets $DATASETS_ARG \
    --output "$OUTPUT_REPO" \
    $([ "$PUBLIC" = "true" ] && echo "--public" || echo "--private")

echo ""
echo "Merge complete!"

if [ "$PUSH_TO_HUB" = "true" ]; then
    echo "Pushing to Hugging Face Hub..."
    pixi run -e "$PIXI_ENV" python3 << PYTHON_SCRIPT
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("$OUTPUT_REPO")
ds.push_to_hub(private=$PRIVATE)
print("Push complete!")
PYTHON_SCRIPT
fi

echo "Done!"
