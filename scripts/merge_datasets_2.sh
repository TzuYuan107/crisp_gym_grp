#!/bin/bash
# Script to merge multiple LeRobot datasets and optionally push to Hub

# ============================================================================
# CONFIGURATION - MODIFY THESE VARIABLES TO CHANGE DATASETS
# ============================================================================

# List of source dataset repo IDs to merge (space-separated)
DATASETS=(
    "cbrian/task2_mode0_obst_NONE_item_D15X13_dist_W19X8_T18X4_R3X12_P2X3_B3X4_epi_10_1"
    "cbrian/task2_mode0_obst_NONE_item_D15X13_dist_W19X8_T18X4_R3X12_P2X3_B3X4_epi_10_2"
    "cbrian/task2_mode0_obst_NONE_item_D15X13_dist_W19X8_T18X4_R3X12_P2X3_B3X4_epi_10_3"
    "cbrian/task2_mode0_obst_NONE_item_D15X13_dist_W19X8_T18X4_R3X12_P2X3_B3X4_epi_10_4"
    "cbrian/task2_mode0_obst_NONE_item_D15X13_dist_W19X8_T18X4_R3X12_P2X3_B3X4_epi_10_5"
    "cbrian/task2_mode1_obst_G10.5X6.5_item_D15X13_S10X13_dist_W19X8_R18X4_T3X12_B2X3_P3X4_epi_10_1"
    "cbrian/task2_mode2_obst_G10.5X6.5_item_D15X13_S10X13_dist_W19X8_P18X4_T3X12_B2X3_R3X4_epi_10_1"
    "cbrian/task2_mode3_obst_G10.5X6.5_item_D15X13_S10X13_dist_W19X8_P18X4_T3X12_B2X3_R3X4_epi_10_1"
    "cbrian/task2_mode2_obst_G12X5_item_D15X13_S10X13_dist_T19X8_B18X4_R3X12_W2X3_P3X4_epi_10_1"
    "cbrian/task2_mode2.5_obst_G12X5_item_D15X13_S10X13_dist_T19X8_B18X4_R3X12_W2X3_P3X4_epi_10_1"
    "cbrian/task2_mode3_obst_G12X5_item_D15X13_S10X13_dist_T19X8_B18X4_R3X12_W2X3_P3X4_epi_10_1"
    "cbrian/task2_mode2_obst_G12X7.5_item_D15X13_S10X13_dist_W19X8_P18X4_R3X12_B2X3_T3X4_epi_10_1"
    "cbrian/task2_mode2.5_obst_G12X7.5_item_D15X13_S10X13_dist_W19X8_P18X4_R3X12_B2X3_T3X4_epi_10_1"
    "cbrian/task2_mode3_obst_G12X7.5_item_D15X13_S10X13_dist_W19X8_P18X4_R3X12_B2X3_T3X4_epi_10_1"
    "cbrian/task2_mode1_obst_G8X5_item_D15X13_S10X13_dist_B19X8_T18X4_R3X12_W2X3_P3X4_epi_10_1"
    "cbrian/task2_mode1.5_obst_G8X5_item_D15X13_S10X13_dist_B19X8_T18X4_R3X12_W2X3_P3X4_epi_10_1"
    "cbrian/task2_mode2_obst_G8X5_item_D15X13_S10X13_dist_B19X8_T18X4_R3X12_W2X3_P3X4_epi_10_1"
    "cbrian/task2_mode1_obst_G8X7.5_item_D15X13_S10X13_dist_W19X8_R18X4_P3X12_T2X3_B3X4_epi_10_1"
    "cbrian/task2_mode1.5_obst_G8X7.5_item_D15X13_S10X13_dist_W19X8_R18X4_P3X12_T2X3_B3X4_epi_10_1"
    "cbrian/task2_mode2_obst_G8X7.5_item_D15X13_S10X13_dist_W19X8_R18X4_P3X12_T2X3_B3X4_epi_10_1"
)

# Output dataset repo ID
OUTPUT_REPO="cbrian/merge_task2_UM_epi_200"

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
