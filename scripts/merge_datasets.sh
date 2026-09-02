#!/bin/bash
# Script to merge multiple LeRobot datasets and optionally push to Hub

# ============================================================================
# CONFIGURATION - MODIFY THESE VARIABLES TO CHANGE DATASETS
# ============================================================================

# List of source dataset repo IDs to merge (space-separated)
DATASETS=(
    "cbrian/task1_mode2_obst_SP10.5X7.5_item_B18X12_S10X13_dist_CO19X8_R18X4_M3X12_ST2X3_CA3X4_epi_10_1"
    "cbrian/task1_mode2_obst_SP10.5X7.5_item_B18X12_S10X13_dist_M19X8_R18X4_CO3X12_ST2X3_CA3X4_epi_10_2"
    "cbrian/task1_mode2.5_obst_SP13X6_item_B18X12_S10X13_dist_ST19X8_M18X4_CO3X12_R2X3_CA3X4_epi_10_1"
    "cbrian/task1_mode3_obst_SP13X6_item_B18X12_S10X13_dist_ST19X8_M18X4_CO3X12_R2X3_CA3X4_epi_10_1"
    "cbrian/task1_mode2.5_obst_SP13X8.5_item_B18X12_S10X13_dist_ST19X8_M18X4_R3X12_CO2X3_CA3X4_epi_10_1"
    "cbrian/task1_mode3_obst_SP13X8.5_item_B18X12_S10X13_dist_ST19X8_M18X4_R3X12_CO2X3_CA3X4_epi_10_1"
    "cbrian/task1_mode1_obst_SP8X6_item_B18X12_S10X13_dist_CO19X8_R18X4_ST3X12_M2X3_CA3X4_epi_10_1"
    "cbrian/task1_mode1.5_obst_SP8X6_item_B18X12_S10X13_dist_CO19X8_R18X4_ST3X12_M2X3_CA3X4_epi_10_1"
    "cbrian/task1_mode1_obst_SP8X8.5_item_B18X12_S10X13_dist_CO19X8_R18X4_M3X12_ST2X3_CA3X4_epi_10_1"
    "cbrian/task1_mode1.5_obst_SP8X8.5_item_B18X12_S10X13_dist_CO19X8_R18X4_M3X12_ST2X3_CA3X4_epi_10_1"
    "cbrian/task2_mode0_obst_NONE_item_D15X13_dist_W19X8_T18X4_R3X12_P2X3_B3X4_epi_10_1"
    "cbrian/task2_mode0_obst_NONE_item_D15X13_dist_W19X8_T18X4_R3X12_P2X3_B3X4_epi_10_2"
    "cbrian/task2_mode0_obst_NONE_item_D15X13_dist_W19X8_T18X4_R3X12_P2X3_B3X4_epi_10_3"
    "cbrian/task2_mode0_obst_NONE_item_D15X13_dist_W19X8_T18X4_R3X12_P2X3_B3X4_epi_10_4"
    "cbrian/task2_mode0_obst_NONE_item_D15X13_dist_W19X8_T18X4_R3X12_P2X3_B3X4_epi_10_5"
    "cbrian/task2_mode0_obst_NONE_item_D15X13_dist_R19X8_T18X4_W3X12_B2X3_P3X4_epi_50_6"
)

# Output dataset repo ID
OUTPUT_REPO="cbrian/merge_task1_MM_epi_100_task2_ND_epi_100"

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
