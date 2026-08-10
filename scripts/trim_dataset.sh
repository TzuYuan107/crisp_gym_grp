#!/bin/bash
 
# Exit immediately if a command fails
set -e
 
# ============================================================
# EDIT THESE VALUES
# ============================================================
OLD_REPO_ID="cbrian/dataset_env_starbuckswall3vegetabledistractor_task_pickcube_lift_epi_40_ctrl_cartesian"
NEW_REPO_ID=$(echo "$OLD_REPO_ID" | sed -E "s/epi_${TOTAL_EPISODES}/epi_${REMAINING_EPISODES}/")
TOTAL_EPISODES=40
NUM_TO_DELETE=10
 
# Set to "true" to pick random episode indices, "false" for the first
# NUM_TO_DELETE episodes (0, 1, 2, ...)
RANDOM_SELECTION=false
# ============================================================
 
REMAINING_EPISODES=$((TOTAL_EPISODES - NUM_TO_DELETE))
 
# Derive the new repo id by swapping the episode count in the name.
# If your naming pattern differs, just hardcode NEW_REPO_ID directly instead.
NEW_REPO_ID=$(echo "$OLD_REPO_ID" | sed -E "s/epi_${TOTAL_EPISODES}/epi_${REMAINING_EPISODES}/")
 
if [ "$RANDOM_SELECTION" = "true" ]; then
  EPISODE_INDICES=$(python3 -c "
import random
print(random.sample(range($TOTAL_EPISODES), $NUM_TO_DELETE))
")
else
  EPISODE_INDICES=$(python3 -c "print(list(range($NUM_TO_DELETE)))")
fi
 
echo "Old repo id:       $OLD_REPO_ID"
echo "New repo id:        $NEW_REPO_ID"
echo "Deleting episodes:  $EPISODE_INDICES"
echo ""
 
# Resolve the local LeRobot cache directory (respects HF_LEROBOT_HOME if set)
CACHE_DIR="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
SRC_PATH="$CACHE_DIR/$OLD_REPO_ID"
 
# Download the old dataset from the Hub if it isn't already cached locally.
# This is read-only against the Hub repo -- it never modifies it.
if [ ! -d "$SRC_PATH" ]; then
  echo "Local cache not found for $OLD_REPO_ID -- downloading from the Hub..."
  python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
LeRobotDataset('$OLD_REPO_ID')
print('Download complete.')
"
fi
 
if [ ! -d "$SRC_PATH" ]; then
  echo "ERROR: dataset still not found locally at: $SRC_PATH"
  echo "Check that OLD_REPO_ID is correct and that you're logged in (hf auth login) if the dataset is private."
  exit 1
fi
 
echo ""
echo "Step 1: deleting episodes locally (in place, NOT pushed to the Hub)"
echo ""
 
# No --new_repo_id, no --push_to_hub.
# This edits the local cache at $SRC_PATH in place. A backup of the
# untouched original is kept locally at "${SRC_PATH}_old" until the
# operation validates successfully. Nothing on the Hub is touched here.
lerobot-edit-dataset \
  --repo_id "$OLD_REPO_ID" \
  --operation.type delete_episodes \
  --operation.episode_indices "$EPISODE_INDICES"
 
echo ""
echo "Step 2: pushing the locally-edited dataset to the Hub under the new repo id"
echo ""
 
# Load the now-edited local dataset (at $SRC_PATH) and push it to a
# DIFFERENT repo id, so the original repo on the Hub is never touched.
python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset('$OLD_REPO_ID')
dataset.push_to_hub(repo_id='$NEW_REPO_ID')
print('Push complete.')
"
 
echo ""
echo "Done."
echo "The Hub repo '$OLD_REPO_ID' was never written to -- only read once to download it."
echo "Your local cache at '$SRC_PATH' now contains the trimmed (${REMAINING_EPISODES}-episode) data."
echo "A backup of the original 40-episode local copy is at '${SRC_PATH}_old' if you need to recover it."
echo "New dataset pushed to: https://huggingface.co/datasets/$NEW_REPO_ID"