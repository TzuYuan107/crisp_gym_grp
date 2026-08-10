from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Replace with your actual dataset repo ID or local path
dataset = LeRobotDataset("cbrian/dataset_merged_ID_MDWC4_numepi_120_ctrl_cartesian")

# Look at the statistics for state and action features
print("Action Stats Keys:", dataset.meta.stats["action"].keys())
if "observation.state" in dataset.meta.stats:
    print("State Stats Keys:", dataset.meta.stats["observation.state"].keys())