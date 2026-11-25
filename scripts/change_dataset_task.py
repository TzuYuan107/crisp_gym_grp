#!/usr/bin/env python3
"""Change the task description for all episodes in a LeRobot dataset.

Usage: python scripts/change_dataset_task.py --repo-id SOURCE/REPO --output-repo-id OUT/REPO --task "new task"
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import validate_repo_id
from lerobot.datasets.dataset_tools import modify_tasks
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME


def main() -> None:
    """Copy a LeRobot dataset and replace its episode and frame task metadata."""
    p = argparse.ArgumentParser(description="Replace task description for all episodes in a dataset")
    p.add_argument("--repo-id", required=True, help="Source dataset repo id")
    p.add_argument("--output-repo-id", required=True, help="Output local repo id (will be created under HF_LEROBOT_HOME)")
    p.add_argument("--task", required=True, help="New task description to set for every episode")
    p.add_argument("--force-overwrite", action="store_true", help="Overwrite output path if it exists locally")
    p.add_argument("--push-to-hub", action="store_true", help="Push the resulting dataset to the Hugging Face Hub")
    p.add_argument("--private", action="store_true", help="Create the repo as private when pushing to the Hub")
    p.add_argument("--push-videos", action="store_true", help="Also push the videos/ directory to the Hub")
    p.add_argument("--branch", default=None, help="Branch/revision to push to on the Hub")
    p.add_argument("--tags", default=None, help="Comma-separated tags to add to the dataset card")
    args = p.parse_args()

    src_repo = args.repo_id
    out_repo = args.output_repo_id
    validate_repo_id(out_repo)
    new_task = args.task

    out_root = HF_LEROBOT_HOME / out_repo
    if out_root.exists() and not args.force_overwrite:
        sys.exit(f"Output path {out_root} already exists. Use --force-overwrite to replace.")

    out_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_root.parent) as tmp:
        staging = Path(tmp) / "dataset"
        snapshot_download(repo_id=src_repo, repo_type="dataset", local_dir=staging)
        # 0.7 updates every episode shard as well as frame indices and typed metadata.
        modify_tasks(LeRobotDataset(out_repo, root=staging), new_task=new_task)
        if out_root.exists():
            shutil.rmtree(out_root)
        staging.rename(out_root)

    # meta/stats.json's task_index entry (min/max/mean/etc.) reflects whatever distribution of
    # indices existed before the remap above -- now stale if there was more than one. Recompute
    # it as the trivial all-zero stats a single-valued column has (count is unaffected).
    stats_path = Path(out_root) / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        if "task_index" in stats:
            count = stats["task_index"].get("count", [0])
            stats["task_index"] = {
                "min": [0], "max": [0], "mean": [0.0], "std": [0.0], "count": count,
                "q01": [0.0], "q10": [0.0], "q50": [0.0], "q90": [0.0], "q99": [0.0],
            }
            stats_path.write_text(json.dumps(stats, indent=2))
            print("Corrected meta/stats.json task_index stats to match.")

    print("Done. Local modified dataset is at:", out_root)

    if args.push_to_hub:
        print(f"Pushing {out_repo} to the Hugging Face Hub (this may take a while)...")
        tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
        # Instantiate LeRobotDataset with the local root so push_to_hub uploads files correctly
        ds = LeRobotDataset(out_repo, root=out_root)
        ds.push_to_hub(branch=args.branch, tags=tags, private=args.private, push_videos=args.push_videos)
        print("Push complete.")


if __name__ == "__main__":
    main()
