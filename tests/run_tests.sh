#!/usr/bin/env bash
# Run the repo unit tests. Usage: tests/run_tests.sh [pixi-env] (default: jazzy-lerobot)
set -euo pipefail

cd "$(dirname "$0")/.."
ENV_NAME="${1:-jazzy-lerobot}"

pixi run -e "$ENV_NAME" python -m unittest discover -s tests -v
