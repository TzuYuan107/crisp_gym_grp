#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--docker] [--force]"
    echo "Build docker/pi05.sif with Apptainer, or locally with --docker."
    echo "Use --force to replace an existing image after a successful build."
}

use_docker=false
force=false
for arg in "$@"; do
    case "$arg" in
        --docker) use_docker=true ;;
        --force) force=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
output="$script_dir/pi05.sif"

if [[ -e "$output" && "$force" != true ]]; then
    echo "Image already exists: $output. Use --force to replace it." >&2
    exit 1
fi

if [[ "$use_docker" == true ]]; then
    command -v docker >/dev/null || { echo "Docker is required for --docker." >&2; exit 1; }
else
    command -v apptainer >/dev/null || { echo "Apptainer is required; use --docker for a local Docker build." >&2; exit 1; }
fi

# Build alongside the destination so a failed build cannot damage the old image.
build_dir="$(mktemp -d "$script_dir/.pi05-build.XXXXXXXX")"
trap 'rm -rf -- "$build_dir"' EXIT
relative_image="docker/$(basename "$build_dir")/pi05.sif"
cd "$repo_root"

if [[ "$use_docker" == true ]]; then
    docker run --rm --privileged \
        --volume "$repo_root:/work" --workdir /work \
        --entrypoint /bin/sh kaczmarj/apptainer:latest \
        -c 'apptainer build "$1" docker/pi05.def && chown "$2:$3" "$1"' \
        sh "$relative_image" "$(id -u)" "$(id -g)"
else
    apptainer build --ignore-fakeroot-command "$relative_image" docker/pi05.def
fi

if [[ ! -f "$build_dir/pi05.sif" ]]; then
    echo "Build did not produce an image." >&2
    exit 1
fi
mv -f -- "$build_dir/pi05.sif" "$output"
echo "Built $output"
