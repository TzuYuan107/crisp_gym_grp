#!/usr/bin/env bash
# Interactively build docker/<model>.sif from docker/<model>.def.
# To support a new policy, add docker/<name>.def; it appears in the menu automatically.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
# shellcheck source=docker/sif_common.sh
source "$script_dir/sif_common.sh"

models=()
for def in "$script_dir"/*.def; do
    [[ -e "$def" ]] && models+=("$(basename "$def" .def)")
done
(( ${#models[@]} > 0 )) || die "No definition files found in $script_dir."

choose model "Which image do you want to build?" "${models[@]}"
choose_runner use_docker

output="$script_dir/$model.sif"
if [[ -e "$output" ]]; then
    ask replace "$output already exists. Replace it after a successful build? (y/n)" "n"
    [[ "$replace" =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 0; }
fi

# Build alongside the destination so a failed build cannot damage the old image.
build_dir="$(mktemp -d "$script_dir/.$model-build.XXXXXXXX")"
trap 'rm -rf -- "$build_dir"' EXIT
relative_image="docker/$(basename "$build_dir")/$model.sif"
cd "$repo_root"

if [[ "$use_docker" == true ]]; then
    docker run --rm --privileged \
        --volume "$repo_root:/work" --workdir /work \
        --entrypoint /bin/sh kaczmarj/apptainer:latest \
        -c 'apptainer build "$1" "$4" && chown "$2:$3" "$1"' \
        sh "$relative_image" "$(id -u)" "$(id -g)" "docker/$model.def"
else
    apptainer build --ignore-fakeroot-command "$relative_image" "docker/$model.def"
fi

[[ -f "$build_dir/$model.sif" ]] || die "Build did not produce an image."
mv -f -- "$build_dir/$model.sif" "$output"
echo "Built $output"
echo "To share it with a cluster without file transfer, run docker/push_sif.sh."
