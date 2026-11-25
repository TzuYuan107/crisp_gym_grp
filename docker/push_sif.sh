#!/usr/bin/env bash
# Interactively push a docker/<model>.sif (built by build_sif.sh) to an OCI registry via ORAS.
#
# On the cluster (Apptainer only, no Docker needed), pull it with:
#   apptainer pull docker/<model>.sif oras://<registry>/<account>/<model>:<tag>
# For a private package, log in first on the cluster:
#   apptainer registry login --username <account> oras://<registry>
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=docker/sif_common.sh
source "$script_dir/sif_common.sh"

models=()
for sif in "$script_dir"/*.sif; do
    [[ -e "$sif" ]] && models+=("$(basename "$sif" .sif)")
done
(( ${#models[@]} > 0 )) || die "No .sif images in $script_dir. Run docker/build_sif.sh first."

choose model "Which image do you want to push?" "${models[@]}"
ask account "Registry account (your GitHub username for ghcr.io)" "${REGISTRY_ACCOUNT:-}"
[[ -n "$account" ]] || die "An account is required."
ask registry "Registry host" "${REGISTRY_HOST:-ghcr.io}"
ask tag "Image tag" "${SIF_TAG:-0.7.0}"
choose_runner use_docker

sif="$script_dir/$model.sif"
# Registry repository names must be lowercase.
account="${account,,}"
target="oras://$registry/$account/$model:$tag"

echo "Pushing $sif to $target"
echo "You will be prompted for a password: paste an access token with write permission."
if [[ "$use_docker" == true ]]; then
    # Credentials live only in this throwaway container.
    docker run --rm -it \
        --volume "$sif:/image.sif:ro" \
        --entrypoint /bin/sh kaczmarj/apptainer:latest \
        -c 'apptainer registry login --username "$1" "oras://$2" && apptainer push /image.sif "$3"' \
        sh "$account" "$registry" "$target"
else
    apptainer registry login --username "$account" "oras://$registry"
    apptainer push "$sif" "$target"
fi

echo "Pushed. On the cluster, run:"
echo "  apptainer pull docker/$model.sif $target"
if [[ "$registry" == "ghcr.io" ]]; then
    # GitHub offers no API to set visibility; a new package is private until changed in the web UI.
    echo "New GHCR packages are private. For an anonymous pull on the cluster, make it public once at:"
    echo "  https://github.com/users/$account/packages/container/$model/settings  (Danger Zone -> Change visibility)"
    echo "Organisation packages: https://github.com/orgs/$account/packages/container/$model/settings"
    echo "Otherwise run 'apptainer registry login --username $account oras://ghcr.io' on the cluster first."
fi
