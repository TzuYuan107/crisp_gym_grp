#!/usr/bin/env bash
# Push docker/pi05.sif (built by build_pi05_sif.sh) to an OCI registry via ORAS.
#
# On the cluster (Apptainer only, no Docker needed), pull it with:
#   apptainer pull docker/pi05.sif oras://<registry>/<account>/pi05:<tag>
# For a private package, log in first on the cluster:
#   apptainer registry login --username <account> oras://<registry>
set -euo pipefail

usage() {
    echo "Usage: $0 --account ACCOUNT [--registry HOST] [--tag TAG] [--docker]"
    echo "  --account   Registry user or organisation (required, or set REGISTRY_ACCOUNT)"
    echo "  --registry  OCI registry host (default: ghcr.io, or REGISTRY_HOST)"
    echo "  --tag       Image tag (default: 0.7.0, or SIF_TAG)"
    echo "  --docker    Push with a Docker-hosted Apptainer when Apptainer is not installed"
    echo "You will be prompted for a password or access token with write permission."
}

account="${REGISTRY_ACCOUNT:-}"
registry="${REGISTRY_HOST:-ghcr.io}"
tag="${SIF_TAG:-0.7.0}"
use_docker=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --account) account="${2:?--account needs a value}"; shift 2 ;;
        --registry) registry="${2:?--registry needs a value}"; shift 2 ;;
        --tag) tag="${2:?--tag needs a value}"; shift 2 ;;
        --docker) use_docker=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$account" ]]; then
    echo "An account is required." >&2; usage >&2; exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sif="$script_dir/pi05.sif"
# Registry repository names must be lowercase.
account="${account,,}"
target="oras://$registry/$account/pi05:$tag"

[[ -f "$sif" ]] || { echo "Missing $sif. Run docker/build_pi05_sif.sh first." >&2; exit 1; }

echo "Pushing $sif to $target"
if [[ "$use_docker" == true ]]; then
    command -v docker >/dev/null || { echo "Docker is required for --docker." >&2; exit 1; }
    # Credentials live only in this throwaway container.
    docker run --rm -it \
        --volume "$sif:/pi05.sif:ro" \
        --entrypoint /bin/sh kaczmarj/apptainer:latest \
        -c 'apptainer registry login --username "$1" "oras://$2" && apptainer push /pi05.sif "$3"' \
        sh "$account" "$registry" "$target"
else
    command -v apptainer >/dev/null || { echo "Apptainer is required; use --docker if it is not installed." >&2; exit 1; }
    apptainer registry login --username "$account" "oras://$registry"
    apptainer push "$sif" "$target"
fi

echo "Pushed. On the cluster, run:"
echo "  apptainer pull docker/pi05.sif $target"
if [[ "$registry" == "ghcr.io" ]]; then
    # GitHub offers no API to set visibility; a new package is private until changed in the web UI.
    echo "New GHCR packages are private. For an anonymous pull on the cluster, make it public once at:"
    echo "  https://github.com/users/$account/packages/container/pi05/settings  (Danger Zone -> Change visibility)"
    echo "Organisation packages: https://github.com/orgs/$account/packages/container/pi05/settings"
    echo "Otherwise run 'apptainer registry login --username $account oras://ghcr.io' on the cluster first."
fi
