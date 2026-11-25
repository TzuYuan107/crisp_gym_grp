#!/usr/bin/env bash
# Shared prompts for build_sif.sh and push_sif.sh. Source this file; do not run it.

die() { echo "$*" >&2; exit 1; }

# ask VAR "Question" [default]: read a line into VAR, using the default on empty input.
ask() {
    local __var="$1" __prompt="$2" __default="${3:-}" __reply
    if [[ -n "$__default" ]]; then __prompt="$__prompt [default: $__default, press Enter to accept]"; fi
    read -r -p "$__prompt: " __reply || die "No input available; run this script in a terminal."
    printf -v "$__var" '%s' "${__reply:-$__default}"
}

# choose VAR "Question" option...: numbered menu, stores the selected option in VAR.
choose() {
    local __var="$1" __prompt="$2" __reply i
    shift 2
    local -a __options=("$@")
    echo "$__prompt"
    for i in "${!__options[@]}"; do echo "  $((i + 1))) ${__options[$i]}"; done
    while true; do
        read -r -p "Select 1-${#__options[@]}: " __reply || die "No input available; run this script in a terminal."
        if [[ "$__reply" =~ ^[0-9]+$ ]] && (( __reply >= 1 && __reply <= ${#__options[@]} )); then
            printf -v "$__var" '%s' "${__options[$((__reply - 1))]}"
            return
        fi
        echo "Invalid choice."
    done
}

# choose_runner USE_DOCKER_VAR: ask whether to use Apptainer directly or a Docker-hosted one.
choose_runner() {
    local __choice
    choose __choice "Run with:" \
        "Apptainer on this machine (e.g. cluster)" \
        "Docker (local machine without Apptainer; runs a privileged container)"
    if [[ "$__choice" == Docker* ]]; then
        command -v docker >/dev/null || die "Docker is not installed."
        printf -v "$1" '%s' true
    else
        command -v apptainer >/dev/null || die "Apptainer is not installed; choose the Docker option."
        printf -v "$1" '%s' false
    fi
}
