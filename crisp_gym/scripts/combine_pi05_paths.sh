#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

python crisp_gym/scripts/combine_safety_paths.py \
    --base-dir "outputs/safety/VLA-SF(Diverse)" \
    "outputs/safety/VLA-SF(Diverse)/trajectory_actual.npy" \
    "outputs/safety/VLA (Normal)/trajectory_actual.npy" \
    "outputs/safety/VLA-SF (Normal)/trajectory_actual.npy" \
    --offset-x -0.12 \
    --offset-y 0.00 \
    --ellipsoid-scale 1.11 \
    --point-alpha 0.4 \
    --xlim 0.2 0.7 \
    --colors tab:blue darkorange orangered \
    --labels "VLA-SF (Diverse)" "VLA (normal)" "VLA-SF (normal)" \
    --legend-loc "lower left" \
    --legend-bbox 0.0 0.03 \
    --legend-order "VLA (normal)" "VLA-SF (normal)" "VLA-SF (Diverse)" "obstacle point cloud" "obstacle area" "goal area" "normal distribution" "safe deployment distribution" \
    --highlight-endpoints "outputs/safety/VLA-SF(Diverse)/trajectory_actual.npy" "outputs/safety/VLA (Normal)/trajectory_actual.npy" \
    --highlight-padding 4.2 \
    --envelope-trajectories "outputs/safety/VLA (Normal)/trajectory_actual.npy" \
    --envelope-color orange \
    --envelope-alpha 0.45 \
    --deployment-region \
    --deployment-scale 1.4 \
    --deployment-xlim 0.25 0.75 \
    --deployment-color skyblue \
    --deployment-alpha 0.10 \
    --out outputs/safety/combined_pi05_MM_ND_novlsa.png
