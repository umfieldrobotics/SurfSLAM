#!/usr/bin/env bash
# Reproduce the paper experiments: run SurfSLAM on all three TBNMS scenes,
# 5 repeats each, with the stereo ablation overrides.
# Requires the SUDS_SLAM dataset mounted at the paths in cfg/tbnms/*.yaml
# (see cfg/dataset_paths.yaml and docs/docker.md).

CONFIGS=(tbnms/mono_long.yaml tbnms/mono_boiler.yaml tbnms/mono_start.yaml)

OVERRIDES=stereo_ablation.yaml

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

overrides=$SCRIPT_DIR/../cfg/$OVERRIDES
(
  cd "$SCRIPT_DIR/../examples/" || exit 1
  for c in "${CONFIGS[@]}"; do
    config_path=$SCRIPT_DIR/../cfg/$c

    python3 run_surfslam.py "$config_path" \
      --overrides "$overrides" \
      --num_repeats 5
  done
)