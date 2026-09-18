#!/usr/bin/env bash
# Reproduce the TBNMS comparison: SurfSLAM ablations + baselines swept over two
# trajectory-pose-window sizes, plus the TurtlMap baseline.
#
# --log_dir_root SETS system.log_dir_prefix rather than appending to it. That matters: the
# ablation configs override log_dir_prefix to ./ablation_outputs/ in their `changes:` block,
# but the baseline configs (cfg/tbnms/*.yaml) have no `changes:` block and fall back to
# cfg/defaults.yaml's ./outputs/. Appending split one arm across two trees and made the
# baselines invisible to analysis/ablation_eval_cfg.yaml. One arm = one tree.
#
# TurtlMap runs ONCE, outside the window sweep: cfg/turtlmap_baseline_defaults.yaml disables
# loop closure and frame-to-frame registration, so nothing ever queries the trajectory pose
# window and the 10k/200k distinction cannot affect it.

# Each config writes to fast local disk, then its finished run dirs are relocated to
# `slam_results` from cfg/dataset_paths.yaml as soon as that config completes -- keeping the
# local disk free during a ~11 h job. Set SLAM_RESULTS_ROOT to override; leave `slam_results`
# unset (null) to keep everything local.

# Swept over QUEUE_SIZES.
CONFIGS=(
  tbnms_ablations/monohansett_boiler_dvl_ablation.yaml
  tbnms_ablations/monohansett_boiler_barometer_ablation.yaml
  tbnms_ablations/monohansett_engine_dvl_ablation.yaml
  tbnms_ablations/monohansett_engine_barometer_ablation.yaml
  tbnms_ablations/monohansett_long_dvl_ablation.yaml
  tbnms_ablations/monohansett_long_barometer_ablation.yaml
  tbnms/monohansett_boiler.yaml
  tbnms/monohansett_engine.yaml
  tbnms/monohansett_long.yaml
)

# Run once, NOT swept -- see header.
TURTLMAP_CONFIGS=(
  tbnms_turtlmap/monohansett_boiler.yaml
  tbnms_turtlmap/monohansett_engine.yaml
  tbnms_turtlmap/monohansett_long.yaml
)

QUEUE_SIZES=(10000 200000)

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

(
  cd "$SCRIPT_DIR/../" || exit 1

  log_dir="./ablation_outputs/logs"
  mkdir -p "$log_dir"

  failures=()

  # Where finished runs get parked. Single source of truth is cfg/dataset_paths.yaml.
  REMOTE_ROOT="${SLAM_RESULTS_ROOT:-$(python3 -c "import yaml;print(yaml.safe_load(open('cfg/dataset_paths.yaml')).get('slam_results') or '')" 2>/dev/null)}"
  if [ -n "$REMOTE_ROOT" ]; then
    echo "=== Finished runs will be moved to $REMOTE_ROOT"
  else
    echo "=== slam_results unset; keeping all results on local disk"
  fi

  # Move one tree's completed run dirs off local disk. Called after each config so the
  # local footprint stays at roughly one config's worth rather than the whole sweep.
  # A failed move leaves the local copy in place and is reported -- never silently dropped.
  stash_results() {
    local tree="$1"                       # e.g. window_10000, turtlmap
    local src="./ablation_outputs/$tree"
    [ -z "$REMOTE_ROOT" ] && return 0
    [ -d "$src" ] || return 0

    mkdir -p "$REMOTE_ROOT/$tree" || { echo "!!! cannot create $REMOTE_ROOT/$tree"; failures+=("stash mkdir $tree"); return 1; }
    for d in "$src"/*/; do
      [ -d "$d" ] || continue
      local base
      base=$(basename "$d")
      if [ -e "$REMOTE_ROOT/$tree/$base" ]; then
        echo "!!! $REMOTE_ROOT/$tree/$base already exists; leaving local copy"
        failures+=("stash collision $tree/$base")
        continue
      fi
      if mv "$d" "$REMOTE_ROOT/$tree/"; then
        echo "    moved $tree/$base -> $REMOTE_ROOT/$tree/"
      else
        echo "!!! failed to move $tree/$base; leaving local copy"
        failures+=("stash failed $tree/$base")
      fi
    done
  }

  for qsize in "${QUEUE_SIZES[@]}"; do
    arm="window_${qsize}"
    echo "===================================================================="
    echo "=== ARM $arm  ($(date))"
    echo "===================================================================="

    for c in "${CONFIGS[@]}"; do
      config_path=$SCRIPT_DIR/../cfg/$c
      name=$(basename "$c" .yaml)

      echo "--- $arm / $name  ($(date))"
      python3 examples/run_surfslam.py "$config_path" \
        --num_repeats 5 \
        --traj_queue_size "$qsize" \
        --log_dir_root "./ablation_outputs/$arm/" 2>&1 | tee "$log_dir/${arm}_${name}.log"

      # tee is last in the pipe, so consult PIPESTATUS for python's status.
      rc=${PIPESTATUS[0]}
      if [ "$rc" -ne 0 ]; then
        echo "!!! $arm / $name exited with status $rc"
        failures+=("$arm/$name (exit $rc)")
      fi

      stash_results "$arm"
    done
  done

  # TurtlMap baseline: one pass, no window sweep.
  echo "===================================================================="
  echo "=== TURTLMAP baseline  ($(date))"
  echo "===================================================================="

  for c in "${TURTLMAP_CONFIGS[@]}"; do
    config_path=$SCRIPT_DIR/../cfg/$c
    name=$(basename "$c" .yaml)

    echo "--- turtlmap / $name  ($(date))"
    python3 examples/run_surfslam.py "$config_path" \
      --num_repeats 5 \
      --log_dir_root "./ablation_outputs/turtlmap/" 2>&1 | tee "$log_dir/turtlmap_${name}.log"

    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
      echo "!!! turtlmap / $name exited with status $rc"
      failures+=("turtlmap/$name (exit $rc)")
    fi

    stash_results "turtlmap"
  done

  echo "===================================================================="
  if [ ${#failures[@]} -eq 0 ]; then
    echo "=== All configs completed ($(date))"
  else
    echo "=== ${#failures[@]} config(s) failed:"
    printf '===   %s\n' "${failures[@]}"
  fi
  echo "=== Logs in $log_dir"
  [ -n "$REMOTE_ROOT" ] && echo "=== Results in $REMOTE_ROOT"
)
