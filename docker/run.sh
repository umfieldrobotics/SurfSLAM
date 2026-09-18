#!/usr/bin/env bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
IMAGE_TAG=surfslam:slam_$(whoami)   # must match IMAGE_TAG in build_user.sh
CONTAINER_NAME=surfslam_$(whoami)

# Where cfg/dataset_paths.yaml entries get mounted inside the container.
# surfslam/common/dataset_paths.py re-points registry entries here when it
# detects it is running in Docker.
CONTAINER_DATA_ROOT=/data
DATASET_PATHS_FILE="${SURF_DATASET_PATHS:-$SCRIPT_DIR/../cfg/dataset_paths.yaml}"

capabilities_str=\""capabilities=compute,utility,graphics,display\""


DOCKER_OPTIONS=""
DOCKER_OPTIONS+="-it "
DOCKER_OPTIONS+="-e DISPLAY=$DISPLAY "
DOCKER_OPTIONS+="-v /tmp/.X11-unix:/tmp/.X11-unix "
DOCKER_OPTIONS+="-v $SCRIPT_DIR/../:/home/$(whoami)/SurfSLAM "
DOCKER_OPTIONS+="-v $HOME/.Xauthority:/home/$(whoami)/.Xauthority "
DOCKER_OPTIONS+="-v /etc/group:/etc/group:ro "
# Tell the code inside the container to resolve dataset roots under $CONTAINER_DATA_ROOT.
DOCKER_OPTIONS+="-e SURF_IN_DOCKER=1 "
DOCKER_OPTIONS+="-e SURF_DOCKER_DATA_ROOT=$CONTAINER_DATA_ROOT "
DOCKER_OPTIONS+="--name $CONTAINER_NAME "
DOCKER_OPTIONS+="--privileged "
DOCKER_OPTIONS+="--gpus=all "
DOCKER_OPTIONS+="-e NVIDIA_DRIVER_CAPABILITIES=all "
DOCKER_OPTIONS+="--net=host "
DOCKER_OPTIONS+="--runtime=nvidia "
DOCKER_OPTIONS+="-e SDL_VIDEODRIVER=x11 "
DOCKER_OPTIONS+="-u $(id -u):$(id -g) "
DOCKER_OPTIONS+="-w /home/$(whoami)/SurfSLAM "
DOCKER_OPTIONS+="--shm-size 32G "

for cam in /dev/video*; do
  DOCKER_OPTIONS+="--device=${cam} "
done

for f in /mnt/*; do
  DOCKER_OPTIONS+="-v $f:/mnt/$(basename $f) "
done

# Legacy single-directory mount, kept for convenience when set.
if [ -n "${SURFSLAM_DATASET_PATH:-}" ]; then
  DOCKER_OPTIONS+="-v $SURFSLAM_DATASET_PATH:/home/$(whoami)/data "
fi

# --- dataset mounts, from cfg/dataset_paths.yaml -----------------------------
# Each `key: /host/path` entry is bind-mounted at $CONTAINER_DATA_ROOT/<key>, so
# `suds_slam: /mnt/big_disk/SUDS_SLAM` becomes /data/suds_slam. Keys left null
# (or pointing at something that doesn't exist) are skipped.
if [ -f "$DATASET_PATHS_FILE" ]; then
  echo "Dataset mounts (from $DATASET_PATHS_FILE):"
  while read -r key value; do
    value="${value%\"}"; value="${value#\"}"
    value="${value%\'}"; value="${value#\'}"

    # YAML nulls, bare ~ means $HOME
    case "$value" in
      null|Null|NULL|"~"|"") continue ;;
    esac

    value="${value/#\~/$HOME}"

    if [ ! -d "$value" ]; then
      echo "  $key: skipped, $value is not a directory"
      continue
    fi

    DOCKER_OPTIONS+="-v $value:$CONTAINER_DATA_ROOT/$key "
    echo "  $key: $value -> $CONTAINER_DATA_ROOT/$key"
  done < <(
    sed -e 's/#.*$//' -e 's/[[:space:]]*$//' "$DATASET_PATHS_FILE" \
      | grep -E '^[A-Za-z_][A-Za-z0-9_]*:[[:space:]]*[^[:space:]]' \
      | sed -E 's/^([A-Za-z_][A-Za-z0-9_]*):[[:space:]]*/\1 /'
  )
else
  echo "No $DATASET_PATHS_FILE - no dataset mounts. Copy cfg/dataset_paths.example.yaml to it."
fi

echo $CONTAINER_NAME

if [ ${1:-""} == "restart" ]; then 
  echo "Restarting Container"
  docker rm -f $CONTAINER_NAME
  docker run $DOCKER_OPTIONS $IMAGE_TAG /bin/bash
# https://stackoverflow.com/questions/38576337/how-to-execute-a-bash-command-only-if-a-docker-container-with-a-given-name-does
elif [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then # If container isn't running
    
    # If it exists, but needs to be started
    if [  "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then

          echo "Resuming Container"
          docker start $CONTAINER_NAME
          docker exec -it -w /home/$(whoami)/SurfSLAM $CONTAINER_NAME /entrypoint.sh
    else
      echo "Running Container"
      docker run $DOCKER_OPTIONS $IMAGE_TAG
    fi
else
  echo "Attaching to existing container"
  docker exec -it -w /home/$(whoami)/SurfSLAM $CONTAINER_NAME /entrypoint.sh
fi
