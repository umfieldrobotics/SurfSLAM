#!/usr/bin/env bash
#
# Build the per-user image: the shared base plus a user matching your host
# UID, so files written from inside the container are owned by you.


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

BASE_IMAGE="sethgi/surfslam:slam"
IMAGE_TAG="surfslam:slam_$(whoami)"   # must match IMAGE_TAG in run.sh

if ! docker image inspect "$BASE_IMAGE" > /dev/null 2>&1; then
  echo "Base image $BASE_IMAGE not found; building it first (this takes a while)."
  "$SCRIPT_DIR/build_base.sh" || exit 1
fi

DOCKER_OPTIONS=""
DOCKER_OPTIONS+="-t $IMAGE_TAG "
DOCKER_OPTIONS+="-f $SCRIPT_DIR/container_user.Dockerfile "
DOCKER_OPTIONS+="--build-arg BASE_IMAGE=$BASE_IMAGE "
DOCKER_OPTIONS+="--build-arg USER_ID=$(id -u) --build-arg USER_NAME=$(whoami) "

DOCKER_CMD="docker build $DOCKER_OPTIONS $SCRIPT_DIR"
echo $DOCKER_CMD
exec $DOCKER_CMD
