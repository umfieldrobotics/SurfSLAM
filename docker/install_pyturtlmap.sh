#!/usr/bin/env bash
#
# Build and install the pyturtlmap python bindings inside the container.
# Called by entrypoint.sh on every launch, but exits immediately if the
# bindings are already importable, so it only does work on first launch
# (or after the module is removed). Safe to re-run by hand at any time:
#   ./docker/install_pyturtlmap.sh          # no-op if already installed
#   FORCE=1 ./docker/install_pyturtlmap.sh  # rebuild unconditionally

set -e

if [ -z "${FORCE:-}" ] && python3 -c "import pyturtlmap" > /dev/null 2>&1; then
  exit 0
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SRC_DIR="$SCRIPT_DIR/../turtlmap"

# Separate build tree from turtlmap/build so a host-side build (with host
# paths baked into its CMake cache) never collides with the container build.
BUILD_DIR="$SRC_DIR/build-docker"

echo "pyturtlmap not importable; building the TurtlMap python bindings (first launch only)..."
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON
cmake --build "$BUILD_DIR" --target pyturtlmap -j"$(nproc)"
cmake --install "$BUILD_DIR/turtlmap"

python3 -c "import pyturtlmap" \
  && echo "pyturtlmap installed to $HOME/.local/lib/python3.*/site-packages"
