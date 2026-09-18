#!/bin/sh

sudo ldconfig

# On first launch, build + install the pyturtlmap python bindings from the
# mounted repo. No-op on later launches (see docker/install_pyturtlmap.sh).
REPO_DIR="$HOME/SurfSLAM"
if [ -f "$REPO_DIR/docker/install_pyturtlmap.sh" ]; then
  bash "$REPO_DIR/docker/install_pyturtlmap.sh" || \
    echo "WARNING: pyturtlmap build failed; run docker/install_pyturtlmap.sh manually."
fi

exec /bin/bash
