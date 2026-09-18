# Docker Setup

All Python in this repo runs inside a Docker container. Everything lives in [docker/](../docker/).

The setup is two images:

1. **Base image** (`sethgi/surfslam:slam`) — CUDA 12.8 + PyTorch 2.9.1 + all Python/apt dependencies, including flash-attn compiled from source. Slow to build, user-agnostic, built once per machine and shared.
2. **User image** (`surfslam:slam_$(whoami)`) — thin layer on the base that adds a user matching your host UID, so files written from inside the container are owned by you.

## Quick start

```bash
./docker/build_base.sh    # once per machine (slow; skipped by build_user.sh if already present)
./docker/build_user.sh    # once per user
./docker/run.sh           # start / resume / attach
```

`run.sh` is idempotent: it starts the container if needed, resumes it if stopped, or attaches another shell if it's already running. Use `./docker/run.sh restart` to remove and recreate the container (needed after rebuilding an image or changing mounts).

## What run.sh mounts

- The repo, at `/home/$(whoami)/SurfSLAM`
- Every directory under `/mnt/*`, at the same path
- Each entry in `cfg/dataset_paths.yaml` (`key: /host/path`), at `/data/<key>`; null or missing paths are skipped. Override the file with `SURF_DATASET_PATHS`. See [data_format.md](data_format.md).
- `$SURFSLAM_DATASET_PATH` (if set), at `/home/$(whoami)/data` (legacy single-directory mount)
- X11 sockets/auth for GUI apps

It also sets `SURF_IN_DOCKER=1` and `SURF_DOCKER_DATA_ROOT=/data` so `surfslam/common/dataset_paths.py` resolves dataset roots under `/data` when running in the container.

## Rebuilding the base image (e.g. different NVIDIA compute architecture)

The base image compiles flash-attn from source. Since no GPU is visible at `docker build` time, the build can't detect your GPU and compiles for a default set of compute architectures — if your GPU's architecture isn't in that set (or you switch machines, e.g. Ampere → Ada → Blackwell), flash-attn will fail at runtime with missing-kernel errors and the base image must be rebuilt for your architecture.

Set `TORCH_CUDA_ARCH_LIST` for the build, e.g. by adding to `container_base.Dockerfile` before the flash-attn install:

```dockerfile
ENV TORCH_CUDA_ARCH_LIST="8.6"   # 8.0/8.6 Ampere, 8.9 Ada, 9.0 Hopper, 12.0 Blackwell
```

then rebuild and recreate everything:

```bash
./docker/build_base.sh
./docker/build_user.sh
./docker/run.sh restart
```

Find your architecture with `nvidia-smi --query-gpu=compute_cap --format=csv`. Listing multiple values (`"8.6;8.9"`) builds a fatter image that works on all of them.

If you change the base image tag, keep it consistent in three places: `IMAGE_TAG` in `build_base.sh`, `BASE_IMAGE` in `build_user.sh`, and the `BASE_IMAGE` default in `container_user.Dockerfile`.
