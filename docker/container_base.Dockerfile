ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux
ENV TZ=America

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# --------------------------------------------------------------------------- #
# Basic system packages (no ROS here)
# --------------------------------------------------------------------------- #
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    tzdata \
    ca-certificates \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    tmux \
    htop \
    pkg-config \
    ninja-build \
    gdb \
    unzip \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libgl1-mesa-dev \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxi6 \
    libxinerama1 \
    libfontconfig1 \
    libsm6 \
    libgtk-3-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libeigen3-dev \
    libboost-all-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libtbb-dev \
    libyaml-cpp-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-distutils \
    python3-numpy \
    pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------- #
# (Optional) GTSAM from source if you need it inside the image
# Comment this block out if you are providing GTSAM another way.
# --------------------------------------------------------------------------- #
RUN wget -qO /tmp/gtsam-4.2.zip https://github.com/borglab/gtsam/archive/refs/tags/4.2.zip && \
    cd /tmp && unzip gtsam-4.2.zip && cd gtsam-4.2 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DGTSAM_TANGENT_PREINTEGRATION=OFF \
    -DGTSAM_USE_SYSTEM_EIGEN=ON && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf /tmp/gtsam-4.2* /tmp/gtsam-4.2.zip

# --------------------------------------------------------------------------- #
# Python packages (Open3D + misc, avoid PyYAML uninstall issue)
# --------------------------------------------------------------------------- #
# PyTorch 2.9.1 + CUDA 12.8 (cu128) wheels ship SASS/PTX for Blackwell (sm_120),
# so torch/vision/audio all match and run on the RTX PRO 6000. torch.version.cuda
# is 12.8, matching nvcc from the CUDA 12.8 base image (pytorch3d/faiss verify this).
RUN python3 -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
# Lock the torch trio to the EXACT cu128 local build. Later `pip install`s
# (timm, kornia, kaolin, ...) resolve against default PyPI, whose `torch 2.9.1`
# is also a cu128 build pulling nvidia-*-cu12 (12.8) runtime wheels. A bare
# `torch==2.9.1` pin can still let pip reshuffle the install set, so keep the
# +cu128 pin aligned with the base image.
RUN printf 'torch==2.9.1+cu128\ntorchvision==0.24.1+cu128\ntorchaudio==2.9.1+cu128\n' > /opt/pip-constraints.txt
ENV PIP_CONSTRAINT=/opt/pip-constraints.txt
# Expose the cu128 wheel index to every later pip step so the +cu128 pin above is
# actually satisfiable (PyPI has no +cu128 build). Without this, resolving any
# torch-dependent package (timm/kornia/kaolin) fails with ResolutionImpossible.
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128
RUN python3 -m pip install --upgrade pip setuptools \
    && python3 -m pip install \
        pybind11 \
        scipy \
        numpy \
        --no-cache-dir --ignore-installed PyYAML "open3d==0.19.0" \
        matplotlib \
        h5py \
        opencv-python \
        imageio \
        scikit-image \
        opt_einsum \
        timm==1.0.15 \
        attrdict \
        kornia \
        small_gicp \
        point-cloud-utils

RUN cd /tmp \
    && wget https://github.com/Kitware/CMake/releases/download/v3.31.3/cmake-3.31.3-linux-x86_64.sh \
    && chmod +x cmake-3.31.3-linux-x86_64.sh \
    && ./cmake-3.31.3-linux-x86_64.sh --skip-license --prefix=/usr/local \
    && rm cmake-3.31.3-linux-x86_64.sh

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    swig \
    && rm -rf /var/lib/apt/lists/*

# faiss GPU: build kernels for Ampere (80/86), Hopper (90) and Blackwell (120,
# RTX PRO 6000). nvcc from the CUDA 13 base image is required for sm_120.
RUN cd /tmp \
    && wget https://github.com/facebookresearch/faiss/archive/refs/tags/v1.13.0.tar.gz \
    && tar -xvf v1.13.0.tar.gz \
    && cd faiss-1.13.0 \
    && mkdir build && cd build \
    && cmake .. -DFAISS_ENABLE_GPU=ON \
        -DFAISS_ENABLE_PYTHON=ON \
        -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DFAISS_OPT_LEVEL=avx2 \
        -DCMAKE_CUDA_ARCHITECTURES="80;86;90;120" \
    && make -j$(nproc) \
    && cd faiss/python \
    && python3 -m pip install . \
    && rm -rf /tmp/faiss

# Install Node.js and npm for three.js web visualization
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get -y install nodejs \
    && rm -rf /var/lib/apt/lists/*

# Target GPU archs for source-built CUDA extensions (pytorch3d, kaolin):
# Ampere (8.0/8.6), Hopper (9.0) and Blackwell (12.0). +PTX gives forward compat.
# FORCE_CUDA ensures kernels build even though no GPU is visible during `docker build`.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;12.0+PTX"
ENV FORCE_CUDA=1

# pytorch3d from source so it compiles against torch 2.9.1 / cu128 for sm_120
# (the old +pt2.4.0cu124 prebuilt wheel is incompatible with this torch).
# System-wide (not --user) so every user built on this base sees it.
RUN cd /tmp \
    && git clone https://github.com/facebookresearch/pytorch3d.git \
    && cd pytorch3d \
    && python3 -m pip install --no-build-isolation . \
    && cd /tmp \
    && rm -rf /tmp/pytorch3d

RUN pip3 install h5py attridict

RUN python3 -m pip install "numpy<2" evo

# kaolin: NVIDIA publishes no prebuilt wheel for torch 2.9.1+cu128 (their S3 index
# only covers older torch/cuda combos), so build v0.18.0 from source for Blackwell.
# Inherits TORCH_CUDA_ARCH_LIST / FORCE_CUDA from the ENV above; IGNORE_TORCH_VER
# bypasses kaolin's strict torch-version pin. This is the most likely step to need
# iteration against torch 2.9 -- if it breaks, bump the kaolin tag or pin torch lower.
# If NVIDIA later ships a matching wheel, revert to:
#   pip install kaolin==<ver> -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.9.1_cu128.html
RUN git clone --recursive https://github.com/NVIDIAGameWorks/kaolin.git /tmp/kaolin \
    && cd /tmp/kaolin \
    && git checkout v0.18.0 \
    && git submodule update --init --recursive \
    && python3 -m pip install cython \
    && IGNORE_TORCH_VER=1 python3 -m pip install --no-cache-dir --no-build-isolation . \
    && cd / \
    && rm -rf /tmp/kaolin
