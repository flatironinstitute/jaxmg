FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.11 python3.11-venv \
    python3.12 python3.12-venv \
    python3.12 python3.13-venv \
    python3.12 python3.14-venv \
    python3-build \
    python3-pip \
    cmake \
    git && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/* && \
    ln -sn /usr/local/cuda/include /usr/local/cuda/extras/CUPTI/include
