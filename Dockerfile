# syntax = docker/dockerfile:1.3
FROM mambaorg/micromamba:1.5.10

USER root

RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制环境文件（变化较少）
COPY environment.yml /tmp/environment.yml

# 使用缓存挂载创建环境（避免重复下载）
RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba env create -f /tmp/environment.yml && \
    micromamba clean --all --yes

ENV MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

ARG DOWNLOAD_FIJI=0
RUN if [ "$DOWNLOAD_FIJI" = "1" ]; then \
      wget https://downloads.imagej.net/fiji/stable/fiji-stable-linux64-jdk.zip -O /tmp/fiji.zip && \
      unzip /tmp/fiji.zip -d /opt && \
      rm /tmp/fiji.zip; \
    fi

ENV FIJI_PATH=/opt/Fiji.app
ENV FIJI_EXE=/opt/Fiji.app/ImageJ-linux64
ENV PYTHONPATH=/app

# 复制代码（最后，变化最频繁）
COPY . /app

CMD ["micromamba", "run", "-n", "fiji-stitcher", "python", "main.py", "--batch"]