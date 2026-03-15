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

COPY environment.yml /tmp/environment.yml
RUN micromamba env create -f /tmp/environment.yml && \
    micromamba run -n fiji-stitcher python -m pip uninstall -y torch torchvision torchaudio || true && \
    micromamba run -n fiji-stitcher python -m pip install --no-cache-dir \
      torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
      --index-url https://download.pytorch.org/whl/cu126 && \
    micromamba clean --all --yes

ENV MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

RUN wget https://downloads.imagej.net/fiji/stable/fiji-stable-linux64-jdk.zip -O /tmp/fiji.zip && \
    unzip /tmp/fiji.zip -d /opt && \
    rm /tmp/fiji.zip

ENV FIJI_PATH=/opt/Fiji.app
ENV FIJI_EXE=/opt/Fiji.app/ImageJ-linux64
ENV PYTHONPATH=/app

COPY . /app

CMD ["micromamba", "run", "-n", "fiji-stitcher", "python", "main.py", "--batch"]
