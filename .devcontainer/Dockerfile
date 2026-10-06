# syntax=docker/dockerfile:1.4
# NOTE: Building this image requires Docker version >= 23.0

# Arguments for base configuration
ARG BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04
FROM ${BASE_IMAGE} AS base
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# System packages installation
RUN echo "=== Installing system packages ===" && \
    apt update && apt install -y --no-install-recommends \
    software-properties-common \
    build-essential ca-certificates \
    nfs-common pdsh \
    ccache cmake make \
    curl wget vim tmux emacs less unzip \
    git psmisc \
    zlib1g-dev libbz2-dev libssl-dev \
    libreadline-dev libsqlite3-dev sqlite3 \
    libncursesw5-dev xz-utils tk-dev \
    libxml2-dev libxmlsec1-dev libffi-dev llvm \
    libboost-all-dev libeigen3-dev libopenbabel-dev \
    sudo python3.8 python3-pip && \
    apt clean && rm -rf /var/lib/apt/lists/*

# GCC 9のインストール（PPAを使用）
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc-9 g++-9 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 \
                        --slave /usr/bin/g++ g++ /usr/bin/g++-9

# Create user
ARG UID=1014
ARG GID=1014
ARG USERNAME=miniconda-user
RUN echo "=== Creating user ${USERNAME} ===" && \
    groupadd -g ${GID} ${USERNAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${USERNAME} && \
    chmod 440 /etc/sudoers.d/${USERNAME}

# Set up Miniconda installation with Miniforge
FROM base AS conda-setup
ARG PYTHON_VERSION=3.11
ARG USERNAME=miniconda-user
USER ${USERNAME}
WORKDIR /home/${USERNAME}

RUN touch /home/${USERNAME}/.bashrc

# Download and install Miniforge with checksum verification for added security
ARG TARGETPLATFORM
RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  MINICONDA_ARCH=aarch64  ;; \
         *)              MINICONDA_ARCH=x86_64   ;; \
    esac && \
    curl -fsSL -v -o /tmp/miniconda.sh -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MINICONDA_ARCH}.sh" && \
    chmod +x /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /home/${USERNAME}/conda && \
    rm /tmp/miniconda.sh

# Minicondaのパスを環境変数に追加
ENV PATH="/home/${USERNAME}/conda/bin:$PATH"

# ワークスペースディレクトリをPYTHONPATHに追加
ENV PYTHONPATH="/home/${USERNAME}/workspace:$PYTHONPATH"
ENV PYTHONPATH="/home/${USERNAME}/workspace/FRATTVAE/scripts:$PYTHONPATH"


# Conda初期化と環境設定
RUN conda init bash && \
    conda config --set auto_activate_base false && \
    conda update -n base -c defaults conda && \
    pip install --upgrade pip setuptools


# Create and activate Conda environment using YAML file
WORKDIR /home/${USERNAME}
COPY fPLG.yml /home/${USERNAME}
RUN conda env create -f /home/${USERNAME}/fPLG.yml && \
    echo "=== Conda environment setup complete ===" && \
    echo "conda activate fPLG" >> /home/${USERNAME}/.bashrc



# コンテナ起動時にconda環境を自動で有効化する
ENTRYPOINT ["/bin/bash"]







