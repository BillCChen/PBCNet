# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set default RUN shell to /bin/bash
SHELL ["/bin/bash", "-cu"]


# Set environment variables
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8


# Install basic packages for compiling and building
ENV DEBIAN_FRONTEND=noninteractive
# change apt source to Tsinghua mirror (optional)
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++-7 \
    git \
    curl \
    wget \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    librdmacm1 \
    libibverbs1 \
    libboost-all-dev \
    ibverbs-providers \
    tzdata \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fontconfig \
    language-pack-en \
    sysstat \
    gnupg \
    lsb-release \
    sudo \
    && rm -rf /var/lib/apt/lists/*


# Install Miniconda & use Python 3.9
ARG python=3.9
ENV PYTHON_VERSION=${python}
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/install-conda.sh \
    && chmod +x /tmp/install-conda.sh \
    && bash /tmp/install-conda.sh -b -f -p /usr/local \
    && rm -f /tmp/install-conda.sh \
    && conda install -y python=${PYTHON_VERSION} \
    && conda clean -y --all


# Setup TUNA mirror (optional)
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN mkdir -p ~/.conda && \
    echo "channels:" > ~/.condarc && \
    echo "  - defaults" >> ~/.condarc && \
    echo "  - conda-forge" >> ~/.condarc && \
    echo "  - https://conda.anaconda.org/dglteam" >> ~/.condarc && \
    echo "channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda" >> ~/.condarc && \
    echo "default_channels:" >> ~/.condarc && \
    echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main" >> ~/.condarc && \
    echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free" >> ~/.condarc && \
    echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r" >> ~/.condarc && \
    echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro" >> ~/.condarc && \
    echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2" >> ~/.condarc && \
    echo "custom_channels:" >> ~/.condarc && \
    echo "  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
    echo "  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
    echo "  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
    echo "  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
    echo "  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
    echo "  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
    echo "  nvidia: https://mirrors.sustech.edu.cn/anaconda-extra/cloud" >> ~/.condarc && \
    echo "show_channel_urls: true" >> ~/.condarc


# By default, install packages from `requirements.txt` with pip.
COPY . /app
RUN conda env create -f /app/environment.yml \
  && conda clean -y --all

# install plip with pip in the interformer environment
RUN conda run -n PBCNet pip install lmdb==1.5.1
RUN conda run -n PBCNet \
  conda install dgl-cuda11.3=0.8.1 
RUN conda run -n PBCNet \
  pip install dgllife==0.2.6
RUN conda run -n PBCNet \
  conda install -c https://conda.anaconda.org/pyg pyg



# Compile Docking sampling program from source code
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
RUN nvcc --version


# Set working directory to /project
WORKDIR "/app"