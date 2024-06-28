ARG CUDA="11.2.2"
ARG CUDNN="8"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-runtime-ubuntu20.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities.
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory.
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it.
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory.
ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config \
 && chmod -R 777 $HOME

# Download and install Micromamba.
RUN curl -sL https://micro.mamba.pm/api/micromamba/linux-64/1.1.0 \
  | sudo tar -xvj -C /usr/local bin/micromamba
ENV MAMBA_EXE=/usr/local/bin/micromamba \
    MAMBA_ROOT_PREFIX=/home/user/micromamba \
    CONDA_PREFIX=/home/user/micromamba \
    PATH=/home/user/micromamba/bin:$PATH

# Set up the base Conda environment by installing PyTorch and friends.
COPY conda-linux-64.lock /app/conda-linux-64.lock
RUN micromamba create -qy -n base -f /app/conda-linux-64.lock \
 && rm /app/conda-linux-64.lock \
 && micromamba shell init --shell=bash --prefix="$MAMBA_ROOT_PREFIX" \
 && micromamba clean -qya

#docker build
USER root
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="Asia/Seoul"
RUN export TZ
RUN apt-get -y update && \
  apt-get -y install libgl1-mesa-glx && \
  apt-get -y install libglib2.0-0

# build
ARG LOCAL_PATH="/home/inhamath/Clustering"

COPY ./mlflow_model/requirment.txt ${LOCAL_PATH}/requirment.txt
WORKDIR ${LOCAL_PATH}
RUN pip install --upgrade pip && \
 pip install -r requirment.txt

CMD ["/bin/bash"]