FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

USER root

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install python3-pip -y && \
    apt install python3.8 -y && apt-get -y install libgl1-mesa-glx

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="Asia/Seoul"
RUN apt install git -y && apt-get install libglib2.0-0 -y


RUN python3 -m pip install --upgrade pip

# python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install ftfy regex tqdm

RUN apt-get install unzip
