# Matterport3DSimulator
# Requires nvidia gpu with driver 396.37 or higher
# On robolidar, ./nvidia_find_vers.sh
ARG CUDA_VERSION=11.4.2
ARG NVIDIA_VERSION=535.183.01

FROM nvidia/cudagl:${CUDA_VERSION}-devel-ubuntu18.04

# Install cudnn
ENV CUDNN_VERSION 7.6.4.38
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
RUN rm /etc/apt/sources.list.d/cuda.list

# Install a few libraries to support both EGL and OSMESA options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget doxygen curl libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python-opencv python3-setuptools python3-dev python3-pip
RUN apt-get update && apt-get -y install python3-pip libvulkan1 python3-venv vim pciutils wget git kmod vim git
# RUN pip3 install --upgrade pip
RUN apt-get update && apt-get install -y \
    libgtkglext1 libgtkglext1-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev nano wget doxygen curl \
    libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python-opencv \
    python3-setuptools python3-dev
# RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip3 install opencv-python==4.1.0.25 numpy==1.13.3 pandas==0.24.1 networkx==2.2

#install latest cmake
ADD https://cmake.org/files/v3.12/cmake-3.12.2-Linux-x86_64.sh /cmake-3.12.2-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.12.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

ENV PYTHONPATH=/root/mount/Matterport3DSimulator/build
