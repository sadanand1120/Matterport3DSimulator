# Matterport3DSimulator
# Requires nvidia gpu with driver 396.37 or higher
# On robolidar, ./nvidia_find_vers.sh
ARG CUDA_VERSION=11.4.2
ARG NVIDIA_VERSION=535.183.01

FROM docker.io/nvidia/cudagl:${CUDA_VERSION}-devel-ubuntu18.04

# Install cudnn
ENV CUDNN_VERSION 7.6.4.38
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
RUN rm /etc/apt/sources.list.d/cuda.list

# Install a few libraries to support both EGL and OSMESA options
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore
RUN apt-get update && apt-get install -y wget doxygen curl libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python-opencv python3-setuptools python3-dev python3-pip \
    python3-pip libvulkan1 python3-venv vim pciutils wget git kmod vim git \
    libgtkglext1 libgtkglext1-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev nano wget doxygen curl \
    libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python-opencv \
    python3-setuptools python3-dev feh \
    libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 libxcb-xinerama0 \
    qt5-default libxcb-xinerama0 libxkbcommon-x11-0 libxcb1 libxcb-xinput0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xfixes0 \
    && rm -rf /var/lib/apt/lists/*

# Install gcc-9 and g++-9: needed for vllm
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y gcc-9 g++-9 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
    && gcc --version \
    && rm -rf /var/lib/apt/lists/*

# Install Git Large File Storage (git-lfs)
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs

#install latest cmake
ADD https://cmake.org/files/v3.12/cmake-3.12.2-Linux-x86_64.sh /cmake-3.12.2-Linux-x86_64.sh
RUN mkdir /opt/cmake && \
    sh /cmake-3.12.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && \
    cmake --version

# Install Miniconda to /root/anaconda3
RUN wget -O Miniconda3-py310_24.7.1-0-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh && \
    bash Miniconda3-py310_24.7.1-0-Linux-x86_64.sh -b -p /root/anaconda3 && \
    rm Miniconda3-py310_24.7.1-0-Linux-x86_64.sh && \
    /root/anaconda3/bin/conda config --set auto_activate_base false

# conda bash completion
RUN /root/anaconda3/bin/conda install -c conda-forge conda-bash-completion -y

RUN echo 'if [ ! -f /initialized ] && [ -f /root/mount/Matterport3DSimulator/docker_bashrc.txt ]; then' >> ~/.bashrc && \
    echo '    cat /root/mount/Matterport3DSimulator/docker_bashrc.txt >> ~/.bashrc' >> ~/.bashrc && \
    echo '    touch /initialized' >> ~/.bashrc && \
    echo 'fi' >> ~/.bashrc

RUN apt-get update && apt-get upgrade -y && apt-get autoremove -y && apt-get autoclean -y && apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install 2>/dev/null