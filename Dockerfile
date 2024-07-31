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
RUN apt-get update && apt-get install -y wget doxygen curl libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python-opencv python3-setuptools python3-dev python3-pip
RUN apt-get update && apt-get -y install python3-pip libvulkan1 python3-venv vim pciutils wget git kmod vim git
# RUN pip3 install --upgrade pip
RUN apt-get update && apt-get install -y \
    libgtkglext1 libgtkglext1-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev nano wget doxygen curl \
    libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python-opencv \
    python3-setuptools python3-dev
# RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip3 install opencv-python==4.1.0.25 numpy==1.13.3 pandas==0.24.1 networkx==2.2
RUN apt-get update && apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 libxcb-xinerama0

# Install gcc-9 and g++-9: needed for vllm
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && apt-get update
RUN apt-get install -y gcc-9 g++-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
RUN gcc --version

#install latest cmake
ADD https://cmake.org/files/v3.12/cmake-3.12.2-Linux-x86_64.sh /cmake-3.12.2-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.12.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

ENV PYTHONPATH=/root/mount/Matterport3DSimulator/build

## Navillm + VLLM conda environment
# curl -O https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
# bash Anaconda3-2024.06-1-Linux-x86_64.sh
# conda config --set auto_activate_base False
### setup following env vars:
    # export OPENAI_API_KEY="<key>"  # UTCS AMRL OpenAI API key
    # export HF_API_KEY="<key>"  # Huggingface
    # export TF_CPP_MIN_LOG_LEVEL='2'
    # alias cleanpycache='find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null'
    # export AM_I_DOCKER=False
    # export BUILD_WITH_CUDA=True
    # export CUDA_HOME=/usr/local/cuda/
    # export MAX_JOBS=6
    # export PATH="${CUDA_HOME}/bin:$PATH"
    # export VLLM_NCCL_SO_PATH=/usr/lib/x86_64-linux-gnu/libnccl.so.2
    # export VLLM_WORKER_MULTIPROC_METHOD=spawn
    # export VLLM_LOGGING_LEVEL=ERROR
# conda create --name navillm python=3.10
# pip3 install xformers --index-url https://download.pytorch.org/whl/cu118
# pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --extra-index-url https://download.pytorch.org/whl/cu118
# pip3 install packaging
# pip3 install flashinfer -i https://flashinfer.ai/whl/cu118/torch2.4
### Run "pip install -e ." in VLLM directory
# pip3 install pandas networkx opencv-contrib-python simple-colors
# pip3 install easydict==1.10 h5py jsonlines lmdb more_itertools==10.1.0 msgpack_numpy msgpack_python numpy Pillow progressbar33 psutil PyYAML ray requests shapely timm tqdm
# pip3 install transformers sentencepiece
# pip3 install ipdb openai supervision==0.6.0 open3d
# huggingface-cli login --token $HF_API_KEY