#!/bin/bash
set -e

# Navillm + VLLM conda environment
cd /root/mount/Matterport3DSimulator
eval "$(/root/anaconda3/bin/conda shell.bash hook)"
conda create --name vllm python=3.10
conda activate vllm
pip3 install xformers --index-url https://download.pytorch.org/whl/cu118
pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --extra-index-url https://download.pytorch.org/whl/cu118
pip3 install packaging
pip3 install flashinfer -i https://flashinfer.ai/whl/cu118/torch2.4
cd /root/mount/Matterport3DSimulator/thirdparty/vllm
pip3 install -e .
pip3 install pandas networkx opencv-contrib-python simple-colors
pip3 install easydict==1.10 h5py jsonlines lmdb more_itertools==10.1.0 msgpack_numpy msgpack_python numpy Pillow progressbar33 psutil PyYAML ray requests shapely timm tqdm
pip3 install transformers sentencepiece
pip3 install ipdb openai supervision==0.6.0 open3d
huggingface-cli login --token $HF_API_KEY

# sim setup
conda deactivate
conda activate vllm
export PYTHON_EXECUTABLE=$(which python) && export PYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") && export PYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
cd /root/mount/Matterport3DSimulator
mkdir build && cd build
cmake -DEGL_RENDERING=ON -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR -DPYTHON_LIBRARY=$PYTHON_LIBRARY ..
make -j$(nproc)

# testing
cd /root/mount/Matterport3DSimulator
echo "Running test 1"
./build/tests ~Timing
echo "Running test 2"
./build/tests Timing

# GroundedSAM setup
conda deactivate
conda activate vllm
cd /root/mount/Matterport3DSimulator/thirdparty/Grounded-Segment-Anything
python3 -m pip install -e segment_anything
pip3 install --no-build-isolation -e GroundingDINO
