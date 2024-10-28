# vln env has navillm working, however, vllm can work though partially (like internvl wont work)
# vllm env has vllm working, but navillm works though with lower performance
# NOTE: run manually, there were some cmake issues with the script
#!/bin/bash
set -e

ENV_NAME=$1

# Navillm conda environment
cd /root/mount/Matterport3DSimulator
eval "$(/root/anaconda3/bin/conda shell.bash hook)"
conda create --name $ENV_NAME python=3.10
conda activate $ENV_NAME
pip3 install xformers --index-url https://download.pytorch.org/whl/cu118
pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --extra-index-url https://download.pytorch.org/whl/cu118
pip3 install packaging
pip3 install flashinfer -i https://flashinfer.ai/whl/cu118/torch2.4
pip3 install pandas networkx opencv-contrib-python simple-colors
pip3 install easydict==1.10 h5py jsonlines lmdb more_itertools==10.1.0 msgpack_numpy msgpack_python numpy Pillow progressbar33 psutil PyYAML ray requests shapely timm tqdm
pip3 install transformers==4.28.0 sentencepiece==0.1.99
pip3 install ipdb openai supervision==0.6.0 open3d
huggingface-cli login --token $HF_API_KEY

# VLLM setup (conflicts with transformers version needed by navillm, so you MAY not have all parts of vllm working)
if [ $ENV_NAME == "vllm" ]; then
    cd /root/mount/Matterport3DSimulator/thirdparty/vllm
    pip3 install --upgrade transformers sentencepiece
    pip3 install -e .

    # GroundedSAM setup
    conda deactivate
    conda activate $ENV_NAME
    cd /root/mount/Matterport3DSimulator/thirdparty/Grounded-Segment-Anything
    python3 -m pip install -e segment_anything
    pip3 install --no-build-isolation -e GroundingDINO
fi

if [ $ENV_NAME == "vln" ]; then
    # sim setup
    conda deactivate
    conda activate $ENV_NAME
    cd /root/mount/Matterport3DSimulator
    export PYTHON_EXECUTABLE=$(which python) && export PYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") && export PYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
    rm -rf build && mkdir build && cd build
    cmake -DEGL_RENDERING=ON -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR -DPYTHON_LIBRARY=$PYTHON_LIBRARY ..
    make -j$(nproc)

    # sim testing
    cd /root/mount/Matterport3DSimulator
    echo "Running test 1"
    ./build/tests ~Timing
    echo "Running test 2"
    ./build/tests Timing
fi