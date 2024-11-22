export HF_ENDPOINT=https://hf-mirror.com
pip install -r /workspace/segment-anything-2-real-time-main/requirements.txt
apt-get install -y libgl1-mesa-glx
wget -P /workspace/segment-anything-2-real-time-main/checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt





















git clone https://hf-mirror.com/spaces/supersolar/florence-sam-tencent
cd /workspace/florence-sam
pip install -r /workspace/florence-sam/requirements.txt
wget -P /workspace/florence-sam/checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
export HF_ENDPOINT=https://hf-mirror.com
apt-get update
apt-get install -y libgl1-mesa-glx
pip install moviepy
python /workspace/florence-sam/all_in_one.py
unzip /workspace/transnetv.zip -d /workspace/transvnet

pip install tensorflow
export CUDA_VISIBLE_DEVICES=0
conda create -n mt3 python==3.10
conda activate mt3 
conda deactivate 
conda activate base
git clone https://gh-proxy.com/tangxiaobai/segment-anything-2-real-time



unzip /workspace/segment-anything-2-real-time-main.zip -d /workspace/

unzip /workspace/transnetv.zip -d /workspace/transnetv2

cd /workspace/segment-anything-2-real-time-main
pip install -e .


apt-get update
apt-get install cuda-12-4

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
source ~/.bashrc

wget -P /workspace/checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt




export HF_ENDPOINT=https://hf-mirror.com
pip install hydra-core iopath pillow opencv-python moviepy imageio supervision transformers timm einops
apt-get install -y libgl1-mesa-glx

git clone https://hf-mirror.com/spaces/supersolar/florence-sam-tencent



pip install hydra-core iopath pillow opencv-python moviepy imageio
apt-get install -y libgl1-mesa-glx
