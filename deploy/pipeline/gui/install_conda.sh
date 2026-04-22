#!/bin/bash
# 安装Miniconda和ffmpeg-cuda的完整脚本

set -e

echo "=============================================="
echo "安装Miniconda + FFmpeg CUDA"
echo "=============================================="

# 1. 下载并安装Miniconda
echo "[1/4] 下载Miniconda..."
cd /tmp

if [ ! -f "Miniconda3-latest-Linux-x86_64.sh" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi

echo "[2/4] 安装Miniconda到 ~/miniconda3..."
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# 初始化Conda
echo "[3/4] 初始化Conda..."
source $HOME/miniconda3/etc/profile.d/conda.sh

# 配置conda - 先接受服务条款
echo "[3.1/5] 接受Conda服务条款..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

conda config --add channels conda-forge
conda config --set channel_priority strict

# 2. 创建新环境并安装ffmpeg-cuda
echo "[4/5] 创建ffmpeg-cuda环境并安装..."
conda create -n ffmpeg-cuda -c conda-forge/label/cuda118 -c conda-forge ffmpeg-cuda cuda-nvcc -y

echo "[5/5] 验证安装..."
conda activate ffmpeg-cuda

echo ""
echo "=============================================="
echo "安装完成!"
echo "=============================================="
echo ""
echo "激活方法："
echo "  source ~/miniconda3/etc/profile.d/conda.sh"
echo "  conda activate ffmpeg-cuda"
echo ""
echo "验证CUDA支持："
echo "  ffmpeg -decoders | grep cuvid"
echo ""
echo "=============================================="
