#!/bin/bash
# FFmpeg CUDA安装脚本 - 支持CUDA 11.x
# 适用于Ubuntu 20.04/22.04 WSL环境

set -e

echo "=============================================="
echo "FFmpeg CUDA解码支持安装脚本"
echo "=============================================="

# 检查系统版本
echo "[1/6] 检查系统环境..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "系统: $NAME $VERSION"
else
    echo "无法检测系统版本"
    exit 1
fi

# 检查NVIDIA驱动
echo "[2/6] 检查NVIDIA驱动..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "警告: nvidia-smi 未找到"
    echo "如果这是在WSL中，确保Windows上有NVIDIA驱动"
fi

# 检查CUDA版本
echo "[3/6] 检查CUDA版本..."
if [ -d "/usr/local/cuda" ]; then
    echo "CUDA目录: /usr/local/cuda"
    ls -la /usr/local/cuda*/version.txt 2>/dev/null || cat /usr/local/cuda/version.txt 2>/dev/null || echo "CUDA版本文件未找到"
elif [ -d "$HOME/paddle-env" ]; then
    echo "检测到PaddlePaddle环境: $HOME/paddle-env"
    ls $HOME/paddle-env/lib/ 2>/dev/null | grep cuda || echo "检查CUDA库..."
fi

# 方法1: 尝试使用Conda安装ffmpeg-cuda（推荐）
echo "[4/6] 安装FFmpeg CUDA支持..."
echo ""
echo "选项A: 使用Conda安装（推荐）"
echo "  conda install -c conda-forge ffmpeg-cuda"
echo ""
echo "选项B: 使用NVIDIA FFmpeg包"
echo "  sudo apt install ffmpeg-nvidia56  # 对于较新的驱动"
echo "  或"
echo "  sudo apt install ffmpeg-nvidia    # 如果有这个包"
echo ""
echo "选项C: 使用预编译的FFmpeg+NVIDIA库"
echo ""

# 检查是否有apt源可用
if command -v apt-get &> /dev/null; then
    echo "[5/6] 搜索可用的FFmpeg NVIDIA包..."
    apt-cache search ffmpeg | grep -i nvidia || echo "未找到nvidia相关包"
    
    echo ""
    echo "=============================================="
    echo "推荐安装方法（复制执行）："
    echo "=============================================="
    echo ""
    echo "# 方法1: Conda安装（最简单）"
    echo "conda create -n ffmpeg-cuda -c conda-forge ffmpeg-cuda"
    echo "conda activate ffmpeg-cuda"
    echo ""
    echo "# 方法2: 使用ffmpeg-nvidia包（Ubuntu用户）"
    echo "# 先尝试添加NVIDIA的repo"
    echo "sudo apt update"
    echo "sudo apt install ffmpeg-nvidia"
    echo ""
    echo "# 方法3: 手动编译（复杂但最灵活）"
    echo "./build_ffmpeg_cuda.sh"
    echo ""
fi

# 检查当前FFmpeg
echo "[6/6] 当前FFmpeg配置..."
ffmpeg -version 2>/dev/null | head -1 || echo "FFmpeg未安装"
ffmpeg -codecs 2>/dev/null | grep -E "(cuda|nvdec|cuvid)" || echo "未检测到CUDA解码器"

echo ""
echo "=============================================="
echo "安装后验证命令："
echo "=============================================="
echo "ffmpeg -decoders 2>&1 | grep cuda"
echo "ffmpeg -hwaccels 2>&1 | grep cuda"
echo ""
echo "使用示例："
echo "ffmpeg -hwaccel cuda -c:v h264_cuvid -i rtsp://... -f rawvideo -pix_fmt bgr24 -an -threads 1 pipe:1"
echo "=============================================="
