#!/bin/bash
# 从源码编译支持CUDA的FFmpeg
# 适用于Ubuntu 20.04/22.04 + CUDA 11.x
# 需要NVIDIA驱动和CUDA Toolkit

set -e

# 配置参数
FFMPEG_VERSION="6.1.1"
NVIDIA_DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "550")
WORK_DIR="$HOME/ffmpeg_build"
PREFIX="$HOME/ffmpeg_cuda"

echo "=============================================="
echo "编译支持CUDA的FFmpeg"
echo "=============================================="

# 检查NVIDIA驱动
echo "[1/8] 检查NVIDIA驱动..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: nvidia-smi 未找到，请确保NVIDIA驱动已安装"
    exit 1
fi
echo "NVIDIA驱动版本: $NVIDIA_DRIVER_VER"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# 检查CUDA
echo "[2/8] 检查CUDA..."
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
elif [ -d "$HOME/paddle-env" ]; then
    # 尝试从PaddlePaddle环境获取CUDA
    CUDA_LIB=$(find $HOME/paddle-env/lib -name "libcudart.so*" 2>/dev/null | head -1)
    if [ -n "$CUDA_LIB" ]; then
        CUDA_HOME=$(dirname $(dirname $CUDA_LIB))
        echo "从PaddlePaddle环境检测到CUDA: $CUDA_HOME"
    fi
fi

if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
    echo "警告: CUDA_HOME未设置或目录不存在"
    echo "请手动设置: export CUDA_HOME=/usr/local/cuda"
    echo "或者确保CUDA已正确安装"
    # 尝试常见位置
    for dir in /usr/local/cuda /usr/lib/cuda; do
        if [ -d "$dir" ]; then
            export CUDA_HOME="$dir"
            echo "使用: $CUDA_HOME"
            break
        fi
    done
fi

if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME" ]; then
    echo "CUDA_HOME: $CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
else
    echo "错误: 无法找到CUDA安装目录"
    exit 1
fi

# 安装依赖
echo "[3/8] 安装编译依赖..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    nasm \
    yasm \
    libfdk-aac-dev \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libmp3lame-dev \
    libopus-dev \
    libass-dev \
    libfreetype6-dev \
    libvdpau-dev \
    libva-dev \
    libdrm-dev \
    libx11-dev \
    libxfixes-dev \
    libxvidcore-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    libnvidia-container-dev \
    libnvidia-container1 \
    || echo "部分依赖安装失败，继续..."

# 创建工作目录
echo "[4/8] 创建工作目录..."
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# 下载FFmpeg源码
echo "[5/8] 下载FFmpeg $FFMPEG_VERSION..."
if [ ! -d "ffmpeg-$FFMPEG_VERSION" ]; then
    wget -q https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz
    tar xf ffmpeg-$FFMPEG_VERSION.tar.xz
fi
cd ffmpeg-$FFMPEG_VERSION

# 配置编译选项（包含CUDA支持）
echo "[6/8] 配置编译选项..."
./configure \
    --prefix="$PREFIX" \
    --pkg-config-flags="--static" \
    --extra-cflags="-I$CUDA_HOME/include -O3" \
    --extra-ldflags="-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib" \
    --extra-libs="-lnvcuvid" \
    --bindir="$PREFIX/bin" \
    --enable-gpl \
    --enable-libx264 \
    --enable-libx265 \
    --enable-libvpx \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-libfdk-aac \
    --enable-libass \
    --enable-libfreetype \
    --enable-nonfree \
    --enable-cuda \
    --enable-cuda-sdk \
    --enable-cuvid \
    --enable-nvenc \
    --enable-nvdec \
    --enable-libnpp \
    --enable-shared \
    --disable-doc \
    --disable-programs

# 编译
echo "[7/8] 编译（这可能需要15-30分钟）..."
make -j$(nproc)

# 安装
echo "[8/8] 安装..."
sudo make install
sudo ldconfig

# 创建激活脚本
cat > "$HOME/activate_ffmpeg_cuda.sh" << 'EOF'
#!/bin/bash
export PATH="$HOME/ffmpeg_cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/ffmpeg_cuda/lib:$LD_LIBRARY_PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH"
echo "FFmpeg CUDA已激活: $(which ffmpeg)"
ffmpeg -version | head -1
EOF

chmod +x "$HOME/activate_ffmpeg_cuda.sh"

echo ""
echo "=============================================="
echo "安装完成!"
echo "=============================================="
echo ""
echo "激活方法："
echo "  source $HOME/activate_ffmpeg_cuda.sh"
echo ""
echo "验证CUDA支持："
echo "  ffmpeg -hwaccels 2>&1 | grep cuda"
echo "  ffmpeg -decoders 2>&1 | grep cuvid"
echo ""
echo "测试GPU解码："
echo "  ffmpeg -hwaccel cuda -c:v h264_cuvid -i rtsp://... -f null -"
echo ""
