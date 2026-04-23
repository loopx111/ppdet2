#!/bin/bash
# PaddleDetection Worker 快速测试脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  PaddleDetection Worker 测试脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 配置
CAM_ID="cam1"
RTSP_URL="${1:-rtsp://127.0.0.1:8554/live/cam1}"
DEVICE="${2:-gpu}"
SKIP_FRAMES=3

echo -e "\n${YELLOW}[配置]${NC}"
echo "  摄像头ID: $CAM_ID"
echo "  RTSP地址: $RTSP_URL"
echo "  设备: $DEVICE"
echo "  跳帧数: $SKIP_FRAMES"

# 检查 Python 环境
echo -e "\n${YELLOW}[1/5] 检查 Python 环境...${NC}"
python -c "import paddle" 2>/dev/null || { echo -e "${RED}错误: PaddlePaddle 未安装${NC}"; exit 1; }
echo -e "${GREEN}✓ PaddlePaddle 已安装${NC}"

python -c "import redis" 2>/dev/null || { echo -e "${YELLOW}⚠ Redis 模块未安装${NC}"; }
python -c "import cv2" 2>/dev/null || { echo -e "${RED}错误: OpenCV 未安装${NC}"; exit 1; }
echo -e "${GREEN}✓ OpenCV 已安装${NC}"

# 检查 Redis
echo -e "\n${YELLOW}[2/5] 检查 Redis 服务...${NC}"
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &>/dev/null; then
        echo -e "${GREEN}✓ Redis 服务正在运行${NC}"
    else
        echo -e "${YELLOW}⚠ Redis 服务未运行 (结果将只输出到控制台)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Redis CLI 未安装 (结果将只输出到控制台)${NC}"
fi

# 检查 PaddleDetection
echo -e "\n${YELLOW}[3/5] 检查 PaddleDetection 模块...${NC}"
cd "$(dirname "$0")/../.."
if [ -d "deploy/python" ]; then
    echo -e "${GREEN}✓ PaddleDetection 路径正确${NC}"
else
    echo -e "${RED}错误: PaddleDetection 路径不正确${NC}"
    exit 1
fi

# 检查 FFmpeg
echo -e "\n${YELLOW}[4/5] 检查 FFmpeg...${NC}"
if command -v ffmpeg &> /dev/null; then
    echo -e "${GREEN}✓ FFmpeg 已安装${NC}"
else
    echo -e "${YELLOW}⚠ FFmpeg 未安装，将使用 OpenCV 原生捕获${NC}"
fi

# 运行 Worker
echo -e "\n${YELLOW}[5/5] 启动 PaddleDetection Worker...${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  按 Ctrl+C 停止${NC}"
echo -e "${GREEN}========================================${NC}\n"

python deploy/pipeline/worker/paddle_worker.py \
    --cam-id "$CAM_ID" \
    --rtsp "$RTSP_URL" \
    --device "$DEVICE" \
    --skip-frames "$SKIP_FRAMES" \
    --redis-host localhost \
    --redis-port 6379
