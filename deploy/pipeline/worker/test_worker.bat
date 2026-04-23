@echo off
chcp 65001 >nul
echo ========================================
echo   PaddleDetection Worker 测试脚本
echo ========================================
echo.

REM 配置
set CAM_ID=cam1
set RTSP_URL=rtsp://127.0.0.1:8554/live/cam1
set DEVICE=gpu
set SKIP_FRAMES=3

echo [配置]
echo   摄像头ID: %CAM_ID%
echo   RTSP地址: %RTSP_URL%
echo   设备: %DEVICE%
echo   跳帧数: %SKIP_FRAMES%
echo.

REM 检查 Python 环境
echo [1/5] 检查 Python 环境...
python -c "import paddle" 2>nul
if errorlevel 1 (
    echo 错误: PaddlePaddle 未安装
    pause
    exit /b 1
)
echo ✓ PaddlePaddle 已安装
python -c "import cv2" 2>nul
if errorlevel 1 (
    echo 错误: OpenCV 未安装
    pause
    exit /b 1
)
echo ✓ OpenCV 已安装
echo.

REM 检查 Redis
echo [2/5] 检查 Redis 服务...
redis-cli ping >nul 2>nul
if errorlevel 1 (
    echo ⚠ Redis 服务未运行 ^(结果将只输出到控制台^)
) else (
    echo ✓ Redis 服务正在运行
)
echo.

REM 检查 PaddleDetection
echo [3/5] 检查 PaddleDetection...
cd /d "%~dp0\..\.."
if exist "deploy\python" (
    echo ✓ PaddleDetection 路径正确
) else (
    echo 错误: PaddleDetection 路径不正确
    pause
    exit /b 1
)
echo.

REM 检查 FFmpeg
echo [4/5] 检查 FFmpeg...
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo ⚠ FFmpeg 未安装，将使用 OpenCV 原生捕获
) else (
    echo ✓ FFmpeg 已安装
)
echo.

REM 运行 Worker
echo [5/5] 启动 PaddleDetection Worker...
echo ========================================
echo   按 Ctrl+C 停止
echo ========================================
echo.

python deploy\pipeline\worker\paddle_worker.py ^
    --cam-id %CAM_ID% ^
    --rtsp "%RTSP_URL%" ^
    --device %DEVICE% ^
    --skip-frames %SKIP_FRAMES% ^
    --redis-host localhost ^
    --redis-port 6379

pause
