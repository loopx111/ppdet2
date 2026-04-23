@echo off
REM PP-YOLOE+ 模型导出脚本 (Windows)
REM 用于导出带 TensorRT 优化的推理模型

echo ==========================================
echo PP-YOLOE+ 模型导出脚本 (TensorRT优化)
echo ==========================================

set WORK_DIR=%~dp0
cd /d %WORK_DIR%

REM PP-YOLOE+_l 高精度模型
set PPYOLOE_PLUS_L_URL=https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_80e_coco.pdparams
set PPYOLOE_PLUS_L_CONFIG=configs\ppyoloe\ppyoloe_plus_crn_l_80e_coco.yml

echo.
echo 导出 PP-YOLOE+_l (COCO mAP: 52.9, V100 TRT-FP16: 149.2 FPS)
echo 配置: %PPYOLOE_PLUS_L_CONFIG%
echo 权重: %PPYOLOE_PLUS_L_URL%

python tools\export_model.py ^
    -c %PPYOLOE_PLUS_L_CONFIG% ^
    -o weights=%PPYOLOE_PLUS_L_URL% ^
    trt=True

if exist "output_inference\ppyoloe_plus_crn_l_80e_coco" (
    if not exist "models" mkdir models
    move output_inference\ppyoloe_plus_crn_l_80e_coco models\
    echo 模型已移动到: models\ppyoloe_plus_crn_l_80e_coco
)

echo.
echo ==========================================
echo 导出完成!
echo ==========================================
echo.
echo 已导出的模型:
if exist models (
    dir /b models
) else (
    echo models目录不存在
)
echo.
echo 使用方法:
echo   python deploy\pipeline\worker\paddle_worker.py ^
echo     --cam-id cam1 ^
echo     --rtsp rtsp://xxx ^
echo     --mode ppyoloe_plus_l

pause
