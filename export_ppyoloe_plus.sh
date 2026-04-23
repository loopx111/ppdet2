#!/bin/bash
# PP-YOLOE+ 模型导出脚本
# 用于导出带 TensorRT 优化的推理模型

set -e

# 工作目录
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
PDDET_ROOT="$WORK_DIR"

cd "$PDDET_ROOT"

echo "=========================================="
echo "PP-YOLOE+ 模型导出脚本 (TensorRT优化)"
echo "=========================================="

# PP-YOLOE+_l 高精度模型
PPYOLOE_PLUS_L_URL="https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_80e_coco.pdparams"
PPYOLOE_PLUS_L_CONFIG="configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml"

# PP-YOLOE+_m 中等精度模型
PPYOLOE_PLUS_M_URL="https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_coco.pdparams"
PPYOLOE_PLUS_M_CONFIG="configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml"

# PP-YOLOE+_s 轻量模型
PPYOLOE_PLUS_S_URL="https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco.pdparams"
PPYOLOE_PLUS_S_CONFIG="configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml"

export_model() {
    local NAME=$1
    local CONFIG=$2
    local URL=$3
    
    echo ""
    echo "[$NAME] 导出中..."
    echo "配置: $CONFIG"
    echo "权重: $URL"
    
    python tools/export_model.py \
        -c "$CONFIG" \
        -o weights="$URL" \
        trt=True
    
    echo "[$NAME] 导出完成!"
}

# 创建模型目录
mkdir -p models

# 导出 PP-YOLOE+_l (高精度，FPS: 149.2 on V100 TRT-FP16)
echo ""
echo ">>> 导出 PP-YOLOE+_l (COCO mAP: 52.9, V100 TRT-FP16: 149.2 FPS)"
export_model "ppyoloe_plus_l" "$PPYOLOE_PLUS_L_CONFIG" "$PPYOLOE_PLUS_L_URL"

# 移动到 models 目录
if [ -d "output_inference/ppyoloe_plus_crn_l_80e_coco" ]; then
    mv output_inference/ppyoloe_plus_crn_l_80e_coco models/
    echo "模型已移动到: models/ppyoloe_plus_crn_l_80e_coco"
fi

# 可选：导出 PP-YOLOE+_m (中等精度)
# export_model "ppyoloe_plus_m" "$PPYOLOE_PLUS_M_CONFIG" "$PPYOLOE_PLUS_M_URL"

# 可选：导出 PP-YOLOE+_s (轻量)
# export_model "ppyoloe_plus_s" "$PPYOLOE_PLUS_S_CONFIG" "$PPYOLOE_PLUS_S_URL"

echo ""
echo "=========================================="
echo "导出完成!"
echo "=========================================="
echo ""
echo "已导出的模型:"
ls -la models/
echo ""
echo "使用方法:"
echo "  python deploy/pipeline/worker/paddle_worker.py \\"
echo "    --cam-id cam1 \\"
echo "    --rtsp rtsp://xxx \\"
echo "    --mode ppyoloe_plus_l  # 使用PP-YOLOE+_l模型"
