"""
API配置文件
"""
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent.parent.parent

# PaddleDetection配置
PPDET_CONFIG = BASE_DIR / "config" / "infer_cfg_pphuman.yml"
PPDET_MODEL_CACHE = Path.home() / ".cache" / "paddle" / "infer_weights"

# ZLMediaKit配置
ZL_MEDIAKIT_RTSP_PORT = 554
ZL_MEDIAKIT_HTTP_PORT = 8080
ZL_MEDIAKIT_WS_PORT = 8088

# 默认RTSP流地址（来自ZLMediaKit代理）
# 注意：直接使用ZLMediaKit代理地址，避免从摄像头直接拉流
DEFAULT_RTSP_PROXY = "rtsp://127.0.0.1:554/live/cam1"

# ZLMediaKit代理地址映射
# 格式：代理流名称 -> ZLMediaKit拉流地址
CAMERA_URLS = {
    "cam1": "rtsp://127.0.0.1:554/live/cam1",
}

# 服务配置
API_HOST = "0.0.0.0"
API_PORT = 8000

# 推理配置
INFERENCE_CONFIG = {
    "device": "gpu",  # gpu 或 cpu
    "warmup_frame": 5,
    "batch_size": 1,
    "skip_frame_num": 2,
    "threshold": 0.5,
}

# CORS配置
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
    "*",  # 生产环境应限制
]

# 日志配置
LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "logs" / "api.log"

# 会话配置
SESSION_TIMEOUT = 3600  # 秒
MAX_SESSIONS = 10
