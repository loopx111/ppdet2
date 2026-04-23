#!/bin/bash
#
# ZLMediaKit 拉流脚本 - 健壮版（自动重连）
#

STREAMS_CONF="/usr/local/ZLMediaKit/streams.conf"
LOG_FILE="/var/log/zlmediakit_pull.log"
FFMPEG="/root/miniconda3/envs/ffmpeg-cuda/bin/ffmpeg"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 拉流函数
pull_stream() {
    local name="$1"
    local url="$2"
    
    log ">>> 开始拉流: $name"
    log "    源: $url"
    
    while true; do
        log "    [重连] 正在连接..."
        
        $FFMPEG -loglevel warning -rtsp_transport tcp \
            -reconnect 1 \
            -reconnect_streamed 1 \
            -reconnect_delay_max 30 \
            -i "$url" \
            -c:v copy -c:a copy \
            -f rtsp \
            -rtsp_transport tcp \
            "rtsp://127.0.0.1:554/live/$name" \
            2>&1 | while read line; do
                log "    FFmpeg: $line"
            done
        
        log "    [断开] FFmpeg退出，5秒后重连..."
        sleep 5
    done
}

log "========================================="
log "ZLMediaKit 拉流服务启动"
log "========================================="

# 等待ZLMediaKit启动
sleep 3

# 检查配置文件
if [ ! -f "$STREAMS_CONF" ]; then
    log "ERROR: 配置文件不存在: $STREAMS_CONF"
    exit 1
fi

# 读取并启动所有流
while IFS='|' read -r name url; do
    # 跳过注释和空行
    [[ "$name" =~ ^# ]] && continue
    [[ -z "$name" ]] && continue
    
    log "配置流: $name -> $url"
    pull_stream "$name" "$url" &
    sleep 1
done < "$STREAMS_CONF"

log "所有流已启动，PID: $(jobs -p)"

# 等待所有后台任务
wait
