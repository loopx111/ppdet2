#!/bin/bash
#
# ZLMediaKit 拉流脚本 - 修复版
#

STREAMS_CONF="/usr/local/ZLMediaKit/streams.conf"
LOG_FILE="/var/log/zlmediakit_pull.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

pull_stream() {
    local name="$1"
    local url="$2"
    
    log ">>> 开始拉流: $name"
    
    while true; do
        log "    正在连接..."
        
        # 关键：-rtsp_transport tcp 放在 -i 前面
        ffmpeg -rtsp_transport tcp \
            -re \
            -i "$url" \
            -c:v copy -c:a copy \
            -f rtsp \
            -rtsp_transport tcp \
            "rtsp://127.0.0.1:554/live/$name" \
            2>&1 | while read line; do
                echo "    FFmpeg: $line"
            done
        
        log "    连接断开，5秒后重连..."
        sleep 5
    done
}

log "=== ZLMediaKit 拉流服务启动 ==="

# 等待ZLMediaKit启动
sleep 3

# 读取配置并启动流
while IFS='|' read -r name url; do
    [[ "$name" =~ ^# ]] && continue
    [[ -z "$name" ]] && continue
    
    log "配置: $name -> $url"
    pull_stream "$name" "$url" &
    sleep 1
done < "$STREAMS_CONF"

wait
