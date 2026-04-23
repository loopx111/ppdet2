#!/bin/bash
#
# ZLMediaKit 拉流脚本 - 简化版
#

RTSP_URL="rtsp://admin:cecell123@192.168.1.64:554/Streaming/Channels/102"
STREAM_NAME="ce64cam"
ZLM_HOST="127.0.0.1"
ZLM_PORT="554"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "=== 开始拉流 ==="
log "源地址: $RTSP_URL"
log "目标: rtsp://$ZLM_HOST:$ZLM_PORT/live/$STREAM_NAME"

# 等待ZLMediaKit启动
sleep 3

# 使用ffmpeg持续拉流并推送到ZLMediaKit
exec ffmpeg -rtsp_transport tcp \
    -reconnect 1 \
    -reconnect_streamed 1 \
    -reconnect_delay_max 30 \
    -i "$RTSP_URL" \
    -c:v copy -c:a copy \
    -f rtsp \
    -rtsp_transport tcp \
    rtsp://$ZLM_HOST:$ZLM_PORT/live/$STREAM_NAME
