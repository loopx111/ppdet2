#!/bin/bash
#
# ZLMediaKit 自动拉流脚本
# 配置文件: /usr/local/ZLMediaKit/streams.conf
#

STREAMS_CONF="/usr/local/ZLMediaKit/streams.conf"
LOG_FILE="/var/log/zlmediakit_pull.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 读取配置文件中的流列表
pull_stream() {
    local name="$1"
    local url="$2"
    
    log "开始拉流: $name -> $url"
    
    # 使用ffmpeg持续推送到ZLMediaKit
    ffmpeg -rtsp_transport tcp \
           -reconnect 1 \
           -reconnect_streamed 1 \
           -reconnect_delay_max 30 \
           -i "$url" \
           -c:v copy -c:a copy \
           -f rtsp \
           rtsp://127.0.0.1:554/live/"$name" \
           >> "$LOG_FILE" 2>&1 &
    
    echo $!
}

# 等待ZLMediaKit启动
sleep 3

log "=== ZLMediaKit 拉流服务启动 ==="

# 读取配置文件
if [ -f "$STREAMS_CONF" ]; then
    while IFS='|' read -r name url; do
        # 跳过注释和空行
        [[ "$name" =~ ^# ]] && continue
        [[ -z "$name" ]] && continue
        
        pull_stream "$name" "$url"
        sleep 2
    done < "$STREAMS_CONF"
else
    log "配置文件 $STREAMS_CONF 不存在，使用默认配置"
fi

log "=== 所有流已启动 ==="

# 保持脚本运行
wait
