#!/bin/bash
#
# ZLMediaKit 开机自动拉流脚本
# 通过API让ZLMediaKit直接拉取摄像头流
#

API_URL="http://127.0.0.1:80/index/api/addStreamProxy"
SECRET="Tyao9jriLj5CYRfSxG6nj121P9EaRrGU"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "等待 ZLMediaKit 启动..."
sleep 5

log "开始自动拉流..."

# 摄像头配置 (流名称|RTSP地址)
# 如需添加多个摄像头，复制下面的curl命令即可

# 摄像头1
curl -s -X POST "$API_URL" \
  -d "secret=$SECRET&vhost=__defaultVhost__&app=live&stream=cam1&url=rtsp://admin:cecell123@192.168.3.64:554/Streaming/Channels/102&rtp_type=0" \
  && log "cam1 拉流成功" || log "cam1 拉流失败"

# 如需添加更多摄像头，取消下面注释并修改：
# curl -s -X POST "$API_URL" \
#   -d "secret=$SECRET&vhost=__defaultVhost__&app=live&stream=cam2&url=rtsp://用户名:密码@IP地址:554/Streaming/Channels/101&rtp_type=0" \
#   && log "cam2 拉流成功" || log "cam2 拉流失败"

log "自动拉流完成"
