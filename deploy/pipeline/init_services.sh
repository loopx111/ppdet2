#!/bin/bash
# 初始化视频流服务脚本 - 开机自启动版本

set -e

echo "=============================================="
echo "初始化视频流服务 (开机自启动版)"
echo "=============================================="

# 检查是否为root用户
if [ "$EUID" -ne 0 ]; then
    echo "请使用 sudo 运行此脚本"
    exit 1
fi

# 1. 安装ZLMediaKit到系统目录
echo "[1/6] 安装ZLMediaKit到系统目录..."
mkdir -p /usr/local/ZLMediaKit
if [ -d "/tmp/ZLMediaKit/release/linux/Release" ]; then
    cp /tmp/ZLMediaKit/release/linux/Release/MediaServer /usr/local/bin/
    cp -r /tmp/ZLMediaKit/release/linux/Release/www /usr/local/ZLMediaKit/
    cp /tmp/ZLMediaKit/release/linux/Release/config.ini /usr/local/ZLMediaKit/
    echo "  [OK] ZLMediaKit 已安装"
else
    echo "  [ERROR] 编译产物未找到，请先编译 ZLMediaKit"
    exit 1
fi

# 2. 安装拉流脚本
echo "[2/6] 安装拉流脚本..."
if [ -f "zlmediakit_pull.sh" ]; then
    cp zlmediakit_pull.sh /usr/local/bin/
    chmod +x /usr/local/bin/zlmediakit_pull.sh
    echo "  [OK] 拉流脚本已安装"
else
    echo "  [ERROR] zlmediakit_pull.sh 不存在"
    exit 1
fi

# 3. 安装流配置文件
echo "[3/6] 安装流配置文件..."
if [ -f "streams.conf" ]; then
    cp streams.conf /usr/local/ZLMediaKit/
    chmod 644 /usr/local/ZLMediaKit/streams.conf
    echo "  [OK] 流配置文件已安装"
    echo "       请编辑 /usr/local/ZLMediaKit/streams.conf 配置你的摄像头地址"
else
    echo "  [ERROR] streams.conf 不存在"
    exit 1
fi

# 4. 安装systemd服务
echo "[4/6] 安装systemd服务..."
if [ -f "zlmediakit.service" ] && [ -f "zlmediakit-pull.service" ]; then
    cp zlmediakit.service /etc/systemd/system/
    cp zlmediakit-pull.service /etc/systemd/system/
    systemctl daemon-reload
    echo "  [OK] systemd 服务已创建"
else
    echo "  [ERROR] systemd 服务文件不存在"
    exit 1
fi

# 5. 启用开机自启动
echo "[5/6] 启用开机自启动..."
systemctl enable zlmediakit
systemctl enable zlmediakit-pull
echo "  [OK] 已设置开机自启动"

# 6. 启动服务
echo "[6/6] 启动服务..."
systemctl start zlmediakit
sleep 2
systemctl start zlmediakit-pull
echo "  [OK] 服务已启动"

echo ""
echo "=============================================="
echo "初始化完成!"
echo "=============================================="
echo ""
echo "服务状态:"
systemctl status zlmediakit --no-pager | head -5
echo ""
echo "服务地址:"
echo "  - ZLMediaKit管理: http://localhost:80"
echo "  - ZLMediaKit API: http://localhost:80/index/api/"
echo "  - RTSP端口: 554"
echo "  - RTMP端口: 1935"
echo ""
echo "播放地址:"
echo "  - RTMP: rtmp://localhost:1935/live/ce64cam"
echo "  - HLS:  http://localhost/live/ce64cam/live.m3u8"
echo "  - WebSocket-FLV: ws://localhost:8088/live/ce64cam.flv"
echo ""
echo "常用命令:"
echo "  查看状态: sudo systemctl status zlmediakit"
echo "  查看日志: sudo journalctl -u zlmediakit -f"
echo "  重启服务: sudo systemctl restart zlmediakit"
echo ""
