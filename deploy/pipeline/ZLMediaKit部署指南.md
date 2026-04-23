# ZLMediaKit 部署指南

ZLMediaKit 是一款高性能的流媒体服务器，支持 RTSP/RTMP/HLS/WebRTC 等多种协议，适合作为视频流统一代理。

## 架构设计

```
[摄像头] → [ZLMediaKit] → 分发到多个客户端
              ↓
    ┌─────────┼─────────┐
    ↓         ↓         ↓
  Paddle    Web前端   移动端
  (RTSP)   (HLS/WS)  (HLS)
```

## 源码编译安装

### 1. 安装依赖

```bash
apt update && apt install -y \
    build-essential \
    cmake \
    git \
    libssl-dev \
    libx264-dev \
    liblua5.3-dev \
    libmariadb-dev \
    libjsoncpp-dev \
    libspdlog-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    libavformat-dev \
    lib麦克风-dev \
    pkg-config
```

### 2. 克隆源码

```bash
cd /tmp
git clone --depth 1 https://github.com/ZLMediaKit/ZLMediaKit.git
cd ZLMediaKit
```

### 3. 初始化子模块

```bash
git submodule update --init
```

### 4. 编译

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 5. 安装到系统目录

```bash
# 安装可执行文件
sudo cp /tmp/ZLMediaKit/release/linux/Release/MediaServer /usr/local/bin/

# 安装配置文件
sudo mkdir -p /usr/local/ZLMediaKit
sudo cp /tmp/ZLMediaKit/release/linux/Release/config.ini /usr/local/ZLMediaKit/
sudo cp -r /tmp/ZLMediaKit/release/linux/Release/www /usr/local/ZLMediaKit/
```

## 低延迟配置

修改 `/usr/local/ZLMediaKit/config.ini`：

### [general] 部分

```ini
[general]
maxStreamWaitMS = 5000
streamNoneReaderDelayMS = 5000
```

### [hls] 部分

```ini
[hls]
segDur = 1
segNum = 1
```

### [rtp] 部分

```ini
[rtp]
lowLatency = 1
```

## 开机自启动配置

### 1. 创建 systemd 服务

**ZLMediaKit 服务** (`/etc/systemd/system/zlmediakit.service`)：

```ini
[Unit]
Description=ZLMediaKit Media Server
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/MediaServer -c /usr/local/ZLMediaKit/config.ini
Restart=always
RestartSec=5
StandardOutput=append:/var/log/zlmediakit.log
StandardError=append:/var/log/zlmediakit.log

[Install]
WantedBy=multi-user.target
```

### 2. 创建自动拉流脚本

**自动拉流脚本** (`/usr/local/bin/zlm_auto_pull.sh`)：

```bash
#!/bin/bash
#
# ZLMediaKit 开机自动拉流脚本
#

API_URL="http://127.0.0.1:80/index/api/addStreamProxy"
SECRET="你的API密钥"  # 从 config.ini 的 api.secret 获取

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "等待 ZLMediaKit 启动..."
sleep 5

log "开始自动拉流..."

# 摄像头1
curl -s -X POST "$API_URL" \
  -d "secret=$SECRET&vhost=__defaultVhost__&app=live&stream=cam1&url=rtsp://admin:密码@摄像头IP:554/Streaming/Channels/102&rtp_type=0" \
  && log "cam1 拉流成功" || log "cam1 拉流失败"

log "自动拉流完成"
```

```bash
sudo chmod +x /usr/local/bin/zlm_auto_pull.sh
```

### 3. 创建自动拉流 systemd 服务

```ini
[Unit]
Description=ZLMediaKit Auto Pull Stream
After=zlmediakit.service
Requires=zlmediakit.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/zlm_auto_pull.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

### 4. 启用服务

```bash
sudo systemctl daemon-reload
sudo systemctl enable zlmediakit
sudo systemctl enable zlm-auto-pull
```

## 手动操作命令

### 启动服务

```bash
# 启动 ZLMediaKit
sudo systemctl start zlmediakit

# 启动自动拉流
sudo systemctl start zlm-auto-pull
```

### 查看状态

```bash
sudo systemctl status zlmediakit
sudo systemctl status zlm-auto-pull
```

### 查看日志

```bash
# ZLMediaKit 日志
sudo journalctl -u zlmediakit -f

# 拉流日志
cat /var/log/zlmediakit.log
```

### 重启服务

```bash
sudo systemctl restart zlmediakit
sudo systemctl restart zlm-auto-pull
```

## API 手动拉流

如果需要手动添加流，使用 API：

```bash
curl -X POST "http://127.0.0.1:80/index/api/addStreamProxy" \
  -d "secret=你的API密钥&vhost=__defaultVhost__&app=live&stream=cam1&url=rtsp://admin:密码@摄像头IP:554/Streaming/Channels/102&rtp_type=0"
```

**参数说明：**
- `secret`: API密钥，从 `/usr/local/ZLMediaKit/config.ini` 的 `api.secret` 获取
- `vhost`: 虚拟主机，填 `__defaultVhost__`
- `app`: 应用名，填 `live`
- `stream`: 流名称，自定义，如 `cam1`
- `url`: 摄像头原始 RTSP 地址
- `rtp_type`: 0=TCP, 1=UDP，推荐 0

## 播放地址

配置成功后，播放地址：

| 类型 | 地址 |
|------|------|
| RTSP | `rtsp://127.0.0.1:554/live/cam1` |
| RTMP | `rtmp://127.0.0.1:1935/live/cam1` |
| HLS | `http://127.0.0.1/live/cam1/live.m3u8` |
| WebSocket-FLV | `ws://127.0.0.1:8088/live/cam1.flv` |

## 播放测试

### FFplay 低延迟播放

```bash
# RTSP 播放
ffplay -fflags nobuffer -flags low_delay rtsp://127.0.0.1:554/live/cam1

# HLS 播放
ffplay -fflags nobuffer -flags low_delay http://127.0.0.1/live/cam1/live.m3u8
```

### VLC 播放

```
# RTSP
rtsp://127.0.0.1:554/live/cam1

# RTMP
rtmp://127.0.0.1:1935/live/cam1
```

## 与 PaddleDetection 配合

### Python 配置

```python
# config.py
CAMERA_RTSP = "rtsp://127.0.0.1:554/live/cam1"
```

### FFmpeg 拉流命令

```bash
ffmpeg -rtsp_transport tcp -i "rtsp://127.0.0.1:554/live/cam1" -c:v copy -f null -
```

## 端口说明

| 端口 | 协议 | 用途 |
|------|------|------|
| 554 | RTSP | 拉流/推流 |
| 1935 | RTMP | 直播推流 |
| 80 | HTTP | Web界面/HLS |
| 443 | HTTPS | HTTPS访问 |
| 9000/10000 | WebRTC | WebRTC信令/数据 |

## 常见问题

### 1. 延迟高

- 使用 ZLMediaKit API 直接拉流（不用外部 FFmpeg）
- 设置 `rtp_type=0` 使用 TCP 传输
- 播放时加参数：`-fflags nobuffer -flags low_delay`

### 2. 401 Unauthorized 错误

- RTSP 地址需要包含用户名密码
- 格式：`rtsp://用户名:密码@IP:554/路径`
- 示例：`rtsp://admin:cecell123@192.168.3.64:554/Streaming/Channels/102`

### 3. 406 Not Acceptable 错误

- 流名称被占用，先停掉之前的推流
- 执行 `pkill -f ffmpeg` 杀死残留进程

### 4. API 调用失败

- 检查 API 密钥是否正确
- 确认 ZLMediaKit 已启动：`sudo systemctl status zlmediakit`

## 获取 API 密钥

启动 ZLMediaKit 后，日志会显示自动生成的密钥：

```
The api.secret is invalid, modified it to: Tyao9jriLj5CYRfSxG6nj121P9EaRrGU
```

或在配置文件中查看：

```bash
grep "secret=" /usr/local/ZLMediaKit/config.ini
```

## Web 管理界面

访问：`http://服务器IP:80`

首次登录使用上述 API 密钥。
