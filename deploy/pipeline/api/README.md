# PaddleDetection API 服务

基于FastAPI的PaddleDetection推理服务，支持前后端分离架构。

## 架构设计

```
[摄像头] → [ZLMediaKit] → RTSP流 → [PaddleDetection API]
                   ↓
          [Web前端/移动端] (HLS/WebRTC)
```

## 功能特性

- ✅ RESTful API 接口
- ✅ WebSocket 实时检测流
- ✅ 多会话管理
- ✅ CORS跨域支持
- ✅ 健康检查
- ✅ 限流扩展点（可接入Redis等）
- ✅ 日志记录

## 快速开始

### 1. 安装依赖

```bash
cd deploy/pipeline/api
pip install -r requirements.txt
```

### 2. 配置

编辑 `config.py` 修改：
- 摄像头RTSP地址
- 服务端口
- 推理设备(gpu/cpu)

### 3. 启动服务

```bash
python main.py
# 或使用uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. 访问API文档

打开浏览器访问：http://localhost:8000/docs

## API接口

### 健康检查
```
GET /health
```

### 单帧检测
```
POST /detection/single
Body: {"rtsp_url": "rtsp://...", "device": "gpu"}
```

### WebSocket实时检测
```
WS /ws/detection/{rtsp_url}
```

返回格式：
```json
{
  "type": "detection",
  "session_id": "abc123",
  "frame_id": 100,
  "timestamp": 1700000000000,
  "fps": 10.5,
  "detections": [
    {
      "track_id": 1,
      "class_id": 0,
      "score": 0.95,
      "bbox": [100, 100, 300, 400]
    }
  ],
  "frame_width": 640,
  "frame_height": 360
}
```

### 会话管理
```
GET  /sessions      # 列出所有会话
DELETE /sessions/{id}  # 关闭会话
```

## 前端集成示例

```javascript
// WebSocket连接
const ws = new WebSocket('ws://localhost:8000/ws/detection/rtsp_url');

// 接收检测结果
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'detection') {
    // 绘制检测框
    drawBoxes(data.detections, data.frame_width, data.frame_height);
  }
};

// 主动关闭
ws.close();
```

## 与ZLMediaKit配合使用

1. 部署ZLMediaKit作为视频流代理
2. 摄像头推流到ZLMediaKit
3. ZLMediaKit分发RTSP/HLS/WebRTC流
4. PaddleDetection API拉取RTSP流进行检测
5. 前端通过HLS/WebRTC获取视频，通过API获取检测结果

详见 `ZLMediaKit部署指南.md`
