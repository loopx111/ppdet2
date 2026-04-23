"""
FastAPI 主服务 - 只做控制层，不做 AI 推理
职责：HTTP API + WebSocket + 告警管理
"""

import asyncio
import json
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import paddle

# ==================== 配置 ====================

API_HOST = "0.0.0.0"
API_PORT = 8000

# Worker 管理器路径
WORKER_MANAGER = Path(__file__).parent.parent / "worker" / "worker_manager.py"

# 摄像头配置
CAMERAS_CONFIG = Path(__file__).parent.parent / "config" / "cameras.json"

# 静态文件
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# 截图目录
SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)


# ==================== 枚举定义 ====================

class EventType(str, Enum):
    PERSON_DETECTED = "person_detected"
    PERSON_COUNT_CHANGE = "person_count_change"
    PERSON_ENTER = "person_enter"
    PERSON_EXIT = "person_exit"
    FIGHT_DETECTED = "fight_detected"
    FALL_DETECTED = "fall_detected"
    SMOKING_DETECTED = "smoking_detected"
    CALLING_DETECTED = "calling_detected"


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"


# ==================== 数据模型 ====================

@dataclass
class Camera:
    """摄像头"""
    id: str
    name: str
    rtsp_url: str
    enabled: bool = True
    scene: str = "person"
    status: str = "stopped"  # stopped, running, error
    worker_pid: Optional[int] = None
    last_person_count: int = 0
    last_update: float = field(default_factory=time.time)


@dataclass
class DetectionResult:
    """检测结果"""
    camera_id: str
    person_count: int
    trackid_number: int
    boxes: List[Dict]
    fps: float
    timestamp: float


@dataclass
class AlertRecord:
    """告警记录"""
    id: str
    camera_id: str
    event_type: str
    level: str
    message: str
    person_count: int
    snapshot_path: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False


# ==================== Pydantic 模型 ====================

class CameraCreate(BaseModel):
    id: str
    name: str
    rtsp_url: str
    scene: str = "person"


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    rtsp_url: Optional[str] = None
    enabled: Optional[bool] = None
    scene: Optional[str] = None


class AlertRuleRequest(BaseModel):
    camera_id: str
    event_type: str
    enabled: bool = True
    threshold: float = 1
    level: str = "warning"


@dataclass
class AlertRule:
    id: str
    camera_id: str
    event_type: str
    enabled: bool
    threshold: float
    level: str
    created_at: float


# ==================== 全局状态 ====================

app = FastAPI(
    title="PaddleDetection 行为分析平台",
    description="控制层 API + AI Worker 管理",
    version="2.0.0"
)

# 摄像头管理
cameras: Dict[str, Camera] = {}
cameras_lock = asyncio.Lock()

# 告警规则
alert_rules: Dict[str, AlertRule] = {}
alert_rules_lock = asyncio.Lock()

# 告警记录
alert_records: List[AlertRecord] = []
MAX_ALERT_RECORDS = 1000
alert_records_lock = asyncio.Lock()

# WebSocket 连接
websocket_connections: Dict[str, List[WebSocket]] = {}  # camera_id -> [ws connections]


# ==================== CORS ====================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 应用生命周期 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    print("=" * 50)
    print("PaddleDetection 控制平台启动")
    print("=" * 50)
    print(f"GPU可用: {paddle.is_compiled_with_cuda()}")
    print(f"PaddlePaddle版本: {paddle.__version__}")
    print(f"Worker管理器: {WORKER_MANAGER}")
    print("=" * 50)
    
    # 加载摄像头配置
    await load_cameras_config()
    
    # 创建默认告警规则
    await create_default_rules()
    
    # 启动结果订阅任务
    asyncio.create_task(subscribe_detection_results())
    
    yield
    
    # 停止所有 Worker
    for cam_id in list(cameras.keys()):
        await stop_camera_worker(cam_id)
    
    print("服务已关闭")


app.router.lifespan_context = lifespan


# ==================== 配置加载 ====================

async def load_cameras_config():
    """加载摄像头配置"""
    global cameras
    
    if CAMERAS_CONFIG.exists():
        try:
            with open(CAMERAS_CONFIG) as f:
                config = json.load(f)
                for cam_data in config.get("cameras", []):
                    cam = Camera(**cam_data)
                    cameras[cam.id] = cam
                print(f"已加载 {len(cameras)} 个摄像头配置")
        except Exception as e:
            print(f"加载摄像头配置失败: {e}")
    
    # 如果没有配置，创建默认配置
    if not cameras:
        for i in range(1, 11):
            cam_id = f"cam{i}"
            cameras[cam_id] = Camera(
                id=cam_id,
                name=f"摄像头 {i}",
                rtsp_url=f"rtsp://127.0.0.1:554/live/{cam_id}",
                scene="person"
            )
        print("已创建 10 个默认摄像头配置")


async def save_cameras_config():
    """保存摄像头配置"""
    try:
        config = {
            "cameras": [asdict(cam) for cam in cameras.values()]
        }
        CAMERAS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        with open(CAMERAS_CONFIG, 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"保存摄像头配置失败: {e}")


# ==================== Worker 管理 ====================

async def start_camera_worker(cam_id: str) -> bool:
    """启动摄像头 AI Worker"""
    if cam_id not in cameras:
        return False
    
    cam = cameras[cam_id]
    if cam.status == "running":
        return True
    
    try:
        # 启动 Worker 进程
        proc = subprocess.Popen(
            [
                "python", str(WORKER_MANAGER),
                "--cam-id", cam_id,
                "--rtsp", cam.rtsp_url,
                "--scene", cam.scene,
                "--device", "gpu" if paddle.is_compiled_with_cuda() else "cpu"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**subprocess.os.environ, "PYTHONUNBUFFERED": "1"}
        )
        
        cam.worker_pid = proc.pid
        cam.status = "running"
        
        print(f"已启动 Worker: {cam_id}, PID={proc.pid}")
        return True
        
    except Exception as e:
        print(f"启动 Worker 失败: {e}")
        cam.status = "error"
        return False


async def stop_camera_worker(cam_id: str) -> bool:
    """停止摄像头 AI Worker"""
    if cam_id not in cameras:
        return False
    
    cam = cameras[cam_id]
    
    try:
        if cam.worker_pid:
            import signal
            try:
                subprocess.os.kill(cam.worker_pid, signal.SIGTERM)
                print(f"已停止 Worker: {cam_id}, PID={cam.worker_pid}")
            except ProcessLookupError:
                pass
        
        cam.worker_pid = None
        cam.status = "stopped"
        return True
        
    except Exception as e:
        print(f"停止 Worker 失败: {e}")
        return False


async def subscribe_detection_results():
    """订阅 Worker 的检测结果 (简化版: 直接订阅 Redis)"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        pubsub = r.pubsub()
        pubsub.psubscribe("detection:*")
        
        for message in pubsub.listen():
            if message["type"] == "pmessage":
                try:
                    data = json.loads(message["data"])
                    camera_id = data.get("camera_id", "")
                    
                    # 更新摄像头状态
                    if camera_id in cameras:
                        cameras[camera_id].last_person_count = data.get("person_count", 0)
                        cameras[camera_id].last_update = time.time()
                    
                    # 检查告警规则
                    await check_alert_rules(data)
                    
                    # 广播到 WebSocket
                    await broadcast_detection(data)
                    
                except json.JSONDecodeError:
                    pass
                    
    except ImportError:
        print("Redis 未安装，使用轮询模式")
    except Exception as e:
        print(f"订阅检测结果失败: {e}")


async def check_alert_rules(data: dict):
    """检查告警规则"""
    camera_id = data.get("camera_id", "")
    person_count = data.get("person_count", 0)
    
    async with alert_rules_lock:
        for rule in alert_rules.values():
            if rule.camera_id != camera_id or not rule.enabled:
                continue
            
            if rule.event_type == EventType.PERSON_DETECTED and person_count >= rule.threshold:
                await create_alert(
                    camera_id=camera_id,
                    event_type=EventType.PERSON_DETECTED,
                    level=AlertLevel(rule.level),
                    message=f"检测到 {person_count} 个人",
                    person_count=person_count
                )
            
            elif rule.event_type == EventType.PERSON_COUNT_CHANGE:
                prev_count = cameras[camera_id].last_person_count if camera_id in cameras else 0
                if person_count != prev_count:
                    if person_count > prev_count:
                        msg = f"人员进入，当前 {person_count} 人"
                    else:
                        msg = f"人员离开，当前 {person_count} 人"
                    await create_alert(
                        camera_id=camera_id,
                        event_type=EventType.PERSON_COUNT_CHANGE,
                        level=AlertLevel(rule.level),
                        message=msg,
                        person_count=person_count
                    )


async def create_default_rules():
    """创建默认告警规则"""
    for i in range(1, 11):
        cam_id = f"cam{i}"
        rules = [
            AlertRule(
                id=f"rule_{cam_id}_001",
                camera_id=cam_id,
                event_type=EventType.PERSON_DETECTED,
                enabled=True,
                threshold=1,
                level=AlertLevel.INFO,
                created_at=time.time()
            ),
            AlertRule(
                id=f"rule_{cam_id}_002",
                camera_id=cam_id,
                event_type=EventType.PERSON_COUNT_CHANGE,
                enabled=True,
                threshold=1,
                level=AlertLevel.WARNING,
                created_at=time.time()
            ),
        ]
        for rule in rules:
            alert_rules[rule.id] = rule


# ==================== WebSocket 广播 ====================

async def broadcast_detection(data: dict):
    """广播检测结果"""
    camera_id = data.get("camera_id", "")
    
    disconnected = []
    for ws in websocket_connections.get(camera_id, []):
        try:
            await ws.send_json({"type": "detection", "data": data})
        except:
            disconnected.append(ws)
    
    for ws in disconnected:
        if ws in websocket_connections.get(camera_id, []):
            websocket_connections[camera_id].remove(ws)


async def broadcast_alert(record: AlertRecord):
    """广播告警"""
    message = {"type": "alert", "data": asdict(record)}
    
    disconnected = []
    for cam_id in websocket_connections:
        for ws in websocket_connections[cam_id]:
            try:
                await ws.send_json(message)
            except:
                disconnected.append(ws)
    
    for ws in disconnected:
        for cam_id in websocket_connections:
            if ws in websocket_connections[cam_id]:
                websocket_connections[cam_id].remove(ws)


async def create_alert(camera_id, event_type, level, message, person_count, snapshot_path=None):
    """创建告警"""
    record = AlertRecord(
        id=str(uuid.uuid4())[:8],
        camera_id=camera_id,
        event_type=event_type.value,
        level=level.value,
        message=message,
        person_count=person_count,
        snapshot_path=snapshot_path,
        timestamp=time.time()
    )
    
    async with alert_records_lock:
        alert_records.append(record)
        if len(alert_records) > MAX_ALERT_RECORDS:
            alert_records.pop(0)
    
    await broadcast_alert(record)
    return record


# ==================== 工具函数 ====================

def save_snapshot(camera_id: str, event_type: str) -> Optional[str]:
    """保存截图"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{camera_id}_{event_type}_{timestamp}.jpg"
        filepath = SNAPSHOT_DIR / filename
        # TODO: 从 Redis 获取最新帧
        return f"/snapshots/{filename}"
    except Exception as e:
        print(f"保存截图失败: {e}")
        return None


# ==================== API 路由 ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """首页"""
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text())
    return {"message": "Web前端未找到"}


@app.get("/health")
async def health_check():
    """健康检查"""
    running_count = sum(1 for c in cameras.values() if c.status == "running")
    return {
        "status": "healthy",
        "gpu_available": paddle.is_compiled_with_cuda(),
        "total_cameras": len(cameras),
        "running_cameras": running_count,
        "alert_rules_count": len(alert_rules),
        "alert_records_count": len(alert_records),
        "websocket_connections": sum(len(v) for v in websocket_connections.values())
    }


# ==================== 摄像头 API ====================

@app.get("/cameras")
async def get_cameras():
    """获取所有摄像头"""
    return {
        "cameras": [asdict(cam) for cam in cameras.values()]
    }


@app.get("/cameras/{cam_id}")
async def get_camera(cam_id: str):
    """获取单个摄像头"""
    if cam_id not in cameras:
        raise HTTPException(status_code=404, detail="摄像头不存在")
    return asdict(cameras[cam_id])


@app.post("/cameras")
async def create_camera(cam: CameraCreate):
    """创建摄像头"""
    if cam.id in cameras:
        raise HTTPException(status_code=400, detail="摄像头已存在")
    
    new_cam = Camera(
        id=cam.id,
        name=cam.name,
        rtsp_url=cam.rtsp_url,
        scene=cam.scene
    )
    cameras[cam.id] = new_cam
    await save_cameras_config()
    return asdict(new_cam)


@app.put("/cameras/{cam_id}")
async def update_camera(cam_id: str, update: CameraUpdate):
    """更新摄像头"""
    if cam_id not in cameras:
        raise HTTPException(status_code=404, detail="摄像头不存在")
    
    cam = cameras[cam_id]
    if update.name is not None:
        cam.name = update.name
    if update.rtsp_url is not None:
        cam.rtsp_url = update.rtsp_url
    if update.enabled is not None:
        cam.enabled = update.enabled
    if update.scene is not None:
        cam.scene = update.scene
    
    await save_cameras_config()
    return asdict(cam)


@app.delete("/cameras/{cam_id}")
async def delete_camera(cam_id: str):
    """删除摄像头"""
    if cam_id not in cameras:
        raise HTTPException(status_code=404, detail="摄像头不存在")
    
    await stop_camera_worker(cam_id)
    del cameras[cam_id]
    await save_cameras_config()
    return {"message": "摄像头已删除"}


@app.post("/cameras/{cam_id}/start")
async def start_camera(cam_id: str):
    """启动摄像头检测"""
    if cam_id not in cameras:
        raise HTTPException(status_code=404, detail="摄像头不存在")
    
    success = await start_camera_worker(cam_id)
    if success:
        return {"message": f"摄像头 {cam_id} 已启动", "status": cameras[cam_id].status}
    raise HTTPException(status_code=500, detail="启动失败")


@app.post("/cameras/{cam_id}/stop")
async def stop_camera(cam_id: str):
    """停止摄像头检测"""
    if cam_id not in cameras:
        raise HTTPException(status_code=404, detail="摄像头不存在")
    
    success = await stop_camera_worker(cam_id)
    if success:
        return {"message": f"摄像头 {cam_id} 已停止", "status": cameras[cam_id].status}
    raise HTTPException(status_code=500, detail="停止失败")


# ==================== WebSocket ====================

@app.websocket("/ws/{cam_id}")
async def websocket_endpoint(websocket: WebSocket, cam_id: str):
    """WebSocket 实时通信"""
    if cam_id not in cameras:
        await websocket.close(code=4004)
        return
    
    await websocket.accept()
    
    if cam_id not in websocket_connections:
        websocket_connections[cam_id] = []
    websocket_connections[cam_id].append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in websocket_connections.get(cam_id, []):
            websocket_connections[cam_id].remove(websocket)


# ==================== 告警规则 API ====================

@app.get("/api/rules")
async def get_rules():
    """获取所有告警规则"""
    return {"rules": [asdict(r) for r in alert_rules.values()]}


@app.post("/api/rules")
async def create_rule(rule_req: AlertRuleRequest):
    """创建告警规则"""
    rule = AlertRule(
        id=str(uuid.uuid4())[:8],
        camera_id=rule_req.camera_id,
        event_type=rule_req.event_type,
        enabled=rule_req.enabled,
        threshold=rule_req.threshold,
        level=rule_req.level,
        created_at=time.time()
    )
    alert_rules[rule.id] = rule
    return asdict(rule)


@app.put("/api/rules/{rule_id}")
async def update_rule(rule_id: str, rule_req: AlertRuleRequest):
    """更新告警规则"""
    if rule_id not in alert_rules:
        raise HTTPException(status_code=404, detail="规则不存在")
    
    alert_rules[rule_id] = AlertRule(
        id=rule_id,
        camera_id=rule_req.camera_id,
        event_type=rule_req.event_type,
        enabled=rule_req.enabled,
        threshold=rule_req.threshold,
        level=rule_req.level,
        created_at=alert_rules[rule_id].created_at
    )
    return asdict(alert_rules[rule_id])


@app.delete("/api/rules/{rule_id}")
async def delete_rule(rule_id: str):
    """删除告警规则"""
    if rule_id not in alert_rules:
        raise HTTPException(status_code=404, detail="规则不存在")
    del alert_rules[rule_id]
    return {"message": "规则已删除"}


# ==================== 告警记录 API ====================

@app.get("/api/alerts")
async def get_alerts(
    camera_id: str = Query(None),
    event_type: str = Query(None),
    limit: int = Query(100, le=1000)
):
    """获取告警记录"""
    records = list(alert_records)
    
    if camera_id:
        records = [r for r in records if r.camera_id == camera_id]
    if event_type:
        records = [r for r in records if r.event_type == event_type]
    
    records = sorted(records, key=lambda x: x.timestamp, reverse=True)[:limit]
    return {"total": len(records), "alerts": [asdict(r) for r in records]}


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """确认告警"""
    for record in alert_records:
        if record.id == alert_id:
            record.acknowledged = True
            return {"message": "告警已确认"}
    raise HTTPException(status_code=404, detail="告警不存在")


@app.delete("/api/alerts")
async def clear_alerts():
    """清空告警记录"""
    alert_records.clear()
    return {"message": "告警记录已清空"}


# ==================== 统计 API ====================

@app.get("/api/stats")
async def get_stats():
    """获取统计数据"""
    running_count = sum(1 for c in cameras.values() if c.status == "running")
    
    type_counts = {}
    for r in alert_records:
        type_counts[r.event_type] = type_counts.get(r.event_type, 0) + 1
    
    return {
        "total_cameras": len(cameras),
        "running_cameras": running_count,
        "total_alerts": len(alert_records),
        "alert_by_type": type_counts,
        "websocket_connections": sum(len(v) for v in websocket_connections.values())
    }


# ==================== 启动 ====================

def main():
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        workers=1,
        log_level="info"
    )


if __name__ == "__main__":
    main()
