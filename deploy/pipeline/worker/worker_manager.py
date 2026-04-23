"""
Worker 管理器 - 独立进程
管理多个 PaddleDetection Worker 进程
"""

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# PaddleDetection 路径
PADDLEDET_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PADDLEDET_ROOT))

import paddle


class WorkerManager:
    """Worker 进程管理器"""
    
    def __init__(self, device: str = "gpu", skip_frames: int = 3):
        self.device = device
        self.skip_frames = skip_frames
        self.workers: Dict[str, subprocess.Popen] = {}
        self.running = True
        
        # 信号处理
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def start_worker(self, cam_id: str, rtsp_url: str, scene: str = "person") -> bool:
        """启动单个 Worker"""
        if cam_id in self.workers:
            print(f"[Manager] Worker {cam_id} 已在运行")
            return True
        
        try:
            # 构建命令
            cmd = [
                sys.executable,
                str(PADDLEDET_ROOT / "deploy" / "pipeline" / "worker" / "paddle_worker.py"),
                "--cam-id", cam_id,
                "--rtsp", rtsp_url,
                "--scene", scene,
                "--device", self.device,
                "--skip-frames", str(self.skip_frames)
            ]
            
            # 启动进程
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                env={**subprocess.os.environ, "PYTHONUNBUFFERED": "1"}
            )
            
            self.workers[cam_id] = proc
            print(f"[Manager] 已启动 Worker {cam_id}, PID={proc.pid}")
            return True
            
        except Exception as e:
            print(f"[Manager] 启动 Worker {cam_id} 失败: {e}")
            return False
    
    def stop_worker(self, cam_id: str) -> bool:
        """停止单个 Worker"""
        if cam_id not in self.workers:
            return True
        
        proc = self.workers[cam_id]
        
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        
        del self.workers[cam_id]
        print(f"[Manager] 已停止 Worker {cam_id}")
        return True
    
    def stop_all(self):
        """停止所有 Worker"""
        print(f"[Manager] 正在停止所有 Worker ({len(self.workers)} 个)...")
        for cam_id in list(self.workers.keys()):
            self.stop_worker(cam_id)
        print("[Manager] 所有 Worker 已停止")
    
    def monitor_workers(self):
        """监控 Worker 状态"""
        while self.running:
            for cam_id in list(self.workers.keys()):
                proc = self.workers[cam_id]
                if proc.poll() is not None:
                    print(f"[Manager] Worker {cam_id} 已退出，退出码={proc.returncode}")
                    del self.workers[cam_id]
            
            time.sleep(5)
    
    def _signal_handler(self, signum, frame):
        """信号处理"""
        print(f"[Manager] 收到信号 {signum}")
        self.running = False
        self.stop_all()
    
    def run(self, cameras: list):
        """运行管理器"""
        print("=" * 50)
        print("PaddleDetection Worker 管理器启动")
        print("=" * 50)
        print(f"GPU 可用: {paddle.is_compiled_with_cuda()}")
        print(f"设备: {self.device}")
        print(f"摄像头数量: {len(cameras)}")
        print("=" * 50)
        
        # 启动所有摄像头 Worker
        for cam in cameras:
            cam_id = cam.get("id", "")
            rtsp_url = cam.get("rtsp_url", "")
            scene = cam.get("scene", "person")
            
            if cam_id and rtsp_url:
                self.start_worker(cam_id, rtsp_url, scene)
        
        print(f"已启动 {len(self.workers)} 个 Worker")
        
        # 监控循环
        try:
            self.monitor_workers()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_all()


def main():
    parser = argparse.ArgumentParser(description="PaddleDetection Worker 管理器")
    parser.add_argument("--cam-id", help="单个摄像头 ID (独立运行模式)")
    parser.add_argument("--rtsp", help="RTSP URL (独立运行模式)")
    parser.add_argument("--scene", default="person", help="检测场景")
    parser.add_argument("--device", default="gpu", help="设备 (gpu/cpu)")
    parser.add_argument("--skip-frames", type=int, default=3, help="跳帧数")
    parser.add_argument("--config", help="摄像头配置文件 JSON")
    
    args = parser.parse_args()
    
    manager = WorkerManager(device=args.device, skip_frames=args.skip_frames)
    
    if args.cam_id and args.rtsp:
        # 独立运行单个 Worker 模式
        from worker.paddle_worker import PaddleWorker
        worker = PaddleWorker(
            cam_id=args.cam_id,
            rtsp_url=args.rtsp,
            scene=args.scene,
            device=args.device,
            skip_frames=args.skip_frames
        )
        worker.run()
    else:
        # 管理器模式
        cameras = []
        
        if args.config:
            import json
            with open(args.config) as f:
                config = json.load(f)
                cameras = config.get("cameras", [])
        
        if not cameras:
            # 默认配置
            for i in range(1, 11):
                cameras.append({
                    "id": f"cam{i}",
                    "rtsp_url": f"rtsp://127.0.0.1:554/live/cam{i}",
                    "scene": "person"
                })
        
        manager.run(cameras)


if __name__ == "__main__":
    main()
