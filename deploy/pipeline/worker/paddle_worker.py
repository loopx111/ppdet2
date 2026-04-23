"""
PaddleDetection Worker - 使用底层 API 的独立 AI 推理进程
每个摄像头对应一个 Worker 进程，独立拉流和推理
"""

import argparse
import cv2
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Dict, Optional, Any

# PaddleDetection 路径
# worker 在 deploy/pipeline/worker/，需要向上 4 级
_current_file = Path(__file__).resolve()
PADDLEDET_ROOT = _current_file.parent.parent.parent.parent
DEPLOY_PYTHON = PADDLEDET_ROOT / "deploy" / "python"
PPTRACKING_PYTHON = PADDLEDET_ROOT / "deploy" / "pptracking" / "python"

# 模型缓存目录
MODEL_CACHE_DIR = PADDLEDET_ROOT / "models"

# 添加搜索路径
sys.path.insert(0, str(PPTRACKING_PYTHON))
sys.path.insert(0, str(DEPLOY_PYTHON))

import numpy as np

# NumPy 兼容性修复 - PaddleDetection 2.5 使用了已废弃的 np.float/np.int
np.float = np.float64
np.int = np.int64
np.bool = np.bool_


def download_and_extract_model(url: str, cache_dir: Path) -> Optional[Path]:
    """下载并解压模型"""
    if not url.startswith("http"):
        # 已经是本地路径
        if Path(url).exists():
            return Path(url)
        return None
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成缓存文件名
    url_hash = str(hash(url))
    model_name = url.split("/")[-1].replace(".zip", "")
    extract_dir = cache_dir / model_name
    
    # 检查是否已缓存
    if extract_dir.exists() and (extract_dir / "infer_cfg.yml").exists():
        print(f"[下载] 模型已缓存: {extract_dir}")
        return extract_dir
    
    # 下载
    zip_path = cache_dir / f"{url_hash}.zip"
    
    print(f"[下载] 开始下载模型: {url}")
    print(f"[下载] 保存到: {zip_path}")
    
    try:
        # 使用 curl 下载（更可靠）
        cmd = ["curl", "-L", "-o", str(zip_path), url]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"[下载] curl 失败: {result.stderr}")
            # 尝试用 urllib
            urllib.request.urlretrieve(url, zip_path)
        
        # 解压
        print(f"[下载] 解压到: {extract_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        
        # 清理 zip
        zip_path.unlink()
        
        print(f"[下载] 完成: {extract_dir}")
        return extract_dir
        
    except Exception as e:
        print(f"[下载] 失败: {e}")
        return None


class PaddleMOTDetector:
    """PaddleDetection MOT 检测器 - SDE 分离式 API 封装
    
    SDE 模式特点:
    - 检测器与跟踪器分离
    - 检测器支持 batch_size > 1（多路并行）
    - 跟踪器仍为逐帧处理
    - 适合多路视频流场景
    """
    
    def __init__(
        self,
        model_dir: str,
        tracker_config: str,
        device: str = "GPU",
        run_mode: str = "paddle",
        threshold: float = 0.5,
        skip_frames: int = 2,
        batch_size: int = 1,
        filter_class: list = None,
        cam_id: str = None
    ):
        self.model_dir = model_dir
        self.tracker_config = tracker_config
        self.device = device.upper()
        self.run_mode = run_mode
        self.threshold = threshold
        self.skip_frames = skip_frames
        self.batch_size = batch_size
        self.filter_class = filter_class  # COCO 类别过滤，如 [0] 表示只跟踪人
        self.cam_id = cam_id  # 摄像头 ID（用于日志）
        self.frame_count = 0
        self.detector = None
        self.ready = False
        
        # 下载模型（如果需要）
        actual_model_dir = model_dir
        if model_dir.startswith("http"):
            actual_model_dir = download_and_extract_model(model_dir, MODEL_CACHE_DIR)
            if actual_model_dir is None:
                print("[PaddleMOT] 模型下载失败")
                return
        
        print(f"[PaddleMOT] 模型目录: {actual_model_dir}")
        print(f"[PaddleMOT] 跟踪器配置: {tracker_config}")
        print(f"[PaddleMOT] 设备: {self.device}")
        print(f"[PaddleMOT] batch_size: {self.batch_size}")
        print(f"[PaddleMOT] run_mode: {self.run_mode}")
        if self.filter_class:
            print(f"[PaddleMOT] 类别过滤: {self.filter_class} (COCO: 0=人, 2=车, 3=摩托, 5=公交, 7=卡车)")
        
        # 导入底层 API
        from mot_sde_infer import SDE_Detector
        
        # 初始化检测器
        try:
            self.detector = SDE_Detector(
                model_dir=str(actual_model_dir),
                tracker_config=tracker_config,
                device=self.device,
                run_mode=run_mode,
                batch_size=self.batch_size,
                threshold=threshold,
                save_images=False,
                save_mot_txts=False
            )
            self.ready = True
            print("[PaddleMOT] 检测器初始化成功")
        except Exception as e:
            print(f"[PaddleMOT] 检测器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.ready = False
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """执行单帧检测"""
        if not self.ready:
            return {'boxes': [], 'person_count': 0, 'frame_id': 0}
        
        self.frame_count += 1
        
        try:
            # ============ 耗时统计 ============
            t0 = time.time()
            
            # 预处理
            t_pre = time.time()
            inputs = self.detector.preprocess([frame])
            pre_time = (time.time() - t_pre) * 1000
            
            # 推理
            t_infer = time.time()
            result = self.detector.predict()
            infer_time = (time.time() - t_infer) * 1000
            
            # 后处理
            t_post = time.time()
            det_result = self.detector.postprocess(inputs, result)
            post_time = (time.time() - t_post) * 1000
            
            # 类别过滤 - 如果配置了 filter_class，只保留指定类别
            # det_result['boxes'] 格式: [cls_id, score, x0, y0, x1, y1]
            if self.filter_class is not None:
                boxes = det_result['boxes']
                if len(boxes) > 0:
                    mask = np.isin(boxes[:, 0], self.filter_class)
                    det_result['boxes'] = boxes[mask]
            
            # 跟踪
            t_track = time.time()
            tracking_outs = self.detector.tracking(det_result)
            track_time = (time.time() - t_track) * 1000
            
            total_time = (time.time() - t0) * 1000
            
            # 每 30 帧打印一次详细耗时
            if self.frame_count % 30 == 0:
                detect_count = len(tracking_outs.get('online_ids', [[]])[0]) if isinstance(tracking_outs.get('online_ids'), dict) else len(tracking_outs.get('online_ids', []))
                print(f"[{self.cam_id}] === 耗时统计 (帧#{self.frame_count}) ===")
                print(f"  预处理: {pre_time:6.1f}ms ({pre_time/total_time*100:4.1f}%)")
                print(f"  模型推理: {infer_time:6.1f}ms ({infer_time/total_time*100:4.1f}%)")
                print(f"  后处理: {post_time:6.1f}ms ({post_time/total_time*100:4.1f}%)")
                print(f"  跟踪器: {track_time:6.1f}ms ({track_time/total_time*100:4.1f}%)")
                print(f"  ─────────────────────")
                print(f"  总耗时: {total_time:6.1f}ms | 检测人数: {detect_count}")
            # =================================
            
            # 解析结果
            boxes = []
            online_tlwhs = tracking_outs['online_tlwhs']
            online_scores = tracking_outs['online_scores']
            online_ids = tracking_outs['online_ids']
            
            if isinstance(online_tlwhs, dict):
                for cls_id in online_tlwhs:
                    for i, tlwh in enumerate(online_tlwhs[cls_id]):
                        tid = online_ids[cls_id][i]
                        score = online_scores[cls_id][i]
                        boxes.append({
                            'id': tid,
                            'class': cls_id,
                            'score': float(score),
                            'bbox': [float(tlwh[0]), float(tlwh[1]), 
                                     float(tlwh[0] + tlwh[2]), float(tlwh[1] + tlwh[3])]
                        })
            else:
                for i, tlwh in enumerate(online_tlwhs):
                    boxes.append({
                        'id': online_ids[i],
                        'class': 0,
                        'score': float(online_scores[i]),
                        'bbox': [float(tlwh[0]), float(tlwh[1]), 
                                 float(tlwh[0] + tlwh[2]), float(tlwh[1] + tlwh[3])]
                    })
            
            return {
                'boxes': boxes,
                'person_count': len(boxes),
                'frame_id': self.frame_count
            }
            
        except Exception as e:
            print(f"[PaddleMOT] 检测错误: {e}")
            return {'boxes': [], 'person_count': 0, 'frame_id': self.frame_count}


class RTSPCapture:
    """RTSP 流捕获器"""
    
    def __init__(self, rtsp_url: str, buffer_size: int = 5):
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.capture_thread = None
        self.reconnect_interval = 3
        self.last_reconnect = 0
        self.frame_width = 0
        self.frame_height = 0
        
    def _connect(self) -> bool:
        if self.cap is not None:
            self.cap.release()
        
        # 尝试多种后端
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]:
            self.cap = cv2.VideoCapture(self.rtsp_url, backend)
            if self.cap.isOpened():
                break
        
        if not self.cap.isOpened():
            return False
        
        # 获取帧尺寸
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[RTSP] 已连接: {self.rtsp_url} ({self.frame_width}x{self.frame_height})")
        return True
    
    def _capture_loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                if time.time() - self.last_reconnect > self.reconnect_interval:
                    print(f"[RTSP] 尝试重新连接...")
                    if self._connect():
                        self.last_reconnect = time.time()
                    else:
                        self.last_reconnect = time.time()
                time.sleep(0.5)
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                print(f"[RTSP] 读取失败，重新连接...")
                self.last_reconnect = time.time()
                continue
            
            try:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass
    
    def start(self) -> bool:
        if not self._connect():
            return False
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        return True
    
    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            self.cap = None
    
    @property
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()


class PaddleWorker:
    """PaddleDetection Worker"""
    
    # 模型配置
    # JDE 模式: 一体化模型，batch_size 只能=1，但配置简单
    # SDE 模式: 纯检测器 + 跟踪器，支持 batch_size > 1（需要已导出的推理模型）
    # 
    # PP-YOLOE+_l 模型性能:
    #   - V100 FP32: 78.1 FPS
    #   - V100 TensorRT FP16: 149.2 FPS
    #   - COCO mAP: 53.3
    #   - 导出命令: python tools/export_model.py -c configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml \
    #                -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_80e_coco.pdparams trt=True
    # 
    # COCO 类别参考: class 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
    DEFAULT_MODELS = {
        # JDE 一体化模式（官方提供完整模型包，只跟踪人）
        "mot_jde": {
            "model_dir": "https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip",
            "tracker_config": str(PADDLEDET_ROOT / "deploy" / "pipeline" / "config" / "tracker_config.yml"),
            "mode": "jde",
            "batch_size": 1,
            "run_mode": "paddle",
            "filter_class": None,  # JDE模型已经是行人专用
            "desc": "JDE一体化，配置简单"
        },
        # PP-YOLOE+_l SDE 模式（高精度 + TensorRT 加速 + 只跟踪人）
        "ppyoloe_plus_l_person": {
            "model_dir": str(PADDLEDET_ROOT / "models" / "ppyoloe_plus_crn_l_80e_coco"),
            "tracker_config": str(PADDLEDET_ROOT / "deploy" / "pptracking" / "python" / "tracker_config.yml"),
            "mode": "sde",
            "batch_size": 1,  # TensorRT MOT 模式只支持 batch_size=1
            "run_mode": "trt_fp16",  # TensorRT FP16 加速
            "filter_class": [0],     # 只跟踪人 (COCO class 0)
            "desc": "PP-YOLOE+_l高精度+TRT+只跟踪人"
        },
        # PP-YOLOE+_l SDE 模式（跟踪所有类别）
        "ppyoloe_plus_l_all": {
            "model_dir": str(PADDLEDET_ROOT / "models" / "ppyoloe_plus_crn_l_80e_coco"),
            "tracker_config": str(PADDLEDET_ROOT / "deploy" / "pptracking" / "python" / "tracker_config.yml"),
            "mode": "sde",
            "batch_size": 4,
            "run_mode": "trt_fp16",
            "filter_class": None,    # 跟踪所有类别
            "desc": "PP-YOLOE+_l高精度+TRT+跟踪所有"
        },
        # PP-YOLOE+_s SDE 模式（轻量，速度最快 + 只跟踪人）
        "ppyoloe_plus_s_person": {
            "model_dir": str(PADDLEDET_ROOT / "models" / "ppyoloe_plus_crn_s_80e_coco"),
            "tracker_config": str(PADDLEDET_ROOT / "deploy" / "pptracking" / "python" / "tracker_config.yml"),
            "mode": "sde",
            "batch_size": 8,
            "run_mode": "trt_fp16",
            "filter_class": [0],     # 只跟踪人
            "desc": "PP-YOLOE+_s轻量+TRT+只跟踪人"
        }
    }
    
    # 当前使用的模式
    DEFAULT_MODE = "mot_jde"  # 默认用JDE模式（稳定，无需额外导出）
    
    def __init__(
        self,
        cam_id: str,
        rtsp_url: str,
        scene: str = "person",
        device: str = "gpu",
        skip_frames: int = 3,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        threshold: float = 0.5
    ):
        self.cam_id = cam_id
        self.rtsp_url = rtsp_url
        self.scene = scene
        self.device = device
        self.skip_frames = max(1, skip_frames)
        self.running = True
        self.threshold = threshold
        
        # 统计
        self.frame_count = 0
        self.detect_count = 0
        self.fps_time = time.time()
        self.fps_count = 0
        self.current_fps = 0
        
        # 背景建模 (Fallback)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        
        # Redis 连接
        self.redis_client = None
        self._init_redis(redis_host, redis_port)
        
        # 初始化 MOT 检测器
        self.detector = None
        self._init_detector()
        
        # RTSP 捕获
        self.capture = None
        
        print(f"[{self.cam_id}] Worker 初始化完成")
    
    def _init_redis(self, host: str, port: int):
        try:
            import redis
            self.redis_client = redis.Redis(host=host, port=port, decode_responses=False)
            self.redis_client.ping()
            print(f"[{self.cam_id}] Redis 连接成功")
        except ImportError:
            print(f"[{self.cam_id}] Redis 未安装")
            self.redis_client = None
        except Exception as e:
            print(f"[{self.cam_id}] Redis 连接失败: {e}")
            self.redis_client = None
    
    def _init_detector(self, mode: str = None):
        """初始化 MOT 检测器
        
        Args:
            mode: 模式选择
                - "mot_jde": JDE 一体化模式 (batch_size=1)
                - "ppyoloe_plus_l": PP-YOLOE+_l SDE模式 (高精度+TRT)
                - "ppyoloe_plus_m": PP-YOLOE+_m SDE模式
                - "ppyoloe_plus_s": PP-YOLOE+_s SDE模式 (最快)
                - None: 使用 DEFAULT_MODE
        """
        if mode is None:
            mode = self.DEFAULT_MODE
        
        model_config = self.DEFAULT_MODELS.get(mode, self.DEFAULT_MODELS[self.DEFAULT_MODE])
        
        print(f"[{self.cam_id}] MOT 模式: {mode}")
        print(f"[{self.cam_id}] 描述: {model_config.get('desc', '')}")
        print(f"[{self.cam_id}] batch_size: {model_config.get('batch_size', 1)}")
        print(f"[{self.cam_id}] run_mode: {model_config.get('run_mode', 'paddle')}")
        
        self.detector = PaddleMOTDetector(
            model_dir=model_config["model_dir"],
            tracker_config=model_config["tracker_config"],
            device=self.device,
            run_mode=model_config.get("run_mode", "paddle"),
            threshold=self.threshold,
            batch_size=model_config.get("batch_size", 1),
            filter_class=model_config.get("filter_class"),
            cam_id=self.cam_id  # 传递摄像头 ID
        )
    
    def run(self):
        print(f"[{self.cam_id}] 启动 Worker")
        print(f"[{self.cam_id}] RTSP: {self.rtsp_url}")
        
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # 尝试 RTSP 捕获
        self.capture = RTSPCapture(self.rtsp_url, buffer_size=5)
        if self.capture.start():
            self._run_native_mode()
        else:
            print(f"[{self.cam_id}] RTSP 连接失败，使用背景建模模式")
            self._run_fallback_mode()
        
        self._cleanup()
        print(f"[{self.cam_id}] Worker 已停止")
    
    def _run_native_mode(self):
        print(f"[{self.cam_id}] 运行 RTSP 模式")
        
        while self.running:
            frame = self.capture.read(timeout=1.0)
            if frame is None:
                time.sleep(0.01)
                continue
            
            self.frame_count += 1
            self.fps_count += 1
            
            if time.time() - self.fps_time >= 1.0:
                self.current_fps = self.fps_count
                self.fps_count = 0
                self.fps_time = time.time()
                print(f"[{self.cam_id}] FPS: {self.current_fps}")
            
            if self.frame_count % self.skip_frames == 0:
                self._detect_and_publish(frame)
                self.detect_count += 1
    
    def _run_fallback_mode(self):
        """使用 FFmpeg 或背景建模"""
        print(f"[{self.cam_id}] 运行 Fallback 模式（背景建模）")
        
        ffmpeg_cmd = [
            'ffmpeg', '-rtsp_transport', 'tcp',
            '-fflags', 'nobuffer', '-flags', 'low_delay',
            '-i', self.rtsp_url,
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', '640x360', '-an', 'pipe:1'
        ]
        
        try:
            proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
            print(f"[{self.cam_id}] ffmpeg PID={proc.pid}")
        except Exception as e:
            print(f"[{self.cam_id}] FFmpeg 启动失败: {e}")
            return
        
        frame_size = 640 * 360 * 3
        
        try:
            while self.running:
                raw_frame = b''
                bytes_read = 0
                
                while bytes_read < frame_size and self.running:
                    chunk = proc.stdout.read(min(4096, frame_size - bytes_read))
                    if not chunk:
                        break
                    raw_frame += chunk
                    bytes_read += len(chunk)
                
                if bytes_read != frame_size:
                    if self.running:
                        print(f"[{self.cam_id}] 流结束，重新连接...")
                        time.sleep(1)
                        proc = self._restart_ffmpeg(ffmpeg_cmd)
                        if proc is None:
                            break
                    continue
                
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((360, 640, 3))
                self.frame_count += 1
                
                if self.frame_count % self.skip_frames == 0:
                    self._detect_and_publish(frame)
                
        except Exception as e:
            print(f"[{self.cam_id}] 错误: {e}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    
    def _restart_ffmpeg(self, cmd: list) -> Optional[subprocess.Popen]:
        try:
            return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
        except:
            return None
    
    def _detect_and_publish(self, frame: np.ndarray):
        if self.detector and self.detector.ready:
            result = self.detector.detect(frame)
        else:
            result = self._background_detect(frame)
        
        boxes = result.get('boxes', [])
        person_count = result.get('person_count', len(boxes))
        
        if person_count > 0:
            print(f"[{self.cam_id}] 检测到 {person_count} 人")
        
        detection_result = {
            "camera_id": self.cam_id,
            "person_count": person_count,
            "boxes": boxes,
            "fps": self.current_fps,
            "frame_id": self.frame_count,
            "timestamp": time.time()
        }
        
        self._publish_result(detection_result)
    
    def _background_detect(self, frame: np.ndarray) -> dict:
        """背景建模检测"""
        try:
            fg_mask = self.bg_subtractor.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            count = 0
            
            for contour in contours:
                if cv2.contourArea(contour) < 500:
                    continue
                count += 1
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    "id": count, "class": 0, "score": 0.5,
                    "bbox": [float(x), float(y), float(x + w), float(y + h)]
                })
            
            return {'boxes': detections[:10], 'person_count': count}
        except:
            return {'boxes': [], 'person_count': 0}
    
    def _publish_result(self, result: dict):
        data = json.dumps(result, default=str)
        
        if self.redis_client:
            try:
                self.redis_client.publish(f"detection:{self.cam_id}", data)
                self.redis_client.set(f"latest:{self.cam_id}", data, ex=10)
            except Exception as e:
                print(f"[{self.cam_id}] Redis 发布失败: {e}")
        
        # 输出标准化格式
        boxes = result.get("boxes", [])
        if boxes:
            boxes_str = ""
            for box in boxes:
                bbox = box.get("bbox", [])
                boxes_str += f"{box['id']}:{box.get('class', 0)}:{box.get('score', 0.5)}:"
                boxes_str += f"{int(bbox[0])}:{int(bbox[1])}:{int(bbox[2])}:{int(bbox[3])},"
            print(f"TRACKING_BOXES:{result.get('frame_id', 0)}:{int(time.time()*1000)}:0:0:{boxes_str.rstrip(',')}")
    
    def _cleanup(self):
        if self.capture:
            self.capture.stop()
        if self.redis_client:
            try:
                self.redis_client.close()
            except:
                pass
    
    def _signal_handler(self, signum, frame):
        print(f"[{self.cam_id}] 收到信号，正在停止...")
        self.running = False
    
    def stop(self):
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="PaddleDetection Worker")
    parser.add_argument("--cam-id", required=True, help="摄像头 ID")
    parser.add_argument("--rtsp", required=True, help="RTSP URL")
    parser.add_argument("--scene", default="person", help="检测场景")
    parser.add_argument("--device", default="gpu", help="设备 (gpu/cpu)")
    parser.add_argument("--skip-frames", type=int, default=3, help="跳帧数")
    parser.add_argument("--redis-host", default="localhost", help="Redis 主机")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis 端口")
    parser.add_argument("--threshold", type=float, default=0.5, help="检测阈值")
    parser.add_argument("--mode", default=None, 
                       choices=["mot_jde", "ppyoloe_plus_l_person", "ppyoloe_plus_l_all", "ppyoloe_plus_s_person"],
                       help="模型模式 (默认mot_jde)")
    
    args = parser.parse_args()
    
    # 使用指定的模式或默认模式
    mode = args.mode or PaddleWorker.DEFAULT_MODE
    
    # 创建 Worker 时初始化检测器
    worker = PaddleWorker.__new__(PaddleWorker)
    worker.cam_id = args.cam_id
    worker.rtsp_url = args.rtsp
    worker.scene = args.scene
    worker.device = args.device
    worker.skip_frames = max(1, args.skip_frames)
    worker.running = True
    worker.threshold = args.threshold
    
    # 统计
    worker.frame_count = 0
    worker.detect_count = 0
    worker.fps_time = time.time()
    worker.fps_count = 0
    worker.current_fps = 0
    
    # 背景建模 (Fallback)
    worker.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    
    # Redis 连接
    worker.redis_client = None
    worker._init_redis(args.redis_host, args.redis_port)
    
    # 初始化 MOT 检测器（使用指定模式）
    worker.detector = None
    worker._init_detector(mode)
    
    # RTSP 捕获
    worker.capture = None
    
    print(f"[{worker.cam_id}] Worker 初始化完成 (mode={mode})")
    
    worker.run()


if __name__ == "__main__":
    main()
