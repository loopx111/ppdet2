#!/usr/bin/env python
"""
PaddleDetection GUI 主程序
功能：实时显示PaddleDetection识别结果，支持多场景切换，显示检测结果列表
"""

import sys
import os
import platform
import cv2
import threading
import queue
import subprocess
import time
import signal
from datetime import datetime
from collections import deque

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import json

class DetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PaddleDetection Real-time Monitoring System")
        self.root.geometry("1400x900")
        
        # 初始化变量
        self.running = False
        self.current_scene = "人物检测"
        self.rtsp_url = "rtsp://admin:cecell123@192.168.3.64:554/Streaming/Channels/102"
        self.detection_process = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.results_queue = queue.Queue()
        self.video_thread = None
        self.results_history = []
        self.last_frame_time = time.time()
        self.fps = 0
        self.frame_count = 0
        
        # 实时统计相关
        self.last_object_count = None
        self.result_update_interval = 2.0  # 最小更新间隔（秒）
        self.last_result_update_time = time.time()
        self.pending_results = []  # 待处理的结果
        
        # 检测框绘制开关 - 初始为关闭，后期逐步添加功能
        self.draw_detection_boxes = False  # True: 绘制检测框, False: 不绘制
        
        # 检测框相关变量
        self.last_tracking_boxes = None  # 存储最新的检测框信息
        self.tracking_boxes_dict = {}  # 按帧号存储检测框信息，用于时间戳同步
        
        # 配置文件映射
        self.scene_configs = {
            "人物检测": "infer_cfg_pphuman.yml",
            "人物属性": "examples/infer_cfg_human_attr.yml",
            "打架检测": "examples/infer_cfg_fight_recognition.yml",
            "吸烟检测": "examples/infer_cfg_smoking.yml",
            "摔倒检测": "examples/infer_cfg_fall_down.yml",
            "打电话检测": "examples/infer_cfg_calling.yml"
        }
        
        # 设置中文字体
        self.setup_chinese_font()
        
        # 日志相关设置（已删除GUI日志显示，所有日志输出到终端）
        # 保留日志队列用于可能的其他用途
        self.log_queue = queue.Queue()
        
        # 设置关闭窗口事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 创建GUI
        self.create_widgets()
    
    def setup_chinese_font(self):
        """设置中文字体"""
        # 常见的中文字体列表
        chinese_fonts = [
            "WenQuanYi Zen Hei",     # 文泉驿正黑
            "WenQuanYi Micro Hei",   # 文泉驿微米黑
            "Noto Sans CJK SC",      # Google Noto字体
            "Noto Sans CJK TC",
            "DejaVu Sans",           # 备选
            "Arial Unicode MS",
            "Microsoft YaHei",       # 微软雅黑（如果WSL有Windows字体）
            "SimSun",                # 宋体
            "SimHei",                # 黑体
            "FangSong",              # 仿宋
            "KaiTi",                 # 楷体
            "sans-serif"             # 最后备选
        ]
        
        # 尝试不同的字体
        for font_name in chinese_fonts:
            try:
                # 创建测试标签
                test_label = tk.Label(self.root, text="测试中文", font=(font_name, 12))
                test_label.destroy()
                self.font_family = font_name
                print(f"使用字体: {font_name}")
                break
            except:
                continue
        else:
            # 如果没有找到合适的字体，使用默认
            self.font_family = "TkDefaultFont"
            print("警告: 未找到中文字体，使用默认字体")
        
        # 设置全局样式
        style = ttk.Style()
        style.configure("TLabel", font=(self.font_family, 10))
        style.configure("TButton", font=(self.font_family, 10))
        style.configure("TEntry", font=(self.font_family, 10))
        style.configure("TCombobox", font=(self.font_family, 10))
        style.configure("TLabelframe", font=(self.font_family, 10))
        style.configure("TLabelframe.Label", font=(self.font_family, 10, "bold"))
        
    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置行列权重 - 2列布局（已删除日志列）
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)  # 视频列
        main_frame.columnconfigure(1, weight=1)  # 结果列
        main_frame.rowconfigure(1, weight=1)
        
        # 控制面板（顶部）
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # RTSP URL输入
        ttk.Label(control_frame, text="RTSP URL:").grid(row=0, column=0, padx=(0, 5))
        self.url_var = tk.StringVar(value=self.rtsp_url)
        url_entry = ttk.Entry(control_frame, textvariable=self.url_var, width=50)
        url_entry.grid(row=0, column=1, padx=(0, 10))
        
        # 场景选择
        ttk.Label(control_frame, text="检测场景:").grid(row=0, column=2, padx=(0, 5))
        self.scene_var = tk.StringVar(value=self.current_scene)
        scene_combo = ttk.Combobox(control_frame, textvariable=self.scene_var, 
                                   values=list(self.scene_configs.keys()),
                                   state="readonly", width=15)
        scene_combo.grid(row=0, column=3, padx=(0, 10))
        scene_combo.bind("<<ComboboxSelected>>", self.on_scene_change)
        
        # 启动/停止按钮
        self.start_btn = ttk.Button(control_frame, text="启动检测", command=self.start_detection)
        self.start_btn.grid(row=0, column=4, padx=(0, 10))
        
        self.stop_btn = ttk.Button(control_frame, text="停止检测", command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=5, padx=(0, 10))
        
        # 测试按钮（仅显示原始视频）
        self.test_btn = ttk.Button(control_frame, text="测试原始视频", command=self.start_raw_video_test)
        self.test_btn.grid(row=0, column=6, padx=(0, 10))
        
        # 检测框绘制开关按钮
        self.toggle_boxes_btn = ttk.Button(control_frame, text="开启检测框", command=self.toggle_detection_boxes)
        self.toggle_boxes_btn.grid(row=0, column=7, padx=(0, 10))
        
        # 性能显示
        self.fps_label = ttk.Label(control_frame, text="FPS: 0")
        self.fps_label.grid(row=0, column=8, padx=(0, 10))
        
        # 主显示区域 - 2列布局（已删除日志列）
        # 第1列：视频显示
        video_frame = ttk.LabelFrame(main_frame, text="实时视频流", padding="10")
        video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # 视频显示画布
        self.video_canvas = tk.Canvas(video_frame, width=640, height=360, bg="black")
        self.video_canvas.grid(row=0, column=0)
        
        # 第2列：信息面板（扩展宽度，占用了原来的日志列空间）
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # 实时统计信息
        stats_frame = ttk.LabelFrame(info_frame, text="实时统计", padding="10")
        stats_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 创建统计变量
        self.total_frames_var = tk.StringVar(value="总帧数: 0")
        self.total_objects_var = tk.StringVar(value="总检测数: 0")
        self.avg_fps_var = tk.StringVar(value="平均FPS: 0.0")
        
        ttk.Label(stats_frame, textvariable=self.total_frames_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(stats_frame, textvariable=self.total_objects_var).grid(row=1, column=0, sticky=tk.W)
        ttk.Label(stats_frame, textvariable=self.avg_fps_var).grid(row=2, column=0, sticky=tk.W)
        
        # 检测结果列表（增加高度，占用了原来日志的空间）
        results_frame = ttk.LabelFrame(info_frame, text="检测结果列表", padding="10")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 创建滚动文本框（增加高度）
        self.results_text = scrolledtext.ScrolledText(results_frame, width=40, height=25)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 为结果文本框启用复制功能
        self._enable_results_copy_support()
        
        # 结果控制按钮框架
        results_control_frame = ttk.Frame(results_frame)
        results_control_frame.grid(row=1, column=0, pady=(5, 0))
        
        ttk.Button(results_control_frame, text="清空列表", command=self.clear_results).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(results_control_frame, text="复制结果", command=self.copy_results).pack(side=tk.LEFT)
        
        # 底部：状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_label.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 配置权重
        main_frame.rowconfigure(1, weight=1)
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        
    def on_scene_change(self, event):
        self.current_scene = self.scene_var.get()
        self.status_var.set(f"切换到场景: {self.current_scene}")
        
        # 如果正在运行，重启检测
        if self.running:
            self.stop_detection()
            time.sleep(1)
            self.start_detection()
    
    def toggle_detection_boxes(self):
        """切换检测框绘制开关"""
        self.draw_detection_boxes = not self.draw_detection_boxes
        if self.draw_detection_boxes:
            self.toggle_boxes_btn.config(text="关闭检测框")
            print(f"[DEBUG] 检测框绘制已开启")
        else:
            self.toggle_boxes_btn.config(text="开启检测框")
            print(f"[DEBUG] 检测框绘制已关闭")
    
    def start_detection(self):
        """启动检测"""
        self.rtsp_url = self.url_var.get().strip()
        if not self.rtsp_url:
            messagebox.showerror("错误", "请输入RTSP URL")
            return
        
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # 清空队列
        self.clear_queues()
        
        # 清空检测框信息
        self.last_tracking_boxes = None
        self.tracking_boxes_dict = {}
        
        # 启动视频读取线程
        self.video_thread = threading.Thread(target=self.video_reader, daemon=True)
        self.video_thread.start()
        
        # 启动检测进程（注释掉这行可以测试原始视频流）
        self.start_detection_process()
        
        # 开始更新显示
        self.update_video()
        
        self.status_var.set(f"正在检测中 - {self.current_scene}")
    
    def start_raw_video_test(self):
        """启动原始视频流测试（不进行检测）"""
        self.rtsp_url = self.url_var.get().strip()
        if not self.rtsp_url:
            messagebox.showerror("错误", "请输入RTSP URL")
            return
        
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # 清空队列
        self.clear_queues()
        
        # 清空检测框信息
        self.last_tracking_boxes = None
        self.tracking_boxes_dict = {}
        
        # 启动视频读取线程
        self.video_thread = threading.Thread(target=self.video_reader, daemon=True)
        self.video_thread.start()
        
        # 不启动检测进程，直接显示原始视频流
        print("[DEBUG] 测试模式：仅显示原始RTSP视频流，不进行检测")
        
        # 开始更新显示
        self.update_video()
        
        self.status_var.set("测试模式：原始视频流")
        
    def stop_detection(self):
        """停止检测"""
        self.running = False
        
        # 停止检测进程
        self.stop_detection_process()
        
        # 等待视频线程结束
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2)
        
        # 清空检测框信息
        self.last_tracking_boxes = None
        self.tracking_boxes_dict = {}
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("已停止")
        
    def video_reader(self):
        """视频读取线程 - 使用ffmpeg管道方式实现ffplay级别的低延迟"""
        print(f"[DEBUG] 使用ffmpeg管道方式获取RTSP流: {self.rtsp_url}")
        
        # 构建ffmpeg命令（类似ffplay的低延迟参数）
        # 使用与pipeline.py中一致的分辨率(640x360)以提高兼容性
        ffmpeg_cmd = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',  # 使用TCP传输
            '-fflags', 'nobuffer',  # 无缓冲区
            '-flags', 'low_delay',  # 低延迟模式
            '-avioflags', 'direct',  # 直接I/O
            '-buffer_size', '102400',  # 缓冲区大小
            '-i', self.rtsp_url,  # 输入URL
            '-f', 'rawvideo',  # 输出原始视频
            '-pix_fmt', 'bgr24',  # 像素格式
            # 使用与pipeline.py一致的分辨率(640x360)
            '-s', '640x360',  # 输出分辨率
            '-an',  # 禁用音频
            '-threads', '1',  # 单线程处理
            'pipe:1'  # 输出到标准输出
        ]
        
        print(f"[DEBUG] ffmpeg命令: {' '.join(ffmpeg_cmd)}")
        
        try:
            # 启动ffmpeg进程
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8,  # 大缓冲区
                universal_newlines=False
            )
            
            # 启动一个线程来读取ffmpeg的错误输出
            def read_stderr():
                while self.running and proc.poll() is None:
                    line = proc.stderr.readline()
                    if line:
                        line_str = line.decode('utf-8', errors='ignore').strip()
                        if line_str and not line_str.startswith('frame='):
                            print(f"[FFMPEG] {line_str}")
            
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()
            
            frame_width = 640
            frame_height = 360
            frame_size = frame_width * frame_height * 3  # 3 channels for bgr24
            
            frame_count = 0
            last_print_time = time.time()
            
            # 等待ffmpeg初始化
            time.sleep(0.5)
            
            while self.running and proc.poll() is None:
                # 读取一帧数据
                grab_start = time.time()
                raw_frame = b''
                bytes_read = 0
                
                # 分块读取，防止阻塞
                while bytes_read < frame_size and self.running:
                    chunk = proc.stdout.read(min(4096, frame_size - bytes_read))
                    if not chunk:
                        # 没有数据了，可能流结束了
                        break
                    raw_frame += chunk
                    bytes_read += len(chunk)
                
                grab_end = time.time()
                
                if bytes_read != frame_size:
                    if frame_count == 0:
                        print(f"[DEBUG] 第一次读取失败，尝试备用方案...")
                        # 清理进程
                        if proc.poll() is None:
                            proc.terminate()
                            proc.wait(timeout=2)
                        # 回退到OpenCV方式
                        self.video_reader_fallback()
                        return
                    time.sleep(0.01)  # 短暂等待
                    continue
                
                # 将原始数据转换为numpy数组
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((frame_height, frame_width, 3))
                
                # 计算抓取延迟
                grab_latency = (grab_end - grab_start) * 1000
                if grab_latency > 100:  # 超过100ms记录
                    print(f"[DEBUG] 抓取延迟: {grab_latency:.0f}ms")
                
                # 记录帧抓取时间戳（毫秒）
                frame_capture_timestamp = int(grab_start * 1000)
                
                # 将帧和时间戳一起放入队列
                if not self.frame_queue.full():
                    self.frame_queue.put({
                        'frame': frame,
                        'capture_timestamp': frame_capture_timestamp,
                        'grab_latency': grab_latency
                    }, block=False)
                else:
                    # 队列满时丢弃帧并记录日志
                    queue_size = self.frame_queue.qsize()
                    print(f"[WARN] 队列已满，丢弃一帧 (队列大小: {queue_size}/10)")
                
                # 计算实际帧率
                current_time = time.time()
                frame_count += 1
                
                if current_time - last_print_time >= 1.0:
                    actual_fps = frame_count / (current_time - last_print_time)
                    queue_size = self.frame_queue.qsize()
                    if frame_count > 0:
                        print(f"[DEBUG] 实际帧率: {actual_fps:.1f} FPS, 队列大小: {queue_size}/10, 抓取延迟: {grab_latency:.0f}ms")
                    frame_count = 0
                    last_print_time = current_time
            
            # 清理进程
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=2)
            
            print("[DEBUG] ffmpeg管道已停止")
            
        except Exception as e:
            print(f"[ERROR] ffmpeg管道错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 如果ffmpeg管道失败，回退到备用方案
            self.video_reader_fallback()
    
    def video_reader_fallback(self):
        """备用视频读取方案（OpenCV方式）"""
        print("[DEBUG] 使用备用方案（OpenCV方式）")
        
        # 使用低延迟RTSP参数
        rtsp_options = 'rtsp_transport=tcp;buffer_size=102400;fflags=nobuffer;flags=low_delay'
        full_url = f'{self.rtsp_url}?{rtsp_options}'
        
        # 尝试使用ffmpeg后端
        cap = cv2.VideoCapture(full_url, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            # 如果特殊URL失败，尝试普通URL
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            print(f"[ERROR] RTSP连接失败: {self.rtsp_url}")
            return
        
        # 设置低延迟参数
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"[DEBUG] OpenCV连接成功")
        
        frame_count = 0
        last_print_time = time.time()
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                # 尝试重新连接
                cap.release()
                time.sleep(0.5)
                cap = cv2.VideoCapture(full_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            
            # 调整大小（降低分辨率以减少处理延迟）
            frame = cv2.resize(frame, (640, 360))
            
            # 放入队列（非阻塞）
            if not self.frame_queue.full():
                self.frame_queue.put(frame, block=False)
            else:
                # 队列满时丢弃帧并记录日志
                queue_size = self.frame_queue.qsize()
                print(f"[WARN] OpenCV模式 - 队列已满，丢弃一帧 (队列大小: {queue_size}/10)")
            
            # 计算实际帧率
            frame_count += 1
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                actual_fps = frame_count / (current_time - last_print_time)
                queue_size = self.frame_queue.qsize()
                if frame_count > 0:
                    print(f"[DEBUG] OpenCV模式 - 实际帧率: {actual_fps:.1f} FPS, 队列大小: {queue_size}/10")
                frame_count = 0
                last_print_time = current_time
        
        cap.release()
        print(f"[DEBUG] OpenCV视频流已停止")
        
    def start_detection_process(self):
        """启动检测进程"""
        config_file = self.scene_configs[self.current_scene]
        
        # 获取项目根目录
        # GUI目录结构: project_root/deploy/pipeline/gui
        # 所以需要向上三级
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        
        # 构建绝对路径
        # 根据平台构建不同的路径格式
        if platform.system() == "Linux" or platform.system() == "Darwin":
            # Linux/WSL/macOS：使用WSL路径格式
            # 项目根目录在WSL中是：/mnt/c/paddleDetection/PaddleDetection-release-2.5/PaddleDetection-release-2.5
            project_root_wsl = "/mnt/c/paddleDetection/PaddleDetection-release-2.5/PaddleDetection-release-2.5"
            pipeline_path = os.path.join(project_root_wsl, "deploy/pipeline/pipeline.py")
            config_path = os.path.join(project_root_wsl, "deploy/pipeline/config", config_file)
            pipeline_dir = os.path.join(project_root_wsl, "deploy/pipeline")
        else:
            # Windows：使用Windows路径格式
            pipeline_path = os.path.join(project_root, "deploy/pipeline/pipeline.py")
            config_path = os.path.join(project_root, "deploy/pipeline/config", config_file)
            pipeline_dir = os.path.join(project_root, "deploy/pipeline")
        
        # 输出调试信息
        print(f"[DEBUG] 当前目录: {current_dir}")
        print(f"[DEBUG] 项目根目录: {project_root}")
        print(f"[DEBUG] pipeline.py路径: {pipeline_path}")
        print(f"[DEBUG] 配置文件路径: {config_path}")
        print(f"[DEBUG] pipeline目录: {pipeline_dir}")
        
        # 构建基础命令参数
        base_params = [
            "--config", config_path,
            "--rtsp", self.rtsp_url,
            "--device", "gpu",
            "-o", "MOT.enable=True",
            "-o", "visual=False",
            "-o", "warmup_frame=5"
        ]
        
        # 根据场景添加特定参数
        scene_params = []
        if "人物属性" in self.current_scene:
            scene_params.extend(["-o", "ATTR.enable=True"])
        elif "打架检测" in self.current_scene:
            scene_params.extend(["-o", "VIDEO_ACTION.enable=True"])
        elif "吸烟检测" in self.current_scene:
            scene_params.extend(["-o", "ID_BASED_DETACTION.enable=True"])
        elif "摔倒检测" in self.current_scene:
            scene_params.extend(["-o", "SKELETON_ACTION.enable=True"])
        
        # 构建完整的命令
        if platform.system() == "Linux" or platform.system() == "Darwin":
            # Linux/WSL/macOS：先激活虚拟环境再运行
            # 使用bash命令激活虚拟环境
            env_activate_cmd = "source ~/paddle-env/bin/activate && "
            
            # 构建命令行参数字符串
            all_params = base_params + scene_params
            param_str = ""
            for i in range(0, len(all_params), 2):
                param_str += f" {all_params[i]} {all_params[i+1]}"
            
            full_cmd = f"{env_activate_cmd}python {pipeline_path}{param_str}"
            cmd = ["bash", "-c", full_cmd]
        else:
            # Windows：直接运行
            cmd = ["python", pipeline_path]
            cmd.extend(base_params)
            cmd.extend(scene_params)
        
        self.status_var.set(f"启动检测进程: {self.current_scene}")
        
        # 检查路径是否存在
        # 注意：在WSL中检查Windows路径可能失败，所以只做基本检查
        # 实际检查让pipeline.py自己处理
        print(f"[DEBUG] 最终pipeline路径: {pipeline_path}")
        print(f"[DEBUG] 最终配置文件路径: {config_path}")
        print(f"[DEBUG] 最终工作目录: {pipeline_dir}")
        
        print(f"工作目录: {pipeline_dir}")
        print(f"运行命令: {' '.join(cmd)}")
        
        # 设置环境变量，强制Python的print立即刷新
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'  # 强制Python无缓冲输出
        
        self.detection_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,  # 传递环境变量
            cwd=pipeline_dir  # 设置工作目录为deploy/pipeline
        )
        
        # 启动输出读取线程
        threading.Thread(target=self.read_process_output, daemon=True).start()
        
    def read_process_output(self):
        """读取检测进程输出 - 所有日志都打印到终端"""
        import time
        
        # 初始化变量
        last_trackid_line_time = time.time()
        pending_trackid_results = []
        
        for line in iter(self.detection_process.stdout.readline, ''):
            if not self.running:
                break
            
            line = line.strip()
            if not line:
                continue
                
            # 直接打印到终端 - 所有pipeline输出都到终端
            print(f"[PIPELINE] {line}")
            
            # 解析输出，提取检测结果
            result = self.parse_detection_output(line)
            if result:
                self.results_history.append(result)
                
                # 如果是检测框结果，立即处理
                if result['type'] == 'tracking_boxes':
                    self.add_result_to_list(result)
                
                # 如果是trackid结果，累积处理
                elif result['type'] == 'object_count':
                    current_time = time.time()
                    current_count = result.get('count', '0')
                    current_frame_id = result.get('frame_id', 0)
                    
                    # 初始化变量
                    if not hasattr(self, 'last_object_count'):
                        self.last_object_count = None
                    if not hasattr(self, 'last_result_update_time'):
                        self.last_result_update_time = current_time
                    if not hasattr(self, 'last_frame_id'):
                        self.last_frame_id = 0
                    
                    # 简化更新逻辑：确保定期更新显示
                    should_update = False
                    
                    # 策略1：首次检测到目标
                    if self.last_object_count is None:
                        should_update = True
                        print(f"[DEBUG] 首次检测到目标: {current_count}")
                    
                    # 策略2：目标数量变化
                    elif self.last_object_count != current_count:
                        should_update = True
                        print(f"[DEBUG] 目标数量变化: {self.last_object_count} -> {current_count}")
                    
                    # 策略3：帧号有明显变化（至少处理了10帧）
                    elif current_frame_id - self.last_frame_id >= 10:
                        should_update = True
                        print(f"[DEBUG] 帧号变化: {self.last_frame_id} -> {current_frame_id}")
                    
                    # 策略4：长时间没有更新（2秒）- 缩短时间
                    time_since_last_update = current_time - self.last_result_update_time
                    if time_since_last_update >= 2.0:
                        should_update = True
                        print(f"[DEBUG] 2秒未更新，强制更新")
                    
                    if should_update:
                        self.add_result_to_list(result)
                        self.last_object_count = current_count
                        self.last_result_update_time = current_time
                        self.last_frame_id = current_frame_id
                        
                        # 使用合理的频率计算
                        if time_since_last_update >= 0.1:  # 至少0.1秒才计算频率
                            update_freq = 1.0 / time_since_last_update
                            print(f"[INFO] 结果列表已更新，间隔: {time_since_last_update:.1f}秒, 频率: {update_freq:.1f}Hz")
                        else:
                            print(f"[INFO] 结果列表已更新，当前目标: {current_count}")
                
    def parse_detection_output(self, line):
        """解析检测输出"""
        # 示例解析逻辑，根据实际输出格式调整
        import re
        
        # 获取当前时间戳（日志打印时间）
        current_log_time_ms = int(time.time() * 1000)
        current_log_time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # 解析Frame处理时间信息
        if "Frame" in line and "Total processing time" in line:
            match = re.search(r'Frame\s+(\d+):.*=\s+([\d.]+)ms', line)
            if match:
                frame_num = match.group(1)
                time_str = match.group(2)
                return {
                    "type": "processing_time",
                    "frame": frame_num,
                    "time": time_str,
                    "timestamp": current_log_time_str
                }
        
        # 解析trackid number信息
        elif "trackid number" in line:
            # 查找"trackid number: "之后的内容
            match = re.search(r'trackid number:\s*(\d+)', line)
            if match:
                num = match.group(1)
                
                # 尝试提取帧号 - 从同一行或上一行的线程信息中
                frame_match = re.search(r'frame id:\s*(\d+)', line)
                if frame_match:
                    frame_id = frame_match.group(1)
                else:
                    # 如果没有找到，使用当前时间估算的帧号
                    # 假设9-10 FPS，每秒9-10帧
                    current_time = time.time()
                    start_time = getattr(self, 'start_time', current_time)
                    fps = getattr(self, 'fps', 10.0)
                    frame_id = int((current_time - start_time) * fps)
                
                print(f"[{current_log_time_str}] [DEBUG] 解析到trackid number: {num}, 帧号: {frame_id}")
                return {
                    "type": "object_count",
                    "count": num,
                    "frame_id": frame_id,
                    "timestamp": current_log_time_str,
                    "scene": self.current_scene
                }
            else:
                # 尝试简单分割
                parts = line.split(":")
                if len(parts) >= 2:
                    num = parts[-1].strip()  # 取最后一个冒号后的内容
                    
                    # 估算帧号
                    current_time = time.time()
                    start_time = getattr(self, 'start_time', current_time)
                    fps = getattr(self, 'fps', 10.0)
                    frame_id = int((current_time - start_time) * fps)
                    
                    print(f"[{current_log_time_str}] [DEBUG] 简单解析到trackid number: {num}, 估算帧号: {frame_id}")
                    return {
                        "type": "object_count",
                    "count": num,
                        "frame_id": frame_id,
                        "timestamp": current_log_time_str,
                        "scene": self.current_scene
                    }
        
        # 解析检测框信息 - 简单版，只解析数据，不做时间戳同步
        elif line.startswith("TRACKING_BOXES:"):
            # 即使检测框绘制已关闭，也解析数据，但减少日志输出
            if not self.draw_detection_boxes:
                # 减少日志频率，避免干扰
                import random
                if random.random() < 0.01:  # 1%的概率记录日志
                    print(f"[{current_log_time_str}] [DEBUG] 检测框绘制已关闭，但仍解析TRACKING_BOXES数据")
            
            # 新格式: TRACKING_BOXES: frame_id:frame_capture_timestamp:grab_delay:processing_delay:id:class:score:xmin:ymin:xmax:ymax,id:class:score:xmin:ymin:xmax:ymax,...
            try:
                # 直接提取数据部分
                data_start = line.find("TRACKING_BOXES:") + len("TRACKING_BOXES:")
                boxes_data = line[data_start:].strip()
                
                # 分割帧号、抓取时间戳、抓取延迟、处理延迟和检测框数据
                parts = boxes_data.split(":", 4)  # 最多分割成5部分
                if len(parts) < 5:
                    print(f"[{current_log_time_str}] [DEBUG] TRACKING_BOXES格式错误，少于5部分: {boxes_data}")
                    return None
                    
                frame_id = int(parts[0])
                frame_capture_timestamp = int(parts[1])  # 毫秒时间戳（原始帧抓取时间）
                grab_delay = int(parts[2])  # 抓取延迟（毫秒）
                processing_delay = int(parts[3])  # 处理延迟（毫秒）
                boxes_info_str = parts[4]
                
                # 解析多个检测框
                boxes = []
                if boxes_info_str:
                    # 直接使用split分割，避免多次字符串操作
                    box_items = boxes_info_str.split(",")
                    for box_item in box_items:
                        if box_item:
                            box_data = box_item.split(":")
                            if len(box_data) >= 7:
                                try:
                                    box_info = {
                                        "id": int(box_data[0]),
                                        "class": int(box_data[1]),
                                        "score": float(box_data[2]),
                                        "xmin": int(box_data[3]),
                                        "ymin": int(box_data[4]),
                                        "xmax": int(box_data[5]),
                                        "ymax": int(box_data[6])
                                    }
                                    boxes.append(box_info)
                                except (ValueError, IndexError) as e:
                                    # 解析失败时跳过，不打印太多日志
                                    print(f"[{current_log_time_str}] [DEBUG] 检测框解析失败: {box_item}, 错误: {e}")
                                    pass
                
                # 计算从原始帧抓取到当前日志的时间差
                time_since_capture = current_log_time_ms - frame_capture_timestamp
                
                # 只在特定情况下记录详细的时间戳信息
                if frame_id % 10 == 0:  # 每10帧记录一次
                    print(f"[{current_log_time_str}] [TIMESTAMP_DETAIL] 解析检测框: 帧 {frame_id}, {len(boxes)}个目标")
                    print(f"[{current_log_time_str}] [TIMESTAMP_DETAIL]  原始帧抓取时间: {frame_capture_timestamp}")
                    print(f"[{current_log_time_str}] [TIMESTAMP_DETAIL]  抓取延迟: {grab_delay}ms, 处理延迟: {processing_delay}ms")
                
                return {
                    "type": "tracking_boxes",
                    "frame_id": frame_id,
                    "frame_capture_timestamp": frame_capture_timestamp,  # 原始帧抓取时间戳
                    "grab_delay": grab_delay,
                    "processing_delay": processing_delay,
                    "boxes": boxes,
                    "log_time": current_log_time_ms,  # 记录日志打印时间
                    "log_time_str": current_log_time_str  # 记录日志打印时间字符串
                }
            except Exception as e:
                # 解析失败时打印错误日志
                print(f"[{current_log_time_str}] [ERROR] TRACKING_BOXES解析失败: {e}, 行: {line[:100]}")
                return None

        return None
    
    def calculate_dynamic_delay_compensation(self, current_time_ms):
        """
        计算动态延迟补偿 - 简化版
        固定返回一个较小的补偿值，避免过度补偿
        """
        # 固定返回30ms补偿，这个值足够覆盖大多数传输延迟
        return 30
    
    def find_best_matching_boxes_by_timestamp(self, current_time_ms):
        """
        基于原始帧抓取时间戳查找最匹配的检测框信息
        在当前架构下，GUI和Pipeline各自抓取RTSP流，时间戳无法完全一致
        使用智能匹配：优先匹配时间戳最接近的帧
        """
        if not self.draw_detection_boxes:
            return []
        
        if not self.tracking_boxes_dict:
            return []
        
        # 获取当前视频帧的抓取时间戳（毫秒）
        video_timestamp = getattr(self, 'current_video_frame_timestamp', 0)
        
        if video_timestamp == 0:
            return []
        
        # 由于GUI和Pipeline各自抓取RTSP流，时间戳不会完全一致
        # 我们需要找到时间戳最接近的检测框
        best_match = None
        best_time_diff = float('inf')
        matched_frame_id = 0
        
        for frame_id, box_info in self.tracking_boxes_dict.items():
            box_timestamp = box_info.get('frame_capture_timestamp', 0)
            if box_timestamp == 0:
                continue
            
            time_diff = abs(video_timestamp - box_timestamp)
            
            # 优先匹配时间差小于100ms的检测框
            if time_diff < 100 and time_diff < best_time_diff:
                best_match = box_info
                best_time_diff = time_diff
                matched_frame_id = frame_id
        
        # 如果找到合适的匹配
        if best_match:
            boxes = best_match.get('boxes', [])
            if boxes:
                # 记录匹配信息，但减少日志频率
                if self.frame_count % 30 == 0:
                    box_timestamp = best_match.get('frame_capture_timestamp', 0)
                    grab_delay = best_match.get('grab_delay', 0)
                    processing_delay = best_match.get('processing_delay', 0)
                    pipeline_total_delay = grab_delay + processing_delay
                    
                    # 提供更有用的同步信息，而不是误导性的绝对时间比较
                    if best_time_diff < 50:  # 同步良好
                        # print(f"[SYNC_GOOD] ✓ 时间戳同步良好: 视频帧与检测框时间差={best_time_diff}ms (帧 {matched_frame_id})")
                        # print(f"[SYNC_INFO]   Pipeline处理延迟: {pipeline_total_delay}ms (抓取:{grab_delay}ms + 处理:{processing_delay}ms)")
                        pass
                    elif best_time_diff < 100:  # 同步可接受
                        # print(f"[SYNC_OK] ~ 时间戳同步可接受: 视频帧与检测框时间差={best_time_diff}ms (帧 {matched_frame_id})")
                        # print(f"[SYNC_INFO]   Pipeline处理延迟: {pipeline_total_delay}ms (抓取:{grab_delay}ms + 处理:{processing_delay}ms)")
                        pass
                    else:  # 同步较差
                        print(f"[SYNC_WARN] ! 时间戳同步较差: 视频帧与检测框时间差={best_time_diff}ms (帧 {matched_frame_id})")
                        
                        # 诊断RTSP缓冲区问题
                        if best_time_diff > 1000:  # 超过1秒，很可能是RTSP缓冲区问题
                            print(f"[RTSP_BUFFER_WARNING] !!! 检测到RTSP缓冲区问题 !!!")
                            print(f"[RTSP_BUFFER_WARNING]   时间差 {best_time_diff}ms (约{best_time_diff/1000:.1f}秒) 表明:")
                            print(f"[RTSP_BUFFER_WARNING]   1. Pipeline和GUI的RTSP连接不同步")
                            print(f"[RTSP_BUFFER_WARNING]   2. 可能有 {best_time_diff/1000:.1f}秒 的RTSP缓冲区差异")
                            print(f"[RTSP_BUFFER_WARNING]   3. Pipeline抓取的是 {best_time_diff/1000:.1f}秒前的帧")
                            print(f"[RTSP_BUFFER_WARNING]   解决方案:")
                            print(f"[RTSP_BUFFER_WARNING]     - 检查RTSP服务器缓存配置")
                            print(f"[RTSP_BUFFER_WARNING]     - 减少OpenCV的缓冲区大小")
                            print(f"[RTSP_BUFFER_WARNING]     - 使用低延迟RTSP参数")
                        else:
                            print(f"[SYNC_WARN]   系统时间可能不同步，建议检查Pipeline和GUI时间设置")
                        
                        # print(f"[SYNC_INFO]   Pipeline处理延迟: {pipeline_total_delay}ms (抓取:{grab_delay}ms + 处理:{processing_delay}ms)")
                
                return boxes
        
        # 没有找到匹配
        return []
    
    def add_result_to_list(self, result):
        """添加结果到列表 - 优化版，避免频繁更新"""
        timestamp = result.get('timestamp', '')
        
        if result['type'] == 'processing_time':
            # 处理时间信息，可以记录但不需要频繁显示
            frame_num = result.get('frame', '0')
            time_ms = result.get('time', '0')
            
            # 只显示重要的处理时间（超过100ms）
            if int(time_ms) > 100:
                text = f"[{timestamp}] 帧 {frame_num}: 处理时间 {time_ms}ms (慢)\n"
                # 在主线程中更新UI
                self.root.after(0, self._update_results_text, text)
        
        elif result['type'] == 'object_count':
            count = result.get('count', '0')
            text = f"[{timestamp}] {self.current_scene}: 检测到 {count} 个目标\n"
            print(f"[INFO] 在结果列表中显示: {self.current_scene}: 检测到 {count} 个目标")
            
            # 在主线程中更新UI
            self.root.after(0, self._update_results_text, text)
            
            # 更新实时统计（使用现有的统计变量）
            self.root.after(0, lambda c=count: self._update_object_stats(c))
        
        elif result['type'] == 'tracking_boxes':
            # 检测框数据，仅用于绘制，不显示在结果列表中
            # 即使检测框绘制已关闭也保存数据，以便在开启时立即有数据可用
            frame_id = result.get('frame_id', 0)
            frame_capture_timestamp = result.get('frame_capture_timestamp', 0)  # 原始帧抓取时间戳
            grab_delay = result.get('grab_delay', 0)  # 抓取延迟
            processing_delay = result.get('processing_delay', 0)  # 处理延迟
            boxes = result.get('boxes', [])
            log_time_ms = result.get('log_time', 0)  # 日志打印时间
            log_time_str = result.get('log_time_str', '')  # 日志打印时间字符串
            
            # 保存检测框信息供绘制使用
            current_local_time_ms = int(time.time() * 1000)
            
            # 保存到最新检测框 - 移除误导性的capture_to_gui_delay字段
            self.last_tracking_boxes = {
                "frame_id": frame_id,
                "frame_capture_timestamp": frame_capture_timestamp,  # 原始帧抓取时间戳（毫秒）
                "grab_delay": grab_delay,  # Pipeline抓取延迟（真实延迟）
                "processing_delay": processing_delay,  # Pipeline处理延迟（真实延迟）
                "boxes": boxes,
                "local_time": current_local_time_ms,  # GUI端接收时间（毫秒）
                "pipeline_total_delay": grab_delay + processing_delay,  # Pipeline总延迟（抓取+处理）
                "log_time": log_time_ms,  # 日志打印时间
                "log_time_str": log_time_str  # 日志打印时间字符串
            }
            
            # 按帧号存储检测框信息，用于同步
            self.tracking_boxes_dict[frame_id] = {
                "frame_capture_timestamp": frame_capture_timestamp,  # 原始帧抓取时间戳
                "grab_delay": grab_delay,  # Pipeline抓取延迟（真实延迟）
                "processing_delay": processing_delay,  # Pipeline处理延迟（真实延迟）
                "boxes": boxes,
                "local_time": current_local_time_ms,  # GUI端接收时间
                "pipeline_total_delay": grab_delay + processing_delay,  # Pipeline总延迟（抓取+处理）
                "log_time": log_time_ms,  # 日志打印时间
                "log_time_str": log_time_str  # 日志打印时间字符串
            }
            
            # 清理旧的检测框信息，避免内存泄漏
            if len(self.tracking_boxes_dict) > 50:  # 最多保留50帧
                oldest_frame = min(self.tracking_boxes_dict.keys())
                del self.tracking_boxes_dict[oldest_frame]
            
            # 减少日志频率
            if frame_id % 10 == 0:  # 每10帧记录一次
                current_log_time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{current_log_time_str}] [DEBUG] 保存检测框信息: 帧 {frame_id}, {len(boxes)}个目标")
                print(f"[{current_log_time_str}] [DEBUG]  原始帧抓取时间: {frame_capture_timestamp} ({datetime.fromtimestamp(frame_capture_timestamp/1000).strftime('%H:%M:%S.%f')[:-3]})")
                print(f"[{current_log_time_str}] [DEBUG]  抓取延迟: {grab_delay}ms, 处理延迟: {processing_delay}ms, 日志打印时间: {log_time_str}")
                print(f"[{current_log_time_str}] [DEBUG]  从抓取到GUI延迟: {current_local_time_ms - frame_capture_timestamp}ms")
                print(f"[{current_log_time_str}] [DEBUG]  第一个检测框详细信息: {boxes[0] if boxes else '无'}")
                print(f"[{current_log_time_str}] [DEBUG]  tracking_boxes_dict 大小: {len(self.tracking_boxes_dict)}")
    
    def _update_results_text(self, text):
        """在主线程中更新结果文本"""
        self.results_text.insert(tk.END, text)
        self.results_text.see(tk.END)
        
        # 更新统计
        self.total_frames_var.set(f"总帧数: {len(self.results_history)}")
    
    def _update_object_stats(self, current_count):
        """更新对象统计信息"""
        try:
            # 计算总的检测对象数
            total_objects = sum(int(result.get('count', 0)) 
                               for result in self.results_history 
                               if result['type'] == 'object_count')
            
            # 更新实时统计显示
            self.total_objects_var.set(f"总检测数: {total_objects}")
            
            # 更新平均FPS
            if len(self.results_history) > 0:
                avg_fps = self.fps if self.fps > 0 else 0
                self.avg_fps_var.set(f"平均FPS: {avg_fps:.1f}")
                
        except Exception as e:
            print(f"[ERROR] 更新统计失败: {e}")
    
    def update_video(self):
        """更新视频显示 - 优化版，减少闪烁"""
        if not self.running:
            return
            
        try:
            # 从队列获取最新帧（非阻塞，只取一帧）
            try:
                frame_data = self.frame_queue.get(timeout=0.01)  # 10ms超时
                
                # 解析帧数据（可能是字典或直接的numpy数组）
                if isinstance(frame_data, dict):
                    latest_frame = frame_data.get('frame')
                    frame_capture_timestamp = frame_data.get('capture_timestamp', 0)
                    grab_latency = frame_data.get('grab_latency', 0)
                else:
                    # 兼容旧格式
                    latest_frame = frame_data
                    frame_capture_timestamp = 0
                    grab_latency = 0
                
                # 计算FPS
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                self.last_frame_time = current_time
                
                if elapsed > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1 / elapsed)
                
                # 基于时间戳的精确同步 - 如果检测框绘制已关闭，跳过时间戳计算
                if self.draw_detection_boxes:
                    current_time_ms = int(current_time * 1000)
                    # 记录当前视频帧的抓取时间戳
                    self.current_video_frame_timestamp = frame_capture_timestamp
                else:
                    current_time_ms = 0  # 设置为0，避免任何基于时间戳的计算
                    self.current_video_frame_timestamp = 0
                
                # 更新FPS显示（在主线程中）
                self.root.after(0, lambda: self.fps_label.config(text=f"FPS: {int(self.fps)}"))
                
                # 根据开关状态决定是否绘制检测框
                frame_to_display = latest_frame  # 默认使用原始帧
                
                # 如果检测框绘制已关闭，记录状态并跳过复杂计算
                if not self.draw_detection_boxes:
                    # 减少日志频率
                    if self.frame_count % 120 == 0:  # 每120帧记录一次
                        print(f"[DEBUG] 检测框绘制已关闭，当前帧: {self.frame_count}")
                
                # 只有在开启检测框绘制时才执行绘制逻辑
                elif self.draw_detection_boxes:
                    # 记录检测框绘制已开启（调试用）
                    if self.frame_count % 60 == 0:
                        print(f"[DEBUG] 检测框绘制已开启，检查可用检测框数据...")
                        print(f"[DEBUG] last_tracking_boxes: {self.last_tracking_boxes is not None}")
                        print(f"[DEBUG] tracking_boxes_dict 大小: {len(self.tracking_boxes_dict)}")
                    
                    # 创建副本用于绘制检测框
                    frame_with_boxes = latest_frame.copy()
                    
                    # 使用简化的检测框查找方法
                    boxes = self.find_best_matching_boxes_by_timestamp(current_time_ms)
                    
                    # 绘制检测框
                    if boxes:
                        # 记录绘制开始时间
                        draw_start_time = time.time()
                        draw_start_time_ms = int(draw_start_time * 1000)
                        draw_start_time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        
                        print(f"[DEBUG] 准备绘制 {len(boxes)} 个检测框，开始时间: {draw_start_time_str}")
                        
                        for i, box in enumerate(boxes):
                            xmin = box.get('xmin', 0)
                            ymin = box.get('ymin', 0)
                            xmax = box.get('xmax', 0)
                            ymax = box.get('ymax', 0)
                            score = box.get('score', 0.0)
                            class_id = box.get('class', 0)
                            
                            # 绘制矩形框
                            color = (0, 255, 0)  # 绿色
                            thickness = 2
                            cv2.rectangle(frame_with_boxes, (xmin, ymin), (xmax, ymax), color, thickness)
                            
                            # 添加标签（显示ID和置信度）
                            label = f"ID: {box.get('id', 0)}"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.5
                            font_thickness = 1
                            
                            # 计算ID标签大小
                            label_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                            label_x = xmin
                            label_y = ymin - 10 if ymin - 10 > 10 else ymin + 10
                            
                            # 绘制ID标签背景
                            cv2.rectangle(frame_with_boxes, 
                                         (label_x, label_y - label_size[1] - 5),
                                         (label_x + label_size[0], label_y + 5),
                                         color, -1)
                            
                            # 绘制ID标签文本
                            cv2.putText(frame_with_boxes, label, 
                                       (label_x, label_y), 
                                       font, font_scale, (0, 0, 0), font_thickness)
                            
                            # 添加置信度标签（在ID标签下方）
                            score_label = f"{score:.2f}"
                            score_font_scale = 0.4
                            score_font_thickness = 1
                            
                            # 计算置信度标签大小
                            score_label_size = cv2.getTextSize(score_label, font, score_font_scale, score_font_thickness)[0]
                            score_label_x = xmin
                            score_label_y = label_y + label_size[1] + 5
                            
                            # 绘制置信度标签背景
                            cv2.rectangle(frame_with_boxes,
                                         (score_label_x, score_label_y - score_label_size[1] - 3),
                                         (score_label_x + score_label_size[0], score_label_y + 3),
                                         (255, 0, 0), -1)  # 使用红色背景
                            
                            # 绘制置信度标签文本
                            cv2.putText(frame_with_boxes, score_label,
                                       (score_label_x, score_label_y),
                                       font, score_font_scale, (255, 255, 255), score_font_thickness)
                        
                        # 记录绘制结束时间
                        draw_end_time = time.time()
                        draw_end_time_ms = int(draw_end_time * 1000)
                        draw_end_time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        
                        # 计算绘制耗时
                        draw_duration_ms = (draw_end_time - draw_start_time) * 1000
                        
                        # 只在绘制耗时异常时记录日志
                        if draw_duration_ms > 50:  # 超过50ms认为是异常情况
                            print(f"[DRAW_SLOW] 检测框绘制耗时过长: {draw_duration_ms:.1f}ms, 框数: {len(boxes)}")
                        
                        # 简单记录关键匹配信息
                        if self.last_tracking_boxes and self.frame_count % 30 == 0:
                            frame_capture_timestamp = self.last_tracking_boxes.get('frame_capture_timestamp', 0)
                            video_timestamp = getattr(self, 'current_video_frame_timestamp', 0)
                            if video_timestamp > 0 and frame_capture_timestamp > 0:
                                time_diff = abs(video_timestamp - frame_capture_timestamp)
                                # print(f"[SYNC_INFO] 时间戳同步: 视频帧={video_timestamp}, 检测框={frame_capture_timestamp}, 差={time_diff}ms")
                        
                        frame_to_display = frame_with_boxes
                    else:
                        if self.frame_count % 30 == 0:
                            print(f"[DEBUG] 未找到可用的检测框数据")
                
                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
                
                # 转换为PIL图像
                pil_image = Image.fromarray(frame_rgb)
                tk_image = ImageTk.PhotoImage(pil_image)
                
                # 更新画布（在主线程中）
                def update_canvas(img):
                    # 先清除画布
                    self.video_canvas.delete("all")
                    # 再绘制新图像
                    self.video_canvas.create_image(0, 0, anchor=tk.NW, image=img)
                    self.video_canvas.image = img
                
                self.root.after(0, update_canvas, tk_image)
                
            except queue.Empty:
                # 没有新帧，不更新
                pass
            
            # 检查队列中是否有积压的帧，减少频繁的清除操作
            queue_size = self.frame_queue.qsize()
            if queue_size > 10:  # 提高阈值，减少频繁清除
                frames_cleared = 0
                # 只清除到保留5帧缓冲
                while queue_size > 5:
                    try:
                        self.frame_queue.get(timeout=0.001)
                        frames_cleared += 1
                        queue_size -= 1
                    except queue.Empty:
                        break
                
                if frames_cleared > 0:
                    print(f"[WARN] 清除了 {frames_cleared} 积压帧，当前队列: {queue_size} (可能检测处理速度跟不上视频流)")
            
            # 定期计算延迟统计
            if self.frame_count % 30 == 0:  # 每30帧计算一次
                self._calculate_and_display_latency_stats()
            
        except Exception as e:
            print(f"[ERROR] 更新视频显示失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 根据当前FPS动态调整更新频率
        update_interval = max(5, min(33, int(1000 / max(self.fps, 1))))  # 5-33ms之间
        self.root.after(update_interval, self.update_video)
    
    def stop_detection_process(self):
        """停止检测进程"""
        if self.detection_process:
            try:
                # 发送SIGTERM
                self.detection_process.terminate()
                # 等待进程结束
                self.detection_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 如果进程没有响应，强制终止
                self.detection_process.kill()
                self.detection_process.wait()
            
            self.detection_process = None
    
    def clear_queues(self):
        """清空队列"""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.results_queue.empty():
            try:
                self.results_queue.get_nowait()
            except queue.Empty:
                break
    
    # _enable_copy_support方法已移除，因为不再需要GUI日志文本框
    
    def _enable_results_copy_support(self):
        """为结果文本框启用复制功能"""
        try:
            # 创建全局复制函数，避免每次调用都重新定义
            if not hasattr(self, '_results_copy_function'):
                def results_copy_selection(event=None):
                    try:
                        # 确保文本框有焦点
                        if not self.results_text.focus_get() == self.results_text:
                            self.results_text.focus_set()
                        
                        # 获取选中的文本
                        selected_text = self.results_text.get(tk.SEL_FIRST, tk.SEL_LAST)
                        if selected_text:
                            # 清除剪贴板并设置新内容
                            self.root.clipboard_clear()
                            self.root.clipboard_append(selected_text)
                            # 保持剪贴板所有权
                            self.root.clipboard_append("")  # 添加空字符串确保所有权
                            print(f"[DEBUG] 结果复制成功: {len(selected_text)} 字符")
                            return "break"  # 阻止默认处理
                    except tk.TclError:
                        # 没有选中文本
                        pass
                    except Exception as e:
                        print(f"[ERROR] 结果复制失败: {e}")
                    return "break"  # 阻止默认处理
                
                self._results_copy_function = results_copy_selection
            
            # 绑定Ctrl+C - 确保绑定正确且不会被覆盖
            self.results_text.bind('<Control-c>', self._results_copy_function, add='+')
            self.results_text.bind('<Control-C>', self._results_copy_function, add='+')
            
            # 添加上下文菜单
            def create_results_context_menu(event):
                menu = tk.Menu(self.root, tearoff=0)
                menu.add_command(label="复制", command=self._results_copy_function)
                menu.add_separator()
                menu.add_command(label="全选", 
                                command=lambda: self.results_text.tag_add(tk.SEL, '1.0', tk.END))
                menu.add_command(label="清空", command=self.clear_results)
                
                # 显示菜单
                try:
                    menu.tk_popup(event.x_root, event.y_root)
                finally:
                    menu.grab_release()
            
            # 绑定右键菜单
            self.results_text.bind('<Button-3>', create_results_context_menu)
            
            # 设置选择颜色
            self.results_text.config(selectbackground='#555555', selectforeground='white')
            
            # 添加鼠标中键粘贴支持（可选）
            def paste_from_clipboard(event=None):
                try:
                    clipboard_text = self.root.clipboard_get()
                    if clipboard_text:
                        self.results_text.insert(tk.INSERT, clipboard_text)
                except tk.TclError:
                    pass
                return "break"
            
            self.results_text.bind('<Control-v>', paste_from_clipboard, add='+')
            self.results_text.bind('<Control-V>', paste_from_clipboard, add='+')
            
            print("[DEBUG] 结果复制支持已启用")
            
        except Exception as e:
            print(f"[ERROR] 启用结果复制支持失败: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_results(self):
        """清空结果列表"""
        self.results_text.delete(1.0, tk.END)
        self.results_history = []
        self.total_frames_var.set("总帧数: 0")
    
    # clear_logs和copy_logs方法已移除，因为不再需要GUI日志文本框
    
    def copy_results(self):
        """复制结果内容到剪贴板"""
        try:
            # 获取所有结果内容
            results_content = self.results_text.get(1.0, tk.END)
            if results_content.strip():  # 确保有内容
                # 清除剪贴板并设置新内容
                self.root.clipboard_clear()
                self.root.clipboard_append(results_content)
                # 保持剪贴板所有权
                self.root.clipboard_append("")
                # 更新状态
                char_count = len(results_content)
                line_count = results_content.count('\n')
                self.status_var.set(f"已复制 {line_count} 行结果 ({char_count} 字符) 到剪贴板")
                print(f"[INFO] 复制结果成功: {line_count} 行, {char_count} 字符")
            else:
                self.status_var.set("结果列表为空，无内容可复制")
        except Exception as e:
            error_msg = f"复制结果失败: {e}"
            self.status_var.set(error_msg)
            print(f"[ERROR] {error_msg}")
    
    # toggle_log_pause, toggle_video_filter, toggle_pipeline_filter方法已移除
    # 所有日志现在都输出到终端，不再需要GUI日志过滤功能
    
    def _calculate_and_display_latency_stats(self):
        """计算并显示延迟统计信息 - 修复版：只计算Pipeline内部真实延迟"""
        try:
            # 如果检测框绘制已关闭，跳过复杂的延迟统计计算
            if not self.draw_detection_boxes:
                if self.frame_count % 120 == 0:  # 减少日志频率
                    print(f"[DEBUG] 检测框绘制已关闭，延迟统计计算已完全跳过")
                return
            
            # 收集最近的检测框信息
            if not self.tracking_boxes_dict:
                return
            
            # 提取真实的Pipeline内部延迟信息
            grab_delays = []  # 抓取延迟（Pipeline内部）
            processing_delays = []  # 处理延迟（Pipeline内部）
            pipeline_total_delays = []  # Pipeline总延迟（grab_delay + processing_delay）
            time_sync_diffs = []  # GUI与Pipeline时间戳同步差异
            
            for frame_id, box_info in self.tracking_boxes_dict.items():
                grab_delay = box_info.get('grab_delay', 0)
                processing_delay = box_info.get('processing_delay', 0)
                frame_capture_timestamp = box_info.get('frame_capture_timestamp', 0)
                local_time = box_info.get('local_time', 0)
                
                # Pipeline内部真实延迟
                if grab_delay > 0:
                    grab_delays.append(grab_delay)
                if processing_delay > 0:
                    processing_delays.append(processing_delay)
                
                # Pipeline总延迟
                if grab_delay > 0 or processing_delay > 0:
                    pipeline_total_delay = grab_delay + processing_delay
                    pipeline_total_delays.append(pipeline_total_delay)
                
                # 时间戳同步差异（仅用于诊断，不用于真实延迟）
                if (frame_capture_timestamp > 0 and local_time > 0 and 
                    hasattr(self, 'current_video_frame_timestamp') and 
                    self.current_video_frame_timestamp > 0):
                    # 计算Pipeline时间戳与GUI时间戳的差异
                    time_sync_diff = abs(self.current_video_frame_timestamp - frame_capture_timestamp)
                    time_sync_diffs.append(time_sync_diff)
            
            # 计算统计信息 - 只显示真实的Pipeline内部延迟
            if grab_delays:
                avg_grab_delay = sum(grab_delays) / len(grab_delays)
                max_grab_delay = max(grab_delays)
                min_grab_delay = min(grab_delays)
                
                if self.frame_count % 60 == 0:
                    # print(f"[PIPELINE_STATS] Pipeline抓取延迟: 平均={avg_grab_delay:.0f}ms, "
                    #       f"最小={min_grab_delay}ms, 最大={max_grab_delay}ms")
                    pass
            
            if processing_delays:
                avg_processing_delay = sum(processing_delays) / len(processing_delays)
                max_processing_delay = max(processing_delays)
                min_processing_delay = min(processing_delays)
                
                if self.frame_count % 60 == 0:
                    # print(f"[PIPELINE_STATS] Pipeline处理延迟: 平均={avg_processing_delay:.0f}ms, "
                    #       f"最小={min_processing_delay}ms, 最大={max_processing_delay}ms")
                    pass
            
            if pipeline_total_delays:
                avg_total_delay = sum(pipeline_total_delays) / len(pipeline_total_delays)
                max_total_delay = max(pipeline_total_delays)
                min_total_delay = min(pipeline_total_delays)
                
                if self.frame_count % 60 == 0:
                    # print(f"[PIPELINE_STATS] Pipeline总延迟: 平均={avg_total_delay:.0f}ms, "
                    #       f"最小={min_total_delay}ms, 最大={max_total_delay}ms")
                    pass
            
            # 时间戳同步诊断（仅用于调试）
            if time_sync_diffs and self.frame_count % 120 == 0:
                avg_sync_diff = sum(time_sync_diffs) / len(time_sync_diffs)
                max_sync_diff = max(time_sync_diffs)
                min_sync_diff = min(time_sync_diffs)
                
                if avg_sync_diff > 500:  # 如果平均同步差异超过500ms，显示警告
                    # print(f"[SYNC_WARNING] 系统时间同步差异: 平均={avg_sync_diff:.0f}ms, "
                    #       f"最小={min_sync_diff}ms, 最大={max_sync_diff}ms")
                    # print(f"[SYNC_WARNING] 提示: Pipeline和GUI系统时间可能不同步，这会影响时间戳匹配精度")
                    pass
                    
        except Exception as e:
            # 静默处理错误，不打印错误日志
            pass

    def on_closing(self):
        """关闭窗口事件"""
        self.stop_detection()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = DetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()