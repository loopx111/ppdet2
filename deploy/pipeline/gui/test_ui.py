#!/usr/bin/env python
"""
PaddleDetection GUI 快速测试脚本
用于预览新的CustomTkinter UI界面
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_ui():
    """测试UI界面"""
    print("正在启动PaddleDetection UI...")
    print("=" * 60)
    print("✨ 新的CustomTkinter UI特性：")
    print("   • 现代化圆角设计")
    print("   • 暗色/亮色主题切换")
    print("   • 卡片式布局")
    print("   • 图标按钮和统计卡片")
    print("   • 响应式界面")
    print("=" * 60)
    
    try:
        from main import main
        main()
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("错误", f"UI启动失败:\n{str(e)}")

if __name__ == "__main__":
    test_ui()
