"""
RealSense D435 摄像头实时姿态估计
使用前请确保已安装 pyrealsense2:
pip install pyrealsense2
"""

import os
# 设置 PyTorch CUDA 显存分配策略，避免显存碎片化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from estimater import *
import argparse
import numpy as np
import cv2
import trimesh
import logging
import time
import torch
from concurrent.futures import ThreadPoolExecutor

# 在所有导入完成后，重新设置日志级别为 WARNING
# 这会覆盖 Utils.py 中的 INFO 级别设置
logging.getLogger().setLevel(logging.WARNING)

try:
    import pyrealsense2 as rs
except ImportError:
    print("请先安装 pyrealsense2: pip install pyrealsense2")
    exit(1)


COLOR_OPTICAL_FRAME = "realsense_color_optical_frame"
DEPTH_OPTICAL_FRAME = "realsense_depth_optical_frame"

# Real-time registration defaults.  The full FoundationPose defaults remain
# unchanged in estimater.py; these values trade a small amount of initial-pose
# search coverage for much lower interaction latency on a D435 stream.
DEFAULT_REGISTRATION_ITERATIONS = 2
DEFAULT_ROTATION_GRID_VIEWS = 20
DEFAULT_ROTATION_GRID_INPLANE_STEP = 90
DEFAULT_DIAMETER_SAMPLES = 2000
DEFAULT_MASK_PREVIEW_DELAY_MS = 0


def _synchronize_cuda():
    """Synchronize only at measured boundaries, not on every tracking frame."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class RealSenseD435:
    def __init__(self, width=640, height=480, fps=30,
                 frame_timeout_ms=1500, reconnect_interval=2.0,
                 min_valid_depth_ratio=0.05):
        """
        初始化 RealSense D435 摄像头

        Args:
            width: 图像宽度
            height: 图像高度
            fps: 帧率
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_timeout_ms = frame_timeout_ms
        self.reconnect_interval = reconnect_interval
        self.min_valid_depth_ratio = min_valid_depth_ratio

        self.pipeline = None
        self.profile = None
        self.align = None
        self.depth_scale = None
        self.K = None
        # FoundationPose 使用对齐后的彩色图像/彩色相机内参，因此其输出是
        # color_T_object。手动外参标定使用 librealsense 原始深度点云，保存的
        # 则是 end_T_depth。这里保存 depth_T_color，供发送前统一坐标系。
        self.T_depth_from_color = None
        self.serial_number = None
        self.connected = False
        self.reconnect_count = 0
        self._next_reconnect_time = 0.0
        self._reconnect_event_pending = False
        self._last_frame_number = None
        self._last_frame_timestamp = None
        self._bad_frame_count = 0
        # 最近一帧真正进入跟踪流程时对应的采集时间。这里使用 Unix epoch
        # 秒，保证能够和 Socket 接收端记录的机器人关节时间直接比较。
        self.last_capture_timestamp = None
        self.last_capture_timestamp_source = None
        self.last_sensor_timestamp_ms = None
        self.last_timestamp_domain = None
        self.last_sensor_frame_number = None
        self._reported_timestamp_source = None

        try:
            self._start_pipeline(is_reconnect=False)
        except Exception:
            self._stop_pipeline()
            raise

    def _create_config(self):
        """创建一份新的流配置；重连时不能继续依赖旧 pipeline。"""
        config = rs.config()

        # 绑定首次连接的设备，防止多相机环境中重连到另一台相机。
        if self.serial_number:
            config.enable_device(self.serial_number)

        # 启用彩色流和深度流
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        return config

    def _start_pipeline(self, is_reconnect):
        """启动全新的 pipeline，并重新读取尺度、对齐器和内参。"""
        pipeline = rs.pipeline()
        config = self._create_config()

        # 启动管道
        profile = pipeline.start(config)
        # 启动成功后立即记录，确保后续初始化步骤报错时也能被 stop() 释放。
        self.pipeline = pipeline
        self.profile = profile

        device = profile.get_device()
        if device.supports(rs.camera_info.serial_number):
            serial_number = device.get_info(rs.camera_info.serial_number)
            if self.serial_number is None:
                self.serial_number = serial_number
        self._enable_global_time(device)

        # 获取深度传感器并设置预设
        depth_sensor = device.first_depth_sensor()
        if depth_sensor.supports(rs.option.visual_preset):
            try:
                depth_sensor.set_option(rs.option.visual_preset, 3)  # 高精度预设
            except RuntimeError as exc:
                print(f"警告: 无法设置深度高精度预设: {exc}")

        # 获取深度尺度
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"深度尺度: {self.depth_scale}")

        # 创建对齐对象（深度对齐到彩色）
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.connected = True
        self._last_frame_number = None
        self._last_frame_timestamp = None
        self._bad_frame_count = 0
        self.last_capture_timestamp = None
        self.last_capture_timestamp_source = None
        self.last_sensor_timestamp_ms = None
        self.last_timestamp_domain = None
        self.last_sensor_frame_number = None

        # 获取相机内参
        self._setup_intrinsics()
        self._setup_stream_extrinsics()

        if is_reconnect:
            self.reconnect_count += 1
            self._reconnect_event_pending = True
            print(f"RealSense 重连成功（第 {self.reconnect_count} 次）")

    @staticmethod
    def _enable_global_time(device):
        """尽量让设备帧时间直接映射到主机 Unix epoch。"""
        global_time_option = getattr(rs.option, "global_time_enabled", None)
        if global_time_option is None:
            print("警告: 当前 pyrealsense2 不支持 global_time_enabled")
            return

        enabled_sensor_count = 0
        for sensor in device.query_sensors():
            try:
                if sensor.supports(global_time_option):
                    sensor.set_option(global_time_option, 1.0)
                    enabled_sensor_count += 1
            except RuntimeError as exc:
                print(f"警告: 无法为 RealSense 传感器启用 global time: {exc}")
        if enabled_sensor_count:
            print(
                "RealSense global time 已启用: "
                f"{enabled_sensor_count} 个传感器"
            )

    def _setup_intrinsics(self):
        """设置相机内参"""
        # 获取彩色相机内参
        color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()

        self.K = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])

        print(f"相机内参 K:\n{self.K}")
        print(f"图像尺寸: {intrinsics.width}x{intrinsics.height}")

    @staticmethod
    def _extrinsics_to_homomat(extrinsics):
        """把 librealsense 的 source->target 外参转换为 4x4 列向量矩阵。"""
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(
            extrinsics.rotation, dtype=np.float64
        ).reshape(3, 3)
        transform[:3, 3] = np.asarray(
            extrinsics.translation, dtype=np.float64
        )
        return transform

    def _setup_stream_extrinsics(self):
        """读取 depth_T_color，避免把彩色相机位姿直接套用深度相机外参。"""
        color_profile = self.profile.get_stream(rs.stream.color)
        depth_profile = self.profile.get_stream(rs.stream.depth)
        color_to_depth = color_profile.get_extrinsics_to(depth_profile)
        self.T_depth_from_color = self._extrinsics_to_homomat(color_to_depth)

        rotation = self.T_depth_from_color[:3, :3]
        angle_deg = np.degrees(np.arccos(np.clip(
            (np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0
        )))
        baseline_mm = np.linalg.norm(
            self.T_depth_from_color[:3, 3]
        ) * 1000.0
        print("彩色→深度光学坐标外参 depth_T_color:")
        print(self.T_depth_from_color)
        print(
            f"彩色/深度光学中心差: {baseline_mm:.3f} mm, "
            f"旋转差: {angle_deg:.3f}°"
        )

    def convert_pose_from_color(self, color_T_object, output_frame):
        """将 FoundationPose 的 color_T_object 转成指定 RealSense 光学系。"""
        pose = np.asarray(color_T_object, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("FoundationPose 位姿必须是有限的 4x4 矩阵")

        if output_frame == COLOR_OPTICAL_FRAME:
            return pose.copy()
        if output_frame == DEPTH_OPTICAL_FRAME:
            if self.T_depth_from_color is None:
                raise RuntimeError("尚未读取 RealSense 彩色/深度流外参")
            return self.T_depth_from_color @ pose
        raise ValueError(
            f"不支持的位姿输出坐标系: {output_frame}; "
            f"可选 {COLOR_OPTICAL_FRAME} 或 {DEPTH_OPTICAL_FRAME}"
        )

    def _stop_pipeline(self):
        """释放当前 pipeline；可重复调用。"""
        pipeline = self.pipeline
        self.pipeline = None
        self.profile = None
        self.align = None
        self.connected = False
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                # 设备已经被拔出时 stop 也可能报错，此时资源仍需作废。
                pass

    def _schedule_reconnect(self, reason):
        """作废当前流，并安排一次非阻塞式的后续重连。"""
        print(f"RealSense 数据流异常: {reason}")
        print(f"将在 {self.reconnect_interval:.1f} 秒后尝试自动重连...")
        self._stop_pipeline()
        self._next_reconnect_time = time.monotonic() + self.reconnect_interval

    def _try_reconnect(self):
        if time.monotonic() < self._next_reconnect_time:
            return False

        print("正在重新连接 RealSense D435...")
        try:
            self._start_pipeline(is_reconnect=True)
            return True
        except RuntimeError as exc:
            self._stop_pipeline()
            self._next_reconnect_time = time.monotonic() + self.reconnect_interval
            print(f"RealSense 重连失败: {exc}")
            return False

    def _reject_frame(self, reason):
        """短暂坏帧先丢弃；连续坏帧才重建数据流。"""
        self._bad_frame_count += 1
        print(f"丢弃 RealSense 异常帧 ({self._bad_frame_count}/3): {reason}")
        if self._bad_frame_count >= 3:
            self._schedule_reconnect(reason)
        return None, None

    def consume_reconnect_event(self):
        """仅在重连后的第一帧被主循环消费时返回 True。"""
        if not self._reconnect_event_pending:
            return False
        self._reconnect_event_pending = False
        return True

    @staticmethod
    def _timestamp_domain_name(timestamp_domain):
        """把 pyrealsense2 枚举转换成稳定、可打印的短名称。"""
        text = str(timestamp_domain)
        return text.rsplit(".", 1)[-1]

    def _get_capture_timestamp(self, color_frame, host_arrival_timestamp):
        """返回当前彩色帧在主机 Unix 时钟上的采集时间。

        D400 系列启用 global/system time 时，frame timestamp 已经映射到
        主机 Unix epoch，优先使用它。若设备只给出 hardware clock，或映射
        尚未稳定，则回退到 wait_for_frames() 返回时的主机时间。回退值仍远
        早于 FoundationPose 推理结束时间，因此不会把推理和 Socket 延迟算
        进拍照时刻。
        """
        sensor_timestamp_ms = float(color_frame.get_timestamp())
        timestamp_domain = color_frame.get_frame_timestamp_domain()
        timestamp_domain_name = self._timestamp_domain_name(timestamp_domain)

        system_time_domain = getattr(rs.timestamp_domain, "system_time", None)
        global_time_domain = getattr(rs.timestamp_domain, "global_time", None)
        epoch_domains = tuple(
            domain for domain in (system_time_domain, global_time_domain)
            if domain is not None
        )

        candidate_timestamp = sensor_timestamp_ms / 1000.0
        # 刚启动时 global-time 映射可能尚未稳定。与帧到达时间相差太大的
        # 值不能用于跨进程对时，直接使用有界的主机到达时间回退。
        if (timestamp_domain in epoch_domains and
                abs(candidate_timestamp - host_arrival_timestamp) <= 5.0):
            capture_timestamp = candidate_timestamp
            source = f"realsense_{timestamp_domain_name}"
        else:
            capture_timestamp = float(host_arrival_timestamp)
            source = "host_time_at_frame_arrival"

        if source != self._reported_timestamp_source:
            print(
                "RealSense 拍照时间源: "
                f"{source}（sensor domain={timestamp_domain_name}）"
            )
            self._reported_timestamp_source = source

        return (
            capture_timestamp,
            source,
            sensor_timestamp_ms,
            timestamp_domain_name,
        )

    def get_frames(self):
        """
        获取对齐的彩色和深度帧

        Returns:
            color_frame: BGR 彩色图像 (H, W, 3)
            depth_frame: 深度图像，单位米 (H, W)
        """
        if self.pipeline is None and not self._try_reconnect():
            return None, None

        try:
            # 显式超时，避免设备断开后主线程和 OpenCV 窗口永久卡住。
            frames = self.pipeline.wait_for_frames(self.frame_timeout_ms)
            # 必须在任何对齐、numpy 拷贝或 FoundationPose 推理之前记录。
            host_arrival_timestamp = time.time()

            # 对齐深度到彩色
            aligned_frames = self.align.process(frames)

            # 获取对齐后的帧
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
        except RuntimeError as exc:
            self._schedule_reconnect(str(exc))
            return None, None

        if not color_frame or not depth_frame:
            return self._reject_frame("彩色帧或深度帧缺失")

        frame_number = color_frame.get_frame_number()
        frame_timestamp = color_frame.get_timestamp()
        if (frame_number == self._last_frame_number and
                frame_timestamp == self._last_frame_timestamp):
            return self._reject_frame("收到重复的陈旧帧")

        # 转换为 numpy 数组
        # copy() 可及时释放 SDK 帧句柄，避免长时间运行时占满内部帧池。
        color_image = np.asanyarray(color_frame.get_data()).copy()
        depth_image = np.asanyarray(depth_frame.get_data()).copy()

        # 转换深度单位为米
        depth_image = depth_image.astype(np.float32) * self.depth_scale

        expected_color_shape = (self.height, self.width, 3)
        expected_depth_shape = (self.height, self.width)
        if color_image.shape != expected_color_shape:
            return self._reject_frame(
                f"彩色图尺寸异常: {color_image.shape}，期望 {expected_color_shape}"
            )
        if depth_image.shape != expected_depth_shape:
            return self._reject_frame(
                f"深度图尺寸异常: {depth_image.shape}，期望 {expected_depth_shape}"
            )

        valid_depth_ratio = np.count_nonzero(
            np.isfinite(depth_image) & (depth_image > 0)
        ) / depth_image.size
        if valid_depth_ratio < self.min_valid_depth_ratio:
            return self._reject_frame(
                f"有效深度仅 {valid_depth_ratio * 100:.1f}%"
            )

        self._last_frame_number = frame_number
        self._last_frame_timestamp = frame_timestamp
        self._bad_frame_count = 0

        (self.last_capture_timestamp,
         self.last_capture_timestamp_source,
         self.last_sensor_timestamp_ms,
         self.last_timestamp_domain) = self._get_capture_timestamp(
            color_frame, host_arrival_timestamp
        )
        self.last_sensor_frame_number = int(frame_number)

        return color_image, depth_image

    def stop(self):
        """停止摄像头"""
        self._stop_pipeline()


def create_mask_from_depth(depth, min_depth=0.1, max_depth=2.0):
    """
    从深度图创建粗略掩码

    Args:
        depth: 深度图 (H, W)
        min_depth: 最小有效深度（米）
        max_depth: 最大有效深度（米）

    Returns:
        mask: 二值掩码 (H, W)
    """
    mask = ((depth > min_depth) & (depth < max_depth)).astype(np.uint8)
    
    # 添加形态学操作来改善掩码质量
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def validate_and_fix_mask(mask, min_pixels=500):
    """
    验证并修复掩码，确保有足够的像素
    
    Args:
        mask: 输入掩码
        min_pixels: 最小像素数量
        
    Returns:
        fixed_mask: 修复后的掩码
    """
    if mask is None or mask.sum() < min_pixels:
        print(f"警告: 掩码像素不足 ({mask.sum() if mask is not None else 0} < {min_pixels})")
        # 创建一个中心区域的矩形掩码
        h, w = mask.shape if mask is not None else (480, 640)
        center_h = int(h * 0.4)
        center_w = int(w * 0.4)
        start_h = (h - center_h) // 2
        end_h = start_h + center_h
        start_w = (w - center_w) // 2
        end_w = start_w + center_w
        
        fixed_mask = np.zeros((h, w), dtype=np.uint8)
        fixed_mask[start_h:end_h, start_w:end_w] = 1
        return fixed_mask
    
    # 找到最大的连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels > 1:
        # 排除背景（标签0），找到最大的前景区域
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8)
        
        # 再次验证
        if mask.sum() < min_pixels:
            print(f"警告: 处理后掩码仍然太小 ({mask.sum()})")
    
    return mask


def interactive_mask_selection(color, depth, window_name="Select Object"):
    """
    交互式选择目标物体区域
    
    Args:
        color: 彩色图像
        depth: 深度图
        window_name: 窗口名称
        
    Returns:
        mask: 选择的掩码
    """
    display_img = color.copy()
    print("\n" + "="*50)
    print("请选择要跟踪的物体区域")
    print("矩形模式：拖动鼠标框选，按 'c' 确认")
    print("多边形模式：按 'p' 切换，沿物体边缘逐点单击，按 'c' 确认")
    print("多边形按 'u' 撤销上一点；按 'r' 清除重选；按 'q' 退出")
    print("="*50 + "\n")

    roi = None
    drawing = False
    temp_roi = None
    selection_mode = "rectangle"
    polygon_points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal roi, drawing, temp_roi

        if selection_mode == "polygon":
            if event == cv2.EVENT_LBUTTONDOWN:
                polygon_points.append((x, y))
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            temp_roi = (x, y)
            roi = None
        elif event == cv2.EVENT_MOUSEMOVE and drawing and temp_roi is not None:
            temp_img = display_img.copy()
            cv2.rectangle(temp_img, temp_roi, (x, y), (0, 255, 0), 2)
            cv2.imshow(window_name, temp_img)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if temp_roi is not None:
                roi = (temp_roi, (x, y))
                print(f"已选择区域: {roi}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        show_img = display_img.copy()
        if selection_mode == "rectangle" and roi is not None:
            cv2.rectangle(show_img, roi[0], roi[1], (0, 255, 0), 2)
        elif selection_mode == "polygon" and polygon_points:
            pts = np.asarray(polygon_points, dtype=np.int32).reshape(-1, 1, 2)
            if len(polygon_points) >= 3:
                overlay = show_img.copy()
                cv2.fillPoly(overlay, [pts], (0, 180, 0))
                show_img = cv2.addWeighted(overlay, 0.2, show_img, 0.8, 0)
            if len(polygon_points) >= 2:
                cv2.polylines(show_img, [pts], False, (0, 255, 0), 2)
            for point in polygon_points:
                cv2.circle(show_img, point, 3, (0, 0, 255), -1)

        mode_label = "多边形" if selection_mode == "polygon" else "矩形"
        cv2.putText(show_img, f"{mode_label}: P切换 C确认 U撤销 R重选 Q退出",
                    (10, show_img.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(window_name, show_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cv2.destroyWindow(window_name)
            return None
        if key == ord('p'):
            selection_mode = "rectangle" if selection_mode == "polygon" else "polygon"
            roi = None
            polygon_points.clear()
            drawing = False
            temp_roi = None
            continue
        if key == ord('u') and selection_mode == "polygon":
            if polygon_points:
                polygon_points.pop()
            continue
        if key == ord('r'):
            roi = None
            polygon_points.clear()
            drawing = False
            temp_roi = None
            continue
        if key != ord('c'):
            continue

        mask = np.zeros(color.shape[:2], dtype=np.uint8)
        if selection_mode == "polygon":
            if len(polygon_points) < 3:
                print("多边形至少需要 3 个点")
                continue
            pts = np.asarray(polygon_points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 1)
        elif roi is not None:
            pt1, pt2 = roi
            x1 = max(0, min(pt1[0], pt2[0]))
            y1 = max(0, min(pt1[1], pt2[1]))
            x2 = min(color.shape[1], max(pt1[0], pt2[0]))
            y2 = min(color.shape[0], max(pt1[1], pt2[1]))
            mask[y1:y2, x1:x2] = 1
        else:
            continue

        pixel_count = int(mask.sum())
        print(f"已选择 {pixel_count} 个像素")
        if pixel_count < 500:
            print("警告: 选择的区域太小")
            response = input("是否继续？(y/n): ").strip().lower()
            if response != 'y':
                roi = None
                polygon_points.clear()
                continue

        cv2.destroyWindow(window_name)
        return mask


def run_realsense_tracking(mesh_file, min_depth=0.2, max_depth=1.5, debug=1,
                           pose_callback=None,
                           pose_output_frame=COLOR_OPTICAL_FRAME,
                           track_iterations=1,
                           pose_registration_callback=None,
                           registration_iterations=None,
                           rotation_grid_views=None,
                           rotation_grid_inplane_step=None,
                           diameter_samples=None,
                           mask_preview_delay_ms=None):
    """
    运行 RealSense 实时跟踪

    Args:
        mesh_file: 物体 mesh 文件路径
        min_depth: 最小跟踪距离（米）
        max_depth: 最大跟踪距离（米）
        debug: 调试级别
        pose_callback: 可选回调，签名为
            callback(pose_matrix, frame_count, timestamp)。每个有效姿态调用一次，
            timestamp 是该图像帧的采集 Unix 时间，而不是推理完成时间；
            回调异常不会中断相机采集和 FoundationPose 跟踪。
        pose_output_frame: 回调位姿所在的 RealSense 光学坐标系。FoundationPose
            原始输出是彩色光学系；当前眼在手上标定使用深度光学系。
        track_iterations: 每帧姿态细化次数。实时路径默认 1；需要更强细化时可设为 2。
        pose_registration_callback: 可选的注册生命周期回调；初次注册以及按 r
            重新注册成功后各调用一次，用于重置发送端的首帧方向约束。
        registration_iterations: 初始注册的姿态细化次数。实时默认 2。
        rotation_grid_views: 初始旋转候选的球面视角数上限。
        rotation_grid_inplane_step: 初始旋转候选的平面内角度步长（度）。
        diameter_samples: 估算 mesh 直径时最多采样的顶点数。
        mask_preview_delay_ms: 掩码确认后的预览时间；0 表示跳过预览。
    """
    # Keep direct calls and D435test_socket_sender.py consistent with the
    # editable configuration block at the end of this file.
    if registration_iterations is None:
        registration_iterations = REGISTRATION_ITERATIONS
    if rotation_grid_views is None:
        rotation_grid_views = ROTATION_GRID_VIEWS
    if rotation_grid_inplane_step is None:
        rotation_grid_inplane_step = ROTATION_GRID_INPLANE_STEP
    if diameter_samples is None:
        diameter_samples = DIAMETER_SAMPLES
    if mask_preview_delay_ms is None:
        mask_preview_delay_ms = MASK_PREVIEW_DELAY_MS

    if track_iterations < 1:
        raise ValueError("track_iterations 必须至少为 1")
    if registration_iterations < 1:
        raise ValueError("registration_iterations 必须至少为 1")
    if rotation_grid_views < 1:
        raise ValueError("rotation_grid_views 必须至少为 1")
    if not 1 <= rotation_grid_inplane_step <= 360:
        raise ValueError("rotation_grid_inplane_step 必须在 1 到 360 度之间")
    if diameter_samples < 2:
        raise ValueError("diameter_samples 必须至少为 2")
    if mask_preview_delay_ms < 0:
        raise ValueError("mask_preview_delay_ms 不能为负数")

    startup_started = time.perf_counter()

    # 先加载两个网络，使权重 I/O 能与 mesh 处理重叠。
    print("后台加载神经网络模型...")
    model_executor = ThreadPoolExecutor(max_workers=1)
    model_future = model_executor.submit(
        lambda: (ScorePredictor(), PoseRefinePredictor())
    )

    # 加载 mesh
    print(f"加载 mesh: {mesh_file}")
    try:
        mesh = trimesh.load(mesh_file)
    except Exception:
        model_executor.shutdown(wait=True)
        raise
    
    # ==================== OBJ 模型坐标系校正 ====================
    # 必须使用行列式为 +1 的合法旋转矩阵。当前模型不需要校正，因此为单位阵。
    # correction_transform = np.array([
    #     [0,  0,  1],  # 新的 X = 原来的 Z
    #     [0,  1,  0],  # 新的 Y = 原来的 Y
    #     [1, 0,  0]   # 新的 Z = 原来的 -X（反向）
    # ])

    correction_transform = np.array([
        [1,  0,  0],  # 新的 X = 原来的 Z
        [0,  1,  0],  # 新的 Y = 原来的 Y
        [0,  0,  1]   # 新的 Z = 原来的 -X（反向）
    ])

    
    # 应用坐标变换到所有顶点
    mesh.vertices = mesh.vertices @ correction_transform.T
    
    print("OBJ 坐标系校正: 单位旋转（未交换或反射坐标轴）")
    # =============================================================
    
    # 打印 mesh 信息，帮助诊断坐标系问题
    print(f"Mesh 边界框:")
    print(f"  最小点: {mesh.vertices.min(axis=0)}")
    print(f"  最大点: {mesh.vertices.max(axis=0)}")
    print(f"  中心点: {(mesh.vertices.min(axis=0) + mesh.vertices.max(axis=0)) / 2}")

    # NVDiffRast 上下文创建很快，先在主线程完成；模型和估计器随后
    # 在后台初始化，用户框选时可以覆盖这部分耗时。
    try:
        glctx = dr.RasterizeCudaContext()
        print("NVDiffRast 初始化成功")
    except Exception as e:
        print(f"NVDiffRast 初始化失败: {e}")
        print("继续使用 CPU 模式（性能可能较低）")
        glctx = None

    debug_dir = './debug_realsense'
    os.makedirs(debug_dir, exist_ok=True)
    mesh_normals = mesh.vertex_normals

    print("后台加载神经网络并初始化姿态估计器...")

    def initialize_estimator():
        scorer, refiner = model_future.result()
        return FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=debug_dir,
            debug=debug,
            glctx=glctx,
            diameter_n_sample=diameter_samples,
            rotation_grid_min_n_views=rotation_grid_views,
            rotation_grid_inplane_step=rotation_grid_inplane_step,
            rotation_grid_max_n_views=rotation_grid_views,
        )

    estimator_future = model_executor.submit(initialize_estimator)

    # 相机取首帧和用户框选掩码时，估计器继续在后台初始化。
    print("\n启动 RealSense D435...")
    camera = None
    try:
        # 可以尝试降低分辨率以提高性能：width=424, height=240
        camera = RealSenseD435(width=640, height=480, fps=30)
    except Exception:
        model_executor.shutdown(wait=True)
        raise

    # 获取第一帧并选择目标
    print("\n等待第一帧数据...")
    color, depth = None, None
    while color is None or depth is None:
        color, depth = camera.get_frames()
        if color is None or depth is None:
            # 相机刚启动时可能出现短暂空帧；如果此时发生断连，
            # get_frames() 也会继续执行同一套自动重连流程。
            time.sleep(0.01)
    
    # 检查深度图质量
    valid_depth = (depth > 0).sum()
    total_pixels = depth.shape[0] * depth.shape[1]
    print(f"深度图有效像素: {valid_depth}/{total_pixels} ({100*valid_depth/total_pixels:.1f}%)")
    
    if valid_depth < total_pixels * 0.1:
        print("警告: 深度图有效像素过少，请检查 RealSense 连接和权限！")
        response = input("是否继续？(y/n): ").strip().lower()
        if response != 'y':
            camera.stop()
            model_executor.shutdown(wait=True)
            return
    
    # 交互式选择目标
    print("\n" + "="*60)
    print("重要提示：")
    print("1. 请确保物体在画面中清晰可见")
    print("2. 选择框应该紧密包围物体（不要包含太多背景）")
    print("3. 建议在良好光照条件下使用")
    print("4. 物体距离摄像头 0.3-1.0 米效果最佳")
    print("="*60 + "\n")
    
    mask_selection_started = time.perf_counter()
    mask = interactive_mask_selection(color, depth)
    mask_selection_elapsed = time.perf_counter() - mask_selection_started
    
    if mask is None:
        print("未选择目标，退出")
        camera.stop()
        model_executor.shutdown(wait=True)
        return
    
    # 验证并修复掩码
    mask = validate_and_fix_mask(mask, min_pixels=500)
    print(f"最终掩码像素数: {mask.sum()}")
    mask_confirmed_at = time.perf_counter()
    
    # 框选窗口已经显示了最终区域；实时模式默认不再固定等待 1 秒。
    if mask_preview_delay_ms > 0:
        mask_vis = np.stack(
            [mask * 255, mask * 128, mask * 128], axis=-1
        ).astype(np.uint8)
        mask_vis = cv2.addWeighted(color, 0.7, mask_vis, 0.3, 0)
        cv2.imshow('Mask Preview', mask_vis)
        cv2.waitKey(mask_preview_delay_ms)
        cv2.destroyWindow('Mask Preview')

    # 用户完成框选后再等待 estimator future；多数情况下此处已经完成。
    estimator_wait_started = time.perf_counter()
    try:
        estimator = estimator_future.result()
    except Exception:
        camera.stop()
        model_executor.shutdown(wait=True)
        raise
    model_executor.shutdown(wait=True)
    estimator_wait_elapsed = time.perf_counter() - estimator_wait_started
    print("估计器初始化完成！")
    print(f"掩码确认后等待估计器: {estimator_wait_elapsed:.2f} 秒")
    _synchronize_cuda()
    print(
        f"初始化就绪耗时（不含人工框选）: "
        f"{time.perf_counter() - startup_started - mask_selection_elapsed:.2f} 秒"
    )
    
    print(f"\n开始跟踪！")
    print("控制说明：")
    print("  - 按 'q' 退出")
    print("  - 按 'r' 重新注册（重新检测姿态）")
    print("  - 按 's' 保存当前帧到 debug 目录\n")
    
    def register_pose(register_color, register_depth, register_mask):
        _synchronize_cuda()
        register_started = time.perf_counter()
        registered_pose = estimator.register(
            K=camera.K,
            rgb=register_color,
            depth=register_depth,
            ob_mask=register_mask,
            iteration=registration_iterations,
        )
        _synchronize_cuda()
        print(
            f"注册耗时: {time.perf_counter() - register_started:.2f} 秒 "
            f"（{registration_iterations} 次细化）"
        )
        return registered_pose

    def publish_pose(pose_matrix, index, print_values=False):
        if print_values:
            position = pose_matrix[:3, 3]
            rotation = pose_matrix[:3, :3]
            print(f"\nFrame {index}:")
            print(
                "  Position: "
                f"[{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]"
            )
            print(f"  Rotation:\n{rotation}")

        if pose_callback is not None:
            try:
                callback_pose = camera.convert_pose_from_color(
                    pose_matrix, pose_output_frame
                )
                pose_callback(
                    callback_pose,
                    index,
                    camera.last_capture_timestamp,
                )
            except Exception as callback_error:
                print(f"位姿回调失败，已跳过当前帧: {callback_error}")

    # 注册初始姿态，并立即输出注册结果，不再额外等待下一帧跟踪。
    pose = register_pose(color, depth, mask)
    if pose_registration_callback is not None:
        try:
            pose_registration_callback()
        except Exception as callback_error:
            print(f"注册生命周期回调失败: {callback_error}")

    frame_count = 0
    registered = True
    publish_pose(pose, frame_count, print_values=True)
    print(
        f"掩码确认到首个 6D 坐标耗时: "
        f"{time.perf_counter() - mask_confirmed_at:.2f} 秒"
    )
    frame_count += 1

    # 主循环
    while True:
        color, depth = camera.get_frames()

        if color is None or depth is None:
            # get_frames() 会在后台按间隔重连。这里继续处理窗口事件，
            # 保证相机离线期间界面仍可响应 q 键退出。
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        if camera.consume_reconnect_event():
            # 断流期间目标或相机可能已经移动，旧 pose 不能跨数据流沿用。
            # 清空 FoundationPose 内部状态，并在新鲜帧上重新框选注册。
            print("\n相机已恢复。为防止旧姿态污染，将重新框选并注册目标。")
            registered = False
            estimator.pose_last = None
            if hasattr(run_realsense_tracking, 'last_pose'):
                delattr(run_realsense_tracking, 'last_pose')

        try:
            if not registered:
                # 重新注册
                print("\n进入重新注册模式...")
                mask = interactive_mask_selection(color, depth)
                if mask is None:
                    print("已取消重新注册，退出跟踪")
                    break
                mask = validate_and_fix_mask(mask, min_pixels=500)
                pose = register_pose(color, depth, mask)
                if pose_registration_callback is not None:
                    try:
                        pose_registration_callback()
                    except Exception as callback_error:
                        print(f"注册生命周期回调失败: {callback_error}")
                registered = True
                print("重新注册成功！")
            else:
                # 跟踪
                try:
                    pose = estimator.track_one(
                        rgb=color,
                        depth=depth,
                        K=camera.K,
                        iteration=track_iterations
                    )
                    
                    # 检查跟踪质量（仅警告，不自动重新注册）
                    if frame_count > 3 and hasattr(run_realsense_tracking, 'last_pose'):
                        last_pos = run_realsense_tracking.last_pose[:3, 3]
                        curr_pos = pose[:3, 3]
                        displacement = np.linalg.norm(curr_pos - last_pos)
                        
                        # 如果单帧位移超过阈值，仅显示警告
                        if displacement > 0.15:
                            print(f"⚠️  警告: 检测到大幅位移 ({displacement:.3f}m)，跟踪可能不稳定")
                            print("   如需重新注册，请按 'r' 键")
                    
                    run_realsense_tracking.last_pose = pose.copy()

                except Exception as e:
                    print(f"跟踪失败: {e}")
                    import traceback
                    traceback.print_exc()
                    print("请按 'r' 键重新注册")
                    registered = False
                    continue

                # 输出位置和姿态矩阵（直接使用 pose，不需要额外校正）
                position = pose[:3, 3]
                rotation = pose[:3, :3]
                
                print(f"\nFrame {frame_count}:")
                print(f"  Position: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
                print(f"  Rotation:\n{rotation}")

            publish_pose(pose, frame_count)

            # 可视化结果
            # 直接使用 pose，这样坐标轴会显示在 OBJ 文件定义的原点
            vis = draw_posed_3d_box(camera.K, img=color, ob_in_cam=pose, bbox=np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)]))

            # 绘制坐标轴（会显示在 OBJ 原点位置）
            vis = draw_xyz_axis(vis, ob_in_cam=pose, scale=0.1,
                              K=camera.K, thickness=3, transparency=0, is_input_rgb=False)

            # 显示信息
            cv2.putText(vis, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, "Press 'q': quit, 'r': reregister, 's': save",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 显示结果
            cv2.imshow('FoundationPose - RealSense Tracking', vis)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("\n重新注册...")
                registered = False
            elif key == ord('s'):
                save_path = f"{debug_dir}/frame_{frame_count:05d}.png"
                cv2.imwrite(save_path, vis)
                print(f"已保存: {save_path}")

            frame_count += 1

        except Exception as e:
            print(f"跟踪错误: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 清理
    camera.stop()
    cv2.destroyAllWindows()
    print(f"\n跟踪结束，共处理 {frame_count} 帧")


# ==================== 配置区域 ====================
# 在这里设置您的参数
# MESH_FILE = "demo_data/mustard0/mesh/textured_simple.obj"  # 物体 mesh 文件路径
# MESH_FILE = "demo_data/mustard0/mesh/objectfoundationpose.obj"  # 物体 mesh

MESH_FILE = r"D:\FoundationPose-main\FoundationPose-main\meshes_stl\culture_flask1.STL"  # 物体 mesh

MIN_DEPTH = 0.1   # 最小跟踪距离（米）- 从 0.001 改为 0.1，过滤太近的噪声
MAX_DEPTH = 1.5   # 最大跟踪距离（米）- 从 1 改为 1.5，增加范围
DEBUG = 0         # 调试级别 (0-3)
TRACK_ITERATIONS = 1  # 实时跟踪默认单次细化；设为 2 可换取更强细化
REGISTRATION_ITERATIONS = DEFAULT_REGISTRATION_ITERATIONS  # 初始注册细化次数；5 更准但明显更慢
ROTATION_GRID_VIEWS = DEFAULT_ROTATION_GRID_VIEWS  # 初始球面视角数；40 为 FoundationPose 原始值
ROTATION_GRID_INPLANE_STEP = DEFAULT_ROTATION_GRID_INPLANE_STEP  # 平面内旋转步长；原始值为 60
DIAMETER_SAMPLES = DEFAULT_DIAMETER_SAMPLES  # 大 mesh 启动优化；原始值为 10000
MASK_PREVIEW_DELAY_MS = DEFAULT_MASK_PREVIEW_DELAY_MS  # 0 跳过确认后的固定预览等待
# ================================================

if __name__ == '__main__':
    # 检查 mesh 文件
    if not os.path.exists(MESH_FILE):
        print(f"错误: mesh 文件不存在: {MESH_FILE}")
        exit(1)
    
    print(f"使用配置:")
    print(f"  Mesh 文件: {MESH_FILE}")
    print(f"  最小深度: {MIN_DEPTH}m")
    print(f"  最大深度: {MAX_DEPTH}m")
    print(f"  调试级别: {DEBUG}")
    print(f"  初始注册细化次数: {REGISTRATION_ITERATIONS}")
    print(
        f"  初始旋转候选: 最多 {ROTATION_GRID_VIEWS} 个视角，"
        f"平面内步长 {ROTATION_GRID_INPLANE_STEP}°"
    )
    print(f"  每帧细化次数: {TRACK_ITERATIONS}\n")

    # 直接开始运行；本进程刚启动时清空 CUDA 缓存只会增加一次同步。
    if torch.cuda.is_available():
        print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
        print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print()

    # 运行跟踪
    run_realsense_tracking(
        mesh_file=MESH_FILE,
        min_depth=MIN_DEPTH,
        max_depth=MAX_DEPTH,
        debug=DEBUG,
        track_iterations=TRACK_ITERATIONS,
        registration_iterations=REGISTRATION_ITERATIONS,
        rotation_grid_views=ROTATION_GRID_VIEWS,
        rotation_grid_inplane_step=ROTATION_GRID_INPLANE_STEP,
        diameter_samples=DIAMETER_SAMPLES,
        mask_preview_delay_ms=MASK_PREVIEW_DELAY_MS,
    )
