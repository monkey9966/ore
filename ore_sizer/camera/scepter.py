"""DS63 相机适配 (ScepterSDK Python)。

办公室静态验证用: 连接相机 -> 配置滤波器 -> 抓取深度帧 -> 转为 HeightFrame。

使用前提:
1. 下载 ScepterSDK: https://github.com/ScepterSW/ScepterSDK
2. 设置环境变量 SCEPTER_SDK_DIR 指向 ScepterSDK 仓库根目录
   (适配器会把 MultilanguageSDK/Python 加入 sys.path 并加载动态库)
3. 相机与主机同网段(默认 IP 192.168.1.101)

注意: SDK 的动态库按 sys.path[0] 定位, 因此这里通过临时切换 sys.path 加载。
"""

import os
import sys
import time
from typing import Optional, Tuple

import numpy as np

from ..config import PipelineConfig
from ..preprocess import depth_to_heightframe
from ..types import HeightFrame

_camera = None
_intrinsics = None


class ScepterNotAvailable(RuntimeError):
    pass


def _sdk_python_dir() -> str:
    sdk_dir = os.environ.get("SCEPTER_SDK_DIR", "")
    if not sdk_dir:
        raise ScepterNotAvailable(
            "未设置环境变量 SCEPTER_SDK_DIR。请下载 ScepterSDK "
            "(https://github.com/ScepterSW/ScepterSDK) 并将该变量指向仓库根目录。")
    py_dir = os.path.join(sdk_dir, "MultilanguageSDK", "Python")
    if not os.path.isdir(os.path.join(py_dir, "API")):
        raise ScepterNotAvailable(f"{py_dir}/API 不存在, 请检查 SCEPTER_SDK_DIR。")
    return py_dir


def connect(cfg: Optional[PipelineConfig] = None):
    """搜索并连接第一台相机, 配置滤波器, 开流。返回 (camera, intrinsics)。"""
    global _camera, _intrinsics
    if _camera is not None:
        return _camera, _intrinsics

    py_dir = _sdk_python_dir()
    # SDK 用 sys.path[0] 定位动态库, 需临时替换
    old_path0 = sys.path[0]
    sys.path.insert(0, py_dir)
    try:
        from API.ScepterDS_api import ScepterTofCam  # noqa
        from API.ScepterDS_types import (ScConfidenceFilterParams,
                                         ScFlyingPixelFilterParams,
                                         ScTimeFilterParams)
        from API.ScepterDS_enums import ScFrameType, ScSensorType, ScConnectStatus
        from ctypes import c_bool, c_uint16
    except Exception as e:
        raise ScepterNotAvailable(f"加载 ScepterSDK Python API 失败: {e}")
    finally:
        if sys.path[0] == py_dir:
            sys.path.pop(0)

    cam = ScepterTofCam()
    count = cam.scGetDeviceCount(3000)
    if count <= 0:
        raise ScepterNotAvailable("未搜索到相机。请确认: 相机已上电、网线连接、主机与相机同网段(默认192.168.1.101/24)。")
    ret, infolist = cam.scGetDeviceInfoList(count)
    if ret != 0:
        raise ScepterNotAvailable(f"scGetDeviceInfoList 失败 ScStatus({ret})")
    dev = infolist[0]
    if dev.status != ScConnectStatus.SC_CONNECTABLE.value:
        raise ScepterNotAvailable(f"设备状态不可连接: {dev.status} (可能被其他程序占用)")
    ret = cam.scOpenDeviceBySN(dev.serialNumber)
    if ret != 0:
        raise ScepterNotAvailable(f"scOpenDeviceBySN 失败 ScStatus({ret})")

    # ---- 滤波器配置(与算法方案匹配) ----
    conf = ScConfidenceFilterParams(); conf.enable = True; conf.threshold = 30
    cam.scSetConfidenceFilterParams(conf)
    fly = ScFlyingPixelFilterParams(); fly.enable = True; fly.threshold = 15
    cam.scSetFlyingPixelFilterParams(fly)
    cam.scSetSpatialFilterEnabled(c_bool(True))
    # 关键: 关闭 FillHole(算法需要知道真实无效区), 关闭时间滤波(皮带运动场景)
    cam.scSetFillHoleFilterEnabled(c_bool(False))
    tf = ScTimeFilterParams(); tf.enable = False
    cam.scSetTimeFilterParams(tf)

    ret, intr = cam.scGetSensorIntrinsicParameters(ScSensorType.SC_TOF_SENSOR)
    if ret != 0:
        raise ScepterNotAvailable(f"获取内参失败 ScStatus({ret})")

    ret = cam.scStartStream()
    if ret != 0:
        raise ScepterNotAvailable(f"scStartStream 失败 ScStatus({ret})")
    time.sleep(1.0)

    _camera = cam
    _intrinsics = intr
    return cam, intr


def capture_depth(timeout_ms: int = 1500, retries: int = 10) -> Tuple[np.ndarray, dict]:
    """抓取一帧深度图。返回 (depth_mm uint16 (H,W), intrinsics dict)。"""
    py_dir = _sdk_python_dir()
    sys.path.insert(0, py_dir)
    try:
        from API.ScepterDS_enums import ScFrameType
        from ctypes import c_uint16
    finally:
        if sys.path[0] == py_dir:
            sys.path.pop(0)

    cam, intr = connect()
    for _ in range(retries):
        ret, ready = cam.scGetFrameReady(c_uint16(timeout_ms))
        if ret != 0 or not ready.depth:
            continue
        ret, frame = cam.scGetFrame(ScFrameType.SC_DEPTH_FRAME)
        if ret != 0:
            continue
        buf = np.ctypeslib.as_array(frame.pFrameData, shape=(frame.dataLen,))
        depth = buf.view(np.uint16).reshape(frame.height, frame.width).copy()
        return depth, {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy}
    raise ScepterNotAvailable("多次尝试未取到深度帧")


def capture_heightframe(cfg: Optional[PipelineConfig] = None) -> HeightFrame:
    """抓取一帧并转为正射高度图(基准面自动 RANSAC 拟合)。"""
    cfg = cfg or PipelineConfig()
    depth, intr = capture_depth()
    return depth_to_heightframe(depth, intr["fx"], intr["fy"],
                                intr["cx"], intr["cy"], cfg)


def disconnect():
    global _camera
    if _camera is not None:
        try:
            _camera.scStopStream()
            _camera.scCloseDevice()
        finally:
            _camera = None
