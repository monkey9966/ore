"""帧文件读写: 采集与算法解耦, 便于离线回放调参。

.npz 格式约定:
    depth      uint16 (H,W)  透视深度图, 单位 mm, 0=无效   [真机采集]
    fx,fy,cx,cy float        ToF 内参                      [真机采集]
或
    height     float32 (H,W) 正射高度图, NaN=无效          [合成/预转换]
    mm_per_px  float
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from ..config import PipelineConfig
from ..preprocess import depth_to_heightframe
from ..types import HeightFrame


def save_depth_npz(path: Union[str, Path], depth_mm: np.ndarray,
                   fx: float, fy: float, cx: float, cy: float,
                   ir: Optional[np.ndarray] = None) -> None:
    data = dict(depth=depth_mm.astype(np.uint16), fx=fx, fy=fy, cx=cx, cy=cy)
    if ir is not None:
        data["ir"] = ir
    np.savez_compressed(path, **data)


def save_heightframe_npz(path: Union[str, Path], frame: HeightFrame) -> None:
    np.savez_compressed(path, height=frame.height.astype(np.float32),
                        mm_per_px=frame.mm_per_px)


def load_frame(path: Union[str, Path], cfg: PipelineConfig) -> HeightFrame:
    """加载 .npz / .npy / 16bit png 为 HeightFrame。"""
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "height" in data:
            h = data["height"].astype(np.float32)
            valid = np.isfinite(h)
            return HeightFrame(h, valid, float(data["mm_per_px"]),
                               meta={"source": str(path)})
        if "depth" in data:
            return depth_to_heightframe(
                data["depth"], float(data["fx"]), float(data["fy"]),
                float(data["cx"]), float(data["cy"]), cfg)
        raise ValueError(f"{path}: npz 中缺少 height 或 depth 数组")
    if path.suffix == ".npy":
        depth = np.load(path)
        return _depth_with_default_intrinsics(depth, cfg)
    if path.suffix.lower() in (".png", ".tif", ".tiff"):
        import cv2
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"无法读取 {path}")
        if depth.ndim == 3:
            raise ValueError("需要 16bit 单通道深度图(mm)")
        return _depth_with_default_intrinsics(depth.astype(np.uint16), cfg)
    raise ValueError(f"不支持的格式: {path.suffix}")


def _depth_with_default_intrinsics(depth: np.ndarray, cfg: PipelineConfig) -> HeightFrame:
    """无内参时按 DS63 ToF FOV(67°x50°, VGA) 估算内参。"""
    h, w = depth.shape
    fx = w / (2 * np.tan(np.deg2rad(67 / 2)))
    fy = h / (2 * np.tan(np.deg2rad(50 / 2)))
    return depth_to_heightframe(depth, fx, fy, w / 2, h / 2, cfg)
