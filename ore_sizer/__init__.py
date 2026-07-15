"""ore_sizer: 基于 ToF 高度图的矿石粒径测量流水线。

流程: 高度图 -> 预处理 -> 分水岭分割 -> 完整性判定(遮挡剔除) -> 尺寸测量 -> 粒径统计
"""

from .config import PipelineConfig
from .types import HeightFrame, RockInfo, FrameResult
from .pipeline import process_frame

__all__ = [
    "PipelineConfig",
    "HeightFrame",
    "RockInfo",
    "FrameResult",
    "process_frame",
]
