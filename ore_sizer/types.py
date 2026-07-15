"""核心数据结构。所有长度单位均为毫米(mm)。"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class HeightFrame:
    """标准化的高度图帧, 算法唯一输入, 与相机解耦。

    height: 每像素高于基准面(皮带面/地面)的高度, float32, 单位 mm, 无效处为 NaN
    valid:  深度是否有效(置信度/飞点滤波后)
    """

    height: np.ndarray
    valid: np.ndarray
    mm_per_px: float
    trigger_id: int = 0
    meta: Dict = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height.shape


# 边界分类码
BOUNDARY_FREE = 1      # 自由边界: 向下跌落, 真实物理边缘
BOUNDARY_OCCLUDED = 2  # 遮挡边界: 被相邻更高/等高物压住
BOUNDARY_INVALID = 3   # 无效边界: 外侧数据缺失

# 拒绝原因码 -> 中文说明
REJECT_REASONS = {
    "OCCLUDED": "被遮挡/压叠(自由边界占比不足)",
    "INVALID_BOUNDARY": "轮廓贴近深度空洞",
    "TOUCH_BORDER": "触碰视场边缘",
    "LOW_VALID_COVERAGE": "表面有效数据不足",
    "LOW_SOLIDITY": "形状异常(疑似欠分割/轮廓缺损)",
    "BAD_ELLIPSE_FIT": "椭圆拟合差(疑似两块粘连/欠分割)",
    "FERET_ELLIPSE_MISMATCH": "尺寸交叉校验不一致(疑似附着邻块/碎料)",
    "PEAK_ON_EDGE": "最高点贴边(疑似大矿石露出的一角)",
    "TOO_SMALL": "小于统计下限",
}


@dataclass
class RockInfo:
    """单块矿石的评估与测量结果。"""

    label: int
    accepted: bool
    reject_reasons: List[str] = field(default_factory=list)

    # 测量值 (mm), 拒绝的矿石也带参考值(不计入统计)
    max_feret_mm: float = 0.0        # 最大边长(最大Feret直径)
    min_rect_l_mm: float = 0.0       # 最小外接矩形长
    min_rect_w_mm: float = 0.0       # 最小外接矩形宽(次轴, 用于对标筛分)
    equiv_diameter_mm: float = 0.0   # 等面积圆直径
    thickness_mm: float = 0.0        # 峰值高度 - 局部基准
    volume_mm3: float = 0.0          # 体积估计(高度积分)
    area_mm2: float = 0.0

    sizing_mm: float = 0.0           # 分箱依据值(由配置决定)
    size_bin: str = ""               # 所属粒级, 如 "200-300"

    # 判定过程指标(便于现场调参)
    free_ratio: float = 0.0
    occluded_ratio: float = 0.0
    invalid_ratio: float = 0.0
    solidity: float = 0.0
    ellipse_iou: float = 0.0
    valid_coverage: float = 0.0

    # 几何信息(像素坐标, 供可视化/交叉校验)
    meta_ellipse_axes_px: Tuple[float, float] = (0.0, 0.0)
    centroid_px: Tuple[float, float] = (0.0, 0.0)
    bbox_px: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (min_row, min_col, max_row, max_col)
    max_feret_endpoints_px: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None


@dataclass
class FrameResult:
    """单帧处理结果。"""

    frame: HeightFrame
    labels: np.ndarray                 # 分割标签图(0=背景, 含碎料标签)
    rocks: List[RockInfo]              # 候选矿石(不含碎料)
    boundary_class: np.ndarray         # 边界分类图(0/FREE/OCCLUDED/INVALID), 供可视化
    prominence: np.ndarray             # 相对局部基准的凸起图
    debris_labels: List[int] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def accepted_rocks(self) -> List[RockInfo]:
        return [r for r in self.rocks if r.accepted]

    @property
    def measurable_ratio(self) -> float:
        """本帧完整可测率 = 接受数 / 候选数。"""
        return len(self.accepted_rocks) / len(self.rocks) if self.rocks else 0.0
