"""流水线参数配置。所有长度单位均为毫米(mm)。"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PipelineConfig:
    # ---------- 正射高度图网格 ----------
    mm_per_px: float = 3.0            # 高度图网格分辨率

    # ---------- 前景提取 (Stage A) ----------
    base_disk_r_mm: float = 300.0     # 局部基准面开运算半径(大于最大矿石半径)
    fg_prominence_mm: float = 40.0    # 相对局部基准面的凸起阈值, 低于此视为碎料床/背景
    max_hole_fill_mm: float = 30.0    # 小于该等效直径的无效洞做插值填补, 更大的保留为无效

    # ---------- 分割 (Stage B) ----------
    smooth_sigma_mm: float = 9.0      # 分割前高斯平滑
    seed_h_mm: float = 8.0            # 细种子 h-maxima 深度(略高于平滑后噪声)
    hmax_delta_mm: float = 18.0       # 鞍点合并深度: 两区域峰-鞍差小于此则视为同一块矿石
    fg_open_r_px: int = 2             # 前景开运算半径, 切断窄颈连接
    debris_diameter_mm: float = 80.0  # 小于该等效直径的分割区域视为碎料(保留标签用于邻接判断)

    # ---------- 边界分类 (Stage C) ----------
    drop_delta_mm: float = 15.0       # h_in - h_out > delta 判为自由边界(向下跌落)
    boundary_window_px: int = 13      # 边界点邻域采样窗口(奇数)
    waist_max_ratio: float = 0.55     # 轮廓点高度低于矿石(峰-基准)的该比例 => 腰线以下,
                                      # 属并排接触而非压叠, 轮廓可信(等价自由边界)

    # ---------- 完整性判定 (Stage D) ----------
    free_ratio_min: float = 0.95      # 自由边界占比下限
    invalid_ratio_max: float = 0.05   # 无效边界占比上限
    border_margin_px: int = 15        # 距图像边缘裕量, 触碰则拒绝
    valid_coverage_min: float = 0.90  # 掩膜内有效像素占比下限
    solidity_min: float = 0.88        # 面积/凸包面积下限
    ellipse_iou_min: float = 0.90     # 椭圆拟合IoU下限: 拦截欠分割的"葫芦形"双胞胎
                                      # (真实棱角矿石若误拒率高可下调至0.85)
    feret_ellipse_tol: float = 1.08   # 掩膜Feret不得超过拟合椭圆长轴的该倍数
                                      # (交叉校验: 拦截附着邻块导致的尺寸虚大)
    peak_interior_ratio: float = 0.20 # 最高点距边界距离 >= 该比例*等效半径
    min_diameter_mm: float = 90.0     # 尺寸下限(带10mm缓冲, 100mm以下不统计)

    # ---------- 统计 (Stage G) ----------
    bin_edges_mm: List[float] = field(
        default_factory=lambda: [100.0, 200.0, 300.0, 400.0, 500.0]
    )                                  # 粒径分箱边界, 最后一箱为 [500, +inf)
    sizing_metric: str = "min_rect_w"  # 分箱依据: "min_rect_w"(次轴,可与筛分对标) 或 "max_feret"

    # ---------- 传感器 (用于真机数据转换与合成仿真) ----------
    sensor_noise_sigma_mm: float = 6.0     # 深度随机噪声(合成仿真/阈值推导参考)
    roi_border_invalid_px: int = 2         # 图像最外圈无效像素宽度

    @property
    def bins_with_inf(self) -> List[Tuple[float, float]]:
        """返回 [(low, high), ...], 最后一箱 high 为 inf。"""
        edges = list(self.bin_edges_mm)
        out = []
        for i in range(len(edges) - 1):
            out.append((edges[i], edges[i + 1]))
        out.append((edges[-1], float("inf")))
        return out
