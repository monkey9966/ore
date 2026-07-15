"""合成 ToF 场景生成器 (按 DS63 规格仿真)。

生成"传送带上堆叠矿石"的正射高度图 + 每颗矿石的真值(尺寸/可见性),
用于在没有相机与现场的条件下定量验证算法。

仿真内容:
- 碎料床(不统计的 <100mm 细料垫底)
- 随机椭球矿石, 支持堆叠(后放的压在先放的上面)
- DS63 传感器模型: 深度随机噪声、边缘飞点、低反射率无效空洞
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from .config import PipelineConfig
from .types import HeightFrame


@dataclass
class GroundTruthRock:
    rock_id: int
    center_px: Tuple[float, float]        # (row, col)
    true_max_edge_mm: float               # 2*max(a,b): 最大边长真值
    true_width_mm: float                  # 2*min(a,b): 次轴真值
    true_thickness_mm: float
    visible_fraction: float = 1.0         # 顶面未被遮挡的比例
    visible_feret_mm: float = 0.0         # 可见表面的最大Feret直径
    fully_visible: bool = True            # 真值"可测": 可见表面保留了矿石的平面尺寸
    inside_roi: bool = True


@dataclass
class SyntheticScene:
    frame: HeightFrame
    gt_rocks: List[GroundTruthRock] = field(default_factory=list)
    clean_height: Optional[np.ndarray] = None   # 无噪声真值高度图(调试用)


def _smooth_noise(shape, rng, sigma_px, amplitude):
    """生成空间相关的平滑随机场。"""
    n = rng.standard_normal(shape)
    n = ndimage.gaussian_filter(n, sigma_px)
    s = n.std()
    if s > 1e-9:
        n = n / s
    return n * amplitude


def generate_scene(
    cfg: PipelineConfig,
    shape: Tuple[int, int] = (480, 640),
    n_rocks: int = 14,
    stacking: float = 0.35,
    size_range_mm: Tuple[float, float] = (100.0, 550.0),
    n_debris: int = 25,
    seed: int = 0,
    sensor_noise: bool = True,
) -> SyntheticScene:
    """生成一帧合成场景。

    stacking: 0~1, 控制堆叠程度(新矿石落在已有矿石附近的概率)。
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape
    mpp = cfg.mm_per_px

    # ---- 碎料床: ~25mm 起伏的细料垫底 ----
    H = np.maximum(_smooth_noise(shape, rng, 8, 8.0) + 15.0, 0.0)

    # ---- 逐颗放置矿石 ----
    rock_surfaces = []   # (gt, own_surf(局部), slice)
    placed_centers = []

    def sample_center():
        if placed_centers and rng.random() < stacking:
            base = placed_centers[rng.integers(len(placed_centers))]
            r = rng.uniform(20, 90)   # px
            ang = rng.uniform(0, 2 * np.pi)
            cy = base[0] + r * np.sin(ang)
            cx = base[1] + r * np.cos(ang)
        else:
            cy = rng.uniform(0.08 * ny, 0.92 * ny)
            cx = rng.uniform(0.08 * nx, 0.92 * nx)
        return float(np.clip(cy, 0, ny - 1)), float(np.clip(cx, 0, nx - 1))

    def place_ellipsoid(rock_id, size_mm, record_gt=True):
        # 半轴: a 为最长半轴(平面), b 次轴, c 竖直半轴
        a = size_mm / 2.0
        b = a * rng.uniform(0.6, 0.95)
        c = b * rng.uniform(0.55, 0.9)
        theta = rng.uniform(0, np.pi)
        cy, cx = sample_center()

        a_px, b_px = a / mpp, b / mpp
        r_px = int(np.ceil(max(a_px, b_px))) + 2
        y0, y1 = int(cy) - r_px, int(cy) + r_px + 1
        x0, x1 = int(cx) - r_px, int(cx) + r_px + 1
        y0c, x0c = max(y0, 0), max(x0, 0)
        y1c, x1c = min(y1, ny), min(x1, nx)
        if y1c <= y0c or x1c <= x0c:
            return None

        yy, xx = np.mgrid[y0c:y1c, x0c:x1c]
        dy = (yy - cy) * mpp
        dx = (xx - cx) * mpp
        ct, st = np.cos(theta), np.sin(theta)
        u = dx * ct + dy * st
        v = -dx * st + dy * ct
        q = 1.0 - (u / a) ** 2 - (v / b) ** 2
        footprint = q > 0

        if footprint.sum() < 4:
            return None

        # 底座高度: 落点区域已有高度的高分位(堆叠), 略微嵌入
        region = H[y0c:y1c, x0c:x1c]
        z_base = float(np.quantile(region[footprint], 0.75))
        z_center = z_base + 0.8 * c    # 球心在底座上方: 矿石"坐"在堆上, 轻微嵌入

        surf = np.full(region.shape, -np.inf, dtype=np.float64)
        surf[footprint] = z_center + c * np.sqrt(q[footprint])
        # 表面粗糙度
        rough = _smooth_noise(region.shape, rng, 3, 3.0)
        surf[footprint] += rough[footprint]
        # 只保留高出现有表面的部分才形成新表面
        H[y0c:y1c, x0c:x1c] = np.maximum(region, surf)

        if record_gt:
            gt = GroundTruthRock(
                rock_id=rock_id,
                center_px=(cy, cx),
                true_max_edge_mm=2 * a,
                true_width_mm=2 * b,
                true_thickness_mm=z_center + c - z_base,
            )
            rock_surfaces.append((gt, surf.copy(), (slice(y0c, y1c), slice(x0c, x1c))))
        placed_centers.append((cy, cx))
        return True

    rid = 0
    attempts = 0
    while rid < n_rocks and attempts < n_rocks * 10:
        attempts += 1
        size = rng.uniform(*size_range_mm)
        if place_ellipsoid(rid, size, record_gt=True):
            rid += 1

    # ---- 碎料: <100mm 的小石头散布(不记真值, 不统计) ----
    for _ in range(n_debris):
        size = rng.uniform(30.0, 85.0)
        place_ellipsoid(-1, size, record_gt=False)

    clean = H.copy()

    # ---- 计算每颗矿石的真值可见性/可测性 ----
    # 真值"可测"(fully_visible)的定义与测量目标一致:
    #   可见表面的最大 Feret 直径保留了矿石真实平面尺寸(误差<3%),
    #   且顶面大部分未被遮挡、完整在 ROI 内。
    # 矿石裙边埋入碎料床不影响俯视测量, 不应算作"被遮挡"。
    import cv2
    gt_rocks: List[GroundTruthRock] = []
    margin = 12  # 与 border_margin_px 对应的真值判断裕量
    for gt, surf, sl in rock_surfaces:
        own_mask = surf > 0
        n_own = int(own_mask.sum())
        if n_own == 0:
            continue
        # 顶面存活: 最终高度 == 自己的表面(未被后放矿石覆盖)
        alive = np.abs(clean[sl] - surf) < 1.0
        vis_mask = alive & own_mask
        vis = float(vis_mask.sum()) / n_own
        gt.visible_fraction = vis

        vis_width_mm = 0.0
        if vis_mask.sum() >= 5:
            ys, xs = np.nonzero(vis_mask)
            pts = np.stack([xs, ys], axis=1).astype(np.int32)
            hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64)
            diff = hull[:, None, :] - hull[None, :, :]
            feret_px = float(np.sqrt(((diff ** 2).sum(-1)).max()))
            gt.visible_feret_mm = feret_px * mpp
            (_, (rw, rh), _) = cv2.minAreaRect(pts.astype(np.float32))
            vis_width_mm = min(rw, rh) * mpp
        # 可见表面保留了平面尺寸: 最大边长与次轴都没有被遮挡截短
        # (容差含 ~2px 的栅格离散化, 对小矿石按绝对值放宽)
        tol_f = max(0.03 * gt.true_max_edge_mm, 2.5 * mpp)
        tol_w = max(0.03 * gt.true_width_mm, 2.5 * mpp)
        feret_ok = (gt.visible_feret_mm >= gt.true_max_edge_mm - tol_f
                    and vis_width_mm >= gt.true_width_mm - tol_w)

        ys, xs = np.nonzero(own_mask)
        gy0, gy1 = sl[0].start + ys.min(), sl[0].start + ys.max()
        gx0, gx1 = sl[1].start + xs.min(), sl[1].start + xs.max()
        gt.inside_roi = (
            gy0 >= margin and gx0 >= margin
            and gy1 < H.shape[0] - margin and gx1 < H.shape[1] - margin
        )
        gt.fully_visible = feret_ok and vis > 0.75 and gt.inside_roi
        gt_rocks.append(gt)

    # ---- 传感器模型 ----
    valid = np.ones(shape, dtype=bool)
    if sensor_noise:
        # 1) 深度随机噪声(空间弱相关)
        H = H + _smooth_noise(shape, rng, 1.2, cfg.sensor_noise_sigma_mm)
        # 2) 边缘飞点: 高度梯度大的地方, 部分像素无效或深度跳变
        gy, gx = np.gradient(clean)
        grad = np.hypot(gy, gx)
        edge = grad > 25.0
        fly = edge & (rng.random(shape) < 0.35)
        kill = fly & (rng.random(shape) < 0.6)
        jump = fly & ~kill
        valid[kill] = False
        H[jump] += rng.uniform(-80, 80, size=int(jump.sum()))
        # 3) 低反射率空洞: 随机小斑块无效
        holes = _smooth_noise(shape, rng, 5, 1.0) > 2.4
        valid[holes] = False
        # 4) 视场最外圈无效
        b = cfg.roi_border_invalid_px
        valid[:b, :] = valid[-b:, :] = False
        valid[:, :b] = valid[:, -b:] = False

    Hf = H.astype(np.float32)
    Hf[~valid] = np.nan

    frame = HeightFrame(
        height=Hf,
        valid=valid,
        mm_per_px=mpp,
        meta={"synthetic": True, "seed": seed, "n_rocks": n_rocks, "stacking": stacking},
    )
    return SyntheticScene(frame=frame, gt_rocks=gt_rocks, clean_height=clean)
