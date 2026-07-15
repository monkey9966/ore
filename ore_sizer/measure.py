"""Stage E: 单颗矿石尺寸测量。掩膜坐标 -> 毫米。"""

from typing import Tuple

import cv2
import numpy as np

from .config import PipelineConfig
from .types import RockInfo


def max_feret(points: np.ndarray) -> Tuple[float, Tuple[int, int]]:
    """最大 Feret 直径: 凸包上点对的最大距离。points: (N,2) 像素 (x,y)。"""
    hull = cv2.convexHull(points.astype(np.int32)).reshape(-1, 2)
    if len(hull) == 1:
        return 0.0, (0, 0)
    if len(hull) == 2:
        d = float(np.linalg.norm(hull[0] - hull[1]))
        return d, (0, 1)
    # 凸包点数很少(几十), 直接 O(n^2)
    diff = hull[:, None, :] - hull[None, :, :]
    d2 = (diff ** 2).sum(-1)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    return float(np.sqrt(d2[i, j])), (int(i), int(j))


def measure_rock(info: RockInfo, mask: np.ndarray, height: np.ndarray,
                 prominence: np.ndarray, cfg: PipelineConfig,
                 mm_per_px: float) -> RockInfo:
    """填充 RockInfo 的测量字段, 并做尺寸下限检查。"""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1)  # (x, y)

    # 最大 Feret 直径(最大边长)
    hull = cv2.convexHull(pts.astype(np.int32)).reshape(-1, 2)
    feret_px, (i, j) = max_feret(pts)
    info.max_feret_mm = feret_px * mm_per_px
    if len(hull) > max(i, j):
        p1, p2 = hull[i], hull[j]
        info.max_feret_endpoints_px = ((float(p1[0]), float(p1[1])),
                                       (float(p2[0]), float(p2[1])))

    # 最小外接矩形
    rect = cv2.minAreaRect(pts.astype(np.float32))
    (rw, rh) = rect[1]
    l_mm, w_mm = max(rw, rh) * mm_per_px, min(rw, rh) * mm_per_px
    info.min_rect_l_mm, info.min_rect_w_mm = l_mm, w_mm

    # 等效直径 / 面积
    area_mm2 = len(ys) * mm_per_px ** 2
    info.area_mm2 = area_mm2
    info.equiv_diameter_mm = 2.0 * np.sqrt(area_mm2 / np.pi)

    # 厚度与体积(基于相对局部基准的凸起)
    prom_rock = prominence[mask]
    prom_rock = prom_rock[np.isfinite(prom_rock)]
    if len(prom_rock):
        info.thickness_mm = float(np.percentile(prom_rock, 99))
        info.volume_mm3 = float(np.clip(prom_rock, 0, None).sum()) * mm_per_px ** 2

    # 分箱依据
    info.sizing_mm = info.min_rect_w_mm if cfg.sizing_metric == "min_rect_w" \
        else info.max_feret_mm

    # 交叉校验: 掩膜 Feret 明显超过拟合椭圆长轴 => 掩膜带着附着物, 尺寸虚大
    ell_major_px, _ = info.meta_ellipse_axes_px
    if ell_major_px > 0 and feret_px > cfg.feret_ellipse_tol * ell_major_px:
        info.reject_reasons.append("FERET_ELLIPSE_MISMATCH")
        info.accepted = False

    # 厚度交叉校验: 平躺的单块矿石厚度不会超过次轴; 明显"过厚"说明
    # 掩膜其实是上层矿石骑在下层矿石上形成的合并体(轮廓看不出来)
    if info.thickness_mm > cfg.thickness_width_max * max(info.min_rect_w_mm, 1e-3):
        info.reject_reasons.append("TOO_THICK")
        info.accepted = False

    # 尺寸下限
    if info.sizing_mm < cfg.min_diameter_mm:
        info.reject_reasons.append("TOO_SMALL")
        info.accepted = False

    # 粒级归属
    info.size_bin = bin_name(info.sizing_mm, cfg)
    return info


def bin_name(size_mm: float, cfg: PipelineConfig) -> str:
    for lo, hi in cfg.bins_with_inf:
        if lo <= size_mm < hi:
            return f"{lo:.0f}-{hi:.0f}" if np.isfinite(hi) else f">{lo:.0f}"
    return f"<{cfg.bin_edges_mm[0]:.0f}"
