"""Stage C+D: 边界分类与完整性判定(遮挡剔除核心)。

对每个候选矿石, 沿轮廓逐点比较"自身高度 h_in"与"紧邻外侧高度 h_out":
- h_in - h_out > drop_delta  -> FREE     (真实物理边缘, 向下跌落)
- 否则                        -> OCCLUDED (被相邻更高/等高物压住)
- 外侧数据缺失                -> INVALID

分水岭脊线的归属由该规则自动解算: 脊线对"高的一方"是 FREE(它压着别人),
对"低的一方"是 OCCLUDED(它被压)。
"""

from typing import List, Tuple

import cv2
import numpy as np
from scipy import ndimage


def _split_depth(mask: np.ndarray, hs: np.ndarray, min_area_px: float,
                 depths=(6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0)) -> float:
    """多峰欠分割深度: 最大的谷深 d, 使掩膜内存在两个被深度>=d 的谷
    分隔、且第二大子区域达到矿石尺寸的表面峰。单峰返回 0。

    两块矿石被鞍点合并成一个掩膜时, 外形往往仍是光滑椭圆
    (solidity/椭圆IoU 都拦不住), 但表面必然有两个独立峰 —— 这是欠分割
    合并误收的最后一道闸。单块矿石表面粗糙引起的次峰谷浅或面积小,
    得到的 split_depth 低, 不会触发拒绝阈值。
    """
    from skimage import morphology, segmentation as sks

    ys, xs = np.nonzero(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    m = mask[y0:y1, x0:x1]
    h_win = hs[y0:y1, x0:x1]
    h = np.where(m, h_win, h_win.min() - 1e3)

    best = 0.0
    for d in depths:
        peaks = morphology.h_maxima(h, d) & m
        markers, n = ndimage.label(peaks)
        if n < 2:
            break  # 峰数随 d 单调不增, 更深的谷不可能再分裂
        sub = sks.watershed(-h, markers=markers, mask=m)
        areas = np.sort(np.bincount(sub.ravel())[1:])[::-1]
        if len(areas) >= 2 and areas[1] >= min_area_px:
            best = d
    return best


def _ellipse_fit(mask: np.ndarray) -> Tuple[float, float, float]:
    """返回 (IoU, 椭圆长轴px, 椭圆短轴px)。单块凸矿石 IoU 接近 1。"""
    m8 = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0, 0.0, 0.0
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return 0.0, 0.0, 0.0
    ellipse = cv2.fitEllipse(cnt)
    (_, (d1, d2), _) = ellipse
    fit = np.zeros_like(m8)
    cv2.ellipse(fit, ellipse, 1, thickness=-1)
    inter = int(np.logical_and(m8, fit).sum())
    union = int(np.logical_or(m8, fit).sum())
    iou = inter / union if union else 0.0
    return iou, max(d1, d2), min(d1, d2)

from .config import PipelineConfig
from .types import (BOUNDARY_FREE, BOUNDARY_INVALID, BOUNDARY_OCCLUDED,
                    HeightFrame, RockInfo)


def classify_boundary(mask: np.ndarray, height: np.ndarray, valid: np.ndarray,
                      hs: np.ndarray, base: np.ndarray, cfg: PipelineConfig
                      ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """对单个矿石掩膜做边界分类。

    每个轮廓点按两条规则之一判为可信(FREE):
    1. 跌落规则: h_in - h_out > drop_delta —— 边缘悬空向下跌落, 无遮挡;
    2. 腰线规则: 轮廓点高度低于矿石相对高度(峰-基准)的 waist_max_ratio ——
       该处即使与邻居相接, 也是"并排接触"而非"被压住", 轮廓位置仍是真实外形
       (被压的矿石其可见轮廓截止在高处, 会违反此规则)。

    采样几何(避免把矿石自身的"裙边"当外侧):
    - h_in : 窗口内掩膜内部像素的中值(平滑高度)
    - h_out: 窗口内、掩膜外且在保护带(guard)之外像素的中值

    返回 (boundary_pixels_yx, class_per_pixel, free_ratio, occ_ratio, inv_ratio)。
    """
    er = ndimage.binary_erosion(mask, border_value=0)
    boundary = mask & ~er
    bys, bxs = np.nonzero(boundary)
    n = len(bys)
    if n == 0:
        return np.empty((0, 2), int), np.empty(0, int), 0.0, 0.0, 1.0

    guard_px = 2
    w = max(cfg.boundary_window_px // 2, guard_px + 3)
    guard_zone = ndimage.binary_dilation(mask, iterations=guard_px)

    # 矿石峰值与基准(用于腰线规则)
    peak_h = float(np.max(hs[mask]))
    base_h = float(np.median(base[mask]))
    rel_height = max(peak_h - base_h, 1e-3)
    waist_h = base_h + cfg.waist_max_ratio * rel_height

    ny, nx = mask.shape
    cls = np.zeros(n, dtype=np.int32)
    delta = cfg.drop_delta_mm

    for i in range(n):
        y, x = bys[i], bxs[i]
        y0, y1 = max(y - w, 0), min(y + w + 1, ny)
        x0, x1 = max(x - w, 0), min(x + w + 1, nx)
        m_win = mask[y0:y1, x0:x1]
        g_win = guard_zone[y0:y1, x0:x1]
        v_win = valid[y0:y1, x0:x1]
        h_win = height[y0:y1, x0:x1]
        hs_win = hs[y0:y1, x0:x1]

        outer = ~g_win
        n_out = int(outer.sum())
        if n_out == 0:
            cls[i] = BOUNDARY_OCCLUDED
            continue
        out_valid = outer & v_win
        if int(out_valid.sum()) < 0.4 * n_out:
            cls[i] = BOUNDARY_INVALID
            continue

        h_out = float(np.median(h_win[out_valid]))
        h_in = float(np.median(hs_win[m_win]))
        # 规则1: 向下跌落
        if h_in - h_out > delta:
            cls[i] = BOUNDARY_FREE
        # 规则2: 并排接触 —— 自身轮廓与外侧都在腰线以下(两侧裙边相碰),
        # 轮廓位置仍是真实外形; 若外侧高于腰线(被别的矿石压住)则不适用
        elif hs[y, x] <= waist_h and h_out <= waist_h:
            cls[i] = BOUNDARY_FREE
        else:
            cls[i] = BOUNDARY_OCCLUDED

    free_ratio = float((cls == BOUNDARY_FREE).sum()) / n
    occ_ratio = float((cls == BOUNDARY_OCCLUDED).sum()) / n
    inv_ratio = float((cls == BOUNDARY_INVALID).sum()) / n
    return np.stack([bys, bxs], axis=1), cls, free_ratio, occ_ratio, inv_ratio


def evaluate_rock(rid: int, mask: np.ndarray, frame: HeightFrame,
                  hs: np.ndarray, base: np.ndarray,
                  cfg: PipelineConfig,
                  boundary_class_map: np.ndarray) -> RockInfo:
    """对单个候选矿石做完整性判定, 返回 RockInfo(未含尺寸测量)。"""
    info = RockInfo(label=rid, accepted=True)
    height, valid = frame.height, frame.valid

    bpix, cls, fr, ocr, ivr = classify_boundary(mask, height, valid, hs, base, cfg)
    info.free_ratio, info.occluded_ratio, info.invalid_ratio = fr, ocr, ivr
    if len(bpix):
        boundary_class_map[bpix[:, 0], bpix[:, 1]] = cls

    ys, xs = np.nonzero(mask)
    info.bbox_px = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
    info.centroid_px = (float(ys.mean()), float(xs.mean()))
    area_px = len(ys)

    # 1. 自由边界占比
    if fr < cfg.free_ratio_min:
        info.reject_reasons.append("OCCLUDED")
    # 2. 无效边界占比
    if ivr > cfg.invalid_ratio_max:
        info.reject_reasons.append("INVALID_BOUNDARY")
    # 3. 触碰视场边缘
    m = cfg.border_margin_px
    ny, nx = mask.shape
    if (ys.min() < m or xs.min() < m or ys.max() >= ny - m or xs.max() >= nx - m):
        info.reject_reasons.append("TOUCH_BORDER")
    # 4. 掩膜内有效像素率
    cov = float(valid[mask].sum()) / area_px
    info.valid_coverage = cov
    if cov < cfg.valid_coverage_min:
        info.reject_reasons.append("LOW_VALID_COVERAGE")
    # 5. Solidity
    from skimage.morphology import convex_hull_image
    try:
        hull = convex_hull_image(mask)
        solidity = area_px / float(hull.sum())
    except Exception:
        solidity = 0.0
    info.solidity = solidity
    if solidity < cfg.solidity_min:
        info.reject_reasons.append("LOW_SOLIDITY")
    # 5b. 椭圆拟合 IoU: 拦截 solidity 拦不住的"葫芦形"欠分割双胞胎
    iou, ell_major_px, ell_minor_px = _ellipse_fit(mask)
    info.ellipse_iou = iou
    info.meta_ellipse_axes_px = (ell_major_px, ell_minor_px)
    if iou < cfg.ellipse_iou_min:
        info.reject_reasons.append("BAD_ELLIPSE_FIT")
    # 5d. 多峰欠分割: 表面存在两个被深谷分隔的矿石级峰 => 合并体, 拒绝
    min_rock_px = (cfg.debris_diameter_mm / frame.mm_per_px / 2) ** 2 * np.pi
    sd = _split_depth(mask, hs, min_rock_px)
    info.split_depth_mm = sd
    if sd >= cfg.split_h_mm:
        info.reject_reasons.append("MULTI_PEAK")
    # 6. 最高点在掩膜内部
    hs_rock = np.where(mask, hs, -np.inf)
    py, px = np.unravel_index(np.argmax(hs_rock), hs_rock.shape)
    dist_in = ndimage.distance_transform_edt(mask)
    equiv_r_px = np.sqrt(area_px / np.pi)
    if dist_in[py, px] < cfg.peak_interior_ratio * equiv_r_px:
        info.reject_reasons.append("PEAK_ON_EDGE")

    info.accepted = len(info.reject_reasons) == 0
    return info
