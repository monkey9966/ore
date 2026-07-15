"""Stage B: 前景提取 + 标记控制分水岭分割 + 鞍点层次合并。

策略: 先用"细种子"(h-maxima, 深度略高于噪声)过分割, 再按鞍点深度合并 —
两个相邻区域若 min(峰A, 峰B) - 鞍点 < hmax_delta, 说明它们之间没有足够深的
沟壑, 属于同一块矿石表面的起伏, 合并之。碎料/床面凸起与矿石之间的沟壑深,
不会被合并, 从而保有独立标签(后续按尺寸归为碎料)。
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage
from skimage import morphology, segmentation

from .config import PipelineConfig
from .types import HeightFrame


def _remove_small_regions(mask: np.ndarray, min_px: int) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = np.nonzero(sizes >= min_px)[0] + 1
    return np.isin(lab, keep)


def _saddle_merge(labels: np.ndarray, hs: np.ndarray, delta: float) -> np.ndarray:
    """按鞍点深度合并相邻区域(并查集)。"""
    n = int(labels.max())
    if n <= 1:
        return labels

    peak = ndimage.maximum(hs, labels=labels, index=np.arange(1, n + 1))

    # 相邻区域对及其鞍点高度: 对4邻域两方向找跨界像素对
    saddle: Dict[Tuple[int, int], float] = {}
    for axis in (0, 1):
        a = labels
        b = np.roll(labels, -1, axis=axis)
        ha = hs
        hb = np.roll(hs, -1, axis=axis)
        m = (a != b) & (a > 0) & (b > 0)
        if axis == 0:
            m[-1, :] = False
        else:
            m[:, -1] = False
        la, lb = a[m], b[m]
        hmin = np.minimum(ha[m], hb[m])
        for i in range(len(la)):
            key = (min(la[i], lb[i]), max(la[i], lb[i]))
            v = saddle.get(key)
            if v is None or hmin[i] > v:
                saddle[key] = float(hmin[i])

    # 并查集
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # 按鞍点从高到低处理(先合并最"浅"的分界)
    for (i, j), s in sorted(saddle.items(), key=lambda kv: -kv[1]):
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        # 合并后簇的峰取两者最大, 判据用较低的簇峰
        pi, pj = peak[ri - 1], peak[rj - 1]
        if min(pi, pj) - s < delta:
            parent[rj] = ri
            peak[ri - 1] = max(pi, pj)

    mapping = np.zeros(n + 1, dtype=np.int32)
    roots = {}
    nxt = 1
    for lbl in range(1, n + 1):
        r = find(lbl)
        if r not in roots:
            roots[r] = nxt
            nxt += 1
        mapping[lbl] = roots[r]
    return mapping[labels]


def segment(frame: HeightFrame, prominence: np.ndarray, cfg: PipelineConfig
            ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """返回 (labels, smoothed_height, debris_label_ids)。"""
    mpp = frame.mm_per_px

    # ---- 前景: 相对局部基准面凸起超过阈值 ----
    prom = np.where(np.isnan(prominence), -1e3, prominence)
    fg = prom > cfg.fg_prominence_mm
    # 开运算切断窄颈, 去碎点, 填小孔
    if cfg.fg_open_r_px > 0:
        fg = ndimage.binary_opening(fg, morphology.disk(cfg.fg_open_r_px))
    min_obj_px = max(int((cfg.debris_diameter_mm / mpp / 2) ** 2 * np.pi * 0.15), 8)
    fg = _remove_small_regions(fg, min_obj_px)
    fg = ~_remove_small_regions(~fg, min_obj_px)

    # ---- 分割前去尖刺 + 平滑 (NaN 用邻域填充参与平滑) ----
    h = frame.height.copy()
    if np.isnan(h).any():
        idx = ndimage.distance_transform_edt(np.isnan(h), return_distances=False,
                                             return_indices=True)
        h = h[tuple(idx)]
    # 中值滤波去除残留飞点尖刺(轮廓边缘成簇的深度跳变), 防止产生假峰。
    h = ndimage.median_filter(h, size=5)
    sigma_px = cfg.smooth_sigma_mm / mpp
    hs = ndimage.gaussian_filter(h, sigma_px)

    if not fg.any():
        return np.zeros(frame.shape, dtype=np.int32), hs, []

    # ---- 细种子: 深度略高于噪声的所有峰 ----
    hs_fg = np.where(fg, hs, hs.min() - 1.0)
    peaks = morphology.h_maxima(hs_fg, cfg.seed_h_mm)
    peaks &= fg
    # 种子不得贴近前景边界(轮廓处飞点残留的假峰)
    d_in = ndimage.distance_transform_edt(fg)
    peaks &= d_in >= 3
    markers, n_markers = ndimage.label(peaks)
    if n_markers == 0:
        return np.zeros(frame.shape, dtype=np.int32), hs, []

    # ---- 分水岭(过分割) + 鞍点合并 ----
    labels = segmentation.watershed(-hs, markers=markers, mask=fg).astype(np.int32)
    labels = _saddle_merge(labels, hs, cfg.hmax_delta_mm)

    # ---- 碎料区域标记(保留标签, 供邻接/边界判断) ----
    debris_ids: List[int] = []
    min_rock_px = (cfg.debris_diameter_mm / mpp / 2) ** 2 * np.pi
    counts = np.bincount(labels.ravel())
    for rid in range(1, labels.max() + 1):
        if rid < len(counts) and 0 < counts[rid] < min_rock_px:
            debris_ids.append(rid)
    return labels, hs, debris_ids
