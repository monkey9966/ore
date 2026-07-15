"""Stage A: 预处理。

- 真机路径: 深度图(透视) -> 点云 -> RANSAC 基准面 -> 正射高度图
- 通用路径: 高度图小洞填补 / 局部基准面 / 前景凸起图
"""

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from .config import PipelineConfig
from .types import HeightFrame


# ---------------------------------------------------------------------------
# 高度图整备
# ---------------------------------------------------------------------------

def fill_small_holes(frame: HeightFrame, cfg: PipelineConfig) -> HeightFrame:
    """小无效洞用最近有效值填补; 大洞保留 NaN(供完整性判定使用)。"""
    invalid = ~frame.valid
    if not invalid.any():
        return frame
    max_hole_px = np.pi * (cfg.max_hole_fill_mm / frame.mm_per_px / 2) ** 2
    lab, n = ndimage.label(invalid)
    if n == 0:
        return frame
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    small_ids = np.nonzero(sizes <= max_hole_px)[0] + 1
    fill_mask = np.isin(lab, small_ids)
    if fill_mask.any():
        # 最近邻填充
        idx = ndimage.distance_transform_edt(
            ~frame.valid, return_distances=False, return_indices=True
        )
        filled = frame.height[tuple(idx)]
        h = frame.height.copy()
        h[fill_mask] = filled[fill_mask]
        v = frame.valid.copy()
        v[fill_mask] = True
        return HeightFrame(h, v, frame.mm_per_px, frame.trigger_id, frame.meta)
    return frame


def local_base(height: np.ndarray, valid: np.ndarray, cfg: PipelineConfig,
               mm_per_px: float) -> np.ndarray:
    """局部基准面: 分块中值下采样 + 灰度开运算(rolling-ball 近似)。

    半径大于最大矿石半径, 使矿石被"削平"、只留下碎料床/皮带的大尺度形状。
    分块中值(而非插值)下采样可抑制飞点残留的离群深度值, 避免基准面被拖垮。
    """
    ds = 4  # 下采样倍率
    ny, nx = height.shape
    py = (ds - ny % ds) % ds
    px = (ds - nx % ds) % ds
    h = np.pad(height.astype(np.float64), ((0, py), (0, px)),
               constant_values=np.nan)
    h[np.pad(~valid, ((0, py), (0, px)), constant_values=True)] = np.nan
    blocks = h.reshape(h.shape[0] // ds, ds, h.shape[1] // ds, ds)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        small = np.nanmedian(blocks.transpose(0, 2, 1, 3).reshape(
            blocks.shape[0], blocks.shape[2], ds * ds), axis=2)
    # 整块无效处用全图中值填补
    fallback = np.nanmedian(small) if np.isfinite(small).any() else 0.0
    small = np.where(np.isnan(small), fallback, small)

    r_px = max(int(cfg.base_disk_r_mm / mm_per_px / ds), 2)
    yy, xx = np.mgrid[-r_px:r_px + 1, -r_px:r_px + 1]
    disk = (yy ** 2 + xx ** 2) <= r_px ** 2
    opened = ndimage.grey_opening(small, footprint=disk)
    opened = ndimage.gaussian_filter(opened, 2)
    base = ndimage.zoom(opened, ds, order=1)
    base = _match_shape(base, height.shape)
    return base


def _match_shape(a: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=a.dtype)
    ny, nx = min(a.shape[0], shape[0]), min(a.shape[1], shape[1])
    out[:ny, :nx] = a[:ny, :nx]
    if a.shape[0] < shape[0]:
        out[a.shape[0]:, :] = out[a.shape[0] - 1:a.shape[0], :]
    if a.shape[1] < shape[1]:
        out[:, a.shape[1]:] = out[:, a.shape[1] - 1:a.shape[1]]
    return out


def prominence_map(frame: HeightFrame, cfg: PipelineConfig) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (prominence, base): 每点相对局部基准面的凸起量。"""
    base = local_base(frame.height, frame.valid, cfg, frame.mm_per_px)
    prom = frame.height - base
    prom[~frame.valid] = np.nan
    return prom, base


# ---------------------------------------------------------------------------
# 真机路径: 透视深度图 -> 正射高度图
# ---------------------------------------------------------------------------

def fit_plane_ransac(points: np.ndarray, n_iter: int = 200,
                     inlier_thresh_mm: float = 10.0,
                     rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, float]:
    """RANSAC 拟合平面 n·p = d。points: (N,3) mm。返回 (n_unit, d)。"""
    rng = rng or np.random.default_rng(0)
    best_inliers = -1
    best = (np.array([0.0, 0.0, 1.0]), 0.0)
    n_pts = points.shape[0]
    if n_pts < 100:
        raise ValueError("点数不足, 无法拟合基准面")
    for _ in range(n_iter):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = points[idx]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n = n / norm
        d = float(n @ p0)
        dist = np.abs(points @ n - d)
        inl = int((dist < inlier_thresh_mm).sum())
        if inl > best_inliers:
            best_inliers = inl
            best = (n, d)
    # 用内点做最小二乘精修
    n, d = best
    dist = np.abs(points @ n - d)
    inliers = points[dist < inlier_thresh_mm]
    centroid = inliers.mean(axis=0)
    u, s, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    n = vt[2]
    d = float(n @ centroid)
    return n, d


def depth_to_heightframe(depth_mm: np.ndarray,
                         fx: float, fy: float, cx: float, cy: float,
                         cfg: PipelineConfig,
                         plane: Optional[Tuple[np.ndarray, float]] = None,
                         max_range_mm: float = 5000.0) -> HeightFrame:
    """真机深度图(透视投影, uint16 mm) -> 正射高度图 HeightFrame。

    办公室静态验证: 相机俯视地面/桌面上的矿石, 基准面直接从场景 RANSAC 拟合
    (地面是场景中最大的平面)。

    plane: 可传入预先标定的 (n, d); None 则本帧自动拟合。
    """
    h, w = depth_mm.shape
    z = depth_mm.astype(np.float64)
    valid_px = (z > 0) & (z < max_range_mm)

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    pts = np.stack([x[valid_px], y[valid_px], z[valid_px]], axis=1)

    if plane is None:
        # 地面是最远的主平面; 对远端 40% 的点做 RANSAC 更稳
        z_v = pts[:, 2]
        far = pts[z_v > np.quantile(z_v, 0.6)]
        sample = far[np.random.default_rng(0).choice(
            far.shape[0], min(8000, far.shape[0]), replace=False)]
        n, d = fit_plane_ransac(sample)
    else:
        n, d = plane
    # 法向指向相机(高度为正)
    if n[2] > 0:
        n, d = -n, -d

    # 高度 = 点到平面的有符号距离
    heights = pts @ n - d

    # 平面内正射坐标系 (e1, e2)
    e1 = np.cross(n, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([1.0, 0.0, 0.0])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    uu = pts @ e1
    vv = pts @ e2

    mpp = cfg.mm_per_px
    u0, v0 = uu.min(), vv.min()
    nx = int((uu.max() - u0) / mpp) + 1
    ny = int((vv.max() - v0) / mpp) + 1
    if nx * ny > 4_000_000:
        raise ValueError("正射网格过大, 请检查 mm_per_px 或视场")

    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    ui = ((uu - u0) / mpp).astype(np.int32)
    vi = ((vv - v0) / mpp).astype(np.int32)
    # z-buffer: 同一格取最高点(俯视可见表面)
    order = np.argsort(heights)   # 从低到高写入, 高的覆盖低的
    grid[vi[order], ui[order]] = heights[order].astype(np.float32)

    valid = np.isfinite(grid)
    return HeightFrame(grid, valid, mpp, meta={"plane_n": n.tolist(), "plane_d": d})
