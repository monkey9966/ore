"""可视化: 分割/判定结果叠加图、粒径分布直方图。"""

import io
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import PipelineConfig
from .stats import SizeDistribution
from .types import (BOUNDARY_FREE, BOUNDARY_INVALID, BOUNDARY_OCCLUDED,
                    FrameResult)

ACCEPT_COLOR = (60, 220, 60)      # BGR 绿
REJECT_COLOR = (50, 50, 235)      # BGR 红
DEBRIS_COLOR = (140, 140, 140)
FERET_COLOR = (255, 200, 0)       # 青蓝


def height_to_bgr(height: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """高度图 -> 伪彩色底图, 无效区域深灰。"""
    h = height.copy()
    finite = np.isfinite(h)
    if finite.any():
        lo, hi = np.percentile(h[finite], [1, 99.5])
        hi = max(hi, lo + 1.0)
        norm = np.clip((h - lo) / (hi - lo), 0, 1)
    else:
        norm = np.zeros_like(h)
    norm[~finite] = 0.0
    cmap = plt.get_cmap("viridis")
    rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr[~valid] = (60, 60, 60)
    return bgr


def render_overlay(result: FrameResult, cfg: PipelineConfig,
                   show_boundary_class: bool = True,
                   scale: int = 2) -> np.ndarray:
    """渲染叠加图: 接受=绿框, 拒绝=红框, 碎料=灰, 边界分类着色。"""
    frame = result.frame
    img = height_to_bgr(frame.height, frame.valid)

    # 碎料区域淡灰描边
    for rid in result.debris_labels:
        mask = (result.labels == rid).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, DEBRIS_COLOR, 1)

    # 边界分类着色(在轮廓线上)
    if show_boundary_class:
        bc = result.boundary_class
        img[bc == BOUNDARY_FREE] = (90, 230, 90)
        img[bc == BOUNDARY_OCCLUDED] = (0, 120, 255)   # 橙: 遮挡边界
        img[bc == BOUNDARY_INVALID] = (200, 0, 200)    # 紫: 无效边界

    # 放大后再画文字(清晰)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    for rock in result.rocks:
        mask = (result.labels == rock.label).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c * scale for c in cnts]
        color = ACCEPT_COLOR if rock.accepted else REJECT_COLOR
        cv2.drawContours(img, cnts, -1, color, 2)

        cy, cx = rock.centroid_px
        pos = (int(cx * scale), int(cy * scale))
        if rock.accepted:
            # 最大 Feret 线
            if rock.max_feret_endpoints_px:
                (x1, y1), (x2, y2) = rock.max_feret_endpoints_px
                cv2.line(img, (int(x1 * scale), int(y1 * scale)),
                         (int(x2 * scale), int(y2 * scale)), FERET_COLOR, 2)
            txt = f"#{rock.label} {rock.max_feret_mm:.0f}mm"
            sub = f"[{rock.size_bin}]"
        else:
            txt = f"#{rock.label} X"
            sub = ",".join(rock.reject_reasons)[:24]
        cv2.putText(img, txt, (pos[0] - 40, pos[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (pos[0] - 40, pos[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, sub, (pos[0] - 40, pos[1] + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, sub, (pos[0] - 40, pos[1] + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    return img


def render_histogram_png(dist: SizeDistribution, cfg: PipelineConfig) -> bytes:
    """粒径分布直方图 PNG (按数量 + 按体积)。"""
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=110)
    x = np.arange(len(dist.bins))
    w = 0.38
    ax.bar(x - w / 2, [dist.count_pct[b] for b in dist.bins], w,
           label="by count %", color="#4c9be8")
    ax.bar(x + w / 2, [dist.volume_pct[b] for b in dist.bins], w,
           label="by volume %", color="#e8a44c")
    ax.set_xticks(x)
    ax.set_xticklabels([b + "mm" for b in dist.bins], fontsize=9)
    ax.set_ylabel("%")
    ax.set_title(f"Size distribution  (n={dist.n_total}, D50={dist.d50_mm:.0f}mm)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def encode_png(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()
