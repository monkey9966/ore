"""Stage G: 粒径分布统计。"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .config import PipelineConfig
from .measure import bin_name
from .types import RockInfo


@dataclass
class SizeDistribution:
    bins: List[str]
    count: Dict[str, int]
    count_pct: Dict[str, float]
    volume_pct: Dict[str, float]
    n_total: int
    d50_mm: float          # 按数量的中位粒径
    mean_mm: float
    max_mm: float

    def as_dict(self) -> Dict:
        return {
            "bins": self.bins,
            "count": self.count,
            "count_pct": self.count_pct,
            "volume_pct": self.volume_pct,
            "n_total": self.n_total,
            "d50_mm": round(self.d50_mm, 1),
            "mean_mm": round(self.mean_mm, 1),
            "max_mm": round(self.max_mm, 1),
        }


def compute_distribution(rocks: List[RockInfo], cfg: PipelineConfig) -> SizeDistribution:
    """对接受的矿石计算粒径分布(按数量与按估计体积)。"""
    accepted = [r for r in rocks if r.accepted]
    bins = [(f"{lo:.0f}-{hi:.0f}" if np.isfinite(hi) else f">{lo:.0f}")
            for lo, hi in cfg.bins_with_inf]
    count = {b: 0 for b in bins}
    vol = {b: 0.0 for b in bins}
    sizes = []
    for r in accepted:
        b = bin_name(r.sizing_mm, cfg)
        if b in count:
            count[b] += 1
            vol[b] += r.volume_mm3
        sizes.append(r.sizing_mm)

    n = len(accepted)
    tot_v = sum(vol.values()) or 1.0
    count_pct = {b: (100.0 * c / n if n else 0.0) for b, c in count.items()}
    volume_pct = {b: 100.0 * v / tot_v for b, v in vol.items()}
    sizes_arr = np.array(sizes) if sizes else np.array([0.0])
    return SizeDistribution(
        bins=bins,
        count=count,
        count_pct=count_pct,
        volume_pct=volume_pct,
        n_total=n,
        d50_mm=float(np.median(sizes_arr)) if n else 0.0,
        mean_mm=float(sizes_arr.mean()) if n else 0.0,
        max_mm=float(sizes_arr.max()) if n else 0.0,
    )
