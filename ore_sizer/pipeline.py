"""流水线编排: 高度图 -> FrameResult。"""

import time
from typing import Optional

import numpy as np

from .completeness import evaluate_rock
from .config import PipelineConfig
from .measure import measure_rock
from .preprocess import fill_small_holes, prominence_map
from .segmentation import segment
from .types import FrameResult, HeightFrame


def process_frame(frame: HeightFrame, cfg: Optional[PipelineConfig] = None) -> FrameResult:
    cfg = cfg or PipelineConfig()
    timings = {}

    t0 = time.perf_counter()
    frame = fill_small_holes(frame, cfg)
    prom, base = prominence_map(frame, cfg)   # base: 局部基准面(碎料床/皮带)
    timings["preprocess"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    labels, hs, debris_ids = segment(frame, prom, cfg)
    timings["segment"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    boundary_class = np.zeros(frame.shape, dtype=np.uint8)
    rocks = []
    debris_set = set(debris_ids)
    for rid in range(1, int(labels.max()) + 1):
        if rid in debris_set:
            continue
        mask = labels == rid
        if not mask.any():
            continue
        info = evaluate_rock(rid, mask, frame, hs, base, cfg, boundary_class)
        info = measure_rock(info, mask, frame.height, prom, cfg, frame.mm_per_px)
        rocks.append(info)
    timings["evaluate_measure"] = (time.perf_counter() - t0) * 1e3

    return FrameResult(
        frame=frame,
        labels=labels,
        rocks=rocks,
        boundary_class=boundary_class,
        prominence=prom,
        debris_labels=debris_ids,
        timings_ms=timings,
    )
