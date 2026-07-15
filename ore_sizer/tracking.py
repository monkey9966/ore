"""Stage F: 跨帧去重与多帧测量聚合(产线阶段)。

产线上编码器定距触发, 同一颗矿石会出现在连续多帧中。本模块:
1. 按编码器给出的帧间皮带位移, 把上一帧矿石质心平移到当前帧坐标系;
2. 与当前帧接受的矿石做最近邻匹配(距离门限), 匹配上的视为同一颗;
3. 同一颗矿石的多帧测量取中值(抗单帧噪声/飞点);
4. 矿石离开视场(或长时间未再匹配)时"结算"一次, 计入统计。

办公室静态验证不需要本模块; 静止场景重复处理同一帧会重复计数,
应当只处理单帧。

用法(产线主循环):
    tracker = RockTracker(cfg)
    for frame in frames:                        # 编码器触发的帧流
        result = process_frame(frame, cfg)
        shift_px = encoder_shift_mm / cfg.mm_per_px   # 本帧相对上帧的皮带位移
        finalized = tracker.update(result, shift_px)  # 离场结算的矿石
        for track in finalized:
            distribution.add(track.median_info())
    for track in tracker.flush():               # 收尾: 结算所有在跟踪的矿石
        distribution.add(track.median_info())
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .measure import bin_name
from .types import FrameResult, RockInfo


@dataclass
class RockTrack:
    """一颗矿石的跨帧轨迹。"""
    track_id: int
    centroid_px: Tuple[float, float]        # 当前帧坐标系下的质心(随皮带平移预测)
    observations: List[RockInfo] = field(default_factory=list)
    last_seen_frame: int = 0
    missed: int = 0                         # 连续未匹配帧数

    def add(self, info: RockInfo, frame_idx: int):
        self.observations.append(info)
        self.centroid_px = info.centroid_px
        self.last_seen_frame = frame_idx
        self.missed = 0

    def median_info(self, cfg: Optional[PipelineConfig] = None) -> RockInfo:
        """多帧测量取中值, 返回聚合后的 RockInfo(用于统计)。"""
        obs = self.observations
        assert obs, "空轨迹不可结算"
        med = lambda attr: float(np.median([getattr(o, attr) for o in obs]))
        out = RockInfo(label=self.track_id, accepted=True)
        out.max_feret_mm = med("max_feret_mm")
        out.min_rect_l_mm = med("min_rect_l_mm")
        out.min_rect_w_mm = med("min_rect_w_mm")
        out.equiv_diameter_mm = med("equiv_diameter_mm")
        out.thickness_mm = med("thickness_mm")
        out.volume_mm3 = med("volume_mm3")
        out.area_mm2 = med("area_mm2")
        out.sizing_mm = med("sizing_mm")
        out.centroid_px = obs[-1].centroid_px
        if cfg is not None:
            out.size_bin = bin_name(out.sizing_mm, cfg)
        else:
            out.size_bin = obs[-1].size_bin
        return out

    @property
    def n_frames(self) -> int:
        return len(self.observations)


class RockTracker:
    """编码器辅助的跨帧矿石跟踪器。

    belt_axis: 皮带运动方向在图像中的轴, 0=行(向下), 1=列(向右)。
    match_dist_px: 平移预测后的质心匹配门限(应大于分割质心抖动,
        小于矿石最小间距; 默认约 60mm)。
    max_missed: 连续未匹配该帧数后结算(处理偶发漏检/漏分割)。
    """

    def __init__(self, cfg: PipelineConfig, belt_axis: int = 0,
                 match_dist_px: Optional[float] = None, max_missed: int = 2):
        self.cfg = cfg
        self.belt_axis = belt_axis
        self.match_dist_px = match_dist_px or (60.0 / cfg.mm_per_px)
        self.max_missed = max_missed
        self.tracks: List[RockTrack] = []
        self._next_id = 1
        self._frame_idx = 0

    def update(self, result: FrameResult, shift_px: float) -> List[RockTrack]:
        """喂入新一帧结果。shift_px: 本帧相对上一帧的皮带位移(像素,
        沿 belt_axis 正方向)。返回本次结算(离场)的轨迹。"""
        self._frame_idx += 1
        ny, nx = result.frame.shape

        # 1) 预测: 所有轨迹质心按皮带位移平移
        for t in self.tracks:
            cy, cx = t.centroid_px
            if self.belt_axis == 0:
                t.centroid_px = (cy + shift_px, cx)
            else:
                t.centroid_px = (cy, cx + shift_px)

        # 2) 匹配: 当前帧接受矿石 <-> 轨迹, 贪心最近邻
        detections = list(result.accepted_rocks)
        unmatched_det = set(range(len(detections)))
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di in unmatched_det:
                d = np.hypot(t.centroid_px[0] - detections[di].centroid_px[0],
                             t.centroid_px[1] - detections[di].centroid_px[1])
                if d <= self.match_dist_px:
                    pairs.append((d, ti, di))
        pairs.sort()
        matched_tracks = set()
        for d, ti, di in pairs:
            if ti in matched_tracks or di not in unmatched_det:
                continue
            self.tracks[ti].add(detections[di], self._frame_idx)
            matched_tracks.add(ti)
            unmatched_det.discard(di)

        # 3) 未匹配检测 -> 新轨迹
        for di in unmatched_det:
            t = RockTrack(track_id=self._next_id,
                          centroid_px=detections[di].centroid_px)
            t.add(detections[di], self._frame_idx)
            self._next_id += 1
            self.tracks.append(t)

        # 4) 结算: 质心已移出视场, 或连续 max_missed 帧未匹配
        finalized = []
        keep = []
        for ti, t in enumerate(self.tracks):
            if ti not in matched_tracks and t.observations \
                    and t.last_seen_frame < self._frame_idx:
                t.missed += 1
            cy, cx = t.centroid_px
            out_of_view = not (0 <= cy < ny and 0 <= cx < nx)
            if out_of_view or t.missed > self.max_missed:
                finalized.append(t)
            else:
                keep.append(t)
        self.tracks = keep
        return finalized

    def flush(self) -> List[RockTrack]:
        """结算所有仍在跟踪的轨迹(处理结束时调用)。"""
        out, self.tracks = self.tracks, []
        return out
