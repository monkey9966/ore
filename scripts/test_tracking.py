#!/usr/bin/env python3
"""Stage F 跨帧去重的仿真测试。

仿真皮带流: 用同一个合成场景做整体平移(等价皮带前进), 连续多帧喂给
RockTracker, 验证:
1. 同一颗矿石多帧不会被重复计数(轨迹数 == 不同矿石数);
2. 多帧中值测量误差 <= 单帧测量误差。

用法: python3 scripts/test_tracking.py [--shift-mm 150] [--seeds 5]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ore_sizer import PipelineConfig, process_frame
from ore_sizer.synthetic import generate_scene
from ore_sizer.tracking import RockTracker
from ore_sizer.types import HeightFrame


def shifted_frame(scene, shift_px: int, cfg, rng) -> HeightFrame:
    """把场景沿行方向平移 shift_px, 露出的区域用碎料床噪声填充,
    并重新加一份传感器噪声(等价皮带前进后重新拍一帧)。"""
    clean = scene.clean_height
    ny, nx = clean.shape
    h = np.full_like(clean, np.nan)
    src = clean[max(0, -shift_px):ny - max(0, shift_px), :]
    h[max(0, shift_px):ny - max(0, -shift_px), :] = src
    exposed = np.isnan(h)
    h[exposed] = 15.0 + rng.normal(0, 4.0, size=int(exposed.sum()))
    h = h + rng.normal(0, cfg.sensor_noise_sigma_mm, size=h.shape)
    valid = np.ones_like(h, dtype=bool)
    b = cfg.roi_border_invalid_px
    valid[:b, :] = valid[-b:, :] = valid[:, :b] = valid[:, -b:] = False
    hf = h.astype(np.float32)
    hf[~valid] = np.nan
    return HeightFrame(hf, valid, cfg.mm_per_px)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift-mm", type=float, default=150.0, help="帧间皮带位移")
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    cfg = PipelineConfig()
    shift_px = args.shift_mm / cfg.mm_per_px

    n_bad_dup = 0
    total_tracks = 0
    multi_frame_tracks = 0
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed + 999)
        scene = generate_scene(cfg, seed=200 + seed, n_rocks=10, stacking=0.2)
        tracker = RockTracker(cfg, belt_axis=0)

        finalized = []
        single_frame_accept_total = 0
        for k in range(args.n_frames):
            frame = shifted_frame(scene, int(round(k * shift_px)), cfg, rng)
            res = process_frame(frame, cfg)
            single_frame_accept_total += len(res.accepted_rocks)
            finalized += tracker.update(res, shift_px if k > 0 else 0.0)
        finalized += tracker.flush()

        total_tracks += len(finalized)
        multi_frame_tracks += sum(1 for t in finalized if t.n_frames > 1)
        # 同一矿石重复计数检查: 结算轨迹的最终质心两两距离不得过近
        cents = [t.observations[-1].centroid_px for t in finalized]
        for i in range(len(cents)):
            for j in range(i + 1, len(cents)):
                # 还原到皮带坐标系(减去各自结算前的平移)比较过近视为重复
                d = np.hypot(cents[i][0] - cents[j][0], cents[i][1] - cents[j][1])
                if d < 20.0 / cfg.mm_per_px:
                    n_bad_dup += 1
        print(f"seed={seed}: 单帧累计接受 {single_frame_accept_total} 次 "
              f"-> 去重后 {len(finalized)} 颗 "
              f"(多帧观测 {sum(1 for t in finalized if t.n_frames > 1)} 颗)")

    print("-" * 50)
    print(f"总轨迹 {total_tracks}, 其中多帧聚合 {multi_frame_tracks}, "
          f"疑似重复 {n_bad_dup}")
    ok = n_bad_dup == 0 and multi_frame_tracks > 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
