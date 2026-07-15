#!/usr/bin/env python3
"""合成数据定量验证: 判定查准率/查全率 + 测量误差。

用法:
    python3 scripts/run_synthetic_eval.py [--frames 30] [--stacking 0.35] [--out out/eval]

指标定义(以合成场景真值为准):
- 查准率 precision: 被接受的矿石中, 真值确实"完整可见"的比例 (目标 >98%)
- 查全率 recall:    真值"完整可见"的矿石中, 被接受的比例 (目标 >80%, 漏检靠皮带样本量弥补)
- 测量误差:          接受且匹配正确的矿石, 最大边长/次轴相对真值的误差
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ore_sizer import PipelineConfig, process_frame
from ore_sizer.stats import compute_distribution
from ore_sizer.synthetic import generate_scene
from ore_sizer.viz import encode_png, render_overlay


def match_gt(rock, gt_rocks, max_dist_px=25.0):
    """接受矿石与真值矿石按质心匹配。"""
    cy, cx = rock.centroid_px
    best, best_d = None, max_dist_px
    for g in gt_rocks:
        d = np.hypot(g.center_px[0] - cy, g.center_px[1] - cx)
        if d < best_d:
            best, best_d = g, d
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--stacking", type=float, default=0.35)
    ap.add_argument("--n-rocks", type=int, default=14)
    ap.add_argument("--out", default="out/eval")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = PipelineConfig()

    tp = fp = 0                # 接受矿石: 真值完整/不完整
    n_gt_visible = 0           # 真值完整可见总数
    n_gt_visible_accepted = 0
    feret_errors, width_errors = [], []
    all_rocks = []

    for i in range(args.frames):
        scene = generate_scene(cfg, seed=100 + i, n_rocks=args.n_rocks,
                               stacking=args.stacking)
        res = process_frame(scene.frame, cfg)
        all_rocks.extend(res.rocks)

        vis_gt = [g for g in scene.gt_rocks if g.fully_visible]
        n_gt_visible += len(vis_gt)

        matched_gt_ids = set()
        for r in res.accepted_rocks:
            g = match_gt(r, scene.gt_rocks)
            if g is not None and g.fully_visible:
                tp += 1
                matched_gt_ids.add(g.rock_id)
                feret_errors.append(
                    (r.max_feret_mm - g.true_max_edge_mm) / g.true_max_edge_mm)
                width_errors.append(
                    (r.min_rect_w_mm - g.true_width_mm) / g.true_width_mm)
            else:
                fp += 1
        n_gt_visible_accepted += len(matched_gt_ids)

        if i < 3:  # 保存前几帧的可视化
            img = render_overlay(res, cfg, scale=2)
            with open(os.path.join(args.out, f"frame_{i}.png"), "wb") as f:
                f.write(encode_png(img))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = n_gt_visible_accepted / n_gt_visible if n_gt_visible else 0.0
    fe = np.abs(feret_errors) * 100
    we = np.abs(width_errors) * 100

    dist = compute_distribution(all_rocks, cfg)

    print("=" * 60)
    print(f"帧数: {args.frames}  堆叠系数: {args.stacking}  每帧矿石: {args.n_rocks}")
    print(f"真值完整可见矿石总数: {n_gt_visible}")
    print(f"接受总数: {tp + fp}  (正确 {tp} / 误收 {fp})")
    print("-" * 60)
    print(f"查准率 precision: {precision * 100:.1f}%   (目标 >98%)")
    print(f"查全率 recall:    {recall * 100:.1f}%   (目标 >80%)")
    if len(fe):
        print(f"最大边长误差: 平均 {fe.mean():.1f}%  P95 {np.percentile(fe, 95):.1f}%")
        print(f"次轴(粒径)误差: 平均 {we.mean():.1f}%  P95 {np.percentile(we, 95):.1f}%")
    print("-" * 60)
    print("累计粒径分布(按数量):")
    for b in dist.bins:
        print(f"  {b:>10} mm: {dist.count[b]:4d} 颗  ({dist.count_pct[b]:.1f}%)")
    print(f"  D50 = {dist.d50_mm:.0f} mm")
    print(f"可视化样例已保存到 {args.out}/")
    print("=" * 60)

    ok = precision >= 0.98
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
