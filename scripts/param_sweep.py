#!/usr/bin/env python3
"""参数扫描: 在查准率 >= 目标(默认98%)的约束下寻找查全率最优的参数组合。

两层扫描(避免重复跑昂贵阶段):
1. 昂贵层: 影响分割/边界分类的参数(hmax_delta_mm / drop_delta_mm / waist_max_ratio),
   每个组合把流水线完整跑一遍, 记录**每颗候选矿石**的原始判定指标与真值匹配;
2. 便宜层: 完整性判定阈值(free_ratio_min / solidity_min / ellipse_iou_min /
   feret_ellipse_tol), 直接在记录的指标上向量化重放, 秒级完成全网格。

用法:
    python3 scripts/param_sweep.py [--frames 20] [--stacking 0.35] \
        [--precision-floor 0.98] [--val-frames 40] [--jobs 4]

输出: 排名靠前的参数组合(先在扫描集选优, 再在独立种子的验证集复核),
以及可直接粘贴进 PipelineConfig 的推荐值。
"""

import argparse
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ore_sizer import PipelineConfig, process_frame
from ore_sizer.synthetic import generate_scene

# ---------------- 昂贵层网格(流水线要重跑的参数) ----------------
HMAX_DELTA_MM = [8.0, 10.0, 12.0, 15.0, 18.0]
DROP_DELTA_MM = [12.0, 15.0]
WAIST_MAX_RATIO = [0.55, 0.65]

# ---------------- 便宜层网格(判定阈值, 离线重放) ----------------
FREE_RATIO_MIN = [0.85, 0.88, 0.90, 0.92, 0.95]
SOLIDITY_MIN = [0.85, 0.88, 0.90]
ELLIPSE_IOU_MIN = [0.80, 0.83, 0.85, 0.88, 0.90]
FERET_ELLIPSE_TOL = [1.05, 1.08, 1.12]
SPLIT_H_MM = [8.0, 10.0, 12.0, 999.0]        # 999=关闭
THICKNESS_WIDTH_MAX = [1.10, 1.15, 1.20, 999.0]  # 999=关闭

FIXED_REASONS = ("TOUCH_BORDER", "PEAK_ON_EDGE", "TOO_SMALL",
                 "LOW_VALID_COVERAGE", "INVALID_BOUNDARY")


@dataclass
class RockRecord:
    """一颗候选矿石的可重放判定指标。"""
    seed: int
    free_ratio: float
    solidity: float
    ellipse_iou: float
    feret_ratio: float        # 掩膜Feret / 拟合椭圆长轴 (px/px)
    split_depth: float        # 表面多峰分裂深度 mm (0=单峰)
    thick_ratio: float        # 厚度 / 次轴
    fixed_reject: bool        # 不参与扫描的固定规则是否已拒绝
    gt_id: int                # 匹配的真值矿石 id, -1 表示无匹配
    gt_visible: bool          # 匹配的真值是否"完整可见"
    feret_err: float          # 相对真值误差(仅 gt_visible 时有意义)
    width_err: float


def _match_gt(rock, gt_rocks, max_dist_px=25.0):
    cy, cx = rock.centroid_px
    best, best_d = None, max_dist_px
    for g in gt_rocks:
        d = np.hypot(g.center_px[0] - cy, g.center_px[1] - cx)
        if d < best_d:
            best, best_d = g, d
    return best


def run_one(task) -> Tuple[Tuple[float, float, float], int, int, List[RockRecord]]:
    """跑一个 (变体, 种子) 组合, 返回该帧全部候选矿石的记录。"""
    (hmax, drop, waist), seed, stacking, n_rocks = task
    cfg = PipelineConfig(hmax_delta_mm=hmax, drop_delta_mm=drop,
                         waist_max_ratio=waist)
    scene = generate_scene(cfg, seed=seed, n_rocks=n_rocks, stacking=stacking)
    res = process_frame(scene.frame, cfg)

    n_gt_visible = sum(1 for g in scene.gt_rocks if g.fully_visible)
    records = []
    for r in res.rocks:
        g = _match_gt(r, scene.gt_rocks)
        ell_major_px, _ = r.meta_ellipse_axes_px
        feret_px = r.max_feret_mm / cfg.mm_per_px
        feret_ratio = feret_px / ell_major_px if ell_major_px > 0 else 99.0
        fe = we = 0.0
        if g is not None and g.fully_visible:
            fe = (r.max_feret_mm - g.true_max_edge_mm) / g.true_max_edge_mm
            we = (r.min_rect_w_mm - g.true_width_mm) / g.true_width_mm
        records.append(RockRecord(
            seed=seed,
            free_ratio=r.free_ratio,
            solidity=r.solidity,
            ellipse_iou=r.ellipse_iou,
            feret_ratio=feret_ratio,
            split_depth=r.split_depth_mm,
            thick_ratio=r.thickness_mm / max(r.min_rect_w_mm, 1e-3),
            fixed_reject=any(x in r.reject_reasons for x in FIXED_REASONS),
            gt_id=(g.rock_id if g is not None else -1),
            gt_visible=(g is not None and g.fully_visible),
            feret_err=fe, width_err=we,
        ))
    return (hmax, drop, waist), seed, n_gt_visible, records


def replay_thresholds(records: List[RockRecord], n_gt_visible_total: int,
                      fr_min: float, sol_min: float, iou_min: float,
                      feret_tol: float, split_h: float, thick_max: float) -> Dict:
    """在记录的指标上重放完整性判定阈值, 返回 precision/recall/误差。"""
    tp = fp = 0
    accepted_gt: set = set()
    fe, we = [], []
    for r in records:
        ok = (not r.fixed_reject
              and r.free_ratio >= fr_min
              and r.solidity >= sol_min
              and r.ellipse_iou >= iou_min
              and r.feret_ratio <= feret_tol
              and r.split_depth < split_h
              and r.thick_ratio <= thick_max)
        if not ok:
            continue
        if r.gt_visible:
            tp += 1
            accepted_gt.add((r.seed, r.gt_id))
            fe.append(abs(r.feret_err))
            we.append(abs(r.width_err))
        else:
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = len(accepted_gt) / n_gt_visible_total if n_gt_visible_total else 0.0
    return dict(precision=precision, recall=recall, tp=tp, fp=fp,
                feret_err=(float(np.mean(fe)) if fe else 0.0),
                width_err=(float(np.mean(we)) if we else 0.0))


def collect(variants, seeds, stacking, n_rocks, jobs):
    """并行跑昂贵层, 返回 {variant: (n_gt_visible_total, records)}。"""
    tasks = [(v, s, stacking, n_rocks) for v in variants for s in seeds]
    out: Dict[Tuple, List[RockRecord]] = {v: [] for v in variants}
    gt_tot: Dict[Tuple, int] = {v: 0 for v in variants}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for variant, seed, n_gt, recs in ex.map(run_one, tasks, chunksize=2):
            out[variant].extend(recs)
            gt_tot[variant] += n_gt
    return {v: (gt_tot[v], out[v]) for v in variants}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=20, help="扫描集帧数")
    ap.add_argument("--val-frames", type=int, default=40, help="验证集帧数(独立种子)")
    ap.add_argument("--stacking", type=float, default=0.35)
    ap.add_argument("--n-rocks", type=int, default=14)
    ap.add_argument("--precision-floor", type=float, default=0.98)
    ap.add_argument("--jobs", type=int, default=max(os.cpu_count() - 0, 1))
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    variants = list(itertools.product(HMAX_DELTA_MM, DROP_DELTA_MM, WAIST_MAX_RATIO))
    sweep_seeds = [100 + i for i in range(args.frames)]

    print(f"昂贵层: {len(variants)} 个流水线变体 × {len(sweep_seeds)} 帧 "
          f"(jobs={args.jobs}) ...", flush=True)
    data = collect(variants, sweep_seeds, args.stacking, args.n_rocks, args.jobs)

    # 便宜层: 全阈值网格重放
    results = []
    for v, (n_gt, recs) in data.items():
        for fr, sol, iou, tol, sph, thk in itertools.product(
                FREE_RATIO_MIN, SOLIDITY_MIN, ELLIPSE_IOU_MIN,
                FERET_ELLIPSE_TOL, SPLIT_H_MM, THICKNESS_WIDTH_MAX):
            m = replay_thresholds(recs, n_gt, fr, sol, iou, tol, sph, thk)
            results.append((v, (fr, sol, iou, tol, sph, thk), m))

    feasible = [r for r in results if r[2]["precision"] >= args.precision_floor]
    pool = feasible if feasible else results
    if not feasible:
        print(f"! 扫描集上没有组合达到查准率 {args.precision_floor:.0%}, "
              f"退化为按 (precision, recall) 排序展示。")
    pool.sort(key=lambda r: (r[2]["recall"], r[2]["precision"]), reverse=True)

    print(f"\n扫描集 Top{args.top} (precision >= {args.precision_floor:.0%}):")
    hdr = (f"{'hmax':>5} {'drop':>5} {'waist':>5} | "
           f"{'free':>5} {'sol':>5} {'iou':>5} {'ftol':>5} {'sph':>4} {'thk':>5} | "
           f"{'prec':>6} {'recall':>6} {'tp':>4} {'fp':>3} {'feret%':>6}")
    print(hdr); print("-" * len(hdr))
    for v, th, m in pool[:args.top]:
        print(f"{v[0]:5.0f} {v[1]:5.0f} {v[2]:5.2f} | "
              f"{th[0]:5.2f} {th[1]:5.2f} {th[2]:5.2f} {th[3]:5.2f} {th[4]:4.0f} {th[5]:5.2f} | "
              f"{m['precision']*100:5.1f}% {m['recall']*100:5.1f}% "
              f"{m['tp']:4d} {m['fp']:3d} {m['feret_err']*100:5.1f}%")

    # ---- 验证集复核(独立种子, 防过拟合): 取扫描集前若干名重跑 ----
    n_check = min(6, len(pool))
    val_variants = sorted({pool[i][0] for i in range(n_check)})
    val_seeds = [1000 + i for i in range(args.val_frames)]
    print(f"\n验证层: {len(val_variants)} 个变体 × {len(val_seeds)} 帧 ...", flush=True)
    val_data = collect(val_variants, val_seeds, args.stacking, args.n_rocks, args.jobs)

    print(f"\n验证集复核 (独立种子 {val_seeds[0]}..{val_seeds[-1]}):")
    print(hdr); print("-" * len(hdr))
    best = None
    for v, th, m in pool[:n_check]:
        n_gt, recs = val_data[v]
        vm = replay_thresholds(recs, n_gt, *th)
        print(f"{v[0]:5.0f} {v[1]:5.0f} {v[2]:5.2f} | "
              f"{th[0]:5.2f} {th[1]:5.2f} {th[2]:5.2f} {th[3]:5.2f} {th[4]:4.0f} {th[5]:5.2f} | "
              f"{vm['precision']*100:5.1f}% {vm['recall']*100:5.1f}% "
              f"{vm['tp']:4d} {vm['fp']:3d} {vm['feret_err']*100:5.1f}%")
        score = (vm["precision"] >= args.precision_floor, vm["recall"])
        if best is None or score > best[0]:
            best = (score, v, th, vm)

    _, v, th, vm = best
    print("\n推荐配置(验证集上查准率约束下查全率最优):")
    print(f"    hmax_delta_mm     = {v[0]}")
    print(f"    drop_delta_mm     = {v[1]}")
    print(f"    waist_max_ratio   = {v[2]}")
    print(f"    free_ratio_min    = {th[0]}")
    print(f"    solidity_min      = {th[1]}")
    print(f"    ellipse_iou_min   = {th[2]}")
    print(f"    feret_ellipse_tol = {th[3]}")
    print(f"    split_h_mm        = {th[4]}")
    print(f"    thickness_width_max = {th[5]}")
    print(f"验证集: precision={vm['precision']*100:.1f}%  recall={vm['recall']*100:.1f}%  "
          f"feret_err={vm['feret_err']*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
