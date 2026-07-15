#!/usr/bin/env python3
"""DS63 采集脚本: 连拍深度帧存 .npz(含内参), 供离线回放调参。

使用前提(见 ore_sizer/camera/scepter.py 顶部注释):
1. 安装 ScepterSDK 并设置环境变量 SCEPTER_SDK_DIR;
2. 相机与主机同网段(默认 IP 192.168.1.101)。

用法:
    python3 scripts/capture_ds63.py --count 10 --interval 0.5 --out data/capture
    python3 scripts/capture_ds63.py --count 1 --process        # 抓一帧顺便跑流水线

回放:
    from ore_sizer.camera.frameio import load_frame
    frame = load_frame("data/capture/frame_0000.npz", cfg)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ore_sizer import PipelineConfig
from ore_sizer.camera.frameio import save_depth_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10, help="采集帧数")
    ap.add_argument("--interval", type=float, default=0.5, help="帧间隔(秒)")
    ap.add_argument("--out", default="data/capture", help="输出目录")
    ap.add_argument("--prefix", default="frame", help="文件名前缀")
    ap.add_argument("--process", action="store_true",
                    help="每帧顺便跑一遍流水线并打印接受矿石(现场快速核对)")
    args = ap.parse_args()

    from ore_sizer.camera.scepter import (ScepterNotAvailable, capture_depth,
                                          disconnect)

    os.makedirs(args.out, exist_ok=True)
    cfg = PipelineConfig()

    print(f"连接 DS63 并采集 {args.count} 帧 -> {args.out}/ ...")
    saved = 0
    try:
        for i in range(args.count):
            t0 = time.perf_counter()
            depth, intr = capture_depth()
            path = os.path.join(args.out, f"{args.prefix}_{i:04d}.npz")
            save_depth_npz(path, depth, **intr)
            dt = (time.perf_counter() - t0) * 1e3
            n_valid = int((depth > 0).sum())
            print(f"[{i + 1}/{args.count}] {path}  "
                  f"{depth.shape[1]}x{depth.shape[0]}  有效像素 {n_valid}  ({dt:.0f} ms)")
            saved += 1

            if args.process:
                from ore_sizer import process_frame
                from ore_sizer.preprocess import depth_to_heightframe
                frame = depth_to_heightframe(
                    depth, intr["fx"], intr["fy"], intr["cx"], intr["cy"], cfg)
                res = process_frame(frame, cfg)
                print(f"    候选 {len(res.rocks)}  接受 {len(res.accepted_rocks)}: "
                      + ", ".join(f"#{r.label} {r.max_feret_mm:.0f}mm"
                                  for r in res.accepted_rocks))

            if i + 1 < args.count and args.interval > 0:
                time.sleep(args.interval)
    except ScepterNotAvailable as e:
        print(f"错误: {e}", file=sys.stderr)
        return 2 if saved == 0 else 1
    except KeyboardInterrupt:
        print("\n中断, 已保存的帧保留。")
    finally:
        disconnect()

    print(f"完成: 共保存 {saved} 帧。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
