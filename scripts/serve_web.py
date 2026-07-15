#!/usr/bin/env python3
"""启动可视化 Web 页面(办公室验证)。

用法:
    python3 scripts/serve_web.py [--port 8000] [--host 0.0.0.0]
浏览器打开 http://localhost:8000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ore_sizer.webapp import main


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    main(host=args.host, port=args.port, debug=args.debug)
