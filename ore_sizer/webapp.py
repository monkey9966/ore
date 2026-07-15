"""可视化 Web 页面(办公室静态验证的主要工具)。

三种输入:
1. 合成场景(seed/堆叠/数量可调, 无需相机);
2. 上传帧文件(.npz 深度或高度图 / 16bit PNG 深度);
3. DS63 相机取流(需 ScepterSDK, 未接相机时按钮会提示错误)。

显示: 分割叠加图(点击矿石查看详情) + 每颗矿石粒径表(含拒绝原因) + 分布直方图。

启动:
    python3 scripts/serve_web.py [--port 8000]
"""

import io
import json
import threading
import time
import uuid
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

from .config import PipelineConfig
from .pipeline import process_frame
from .stats import compute_distribution
from .types import REJECT_REASONS, FrameResult
from .viz import encode_png, render_histogram_png, render_overlay

app = Flask(__name__)

# 会话缓存: result_id -> dict(result, overlay_png, hist_png, ...)
_store: Dict[str, Dict] = {}
_store_lock = threading.Lock()
_MAX_SESSIONS = 20


def _cfg_from_request(form) -> PipelineConfig:
    cfg = PipelineConfig()
    for name in ("hmax_delta_mm", "drop_delta_mm", "free_ratio_min",
                 "ellipse_iou_min", "solidity_min"):
        v = form.get(name, "").strip()
        if v:
            setattr(cfg, name, float(v))
    return cfg


def _run_and_store(frame, cfg: PipelineConfig, source: str) -> str:
    t0 = time.perf_counter()
    result = process_frame(frame, cfg)
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    dist = compute_distribution(result.rocks, cfg)

    overlay = render_overlay(result, cfg, scale=2)
    rid = uuid.uuid4().hex[:12]
    with _store_lock:
        if len(_store) >= _MAX_SESSIONS:
            oldest = min(_store, key=lambda k: _store[k]["ts"])
            _store.pop(oldest, None)
        _store[rid] = dict(
            ts=time.time(),
            result=result,
            cfg=cfg,
            source=source,
            elapsed_ms=elapsed_ms,
            overlay_png=encode_png(overlay),
            hist_png=render_histogram_png(dist, cfg),
            dist=dist.as_dict(),
            scale=2,
        )
    return rid


def _rock_row(r) -> Dict:
    return {
        "label": r.label,
        "accepted": r.accepted,
        "reject_reasons": r.reject_reasons,
        "reject_text": "; ".join(REJECT_REASONS.get(x, x) for x in r.reject_reasons),
        "max_feret_mm": round(r.max_feret_mm, 1),
        "min_rect_l_mm": round(r.min_rect_l_mm, 1),
        "min_rect_w_mm": round(r.min_rect_w_mm, 1),
        "equiv_diameter_mm": round(r.equiv_diameter_mm, 1),
        "thickness_mm": round(r.thickness_mm, 1),
        "size_bin": r.size_bin if r.accepted else "-",
        "free_ratio": round(r.free_ratio, 3),
        "occluded_ratio": round(r.occluded_ratio, 3),
        "invalid_ratio": round(r.invalid_ratio, 3),
        "solidity": round(r.solidity, 3),
        "ellipse_iou": round(r.ellipse_iou, 3),
        "valid_coverage": round(r.valid_coverage, 3),
        "centroid_px": [round(c, 1) for c in r.centroid_px],
        "bbox_px": list(r.bbox_px),
    }


def _session_payload(rid: str) -> Dict:
    s = _store[rid]
    result: FrameResult = s["result"]
    rocks = sorted((_rock_row(r) for r in result.rocks),
                   key=lambda x: (not x["accepted"], -x["max_feret_mm"]))
    return {
        "result_id": rid,
        "source": s["source"],
        "elapsed_ms": round(s["elapsed_ms"], 1),
        "frame_shape": list(result.frame.shape),
        "mm_per_px": result.frame.mm_per_px,
        "scale": s["scale"],
        "n_candidates": len(result.rocks),
        "n_accepted": len(result.accepted_rocks),
        "n_debris": len(result.debris_labels),
        "rocks": rocks,
        "dist": s["dist"],
        "timings_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
    }


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/synthetic", methods=["POST"])
def api_synthetic():
    from .synthetic import generate_scene
    cfg = _cfg_from_request(request.form)
    seed = int(request.form.get("seed", 1))
    n_rocks = int(request.form.get("n_rocks", 14))
    stacking = float(request.form.get("stacking", 0.35))
    scene = generate_scene(cfg, seed=seed, n_rocks=n_rocks, stacking=stacking)
    rid = _run_and_store(scene.frame, cfg,
                         f"合成场景 seed={seed} 矿石={n_rocks} 堆叠={stacking}")
    return jsonify(_session_payload(rid))


@app.route("/api/upload", methods=["POST"])
def api_upload():
    from .camera.frameio import load_frame
    import tempfile, os
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "未选择文件"}), 400
    cfg = _cfg_from_request(request.form)
    suffix = "." + f.filename.rsplit(".", 1)[-1].lower()
    if suffix not in (".npz", ".npy", ".png", ".tif", ".tiff"):
        return jsonify({"error": f"不支持的格式 {suffix} (支持 .npz/.npy/.png/.tif)"}), 400
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp.name)
        path = tmp.name
    try:
        frame = load_frame(path, cfg)
    except Exception as e:
        return jsonify({"error": f"帧加载失败: {e}"}), 400
    finally:
        os.unlink(path)
    rid = _run_and_store(frame, cfg, f"上传 {f.filename}")
    return jsonify(_session_payload(rid))


@app.route("/api/camera", methods=["POST"])
def api_camera():
    cfg = _cfg_from_request(request.form)
    try:
        from .camera.scepter import capture_heightframe
        frame = capture_heightframe(cfg)
    except Exception as e:
        return jsonify({"error": f"相机取流失败: {e}"}), 502
    rid = _run_and_store(frame, cfg, "DS63 相机")
    return jsonify(_session_payload(rid))


@app.route("/img/<rid>/overlay.png")
def img_overlay(rid):
    s = _store.get(rid)
    if s is None:
        return "expired", 404
    return Response(s["overlay_png"], mimetype="image/png")


@app.route("/img/<rid>/hist.png")
def img_hist(rid):
    s = _store.get(rid)
    if s is None:
        return "expired", 404
    return Response(s["hist_png"], mimetype="image/png")


# ---------------------------------------------------------------------------
# 前端(单页, 无外部依赖)
# ---------------------------------------------------------------------------

PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>ore-sizer 可视化验证</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {
    --bg: #101418; --panel: #1a2026; --panel2: #212a33; --border: #2e3a45;
    --text: #dce5ec; --dim: #8fa1b0; --accent: #37b26c; --red: #e05252;
    --blue: #4c9be8;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.5 "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }
  header { padding: 14px 22px; background: var(--panel); border-bottom: 1px solid var(--border);
           display: flex; align-items: baseline; gap: 14px; }
  header h1 { margin: 0; font-size: 18px; font-weight: 600; }
  header .sub { color: var(--dim); font-size: 12.5px; }
  main { display: grid; grid-template-columns: 300px 1fr; gap: 16px; padding: 16px 22px; }
  @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
          padding: 14px 16px; }
  .card h2 { margin: 0 0 10px; font-size: 14px; font-weight: 600; color: var(--blue); }
  label { display: block; margin: 8px 0 3px; color: var(--dim); font-size: 12.5px; }
  input[type=number], input[type=text] { width: 100%; padding: 6px 8px; border-radius: 6px;
      border: 1px solid var(--border); background: var(--panel2); color: var(--text); }
  input[type=file] { width: 100%; color: var(--dim); font-size: 12.5px; }
  button { margin-top: 12px; width: 100%; padding: 9px 0; border: 0; border-radius: 8px;
           background: var(--accent); color: #fff; font-size: 14px; font-weight: 600;
           cursor: pointer; }
  button.secondary { background: var(--panel2); border: 1px solid var(--border); }
  button:disabled { opacity: .5; cursor: wait; }
  details { border-top: 1px solid var(--border); margin-top: 12px; padding-top: 8px; }
  summary { cursor: pointer; color: var(--dim); font-size: 12.5px; }
  .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .stats { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
  .stat { background: var(--panel2); border: 1px solid var(--border); border-radius: 8px;
          padding: 6px 12px; }
  .stat b { font-size: 17px; }
  .stat span { color: var(--dim); font-size: 12px; display: block; }
  #overlayWrap { position: relative; overflow: auto; max-height: 72vh;
                 border: 1px solid var(--border); border-radius: 8px; background: #000; }
  #overlay { display: block; max-width: 100%; height: auto; cursor: crosshair; }
  #hist { max-width: 100%; border-radius: 8px; margin-top: 12px; }
  table { border-collapse: collapse; width: 100%; font-size: 12.5px; margin-top: 8px; }
  th, td { padding: 5px 8px; border-bottom: 1px solid var(--border); text-align: right;
           white-space: nowrap; }
  th { color: var(--dim); position: sticky; top: 0; background: var(--panel); }
  td:first-child, th:first-child { text-align: left; }
  tr.acc td:first-child { color: var(--accent); }
  tr.rej td:first-child { color: var(--red); }
  tr:hover { background: var(--panel2); cursor: pointer; }
  tr.sel { background: #29405c !important; }
  .tableWrap { max-height: 46vh; overflow: auto; margin-top: 8px; }
  #detail { margin-top: 12px; padding: 10px 12px; background: var(--panel2);
            border: 1px solid var(--border); border-radius: 8px; font-size: 12.5px;
            display: none; }
  #detail table td { border: 0; padding: 2px 8px; }
  .err { color: var(--red); margin-top: 10px; font-size: 13px; }
  .legend { color: var(--dim); font-size: 12px; margin-top: 8px; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 3px;
         margin: 0 4px 0 10px; vertical-align: -1px; }
</style>
</head>
<body>
<header>
  <h1>ore-sizer</h1>
  <div class="sub">ToF 矿石粒径测量 — 办公室验证页面(合成 / 上传 / DS63 相机)</div>
</header>
<main>
  <div>
    <div class="card">
      <h2>1 · 合成场景(无需相机)</h2>
      <div class="row2">
        <div><label>随机种子</label><input type="number" id="seed" value="1"></div>
        <div><label>矿石数</label><input type="number" id="n_rocks" value="14"></div>
      </div>
      <label>堆叠系数 0~1</label><input type="number" id="stacking" value="0.35" step="0.05">
      <button id="btnSyn">生成并处理</button>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>2 · 上传帧文件</h2>
      <label>.npz(depth+内参 或 height) / 16bit PNG 深度</label>
      <input type="file" id="file" accept=".npz,.npy,.png,.tif,.tiff">
      <button id="btnUpload" class="secondary">上传并处理</button>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>3 · DS63 相机取流</h2>
      <div class="legend">需安装 ScepterSDK 并设置 SCEPTER_SDK_DIR</div>
      <button id="btnCam" class="secondary">抓一帧并处理</button>
    </div>
    <div class="card" style="margin-top:14px">
      <details open>
        <summary>算法参数(留空用默认)</summary>
        <div class="row2">
          <div><label>hmax_delta_mm</label><input type="text" id="hmax_delta_mm" placeholder=""></div>
          <div><label>drop_delta_mm</label><input type="text" id="drop_delta_mm" placeholder=""></div>
          <div><label>free_ratio_min</label><input type="text" id="free_ratio_min" placeholder=""></div>
          <div><label>ellipse_iou_min</label><input type="text" id="ellipse_iou_min" placeholder=""></div>
          <div><label>solidity_min</label><input type="text" id="solidity_min" placeholder=""></div>
        </div>
      </details>
      <div class="err" id="err"></div>
    </div>
  </div>

  <div>
    <div class="card">
      <div class="stats" id="stats"></div>
      <div id="overlayWrap"><img id="overlay" alt="等待输入…"></div>
      <div class="legend">
        轮廓: <span class="dot" style="background:#3cdc3c"></span>接受
        <span class="dot" style="background:#eb3232"></span>拒绝
        · 边界分类: <span class="dot" style="background:#5ae65a"></span>自由
        <span class="dot" style="background:#ff7800"></span>遮挡
        <span class="dot" style="background:#c800c8"></span>无效
        · <span class="dot" style="background:#00c8ff"></span>最大Feret线
        · 点击矿石或表格行查看详情
      </div>
      <div id="detail"></div>
      <div class="tableWrap"><table id="tbl"></table></div>
      <img id="hist" alt="">
    </div>
  </div>
</main>
<script>
let CUR = null;   // 当前会话 payload

function cfgFields(fd) {
  for (const k of ["hmax_delta_mm","drop_delta_mm","free_ratio_min",
                   "ellipse_iou_min","solidity_min"]) {
    const v = document.getElementById(k).value.trim();
    if (v) fd.append(k, v);
  }
}

async function call(url, fd, btn) {
  const err = document.getElementById("err");
  err.textContent = "";
  btn.disabled = true;
  const old = btn.textContent; btn.textContent = "处理中…";
  try {
    const resp = await fetch(url, {method: "POST", body: fd});
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || resp.statusText);
    render(data);
  } catch (e) {
    err.textContent = e.message;
  } finally {
    btn.disabled = false; btn.textContent = old;
  }
}

document.getElementById("btnSyn").onclick = (ev) => {
  const fd = new FormData();
  fd.append("seed", document.getElementById("seed").value);
  fd.append("n_rocks", document.getElementById("n_rocks").value);
  fd.append("stacking", document.getElementById("stacking").value);
  cfgFields(fd);
  call("/api/synthetic", fd, ev.target);
};
document.getElementById("btnUpload").onclick = (ev) => {
  const f = document.getElementById("file").files[0];
  if (!f) { document.getElementById("err").textContent = "请先选择文件"; return; }
  const fd = new FormData();
  fd.append("file", f);
  cfgFields(fd);
  call("/api/upload", fd, ev.target);
};
document.getElementById("btnCam").onclick = (ev) => {
  const fd = new FormData();
  cfgFields(fd);
  call("/api/camera", fd, ev.target);
};

function render(d) {
  CUR = d;
  const s = document.getElementById("stats");
  s.innerHTML = `
    <div class="stat"><b>${d.n_accepted}</b><span>接受</span></div>
    <div class="stat"><b>${d.n_candidates - d.n_accepted}</b><span>拒绝</span></div>
    <div class="stat"><b>${d.n_debris}</b><span>碎料区</span></div>
    <div class="stat"><b>${d.dist.d50_mm} mm</b><span>D50</span></div>
    <div class="stat"><b>${d.elapsed_ms} ms</b><span>${d.source}</span></div>`;
  document.getElementById("overlay").src = `/img/${d.result_id}/overlay.png?${Date.now()}`;
  document.getElementById("hist").src = `/img/${d.result_id}/hist.png?${Date.now()}`;
  document.getElementById("detail").style.display = "none";

  const tbl = document.getElementById("tbl");
  let html = `<tr><th>#</th><th>判定</th><th>最大边长mm</th><th>次轴mm</th>
    <th>粒级</th><th>自由边界</th><th>solidity</th><th>椭圆IoU</th><th>拒绝原因</th></tr>`;
  for (const r of d.rocks) {
    html += `<tr class="${r.accepted ? "acc" : "rej"}" data-label="${r.label}">
      <td>#${r.label}</td>
      <td>${r.accepted ? "接受" : "拒绝"}</td>
      <td>${r.max_feret_mm}</td><td>${r.min_rect_w_mm}</td>
      <td>${r.size_bin}</td>
      <td>${(r.free_ratio*100).toFixed(0)}%</td>
      <td>${r.solidity}</td><td>${r.ellipse_iou}</td>
      <td style="text-align:left">${r.reject_text || ""}</td></tr>`;
  }
  tbl.innerHTML = html;
  for (const tr of tbl.querySelectorAll("tr[data-label]"))
    tr.onclick = () => selectRock(parseInt(tr.dataset.label));
}

function selectRock(label) {
  if (!CUR) return;
  const r = CUR.rocks.find(x => x.label === label);
  if (!r) return;
  for (const tr of document.querySelectorAll("#tbl tr[data-label]"))
    tr.classList.toggle("sel", parseInt(tr.dataset.label) === label);
  const det = document.getElementById("detail");
  det.style.display = "block";
  det.innerHTML = `<b>矿石 #${r.label}</b> — ${r.accepted ? "✔ 接受" : "✘ 拒绝: " + r.reject_text}
    <table><tr>
      <td>最大边长 ${r.max_feret_mm} mm</td><td>外接矩形 ${r.min_rect_l_mm}×${r.min_rect_w_mm} mm</td>
      <td>等效直径 ${r.equiv_diameter_mm} mm</td><td>厚度 ${r.thickness_mm} mm</td></tr><tr>
      <td>自由边界 ${(r.free_ratio*100).toFixed(1)}%</td>
      <td>遮挡边界 ${(r.occluded_ratio*100).toFixed(1)}%</td>
      <td>无效边界 ${(r.invalid_ratio*100).toFixed(1)}%</td>
      <td>有效覆盖 ${(r.valid_coverage*100).toFixed(1)}%</td></tr><tr>
      <td>solidity ${r.solidity}</td><td>椭圆IoU ${r.ellipse_iou}</td>
      <td>质心 (${r.centroid_px[0]}, ${r.centroid_px[1]}) px</td><td></td>
    </tr></table>`;
  // 滚动叠加图到该矿石
  const img = document.getElementById("overlay");
  const wrap = document.getElementById("overlayWrap");
  const k = (img.clientWidth / (CUR.frame_shape[1] * CUR.scale)) * CUR.scale;
  wrap.scrollTo({left: r.centroid_px[1] * k - wrap.clientWidth / 2,
                 top: r.centroid_px[0] * k - wrap.clientHeight / 2,
                 behavior: "smooth"});
}

// 点击叠加图: 命中 bbox 最小的矿石
document.getElementById("overlay").onclick = (ev) => {
  if (!CUR) return;
  const img = ev.target;
  const rect = img.getBoundingClientRect();
  const px = (ev.clientX - rect.left) / rect.width * CUR.frame_shape[1];
  const py = (ev.clientY - rect.top) / rect.height * CUR.frame_shape[0];
  let best = null, bestArea = 1e18;
  for (const r of CUR.rocks) {
    const [y0, x0, y1, x1] = r.bbox_px;
    if (py >= y0 && py <= y1 && px >= x0 && px <= x1) {
      const a = (y1 - y0) * (x1 - x0);
      if (a < bestArea) { best = r; bestArea = a; }
    }
  }
  if (best) selectRock(best.label);
};
</script>
</body>
</html>
"""


def main(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    app.run(host=host, port=port, debug=debug, threaded=True)
