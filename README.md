# ore-sizer: ToF 相机传送带矿石粒径测量

用 Vzense DS63 ToF 相机俯拍传送带矿石堆, **只测量轮廓完整可见的矿石**
(被压/被遮挡的自动剔除), 输出每颗矿石的最大边长与粒径分布
(100–200 / 200–300 / 300–400 / 400–500 / ≥500 mm, 100 mm 以下不统计)。

完整技术方案见 **[docs/DESIGN.md](docs/DESIGN.md)**。

## 当前进度

### 已完成

- [x] **核心算法流水线** (`ore_sizer/`), 全部可运行:
  - `preprocess.py` — 深度图→正射高度图(真机路径: 点云 + RANSAC 基准面 + z-buffer 投影)、
    小洞填补、局部基准面(rolling-ball)、凸起图;
  - `segmentation.py` — 前景提取 + 细种子分水岭 + 鞍点层次合并, 碎料(<80mm)单独标记;
  - `completeness.py` — 轮廓逐点边界分类(FREE/OCCLUDED/INVALID, 跌落规则+并排腰线规则)
    与 10 项完整性判定(含 `MULTI_PEAK` 表面双峰欠分割、`TOO_THICK` 厚度交叉校验),
    每颗输出拒绝原因码;
  - `measure.py` — 最大 Feret 直径/最小外接矩形/等效直径/厚度/体积, 椭圆+厚度交叉校验;
  - `stats.py` — 分箱统计、按数量/体积占比、D50;
  - `viz.py` — 结果叠加渲染(接受绿/拒绝红/边界分类着色/Feret线)、分布直方图。
- [x] **合成 ToF 场景生成器** (`ore_sizer/synthetic.py`): 按 DS63 规格仿真
  (噪声/飞点/空洞/碎料床/堆叠), 带每颗矿石尺寸与可见性真值, 无相机即可定量验证;
- [x] **定量验证脚本** (`scripts/run_synthetic_eval.py`): 查准率/查全率/测量误差/分布输出;
- [x] **参数扫描调优** (`scripts/param_sweep.py`): 昂贵层(分割/边界参数, 重跑流水线) ×
  便宜层(判定阈值, 记录指标离线重放)两级网格, 扫描集选优 + 独立种子验证集复核,
  联合选优要求两套种子查准率都达标; 当前默认参数即扫描结果;
- [x] **可视化 Web 页面** (`ore_sizer/webapp.py`, `scripts/serve_web.py`):
  合成场景/上传帧/DS63 取流三种输入, 分割叠加图 + 每颗矿石粒径表(含拒绝原因) +
  分布直方图, 点击矿石(图上或表格)查看判定详情, 可在线改关键参数;
- [x] **DS63 相机适配器** (`ore_sizer/camera/scepter.py`): ScepterSDK 连接、
  滤波器按方案配置(关 FillHole/TimeFilter)、抓帧转高度图; `frameio.py` 帧存取;
- [x] **DS63 采集脚本** (`scripts/capture_ds63.py`): 连拍深度帧存 .npz(含内参),
  `--process` 可现场即时跑流水线核对;
- [x] **跨帧去重 Stage F** (`ore_sizer/tracking.py`): 编码器位移预测 + 质心贪心匹配 +
  多帧测量中值 + 离场结算; 仿真皮带流测试 `scripts/test_tracking.py` 通过(0 重复计数)。

### 当前指标(合成数据, 堆叠系数 0.35)

| 指标 | 扫描集(20帧) | 验证集(独立40帧) | 目标 | 状态 |
|---|---|---|---|---|
| 查准率(接受的矿石中真值确实完整可见) | 100% | 100% | >98% | 达标 |
| 查全率(完整可见矿石中被接受) | 33.3% | 24.3% | >80% | 未达标 |
| 最大边长平均误差 | 2.5% | 2.0% | <5% | 达标 |
| 次轴(粒径)平均误差 | 2.2% | — | <5% | 达标 |

按"宁可少测、不能测错"原则, 参数选优先保查准率: 两套种子共 60 帧 0 误收。
查全率是结构性瓶颈, 靠皮带样本量与跨帧去重弥补部分, 后续切入点:

1. **欠分割(NO_CANDIDATE, 约占漏检 1/3)**: 堆叠矿石高差小时分水岭合并成一块,
   现有形状/双峰校验只能拒掉不能分开 —— 长期方向是深度学习实例分割
   (SAM 零样本或小规模微调)替换 Stage B;
2. **边界误判 OCCLUDED(约 1/4)**: 并排接触的腰线规则对深埋碎料床的矿石失效,
   可尝试自适应腰线或按邻居类型(碎料/矿石)分别判定;
3. **形状校验误杀(BAD_ELLIPSE_FIT, 约 1/5)**: `ellipse_iou_min=0.90` 较严,
   但放宽会引入误收(已扫描证实), 需要更精细的欠分割判别器而非单一 IoU 阈值。

### 未完成(按优先级)

- [ ] **真机联调**: 用真实 DS63 深度帧回放验证 `depth_to_heightframe` 路径(RANSAC 基准面等);
- [ ] **查全率提升**(见上文三个切入点);
- [ ] **筛分对标与校正系数**(现场阶段)。

## 快速开始

```bash
pip install -r requirements.txt

# 可视化 Web 页面(办公室验证主入口), 浏览器打开 http://localhost:8000
python3 scripts/serve_web.py --port 8000

# 合成数据定量验证(无需相机)
python3 scripts/run_synthetic_eval.py --frames 20
# 结果图保存在 out/eval/frame_*.png (绿=接受, 红=拒绝+原因码)

# 参数扫描(查准率>=98%约束下查全率选优, 独立种子复核)
python3 scripts/param_sweep.py --frames 30 --val-frames 40

# DS63 连拍采集(需 ScepterSDK, 设置 SCEPTER_SDK_DIR)
python3 scripts/capture_ds63.py --count 10 --out data/capture --process

# 跨帧去重仿真测试
python3 scripts/test_tracking.py
```

单帧处理 API:

```python
from ore_sizer import PipelineConfig, process_frame
from ore_sizer.synthetic import generate_scene
from ore_sizer.stats import compute_distribution

cfg = PipelineConfig()
scene = generate_scene(cfg, seed=1)          # 或 camera/frameio 加载真机帧
result = process_frame(scene.frame, cfg)
for rock in result.accepted_rocks:
    print(rock.label, rock.max_feret_mm, rock.size_bin)
print(compute_distribution(result.rocks, cfg).as_dict())
```

办公室真机验证(需先装 ScepterSDK, 设置环境变量 `SCEPTER_SDK_DIR`):

```python
from ore_sizer.camera.scepter import capture_heightframe
frame = capture_heightframe()   # 自动连接相机、配置滤波器、RANSAC 拟合地面
```

## 目录结构

```
ore_sizer/            算法包(相机无关)
  config.py           全部参数(一处配置)
  types.py            HeightFrame / RockInfo / FrameResult
  synthetic.py        合成场景生成器(带真值)
  preprocess.py       Stage A + 真机深度图转换
  segmentation.py     Stage B 分割
  completeness.py     Stage C/D 完整性判定(核心)
  measure.py          Stage E 测量
  tracking.py         Stage F 跨帧去重(产线)
  stats.py            Stage G 统计
  viz.py              可视化渲染
  webapp.py           可视化 Web 页面(Flask 单页)
  camera/
    scepter.py        DS63/ScepterSDK 适配器
    frameio.py        帧文件读写(.npz)
scripts/
  serve_web.py        启动 Web 页面
  run_synthetic_eval.py  定量验证
  param_sweep.py      参数扫描调优
  capture_ds63.py     DS63 连拍采集
  test_tracking.py    跨帧去重仿真测试
docs/
  DESIGN.md           完整技术方案
```
