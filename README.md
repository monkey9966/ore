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
    与 8 项完整性判定, 每颗输出拒绝原因码;
  - `measure.py` — 最大 Feret 直径/最小外接矩形/等效直径/厚度/体积, 椭圆交叉校验;
  - `stats.py` — 分箱统计、按数量/体积占比、D50;
  - `viz.py` — 结果叠加渲染(接受绿/拒绝红/边界分类着色/Feret线)、分布直方图。
- [x] **合成 ToF 场景生成器** (`ore_sizer/synthetic.py`): 按 DS63 规格仿真
  (噪声/飞点/空洞/碎料床/堆叠), 带每颗矿石尺寸与可见性真值, 无相机即可定量验证;
- [x] **定量验证脚本** (`scripts/run_synthetic_eval.py`): 查准率/查全率/测量误差/分布输出;
- [x] **DS63 相机适配器** (`ore_sizer/camera/scepter.py`): ScepterSDK 连接、
  滤波器按方案配置(关 FillHole/TimeFilter)、抓帧转高度图; `frameio.py` 帧存取。

### 当前指标(合成数据, 20 帧, 堆叠系数 0.35)

| 指标 | 当前 | 目标 | 状态 |
|---|---|---|---|
| 查准率(接受的矿石中真值确实完整可见) | 93.5% | >98% | 需继续调优 |
| 查全率(完整可见矿石中被接受) | 33.3% | >80% | 需继续调优 |
| 最大边长平均误差 | 2.8% | <5% | 达标 |
| 次轴(粒径)平均误差 | 2.5% | <5% | 达标 |

已知问题(下一步调优的切入点):

1. 剩余误收(~3/46)是"矿石掩膜附着邻块/碎料导致尺寸虚大", 椭圆交叉校验
   (`FERET_ELLIPSE_MISMATCH`)刚加入, 阈值待扫描;
2. 查全率低的主因: 堆叠场景欠分割(NO_CANDIDATE ~24%)与形状校验过严
   (BAD_ELLIPSE_FIT), 需要对 `hmax_delta_mm`/`ellipse_iou_min`/`waist_max_ratio`
   做参数扫描, 找查准率>=98% 约束下查全率最优点。

### 未完成(按优先级)

- [ ] **参数扫描调优**: 以合成数据为基准, 网格扫描上述参数, 查准率 98%+ 前提下拉高查全率;
- [ ] **可视化 Web 页面**(办公室验证的主要工具): 生成合成场景/上传帧/相机取流三种输入,
  显示分割叠加图 + 每颗矿石粒径表(含拒绝原因) + 分布直方图, 点击矿石查看详情;
- [ ] **DS63 采集脚本** `scripts/capture_ds63.py`(适配器已写好, 缺命令行入口: 连拍存 .npz);
- [ ] **跨帧去重(Stage F, 产线阶段)**: 编码器触发 + 质心平移匹配 + 多帧中值, 办公室静态验证不需要;
- [ ] **真机联调**: 用真实 DS63 深度帧回放验证 `depth_to_heightframe` 路径(RANSAC 基准面等);
- [ ] **筛分对标与校正系数**(现场阶段)。

## 快速开始

```bash
pip install -r requirements.txt

# 合成数据定量验证(无需相机)
python3 scripts/run_synthetic_eval.py --frames 20
# 结果图保存在 out/eval/frame_*.png (绿=接受, 红=拒绝+原因码)
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
  stats.py            Stage G 统计
  viz.py              可视化渲染
  camera/
    scepter.py        DS63/ScepterSDK 适配器
    frameio.py        帧文件读写(.npz)
scripts/
  run_synthetic_eval.py  定量验证
docs/
  DESIGN.md           完整技术方案
```
