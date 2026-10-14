# 蓝莓采摘机器人视觉系统 — 工作进展报告

**日期**：2026-10-13  
**仓库**：https://github.com/userwj123/berryseg-picking-robot

---

## 1. 项目目标

构建完整的蓝莓采摘机器人视觉管线：
1. **2D 实例分割**：YOLO11n-seg 检测蓝莓，输出实例级 mask
2. **3D 定位**：RealSense D435 深度图 → 3D 坐标 + 外法线
3. **采摘策略**：距离过滤 → 生熟过滤 → TSP 顺序 → 6DOF 输出给机械臂

---

## 2. 已完成工作

### 2.1 环境搭建与数据准备

| 项目 | 状态 | 说明 |
|------|------|------|
| berryseg 工程复现 | ✅ | 师兄 zip 解压，理解代码结构 |
| BSAIL 数据集下载 | ✅ | Kaggle 1.93GB，YOLO 格式实例分割 |
| DB-1 数据集转换 | ✅ | 1195 张语义 mask → YOLO 实例格式 |
| 预训练权重 | ✅ | yolo11n-seg.pt (6.2MB) |
| GitHub 仓库初始化 | ✅ | userwj123/berryseg-picking-robot |

### 2.2 代码实现

| 模块 | 文件 | 功能 |
|------|------|------|
| 实例分割训练 | `blueberry_seg/train.py` | YOLO11n-seg 训练，支持 CPU/GPU |
| 实例分割推理 | `blueberry_seg/predict.py` | 图像/视频推理，输出 bbox + polygon |
| 3D 定位 | `blueberry_seg/localize_recording.py` | D435 .db3 回放 → 3D 坐标 + 法线 |
| RGB-D 读取 | `blueberry_seg/rgbd.py` | 免 ROS 读 RealSense SQLite 录像 |
| 采摘策略 | `blueberry_seg/picking_strategy.py` | 距离/生熟过滤 + TSP 顺序 + 6DOF 输出 |
| 枝干分割 | `branch_seg/` | DeepLabV3-MobileNetV3（辅助） |

### 2.3 训练与验证

#### 本地 M4 训练（已中断）
- **配置**：CPU，batch 4，imgsz 384，5 epoch
- **结果**：mask mAP50 0.231（DB-1 派生数据）
- **问题**：BSAIL 数据 2757 张，M4 内存不足，3 epoch 后崩溃

#### OpenBayes RTX 5090 训练（已中断）
- **配置**：GPU，batch 16，imgsz 640，50 epoch
- **进度**：epoch 12/50，mask mAP50 0.643，box mAP50 0.698
- **中断原因**：OpenBayes 30 分钟无活动自动关容器

### 2.4 3D 定位验证

- **数据**：自采 D435 `.db3` 录像（278 帧 RGB + 深度）
- **结果**：成功输出 19 个几何候选，选中目标：
  - 3D 坐标：X=-0.085m, Y=-0.135m, Z=0.503m
  - 外法线：[0.220, -0.272, -0.937]
- **可视化**：`reports/localize_20260902_v3/` 含三联图和接触表

### 2.5 采摘策略实现

**流程**：
1. **距离过滤**：>1.0m 忽略（会议纪要要求 0.5m，文献建议 1.0m 平衡）
2. **生熟过滤**：只保留 `ripe`，`unripe`/`semi_ripe`/`unknown` 永不摘
3. **TSP 排序**：最小化机械臂末端总行程（N≤22 精确 DP，N>22 最近邻+2-opt）
4. **外围优先**：邻居少的先摘（减少碰撞，番茄采摘文献）
5. **输出 6DOF**：`[X, Y, Z, nx, ny, nz]` 相机坐标系，米制 + 单位法线

**文献依据**：
- TSP 用于采摘：Peiper & Sarig (1991)、Kiani et al. (2023)、Zhang et al. (2025)
- 外围优先：Yu et al. (2022) 番茄采摘
- 距离过滤优先：会议纪要 (2026-10-08)

---

## 3. 当前问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| OpenBayes 容器被关 | 30 分钟无活动自动清理 | 用 `screen`/`tmux` 保持会话，或设置 keepalive |
| 本地 M4 训练崩溃 | 内存不足（BSAIL 2757 张 + mask） | 减少 batch size 到 2，imgsz 到 256 |
| `scripts/` 缺失 | 师兄 zip 里本来就没有 | 无法补全，需问师兄 |
| `tests/` 缺失 | 原工程就没有 | 无法补全 |
| Git 缺大文件 | data/runs/reports 太大 | 用 Git LFS 或 release 分发 |

---

## 4. 下一步计划

### 短期（今天）
1. **继续训练**：OpenBayes 新容器，用 `screen` 保持会话，完成 50 epoch
2. **拉取权重**：训练完成后下载 `best.pt` 到本地
3. **本地验证**：用训练好的权重跑 3D 定位和采摘策略

### 中期（本周）
1. **集成 ripe_det**：师兄的成熟度检测模型接入采摘策略
2. **手眼标定**：元宵负责，相机坐标系 → 机械臂基坐标系
3. **实地测试**：真实蓝莓植株，验证完整管线

### 长期（论文）
1. **抓取成功率预测**：小果、成簇、表面蜡质的抓取成功率模型
2. **成簇顺序优化**：摘一个带动其他的链式反应建模
3. **论文产出**：Computers and Electronics in Agriculture 或 ICRA

---

## 5. 文件清单（GitHub 仓库）

```
berryseg-picking-robot/
├── blueberry_seg/          # 实例分割 + 3D 定位 + 采摘策略
│   ├── train.py           # YOLO11n-seg 训练
│   ├── predict.py         # 推理
│   ├── localize_recording.py  # 3D 定位回放
│   ├── picking_strategy.py    # 采摘策略
│   ├── rgbd.py            # RealSense .db3 读取
│   └── ...
├── branch_seg/             # 枝干分割（辅助）
├── configs/                # 训练配置
├── weights/                # 预训练权重（需手动下载）
│   └── pretrained_source.json
├── requirements.txt        # 依赖
└── README.md               # 项目说明
```

---

## 6. 参考资源

- **AgriVision 数据集**：https://www.nature.com/articles/s41597-025-06263-3
- **BSAIL 数据集**：https://www.kaggle.com/datasets/zhengkunli3969/blueberry-segmentation-with-segment-anything-model
- **YOLO11**：https://github.com/ultralytics/ultralytics
- **OpenBayes**：https://openbayes.com（GPU 训练平台）
