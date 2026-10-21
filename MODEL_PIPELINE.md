# 蓝莓采摘机器人视觉系统 — 模型方案总结

## 当前状态

### 已完成
- **YOLO11n-seg 实例分割**：BSAIL 数据集训练到 epoch 12/50（OpenBayes 容器被关，中断）
  - mask mAP50: 0.643，box mAP50: 0.698
  - 权重在 OpenBayes 容器里，未拉取到本地
- **YOLO26n 检测（ripe_det）**：师兄给的成熟度检测权重，未在 BSAIL 上测试

### 待完成
- **YOLO11n-seg 训练**：继续 OpenBayes 或本地完成 50 epoch
- **YOLO26+YOLO11 串联管线**：检测+成熟度判断+分割整合

---

## 模型方案：YOLO26 + YOLO11 串联

### 架构

```
输入图像 (D435 RGB)
    │
    ▼
┌─────────────────┐
│  YOLO26n        │  ← 2026 最新检测模型（师兄给的 ripe_det 权重）
│  检测 + 成熟度   │     输出：bbox + 类别（ripe/unripe/semi-ripe）
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  YOLO11n-seg    │  ← 2024 分割模型（BSAIL 训练权重）
│  实例分割        │     输出：实例级 mask（每颗蓝莓的像素轮廓）
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  3D 定位         │  ← D435 深度图
│  深度→3D坐标+法线│     输出：[X, Y, Z, nx, ny, nz]
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  采摘策略        │  ← picking_strategy.py
│  TSP + 过滤      │     输出：6DOF 抓取序列
└─────────────────┘
```

### 分工

| 模型 | 年份 | 功能 | 输入 | 输出 | 权重来源 |
|------|------|------|------|------|---------|
| YOLO26n | 2026 | 检测 + 成熟度分类 | RGB 图像 | bbox + ripe/unripe/semi-ripe | 师兄给的 ripe_det |
| YOLO11n-seg | 2024 | 实例分割 | RGB 图像 | 实例级 mask | BSAIL 训练（中断） |
| D435 | - | 深度估计 | 深度图 | 3D 坐标 + 法线 | RealSense SDK |
| picking_strategy | - | 采摘顺序规划 | 3D 坐标 + 成熟度 | 6DOF 抓取序列 | 自研代码 |

### 数据流

1. **YOLO26n** 检测图像中所有蓝莓，给出 bbox 和成熟度
2. **YOLO11n-seg** 对同一图像做实例分割，给出每颗蓝莓的精确 mask
3. **匹配**：用 bbox 和 mask 的 IoU 关联两个模型的结果
4. **D435 深度图**：将 2D mask 投影到 3D，计算每颗蓝莓的 XYZ + 法线
5. **采摘策略**：过滤距离 >1m、未成熟，TSP 排序，输出 6DOF

---

## 为什么这样组合

| 问题 | 解决方案 |
|------|---------|
| YOLO26 只有检测，没有分割 | 用 YOLO11n-seg 补充分割能力 |
| YOLO11 是 2024 老模型 | 等 Ultralytics 发布 YOLO26-seg 后替换 |
| 需要成熟度判断 | YOLO26n（ripe_det）已具备 |
| 需要 3D 抓取点 | D435 深度 + 法线计算 |

---

## 后续计划

### 短期（本周）
1. **完成 YOLO11n-seg 训练**：OpenBayes 或本地跑完 50 epoch
2. **测试 YOLO26n**：在 BSAIL 或 D435 数据上验证检测+成熟度效果
3. **串联管线**：写代码匹配 YOLO26 bbox 和 YOLO11 mask

### 中期（本月）
1. **实地验证**：D435 实拍蓝莓，跑完整管线
2. **论文撰写**：Computers and Electronics in Agriculture 或 IROS 2026

### 长期（下学期）
1. **YOLO26-seg 自训**：YOLO26 backbone + 分割 head，替代 YOLO11n-seg
2. **软体夹爪集成**：对标 CLASP，加力控选择性采摘

---

## 关键文件

- `blueberry_seg/train.py`：YOLO11n-seg 训练
- `blueberry_seg/predict.py`：推理
- `blueberry_seg/localize_recording.py`：3D 定位
- `blueberry_seg/picking_strategy.py`：采摘策略
- `weights/yolo11n-seg.pt`：YOLO11 预训练权重
- `~/Desktop/成熟blueberry_ripe_det_v1_20260723/weights/best.pt`：YOLO26n ripe_det 权重

---

## 参考文献

- BerryNet (Comput. Electron. Agric. 2025)：https://www.kaggle.com/datasets/zhengkunli3969/blueberry-segmentation-with-segment-anything-model
- AgriVision (Nature Sci Data 2025)：https://www.nature.com/articles/s41597-025-06263-3
- CLASP (ICRA 2025)：https://arxiv.org/abs/2609.18051
